"""RLOO trainer for the Newcomb p-sweep (plan: rloo.py).

Loop mechanics (LoRA setup, disable_adapter KL reference, chunked log-probs, clipped-surrogate
+ KL learn(), micro-batching, fp32-LoRA) are ADAPTED from David Quarel's on-box GRPO reference
`/root/rlvr_4x4/rlvr.py` (used with permission). The key change is the advantage estimator:
RLOO **leave-one-out** baseline `A_i = r_i - mean_{j!=i} r_j` instead of GRPO group-normalisation,
and the reward/task are Newcomb-specific (newcomb_rl.reward + newcomb_eval seams).
"""
from __future__ import annotations

import os
import random
import time

import torch
import torch.nn.functional as F

from newcomb_eval.scorer import ROLE_CDT, ROLE_NON_CDT, resolve_choice

from .reward import ARM_CAUSAL, MODE_REALIZED, compute_reward
from .rl_config import RLOOConfig
from .sampler import NewcombSampler


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def loo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """RLOO leave-one-out advantage for one group: A_i = r_i - mean_{j!=i} r_j.

    Equivalent to (K/(K-1)) * (r_i - mean). K=1 -> zeros (no baseline available).
    """
    K = rewards.shape[0]
    if K < 2:
        return torch.zeros_like(rewards)
    return (K / (K - 1.0)) * (rewards - rewards.mean())


class NewcombRLOO:
    def __init__(self, cfg: RLOOConfig):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Some shared venvs ship a broken bitsandbytes; we never quantise, so stop peft importing it.
        for _mod in ("peft.import_utils", "peft.tuners.lora.model"):
            try:
                _m = __import__(_mod, fromlist=["x"])
                for fn in ("is_bnb_available", "is_bnb_4bit_available"):
                    if hasattr(_m, fn):
                        setattr(_m, fn, lambda *a, **k: False)
            except Exception:
                pass

        self.cfg = cfg
        self.device = torch.device("cuda")
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        self.B, self.S = cfg.payoffs
        self.sampler = NewcombSampler(cfg)
        self.is_instruct = "instruct" in cfg.model_name.lower()

        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(self.device)
        base.config.pad_token_id = self.tok.pad_token_id

        lora = LoraConfig(
            r=cfg.lora_rank, lora_alpha=2 * cfg.lora_rank, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(base, lora)

        # fp32-LoRA: tiny trainable params in fp32 for stabler AdamW; forward runs bf16 via autocast.
        self._use_ac = cfg.fp32_lora
        if self._use_ac:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr
        )
        self.pad = self.tok.pad_token_id

    # -- prompt encoding ---------------------------------------------------
    def _wrap(self, prompt: str) -> str:
        if self.is_instruct and getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt + "\n"

    def _prompt_ids(self, prompts):
        texts = [self._wrap(p) for p in prompts]
        return self.tok(texts, return_tensors="pt", padding=True,
                        add_special_tokens=not self.is_instruct).to(self.device)

    def _ac(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self._use_ac else _null()

    # -- log-probs ---------------------------------------------------------
    def _chunk_logprobs(self, ids, mask):
        with self._ac():
            logits = self.model(ids, attention_mask=mask).logits[:, :-1]
        lp = F.log_softmax(logits.float(), dim=-1)
        return lp.gather(-1, ids[:, 1:, None]).squeeze(-1)

    @torch.no_grad()
    def _seq_logprobs(self, ids, mask, gen_mask, adapters: bool):
        # reference (adapters=False): disable the LoRA adapter -> base model logprobs (no 2nd copy).
        cm = _null() if adapters else self.model.disable_adapter()
        parts, mb = [], self.cfg.micro
        with cm:
            for i in range(0, ids.shape[0], mb):
                lp = self._chunk_logprobs(ids[i:i + mb], mask[i:i + mb])
                parts.append(lp * gen_mask[i:i + mb, 1:])
        return torch.cat(parts, 0)

    # -- rollout -----------------------------------------------------------
    @torch.no_grad()
    def rollout(self):
        c = self.cfg
        prompts = self.sampler.sample(c.P)
        enc = self._prompt_ids([sp.text for sp in prompts])
        Lp = enc.input_ids.shape[1]
        with self._ac():
            gen = self.model.generate(
                **enc, max_new_tokens=c.max_new_tokens, do_sample=True,
                temperature=c.temp, top_p=c.top_p, num_return_sequences=c.K,
                pad_token_id=self.pad,
            )
        completions = self.tok.batch_decode(gen[:, Lp:], skip_special_tokens=True)

        rewards = torch.zeros(c.P * c.K, dtype=torch.float)
        roles = []
        n_k = n_valid = 0
        for i, sp in enumerate(prompts):
            # shared causal fill latent per prompt-group (realized mode only).
            causal_fill = None
            if c.arm == ARM_CAUSAL and c.reward_mode == MODE_REALIZED:
                causal_fill = self.rng.random() < sp.p
            for j in range(c.K):
                idx = i * c.K + j
                role, _tok, valid = resolve_choice(completions[idx], sp.legal_tokens, sp.token_role)
                roles.append(role)
                rewards[idx] = compute_reward(
                    role, sp.p, sp.payoff_big, sp.payoff_small,
                    arm=c.arm, mode=c.reward_mode, rng=self.rng, causal_fill=causal_fill,
                )
                n_valid += int(valid)
                n_k += int(role == ROLE_NON_CDT)
        rewards = rewards.to(self.device)

        adv = torch.zeros_like(rewards)
        for i in range(c.P):
            adv[i * c.K:(i + 1) * c.K] = loo_advantage(rewards[i * c.K:(i + 1) * c.K])

        mask = (gen != self.pad).long()
        mask[:, :Lp] = enc.attention_mask.repeat_interleave(c.K, 0)
        gen_mask = torch.zeros_like(mask, dtype=torch.float)
        gen_mask[:, Lp:] = (gen[:, Lp:] != self.pad).float()

        old_lp = self._seq_logprobs(gen, mask, gen_mask, adapters=True)
        ref_lp = self._seq_logprobs(gen, mask, gen_mask, adapters=False)
        N = c.P * c.K
        return dict(
            ids=gen, mask=mask, gen_mask=gen_mask, adv=adv, old_lp=old_lp, ref_lp=ref_lp,
            reward=rewards.mean().item(), k_rate=n_k / N, invalid_rate=1.0 - n_valid / N,
            gen_len=gen_mask[:, Lp:].sum(-1).mean().item(),
            sample=(prompts[0].text[-80:], completions[0]),
        )

    # -- learn -------------------------------------------------------------
    def learn(self, batch):
        c = self.cfg
        ids, mask, gen_mask, adv, old_lp, ref_lp = (
            batch[k] for k in ("ids", "mask", "gen_mask", "adv", "old_lp", "ref_lp")
        )
        g = gen_mask[:, 1:]
        total = g.sum().clamp_min(1.0)
        self.opt.zero_grad()
        mb = c.micro
        for i in range(0, ids.shape[0], mb):
            sl = slice(i, i + mb)
            gi = g[sl]
            new_lp = self._chunk_logprobs(ids[sl], mask[sl]) * gi
            adv_tok = adv[sl, None] * gi
            ratio = torch.exp(new_lp - old_lp[sl])
            surr = torch.minimum(ratio * adv_tok,
                                 torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv_tok)
            d = ref_lp[sl] - new_lp
            kl = (torch.exp(d) - d - 1) * gi
            (-(surr.sum() - c.kl_coef * kl.sum()) / total).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], c.grad_clip
        )
        self.opt.step()

    # -- in-loop eval ------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        """Greedy K-rate over the FULL p-grid (incl. held-out p) — the learning-curve signal."""
        from newcomb_eval.prompts import build_prompt

        c = self.cfg
        self.model.eval()
        items = self.sampler.items[: c.eval_items] if c.eval_items else self.sampler.items
        grid = c.eval.sweep.p_grid
        by_p = {p: [0, 0, 0] for p in grid}  # p -> [n_k, n_valid, n_total]
        for p in grid:
            for i in range(0, len(items), c.P):
                chunk = items[i:i + c.P]
                rps = [build_prompt(it, p, c.eval.prompt, sweep_seed=10_000, repeat=0,
                                    mode=c.eval.loader_mode) for it in chunk]
                enc = self._prompt_ids([rp.text for rp in rps])
                with self._ac():
                    gen = self.model.generate(**enc, max_new_tokens=c.max_new_tokens,
                                              do_sample=False, pad_token_id=self.pad)
                comp = self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                for rp, ct in zip(rps, comp):
                    role, _t, valid = resolve_choice(ct, rp.legal_tokens, rp.token_role)
                    by_p[p][2] += 1
                    by_p[p][1] += int(valid)
                    by_p[p][0] += int(role == ROLE_NON_CDT)
        self.model.train()
        out = {}
        for p, (nk, nv, nt) in by_p.items():
            out[p] = dict(k_rate=(nk / nv if nv else float("nan")),
                          invalid_rate=1 - nv / nt if nt else float("nan"))
        return out

    # -- train -------------------------------------------------------------
    def train(self):
        c = self.cfg
        wb = None
        if c.wandb:
            try:
                import wandb
                wb = wandb.init(project=c.wandb_project, name=f"{c.arm}-{c.model_name.split('/')[-1]}",
                                config=vars(c), reinit=True)
            except Exception as e:
                print(f"wandb init failed: {e}", flush=True)

        def log_eval(step):
            ev = self.evaluate()
            mean_k = sum(v["k_rate"] for v in ev.values()) / len(ev)
            lo = ev[min(ev)]["k_rate"]
            hi = ev[max(ev)]["k_rate"]
            print(f"[{c.arm}] step={step} eval mean_K={mean_k:.3f}  "
                  f"K@p={min(ev):.2f}->{lo:.2f}  K@p={max(ev):.2f}->{hi:.2f}  "
                  f"slope(hi-lo)={hi-lo:+.2f}", flush=True)
            if wb:
                wb.log({"eval/mean_k": mean_k, "eval/k_lo": lo, "eval/k_hi": hi,
                        "eval/slope": hi - lo,
                        **{f"eval/k_p{p}": v["k_rate"] for p, v in ev.items()}}, step=step)
            return ev

        print(f"[{c.arm}] model={c.model_name} reward_mode={c.reward_mode} "
              f"train_p={self.sampler.train_p} p*={self.sampler.items[0].id and self.sampler._xover[self.sampler.items[0].id].p_star}",
              flush=True)
        log_eval(0)

        t0 = time.time()
        step = 0
        while True:
            if c.minutes > 0 and (time.time() - t0) >= c.minutes * 60:
                break
            if c.minutes == 0 and step >= c.steps:
                break
            step += 1
            self.model.train()
            batch = self.rollout()
            self.learn(batch)
            if step % max(1, c.eval_every) == 0 or step == 1:
                print(f"[{c.arm}] step={step} train reward={batch['reward']:.2f} "
                      f"K={batch['k_rate']:.2f} invalid={batch['invalid_rate']:.2f} "
                      f"gen_len={batch['gen_len']:.1f}", flush=True)
                if wb:
                    wb.log({"train/reward": batch["reward"], "train/k_rate": batch["k_rate"],
                            "train/invalid": batch["invalid_rate"]}, step=step)
            if step % max(1, c.eval_every) == 0:
                log_eval(step)

        final = log_eval(step)
        out_dir = c.arm_dir()
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        print(f"[{c.arm}] saved adapter -> {out_dir}", flush=True)
        if wb:
            wb.finish()
        return final, out_dir
