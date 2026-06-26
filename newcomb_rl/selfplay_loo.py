"""A2 — self-consistency RLOO: the "predictor" is the chooser's OWN rollout samples (no re-thinking).

The lagged reasoning-predictor (selfplay_cot) had to GENERATE a fresh <think> chain per prompt — the
pass that got truncated 60-90% of the time at the 2048 budget (the confound). A2 removes it entirely:

  p_eff(prompt) = the LEAVE-ONE-OUT one-box rate among the chooser's K reasoned samples for that prompt.

i.e. the box-fill accuracy = how often the policy *actually* one-boxes here, estimated from the samples
we already generated. No second model, no second generation, nothing to truncate. The chooser still
reasons (that's where p-tracking lives); we just TALLY its resolved actions. Leave-one-out (per sample,
exclude itself) keeps a sample from grading itself. Reward is the usual evidential EV at that p_eff:
  one-box -> p_eff*B ;  two-box -> S + (1-p_eff)*B.

Trade-off vs selfplay_cot: drops the *lag* (predictor = current self, not a stale snapshot) — fine for
the basic "does the conditional rule stabilise?" question; the lag is only needed for the hysteresis
variant. No snapshot/seed-ref adapters needed. KL-to-base (inherited). Chat template forced on for R1.
No edits to rloo.py / reward.py / selfplay.py. Same m3 binding framing + coarse-grid defaults.
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import torch

from newcomb_eval.scorer import ROLE_INVALID, ROLE_NON_CDT, resolve_choice

from .reward import ARM_EVIDENTIAL, compute_reward
from .rl_config import RLOOConfig
from .rloo import NewcombRLOO, loo_advantage

P_STAR_DEFAULT = 0.8


class SelfConsistRLOO(NewcombRLOO):
    """RLOO where the evidential accuracy p_eff is the chooser's own leave-one-out one-box rate."""

    def __init__(self, cfg: RLOOConfig, *, seed_adapter: str | None = None):
        cfg.arm = ARM_EVIDENTIAL          # reward dispatch; p_eff supplied per-sample below
        super().__init__(cfg)
        if getattr(self.tok, "chat_template", None):
            self.is_instruct = True       # R1's name lacks 'instruct' → force the chat template on
        if seed_adapter:
            self._load_into_default(seed_adapter)

    def _load_into_default(self, adapter_dir: str):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        tgt = next(p for p in self.model.parameters() if p.requires_grad).dtype
        set_peft_model_state_dict(self.model, {k: v.to(tgt) for k, v in sd.items()}, adapter_name="default")
        print(f"[seed] default <- {adapter_dir}", flush=True)

    # -- rollout: identical plumbing to rloo.rollout, but p_eff = leave-one-out self-consistency ----
    @torch.no_grad()
    def rollout(self):
        c = self.cfg
        prompts = self.sampler.sample(c.P)
        enc = self._prompt_ids([sp.text for sp in prompts])
        Lp = enc.input_ids.shape[1]
        with self._ac():
            gen = self.model.generate(
                **enc, max_new_tokens=c.max_new_tokens, do_sample=True, temperature=c.temp,
                top_p=c.top_p, num_return_sequences=c.K, pad_token_id=self.pad)
        completions = self.tok.batch_decode(gen[:, Lp:], skip_special_tokens=True)

        cot = c.eval.prompt.cot
        cot_roles = None
        if cot:  # forced-'Answer:' read of each completion's action (truncation-robust; we only need the label)
            ptxt = [sp.text for sp in prompts for _ in range(c.K)]
            legl = [sp.legal_tokens for sp in prompts for _ in range(c.K)]
            rolm = [sp.token_role for sp in prompts for _ in range(c.K)]
            cot_roles = self._forced_answer_roles(ptxt, completions, legl, rolm)

        # 1) resolve every sample's action
        roles, valids = [], []
        for i, sp in enumerate(prompts):
            for j in range(c.K):
                idx = i * c.K + j
                if cot:
                    role = cot_roles[idx]; valid = role != ROLE_INVALID
                else:
                    role, _t, valid = resolve_choice(completions[idx], sp.legal_tokens, sp.token_role, cot=False)
                roles.append(role); valids.append(valid)

        # 2) p_eff per prompt = LEAVE-ONE-OUT one-box rate among the chooser's own K samples
        rewards = torch.zeros(c.P * c.K, dtype=torch.float)
        peff_means = []
        for i, sp in enumerate(prompts):
            grp = roles[i * c.K:(i + 1) * c.K]
            m = sum(r == ROLE_NON_CDT for r in grp)            # one-box count among K
            peff_means.append(m / c.K)
            for j in range(c.K):
                idx = i * c.K + j
                loo = (m - int(grp[j] == ROLE_NON_CDT)) / (c.K - 1) if c.K > 1 else (m / c.K)
                rewards[idx] = compute_reward(grp[j], loo, sp.payoff_big, sp.payoff_small,
                                              arm=ARM_EVIDENTIAL, mode=c.reward_mode)
        n_k = sum(r == ROLE_NON_CDT for r in roles); n_valid = sum(valids); N = c.P * c.K
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
        return dict(
            ids=gen, mask=mask, gen_mask=gen_mask, adv=adv, old_lp=old_lp, ref_lp=ref_lp,
            reward=rewards.mean().item(), k_rate=n_k / N, invalid_rate=1.0 - n_valid / N,
            gen_len=gen_mask[:, Lp:].sum(-1).mean().item(),
            p_model_mean=sum(peff_means) / len(peff_means),   # mean self-consistency one-box rate
            sample=(prompts[0].text[-80:], completions[0]))


# --------------------------------------------------------------------------- CLI
def build_cfg(args) -> RLOOConfig:
    cfg = RLOOConfig()
    cfg.reward_mode = "ev"
    model = dataclasses.replace(cfg.eval.model, model_name=args.model)
    grid = list(args.p_grid) if args.p_grid else list(cfg.eval.sweep.p_grid)
    if args.drop_pstar:
        grid = [p for p in grid if abs(p - P_STAR_DEFAULT) > 1e-9]
    sweep = dataclasses.replace(cfg.eval.sweep, p_grid=tuple(grid), holdout_p=())
    prompt = dataclasses.replace(cfg.eval.prompt, cot=True)
    cfg.eval = dataclasses.replace(cfg.eval, sweep=sweep, model=model, prompt=prompt,
                                   dataset_path=args.dataset, category_filter=None)
    cfg.steps = args.steps; cfg.K = args.K; cfg.P = args.P; cfg.temp = args.temp
    cfg.max_new_tokens = args.max_new_tokens; cfg.micro = args.micro
    if args.kl_coef is not None:
        cfg.kl_coef = args.kl_coef
    cfg.eval_every = args.eval_every; cfg.eval_items = args.eval_items
    cfg.seed = args.seed; cfg.tag = args.tag
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A2 self-consistency RLOO (predictor = chooser's own samples)")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dataset", default="newcomb_eval/data/dataset_mech_m3.json")
    ap.add_argument("--seed-adapter", dest="seed_adapter", default=None,
                    help="optional LoRA dir to seed the policy from (hysteresis); default = base R1.")
    ap.add_argument("--drop-pstar", dest="drop_pstar", action="store_true")
    ap.add_argument("--p-grid", dest="p_grid", type=float, nargs="+", default=None)
    ap.add_argument("--kl-coef", dest="kl_coef", type=float, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2560)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=10)
    ap.add_argument("--eval-items", dest="eval_items", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    cfg = build_cfg(args)
    print(f"[selfplay-loo] A2 self-consistency | model={cfg.model_name} dataset={args.dataset} "
          f"grid={cfg.eval.sweep.p_grid} K={cfg.K} P={cfg.P} mnt={cfg.max_new_tokens} "
          f"kl={cfg.kl_coef} seed_adapter={args.seed_adapter} tag={args.tag}", flush=True)
    trainer = SelfConsistRLOO(cfg, seed_adapter=args.seed_adapter)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
