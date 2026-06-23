"""STaR / rejection-sampling SFT — install the p-conditional EV competence (Lever 2).

The intercept-vs-slope finding holds even under a *fair* RL objective (Lever 1b), pointing at a
capability ceiling: the 3B does EV reasoning on *some* items but inconsistently. STaR amplifies the
model's *own* correct traces into consistency — testing "is the competence latent/elicitable?".

Pipeline:
  1. harvest  — sample many CoT traces per (item, p); keep a trace iff its forced answer
                (ModelWrapper.answer_2way) is EV-optimal AND |margin| >= margin_min (a judge-free
                "the reasoning drove the answer" filter, not just lucky-correct). Oversample low p
                (cold-start), balance across p, report yield/p.
  2. train    — SFT a LoRA on (prompt -> CoT + "Answer: <optimal label>"), CE on completion tokens
                only. Reuses rloo's PEFT setup so the adapter is rloo-loadable for the 2x2 RL.
  3. eval     — (external) logprob_sweep / cot_inspect on the saved adapter: did a slope form?

If low-p yield ~ 0 (or no slope forms), escalate to a teacher (local 14B/32B) — not built here.

    python -m newcomb_rl.sft --dry-run            # harvest + yield report, no training
    python -m newcomb_rl.sft --tag star --epochs 3
    python -m newcomb_rl.sft --model Qwen/Qwen2.5-0.5B-Instruct --samples-per-cell 4 --limit 2 --dry-run
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import random
from dataclasses import dataclass, field

from newcomb_eval.crossover import crossover_for_item
from newcomb_eval.data.loader import load
from newcomb_eval.logprob_sweep import roles_to_tokens
from newcomb_eval.prompts import build_prompt
from newcomb_eval.scorer import ROLE_CDT, ROLE_NON_CDT

from .reward import ARM_EVIDENTIAL, optimal_role
from .rl_config import RLOOConfig

ANSWER_CUE = "\nAnswer:"


@dataclass
class Example:
    item_id: str
    p: float
    prompt: str          # the rendered user-prompt text (pre chat-template)
    cot: str             # the sampled reasoning
    answer_token: str    # the EV-optimal label for this rendering
    role: str            # optimal role (non_cdt | cdt)
    margin: float


def harvest(cfg: RLOOConfig, wrapper, *, samples_per_cell: int = 8, temperature: float = 0.8,
            margin_min: float = 1.0, max_new_tokens: int = 160, oversample_low: int = 2,
            balance: bool = True, seed0: int = 30_000):
    """STaR harvest: keep the model's own correct + confident CoT traces. Returns (examples, report).

    ``report`` maps p -> {"kept": int, "total": int}. Low-p cells get ``oversample_low``x samples
    (the base rarely produces correct two-box traces there).
    """
    items = load(cfg.eval.dataset_path, cfg.eval)
    rng = random.Random(cfg.seed)
    kept: list[Example] = []
    report = {p: {"kept": 0, "total": 0} for p in cfg.train_p_grid}

    for item in items:
        xo = crossover_for_item(item, cfg.eval.crossover)
        B, S, p_star = xo.payoff_big, xo.payoff_small, xo.p_star
        for p in cfg.train_p_grid:
            optimal = optimal_role(ARM_EVIDENTIAL, p, B, S)
            n = samples_per_cell * (oversample_low if p < p_star else 1)
            for s in range(n):
                rp = build_prompt(item, p, dataclasses.replace(cfg.eval.prompt, cot=True),
                                  sweep_seed=seed0, repeat=s, mode=cfg.eval.loader_mode)
                non_cdt, cdt = roles_to_tokens(rp.token_role)
                cot = wrapper.generate_one(rp.text, max_new_tokens=max_new_tokens, temperature=temperature)
                ans = wrapper.answer_2way(wrapper._format(rp.text) + cot + ANSWER_CUE, non_cdt, cdt)
                role = ROLE_NON_CDT if ans["is_k"] else ROLE_CDT
                report[p]["total"] += 1
                # keep only correct AND confident (margin), and not a degenerate empty CoT
                if role == optimal and abs(ans["margin"]) >= margin_min and cot.strip():
                    report[p]["kept"] += 1
                    kept.append(Example(item.id, p, rp.text, cot,
                                        non_cdt if optimal == ROLE_NON_CDT else cdt, optimal,
                                        round(ans["margin"], 2)))

    if balance:
        kept = _balance_across_p(kept, rng)
    return kept, report


def _balance_across_p(examples, rng) -> list[Example]:
    """Downsample each p bucket to the smallest non-empty bucket (uniform pressure across p)."""
    by_p = collections.defaultdict(list)
    for e in examples:
        by_p[e.p].append(e)
    if not by_p:
        return []
    k = min(len(v) for v in by_p.values())
    out = []
    for v in by_p.values():
        rng.shuffle(v)
        out.extend(v[:k])
    rng.shuffle(out)
    return out


def yield_table(report) -> str:
    lines = ["  p     kept/total  rate"]
    for p in sorted(report):
        k, t = report[p]["kept"], report[p]["total"]
        lines.append(f"  {p:.2f}  {k:3d}/{t:<3d}    {(k/t if t else 0):.2f}")
    return "\n".join(lines)


class NewcombSFT:
    """LoRA SFT trainer; PEFT setup mirrors NewcombRLOO so the adapter is rloo-loadable."""

    def __init__(self, cfg: RLOOConfig):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for _mod in ("peft.import_utils", "peft.tuners.lora.model"):  # neutralise broken bnb
            try:
                _m = __import__(_mod, fromlist=["x"])
                for fn in ("is_bnb_available", "is_bnb_4bit_available"):
                    if hasattr(_m, fn):
                        setattr(_m, fn, lambda *a, **k: False)
            except Exception:
                pass

        self.cfg = cfg
        self.torch = torch
        self.device = torch.device("cuda")
        torch.manual_seed(cfg.seed)
        self.is_instruct = "instruct" in cfg.model_name.lower()
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16, attn_implementation="sdpa").to(self.device)
        base.config.pad_token_id = self.tok.pad_token_id
        lora = LoraConfig(r=cfg.lora_rank, lora_alpha=2 * cfg.lora_rank, lora_dropout=0.0,
                          target_modules=list(cfg.lora_targets))
        self.model = get_peft_model(base, lora)
        self._use_ac = cfg.fp32_lora
        if self._use_ac:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()

    def _wrap(self, prompt: str) -> str:
        if self.is_instruct and getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return prompt + "\n"

    def _example_loss(self, ex: Example):
        """CE on the completion tokens only (prompt masked). Memory-safe via logsumexp."""
        torch = self.torch
        prompt_text = self._wrap(ex.prompt)
        full_text = prompt_text + ex.cot + ANSWER_CUE + " " + ex.answer_token
        p_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        f_ids = self.tok(full_text, add_special_tokens=False).input_ids
        if len(f_ids) <= len(p_ids):
            return None
        ids = torch.tensor([f_ids], device=self.device)
        ac = torch.autocast("cuda", dtype=torch.bfloat16) if self._use_ac else _null()
        with ac:
            logits = self.model(ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        tok_lp = logits.gather(-1, tgt[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        start = max(len(p_ids) - 1, 0)          # completion tokens are predicted from here on
        comp = tok_lp[start:]
        return -comp.mean()

    def train(self, examples: list[Example], *, epochs: int = 3, lr: float = 1e-4, micro: int = 8):
        torch = self.torch
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.model.train()
        order = list(range(len(examples)))
        for ep in range(epochs):
            random.Random(self.cfg.seed + ep).shuffle(order)
            opt.zero_grad()
            running, n_acc, n_seen = 0.0, 0, 0
            for i, idx in enumerate(order):
                loss = self._example_loss(examples[idx])
                if loss is None:
                    continue
                (loss / micro).backward()
                running += loss.item(); n_acc += 1; n_seen += 1
                if n_acc == micro or i == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad(); n_acc = 0
            print(f"[sft] epoch {ep+1}/{epochs}  mean_loss={running/max(n_seen,1):.3f}  "
                  f"n={n_seen}", flush=True)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        print(f"[sft] saved adapter -> {out_dir}", flush=True)


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="STaR/rejection-sampling SFT (Lever 2)")
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--samples-per-cell", dest="spc", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--margin-min", dest="margin_min", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=160)
    ap.add_argument("--oversample-low", dest="oversample_low", type=int, default=2)
    ap.add_argument("--no-balance", dest="balance", action="store_false", default=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", dest="lora_rank", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--tag", default="star")
    ap.add_argument("--dry-run", action="store_true", help="harvest + yield report only, no training")
    args = ap.parse_args(argv)

    cfg = RLOOConfig()
    if args.model:
        cfg.eval = dataclasses.replace(cfg.eval, model=dataclasses.replace(cfg.eval.model, model_name=args.model))
    if args.limit is not None:
        cfg.eval = dataclasses.replace(cfg.eval, limit=args.limit)
    cfg.lora_rank = args.lora_rank

    from newcomb_eval.model import ModelWrapper
    wrapper = ModelWrapper(cfg.model_name, dtype=cfg.eval.model.dtype,
                           device_map=cfg.eval.model.device_map,
                           use_chat_template=cfg.eval.model.use_chat_template)
    examples, report = harvest(cfg, wrapper, samples_per_cell=args.spc, temperature=args.temperature,
                               margin_min=args.margin_min, max_new_tokens=args.max_new_tokens,
                               oversample_low=args.oversample_low, balance=args.balance)
    print(f"[sft] harvest: {len(examples)} examples kept (balanced={args.balance})")
    print(yield_table(report))
    low = [p for p in cfg.train_p_grid if report[p]["kept"] == 0 and p < crossover_for_item(
        load(cfg.eval.dataset_path, cfg.eval)[0], cfg.eval.crossover).p_star]
    if low:
        print(f"[sft] WARNING cold-start: zero correct traces at low p={low} → consider teacher escalation")

    # persist the harvested examples (raw traces — CLAUDE.md persist-generations)
    out_dir = os.path.abspath(os.path.join(cfg.eval.results_dir, "sft"))
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, f"harvest_{args.tag}.jsonl"), "w") as f:
        for e in examples:
            f.write(json.dumps(dataclasses.asdict(e)) + "\n")

    if args.dry_run:
        print("[sft] --dry-run: stopping after harvest.")
        return 0
    if not examples:
        print("[sft] no examples harvested — aborting train (escalate to a teacher).")
        return 1

    del wrapper
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    trainer = NewcombSFT(cfg)
    trainer.train(examples, epochs=args.epochs, lr=args.lr, micro=args.micro)
    trainer.save(os.path.join(cfg.adapter_dir, f"sft_{args.tag}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
