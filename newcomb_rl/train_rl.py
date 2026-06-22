"""Train one RLOO arm and save its LoRA adapter (plan: train_rl.py).

Examples
--------
    # Smoke (tiny model, few steps):
    python -m newcomb_rl.train_rl --arm causal --model Qwen/Qwen2.5-0.5B-Instruct \
        --steps 12 --K 4 --P 4 --eval-every 6

    # Full arm on the 3B:
    python -m newcomb_rl.train_rl --arm causal     --steps 200
    python -m newcomb_rl.train_rl --arm evidential --steps 200
"""
from __future__ import annotations

import argparse
import dataclasses

from .rl_config import RLOOConfig
from .rloo import NewcombRLOO


def build_cfg(args) -> RLOOConfig:
    cfg = RLOOConfig()
    if args.model:
        cfg.eval = dataclasses.replace(
            cfg.eval, model=dataclasses.replace(cfg.eval.model, model_name=args.model)
        )
    cfg.arm = args.arm
    cfg.reward_mode = args.reward_mode
    for fld in ("lr", "kl_coef", "clip", "lora_rank", "K", "P", "temp", "max_new_tokens",
                "micro", "steps", "minutes", "eval_every", "eval_items", "seed",
                "wandb", "fp32_lora"):
        v = getattr(args, fld, None)
        if v is not None:
            setattr(cfg, fld, v)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train one RLOO arm (causal/evidential)")
    ap.add_argument("--arm", choices=["causal", "evidential"], required=True)
    ap.add_argument("--reward-mode", dest="reward_mode", choices=["ev", "realized"], default="ev")
    ap.add_argument("--model", help="HF model id (default: config Qwen2.5-3B-Instruct)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--kl-coef", dest="kl_coef", type=float)
    ap.add_argument("--clip", type=float)
    ap.add_argument("--lora-rank", dest="lora_rank", type=int)
    ap.add_argument("--K", type=int, help="samples per prompt (RLOO group size)")
    ap.add_argument("--P", type=int, help="prompts per rollout")
    ap.add_argument("--temp", type=float)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    ap.add_argument("--micro", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--eval-every", dest="eval_every", type=int)
    ap.add_argument("--eval-items", dest="eval_items", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fp32-lora", dest="fp32_lora", action="store_true", default=None)
    ap.add_argument("--wandb", action="store_true", default=None)
    args = ap.parse_args(argv)

    cfg = build_cfg(args)
    trainer = NewcombRLOO(cfg)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
