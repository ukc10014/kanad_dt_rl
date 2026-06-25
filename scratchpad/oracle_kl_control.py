"""kl_coef=0 control for the Day-4 causal->one-box flip (reconstructs the lost oracle_anchor.py).

Day-4 found: seed the RL'd two-boxer (`causal` adapter, margin ~ -18) into the evidential RLOO
with the predictor accuracy pinned to p=1.0 (perfect predictor => one-box is EV-optimal), and the
policy *flips* 0.00 -> 1.00 in ~40 steps. But at margin -18 the policy samples one-box with prob
~0, so the *reward* can't be moving an action it never explores. The suspected driver is the
**KL-to-base** reference (base one-boxes at p=1): the KL gradient pushes every token toward base
regardless of sampling, eroding the two-box margin until one-boxing becomes samplable.

This driver isolates that. Same setup, only `kl_coef` varies:
  kl_coef=0.0  still flips (K 0->1)  => reward-driven (Day-4 reading wrong)
  kl_coef=0.0  stalls at K=0         => KL-to-base did it (Day-4 reading confirmed)
The matched baseline kl_coef=0.02 (same driver) must reproduce the documented flip, which
validates the reconstruction against the lost original.

Standalone: imports newcomb_rl, edits nothing under it. Seeding mirrors selfplay._load_into_default.
"""
from __future__ import annotations

import argparse
import dataclasses
import os

from newcomb_rl.rl_config import RLOOConfig
from newcomb_rl.rloo import NewcombRLOO


class SeededRLOO(NewcombRLOO):
    """NewcombRLOO that loads a saved LoRA adapter into the live (trainable) 'default' adapter
    before training — so RL starts from a committed disposition instead of the fresh base."""

    def __init__(self, cfg: RLOOConfig, *, seed_adapter: str | None):
        super().__init__(cfg)
        if seed_adapter:
            self._load_into_default(seed_adapter)

    def _load_into_default(self, adapter_dir: str):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        tgt = next(p for p in self.model.parameters() if p.requires_grad).dtype
        sd = {k: v.to(tgt) for k, v in sd.items()}
        before = self._lora_b_norm()
        set_peft_model_state_dict(self.model, sd, adapter_name="default")
        print(f"[seed] default <- {adapter_dir}: sum|lora_B| {before:.3f} -> "
              f"{self._lora_b_norm():.3f}", flush=True)

    def _lora_b_norm(self) -> float:
        return sum(p.float().norm().item()
                   for n, p in self.model.named_parameters() if "lora_B" in n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-adapter", dest="seed_adapter", required=True)
    ap.add_argument("--p", type=float, default=1.0, help="pinned predictor accuracy")
    ap.add_argument("--kl-coef", dest="kl_coef", type=float, required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = RLOOConfig()
    cfg.reward_mode = "ev"
    cfg.arm = "evidential"
    sweep = dataclasses.replace(cfg.eval.sweep, p_grid=(args.p,), holdout_p=())
    model = cfg.eval.model
    if args.model:
        model = dataclasses.replace(model, model_name=args.model)
    cfg.eval = dataclasses.replace(cfg.eval, sweep=sweep, model=model)
    cfg.steps = args.steps
    cfg.K = args.K
    cfg.P = args.P
    cfg.kl_coef = args.kl_coef
    cfg.eval_every = args.eval_every
    cfg.seed = args.seed
    cfg.tag = args.tag

    print(f"[oracle-klctl] arm=evidential p={args.p} kl_coef={args.kl_coef} "
          f"seed_adapter={args.seed_adapter} model={cfg.model_name} "
          f"steps={args.steps} K={args.K} P={args.P} temp={cfg.temp}", flush=True)
    trainer = SeededRLOO(cfg, seed_adapter=args.seed_adapter)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
