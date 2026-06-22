"""RLOO training config (plan: rl_config.py).

Composes the existing newcomb_eval EvalConfig (dataset/model/prompt/crossover/sweep) with
RL-specific hyperparameters. The training p-grid is `p_grid \\ holdout_p` — the held-out p
values are reserved for the generalisation probe and never trained on (PLAN.md §3).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from newcomb_eval.config import EvalConfig

from .reward import ARM_EVIDENTIAL, MODE_EV

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RLOOConfig:
    eval: EvalConfig = field(default_factory=EvalConfig)

    # --- experiment arm ---
    arm: str = ARM_EVIDENTIAL          # "evidential" | "causal"
    reward_mode: str = MODE_EV          # "ev" | "realized"

    # --- RLOO / optimisation (defaults follow common single-GPU LoRA-RL practice) ---
    lr: float = 1e-4
    kl_coef: float = 0.02
    clip: float = 0.2
    lora_rank: int = 16
    K: int = 8                          # samples per prompt (RLOO group size)
    P: int = 8                          # prompts per rollout
    temp: float = 1.0                   # rollout sampling temperature
    top_p: float = 0.95
    max_new_tokens: int = 5             # forced-choice: a label + a little slack
    micro: int = 16                     # micro-batch for logprob/backward passes
    grad_clip: float = 1.0
    fp32_lora: bool = True              # keep tiny LoRA params in fp32 for stabler updates

    # --- loop budget / cadence ---
    steps: int = 200                    # rollout->learn iterations (0 => use minutes)
    minutes: float = 0.0                # wall-clock budget (0 => use steps)
    eval_every: int = 25                # steps between in-loop evals
    eval_items: int | None = None       # cap items in the in-loop eval (None => all)
    seed: int = 0

    # --- io ---
    adapter_dir: str = os.path.join(_PKG_DIR, "..", "results", "adapters")
    wandb: bool = False
    wandb_project: str = "newcomb-rloo"

    # ---- derived ----
    @property
    def train_p_grid(self) -> tuple[float, ...]:
        hold = set(self.eval.sweep.holdout_p)
        return tuple(p for p in self.eval.sweep.p_grid if p not in hold)

    @property
    def model_name(self) -> str:
        return self.eval.model.model_name

    @property
    def payoffs(self) -> tuple[float, float]:
        c = self.eval.crossover
        return c.payoff_big, c.payoff_small

    def arm_dir(self) -> str:
        return os.path.join(self.adapter_dir, self.arm)
