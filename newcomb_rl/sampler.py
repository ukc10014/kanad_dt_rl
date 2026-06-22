"""Training prompt sampler (plan: sampler.py).

Draws (NewcombItem x p) with p restricted to the TRAIN split `p_grid \\ holdout_p`, renders
via newcomb_eval.prompts.build_prompt (abstract tokens, p injected, seeded), and attaches the
payoffs + crossover p* the reward needs. Held-out p values are never drawn here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from newcomb_eval.crossover import crossover_for_item
from newcomb_eval.data.loader import load
from newcomb_eval.prompts import build_prompt

from .rl_config import RLOOConfig


@dataclass
class SampledPrompt:
    text: str
    item_id: str
    p: float
    legal_tokens: list
    token_role: dict
    payoff_big: float
    payoff_small: float
    p_star: float
    draw: int


class NewcombSampler:
    """Seeded sampler over (item, train-p). One instance per training run."""

    def __init__(self, cfg: RLOOConfig):
        self.cfg = cfg
        self.items = load(cfg.eval.dataset_path, cfg.eval)
        self.train_p = list(cfg.train_p_grid)
        if not self.train_p:
            raise ValueError("train_p_grid is empty (all p values are held out?)")
        self.rng = random.Random(cfg.seed)
        self._draw = 0
        # Pre-compute crossover (p*, B, S) per item under the configured mode.
        self._xover = {it.id: crossover_for_item(it, cfg.eval.crossover) for it in self.items}

    def _one(self) -> SampledPrompt:
        item = self.rng.choice(self.items)
        p = self.rng.choice(self.train_p)
        draw = self._draw
        self._draw += 1
        rp = build_prompt(
            item, p, self.cfg.eval.prompt,
            sweep_seed=self.cfg.seed, repeat=draw, mode=self.cfg.eval.loader_mode,
        )
        xo = self._xover[item.id]
        return SampledPrompt(
            text=rp.text,
            item_id=item.id,
            p=p,
            legal_tokens=rp.legal_tokens,
            token_role=rp.token_role,
            payoff_big=xo.payoff_big,
            payoff_small=xo.payoff_small,
            p_star=xo.p_star,
            draw=draw,
        )

    def sample(self, n: int) -> list[SampledPrompt]:
        return [self._one() for _ in range(n)]
