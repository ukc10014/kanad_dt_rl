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

    def _render(self, item, p: float) -> SampledPrompt:
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

    def _one(self) -> SampledPrompt:
        return self._render(self.rng.choice(self.items), self.rng.choice(self.train_p))

    def sample(self, n: int) -> list[SampledPrompt]:
        return [self._one() for _ in range(n)]

    def sample_paired(self, n_pairs: int) -> list[SampledPrompt]:
        """EV-balanced paired draw: 2*n_pairs prompts, each consecutive pair the *same item* at a
        p_low < p* and a p_high > p* (Lever 1b).

        This makes each rollout batch balanced below/above the crossover, so a p-independent policy
        earns ~0 net advantage — the only way to win is to *condition on p* (one-box high, two-box
        low). Pairing on the same item isolates the difference to `p`. If an item's train grid has
        no value on one side of p* (degenerate payoff), fall back to two distinct train-p draws.
        """
        out = []
        for _ in range(n_pairs):
            item = self.rng.choice(self.items)
            p_star = self._xover[item.id].p_star
            lows = [p for p in self.train_p if p < p_star]
            highs = [p for p in self.train_p if p > p_star]
            if lows and highs:
                p_low, p_high = self.rng.choice(lows), self.rng.choice(highs)
            else:  # degenerate split — just take two distinct train-p values
                a, b = sorted(self.rng.sample(self.train_p, k=min(2, len(self.train_p))))
                p_low, p_high = a, b
            out.append(self._render(item, p_low))
            out.append(self._render(item, p_high))
        return out
