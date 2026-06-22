"""Reward functions — the heart of the experiment (plan: "the two reward arms").

The "predictor of accuracy p" is *simulated* by the reward itself: the model chooses, then we
retro-fit a prediction to the realized choice. The selection-dependent payoff IS the oracle.
The two arms are the two horns of the Newcomb paradox written as reward functions:

  evidential (EDT): fill correlates with the realized action — Bernoulli(p) if one-box,
                    Bernoulli(1-p) if two-box. EV: one-box -> p*B, two-box -> S+(1-p)*B.
                    Optimum = track-p (one-box iff p > p*).
  causal    (CDT):  fill is sealed, drawn INDEPENDENT of this action. Two-boxing then dominates
                    by exactly +S for any fill. EV: one-box -> q*B, two-box -> q*B + S.
                    Optimum = always two-box, for ANY q.

Notes
-----
* The causal baseline q*B is action-independent, so it cancels under RLOO's leave-one-out
  baseline and does not affect the gradient; it is included only so reward magnitudes are
  comparable across arms. Default q = p (the stated accuracy, applied to a fixed reference).
  The CDT optimum (two-box) is robust to the exact value of q.
* Invalid completion -> reward 0 (a floor; never coerced to a choice). Tracked separately.
"""
from __future__ import annotations

import random

from newcomb_eval.crossover import ev_cdt, ev_non_cdt
from newcomb_eval.scorer import ROLE_CDT, ROLE_INVALID, ROLE_NON_CDT

ARM_EVIDENTIAL = "evidential"
ARM_CAUSAL = "causal"
MODE_EV = "ev"            # deterministic expected value (low variance) — default
MODE_REALIZED = "realized"  # stochastic realized points (ablation)

INVALID_REWARD = 0.0


# ----------------------------------------------------------------------------- EV (deterministic)
def evidential_reward_ev(role: str, p: float, B: float, S: float) -> float:
    """EV reward, evidential arm. one-box -> p*B ; two-box -> S+(1-p)*B ; invalid -> 0."""
    if role == ROLE_NON_CDT:
        return ev_non_cdt(p, B, S)
    if role == ROLE_CDT:
        return ev_cdt(p, B, S)
    return INVALID_REWARD


def causal_reward_ev(role: str, p: float, B: float, S: float, q: float | None = None) -> float:
    """EV reward, causal arm. one-box -> q*B ; two-box -> q*B + S ; invalid -> 0.

    q (action-independent fill prob) defaults to p. Two-box dominates by +S for any q.
    """
    q = p if q is None else q
    if role == ROLE_NON_CDT:
        return q * B
    if role == ROLE_CDT:
        return q * B + S
    return INVALID_REWARD


# ----------------------------------------------------------------------------- realized (stochastic)
def realized_points(role: str, fill_full: bool, B: float, S: float) -> float:
    """Realized points given whether the big box is full and the chosen action.

    one-box: B if full else 0.  two-box: (B if full else 0) + S.  invalid: 0.
    """
    if role == ROLE_INVALID:
        return INVALID_REWARD
    big = B if fill_full else 0.0
    return big + (S if role == ROLE_CDT else 0.0)


def evidential_fill_prob(role: str, p: float) -> float:
    """P(big box full) under the evidential arm, conditioned on the realized action."""
    if role == ROLE_NON_CDT:
        return p
    if role == ROLE_CDT:
        return 1.0 - p
    return 0.0


# ----------------------------------------------------------------------------- dispatcher
def compute_reward(
    role: str,
    p: float,
    B: float,
    S: float,
    *,
    arm: str = ARM_EVIDENTIAL,
    mode: str = MODE_EV,
    rng: random.Random | None = None,
    causal_fill: bool | None = None,
    q: float | None = None,
) -> float:
    """Single-sample reward.

    EV mode: deterministic expected value (default).
    Realized mode: stochastic points.
      - evidential: fill ~ Bernoulli(evidential_fill_prob(role, p)) drawn from ``rng``.
      - causal:     fill = ``causal_fill`` (a shared per-prompt-group latent the caller samples
                    once, independent of the action). Falls back to Bernoulli(q or p) via rng.
    """
    if mode == MODE_EV:
        if arm == ARM_EVIDENTIAL:
            return evidential_reward_ev(role, p, B, S)
        if arm == ARM_CAUSAL:
            return causal_reward_ev(role, p, B, S, q=q)
        raise ValueError(f"unknown arm {arm!r}")

    if mode == MODE_REALIZED:
        if role == ROLE_INVALID:
            return INVALID_REWARD
        if arm == ARM_EVIDENTIAL:
            r = rng or random
            fill = r.random() < evidential_fill_prob(role, p)
        elif arm == ARM_CAUSAL:
            if causal_fill is None:
                r = rng or random
                causal_fill = r.random() < (p if q is None else q)
            fill = causal_fill
        else:
            raise ValueError(f"unknown arm {arm!r}")
        return realized_points(role, fill, B, S)

    raise ValueError(f"unknown reward mode {mode!r}")


def optimal_role(arm: str, p: float, B: float, S: float) -> str:
    """The EV-optimal role for an arm at (p, B, S) — used by tests and eval metrics.

    evidential -> non_cdt iff p above the crossover, else cdt.  causal -> always cdt.
    """
    if arm == ARM_CAUSAL:
        return ROLE_CDT
    one = evidential_reward_ev(ROLE_NON_CDT, p, B, S)
    two = evidential_reward_ev(ROLE_CDT, p, B, S)
    return ROLE_NON_CDT if one > two else ROLE_CDT
