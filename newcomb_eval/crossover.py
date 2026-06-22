"""Theoretical crossover reference for opaque Newcomb (PLAN.md §4).

Payoff symbols
--------------
    B = payoff_big   : the conditional prize, present iff the predictor forecast one-box.
    S = payoff_small : the guaranteed prize, present regardless of choice.
    p = predictor accuracy in [0, 1] (probability it correctly forecasts your choice).

Expected value under the EDT calculation (the agent conditions on its choice, treating
the predictor's forecast as correlated with that choice with probability p):

    one-box  (non-CDT):  you receive B iff the predictor forecast one-box (prob p).
        EV_nonCDT(p) = p * B

    two-box  (CDT):      you always get S; you also get B iff the predictor *mis*-forecast
                         one-box while you took both (prob 1 - p).
        EV_CDT(p)    = S + (1 - p) * B

Crossover p* solves EV_nonCDT(p) = EV_CDT(p):

    p B = S + (1 - p) B
    2 p B - B = S
    p* = (B + S) / (2 B) = (1 + S/B) / 2

For p > p* the non-CDT (one-box) option has the higher EV; for p < p* the CDT option does.
"""
from __future__ import annotations

from dataclasses import dataclass


def ev_non_cdt(p: float, payoff_big: float, payoff_small: float) -> float:
    """Expected value of the non-CDT (one-box) choice. (payoff_small unused; kept for symmetry.)"""
    return p * payoff_big


def ev_cdt(p: float, payoff_big: float, payoff_small: float) -> float:
    """Expected value of the CDT (two-box) choice."""
    return payoff_small + (1.0 - p) * payoff_big


def crossover_p(payoff_big: float, payoff_small: float) -> float:
    """Return p* where EV_nonCDT(p) == EV_CDT(p): p* = (1 + S/B) / 2.

    Raises ValueError if B <= 0 or if p* falls outside [0, 1] (degenerate payoffs).
    """
    if payoff_big <= 0:
        raise ValueError(f"payoff_big must be positive, got {payoff_big}")
    p_star = (payoff_big + payoff_small) / (2.0 * payoff_big)
    if not (0.0 <= p_star <= 1.0):
        raise ValueError(
            f"degenerate payoffs (B={payoff_big}, S={payoff_small}) -> p*={p_star} outside [0,1]"
        )
    return p_star


@dataclass
class CrossoverResult:
    p_star: float
    payoff_big: float
    payoff_small: float
    mode: str


def crossover_for_item(item, cfg) -> CrossoverResult:
    """Compute p* for an item under ``cfg`` (a CrossoverConfig).

    mode "global"   -> use cfg.payoff_big / cfg.payoff_small (one number for the whole run).
    mode "per_item" -> use the item's own payoffs.
    """
    if cfg.mode == "per_item":
        B, S = item.payoff_big, item.payoff_small
        if B is None or S is None:
            raise ValueError(f"{item.id}: per_item crossover needs item payoffs")
    else:
        B, S = cfg.payoff_big, cfg.payoff_small
    return CrossoverResult(crossover_p(B, S), B, S, cfg.mode)
