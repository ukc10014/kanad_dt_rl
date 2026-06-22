"""Run the Inspect task across the p-grid and aggregate K-rate(p) (PLAN.md §2 sweep.py).

K-rate(p) = mean(is_k) over *valid* samples at each p. Invalid/unparseable completions are
excluded from the K-rate denominator and reported as their own rate (a data-quality signal).
Writes a tidy table: one row per (p, stratum) -> p, k_rate, n, n_valid, invalid_rate, ci_lo, ci_hi.
"""
from __future__ import annotations

import math
import os

import pandas as pd
from inspect_ai import eval as inspect_eval

from .config import EvalConfig
from .crossover import crossover_for_item
from .data.loader import load
from .task import build_task

SCORER_NAME = "newcomb_scorer"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (stable at small n / extremes)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _iter_scored_samples(logs):
    """Yield (p, is_k, is_valid) per sample, robust to whether samples are attached."""
    from inspect_ai.log import read_eval_log

    for log in logs:
        samples = getattr(log, "samples", None)
        if not samples and getattr(log, "location", None):
            samples = read_eval_log(log.location).samples
        for s in samples or []:
            score = (s.scores or {}).get(SCORER_NAME)
            if score is None:
                continue
            sm = score.metadata or {}
            p = sm.get("p", (s.metadata or {}).get("p"))
            yield p, float(sm.get("is_k", 0.0)), bool(sm.get("is_valid", False))


def aggregate(rows) -> pd.DataFrame:
    """Aggregate (p, is_k, is_valid) rows into K-rate(p) with Wilson CIs."""
    df = pd.DataFrame(rows, columns=["p", "is_k", "is_valid"])
    out = []
    for p, g in df.groupby("p"):
        n_total = len(g)
        valid = g[g["is_valid"]]
        n_valid = len(valid)
        k = int(valid["is_k"].sum())
        k_rate = (k / n_valid) if n_valid else float("nan")
        lo, hi = _wilson_ci(k, n_valid)
        out.append(
            {
                "p": p,
                "k_rate": k_rate,
                "n": n_total,
                "n_valid": n_valid,
                "invalid_rate": 1.0 - (n_valid / n_total) if n_total else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(out).sort_values("p").reset_index(drop=True)


def run_sweep(cfg: EvalConfig, model_wrapper, *, log_dir: str | None = None) -> pd.DataFrame:
    """Build + run the Inspect task once over the full cross-product, then aggregate."""
    results_dir = os.path.abspath(cfg.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    log_dir = log_dir or os.path.join(results_dir, "inspect_logs")

    task = build_task(cfg, model_wrapper)
    logs = inspect_eval(
        task,
        model="mockllm/model",  # dummy; our solver drives generation, this is never called
        max_samples=cfg.max_samples,
        log_dir=log_dir,
        display="plain",
    )
    df = aggregate(list(_iter_scored_samples(logs)))

    # p* reference (global): from the first item under the crossover config.
    items = load(cfg.dataset_path, cfg)
    p_star = crossover_for_item(items[0], cfg.crossover).p_star
    df["p_star"] = p_star
    df["is_holdout"] = df["p"].isin(set(cfg.sweep.holdout_p))

    csv_path = os.path.join(results_dir, "k_rate_by_p.csv")
    df.to_csv(csv_path, index=False)
    df.attrs["csv_path"] = csv_path
    df.attrs["p_star"] = p_star
    return df
