"""Tests for the per-item conditioning deep-dive — synthetic data, no GPU, no files."""
from __future__ import annotations

import numpy as np

from newcomb_eval.item_analysis import (
    ols_slope,
    classify,
    bootstrap_mean_ci,
    per_item_slopes,
    slope_distribution,
    scaffold_item_arm_table,
    content_tags,
)
from newcomb_eval.scorer import ROLE_NON_CDT, ROLE_CDT

P_GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99]


# --------------------------------------------------------------------------- OLS slope
def test_ols_slope_recovers_known_line():
    x = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    y = 2.0 * x + 1.0  # exact line, slope 2
    fit = ols_slope(x, y)
    assert abs(fit.slope - 2.0) < 1e-9
    assert fit.se < 1e-9          # perfect fit -> ~0 residual SE
    assert fit.klass == "tracks"  # CI excludes 0, positive
    assert abs(fit.level - float(np.mean(y))) < 1e-12


def test_ols_flat_series_is_flat():
    x = np.array(P_GRID)
    y = np.full_like(x, 0.8) + np.array([0.01, -0.01, 0.0, 0.005, -0.005, 0.0, 0.01, -0.01])
    fit = ols_slope(x, y)
    assert fit.klass == "flat"
    assert fit.ci_lo < 0 < fit.ci_hi


def test_ols_negative_ramp_is_anti():
    x = np.array(P_GRID)
    y = -3.0 * x + 5.0
    fit = ols_slope(x, y)
    assert fit.klass == "anti"
    assert fit.ci_hi < 0


def test_ols_degenerate_inputs_are_flat_nan():
    fit = ols_slope([0.5, 0.6], [1.0, 2.0])  # n<3
    assert fit.klass == "flat"
    assert fit.slope != fit.slope or fit.se != fit.se  # NaN somewhere


def test_classify_thresholds():
    assert classify(0.1, 0.5) == "tracks"
    assert classify(-0.5, -0.1) == "anti"
    assert classify(-0.2, 0.3) == "flat"


# --------------------------------------------------------------------------- bootstrap
def test_bootstrap_mean_ci_deterministic_and_brackets():
    vals = [1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7]
    mean, lo, hi = bootstrap_mean_ci(vals, seed=0)
    mean2, lo2, hi2 = bootstrap_mean_ci(vals, seed=0)
    assert (mean, lo, hi) == (mean2, lo2, hi2)   # deterministic under seed
    assert lo < mean < hi
    assert abs(mean - float(np.mean(vals))) < 1e-9


# --------------------------------------------------------------- the heterogeneity confound
def _ramp_rows(item_id, slope, intercept=0.0):
    return [{"item_id": item_id, "p": p, "margin": slope * p + intercept} for p in P_GRID]


def test_bimodal_set_reported_heterogeneous_not_flat():
    """The exact confound this module exists to catch: a flat *mean* hiding a +/- mix."""
    rows = []
    for i in range(5):
        rows += _ramp_rows(f"pos_{i}", slope=+10.0)
    for i in range(5):
        rows += _ramp_rows(f"neg_{i}", slope=-10.0)
    fits = per_item_slopes(rows, signal="margin")
    dist = slope_distribution(fits, seed=0)
    assert dist["n_items"] == 10
    assert dist["n_tracks"] == 5
    assert dist["n_anti"] == 5
    assert dist["n_flat"] == 0
    assert dist["heterogeneous"] is True
    assert abs(dist["mean_slope"]) < 1e-6  # mean ~0 despite NO flat items


def test_uniformly_flat_set_is_not_heterogeneous():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(8):
        noise = rng.normal(0, 0.005, len(P_GRID))
        rows += [{"item_id": f"f_{i}", "p": p, "margin": 3.0 + n}
                 for p, n in zip(P_GRID, noise)]
    dist = slope_distribution(per_item_slopes(rows, signal="margin"), seed=0)
    assert dist["n_tracks"] == 0 and dist["n_anti"] == 0
    assert dist["heterogeneous"] is False


# --------------------------------------------------------------------------- scaffold
def test_scaffold_item_arm_table_hi_lo_slope():
    rows = []
    # one item that tracks (low K below p*, high K above) under one arm
    for p in P_GRID:
        is_k = 1.0 if p > 0.8 else 0.0
        rows.append({"arm": "free_cot", "item_id": "a", "p": p, "is_k": is_k, "is_valid": True})
    tab = scaffold_item_arm_table(rows)
    assert len(tab) == 1
    r = tab[0]
    assert r["k_low"] == 0.0 and r["k_high"] == 1.0
    assert r["slope"] == 1.0 and r["class"] == "tracks"


def test_scaffold_drops_invalid_rows():
    rows = [{"arm": "no_cot", "item_id": "a", "p": 0.5, "is_k": 1.0, "is_valid": False},
            {"arm": "no_cot", "item_id": "a", "p": 0.9, "is_k": 1.0, "is_valid": True}]
    tab = scaffold_item_arm_table(rows)
    # only the valid (high-p) row survives -> no low bucket -> NaN slope -> flat
    assert tab[0]["class"] == "flat"


# --------------------------------------------------------------------------- content tags
def test_content_tagger_detects_ev_arithmetic():
    t = content_tags("If F is right 50% of the time, 100 x 0.5 = 50 on average.",
                     p=0.5, answer_role=ROLE_CDT, margin=-3.0)
    assert t["does_ev_arithmetic"] is True
    assert t["mentions_p_number"] is True
    # never returns a role/choice
    assert "role" not in t and "choice" not in t and "is_k" not in t


def test_content_tagger_false_on_pure_qualitative_rule():
    t = content_tags("They're correlated, so I should just pick the one-only option.",
                     p=0.99, answer_role=ROLE_NON_CDT, margin=4.0)
    assert t["does_ev_arithmetic"] is False
    assert t["mentions_p_number"] is False  # prose quotes no accuracy figure
    assert t["committed"] is True
    assert t["cot_answer_consistent"] is True  # margin>0 matches non_cdt


def test_content_tagger_consistency_and_commitment():
    # margin near 0 -> non-committal; sign disagrees with recorded role
    t = content_tags("...", p=0.7, answer_role=ROLE_NON_CDT, margin=-0.2)
    assert t["committed"] is False
    assert t["cot_answer_consistent"] is False  # margin<0 but role says non_cdt
