"""Per-item conditioning deep-dive — CPU-only re-analysis of persisted results.

The project headline ("RL moves the *intercept*, not the *slope*; the 3B does the EV reasoning on
*some* items but not reliably") rests on two soft spots this module hardens:

  1. Every published "slope" is an eyeballed hi-minus-lo difference (`df.iloc[-1]-df.iloc[0]`) with
     no error bars and no flat-vs-sloped test. Here a per-item slope is a proper OLS fit with an
     analytic SE / CI, and the aggregate is an *item-bootstrap* mean (item = unit of analysis).
  2. "Item-dependent conditioning" was asserted, never measured. A flat *aggregate* slope can mean
     "every item flat" (capability ceiling) OR "some track, some anti-track, cancelling" — a very
     different story. We classify each item {tracks / flat / anti} and report the mix.

Everything reads already-persisted files under ``results/`` — no model is loaded, no GPU.

**Tracking direction.** One-boxing (``non_cdt``) is EV-optimal iff ``p > p*`` (p*=0.8 here), so a
*tracking* item has a **positive** slope of ``P(non_cdt)`` / ``margin`` / K in ``p``.

**Invariant (CLAUDE.md #1).** The choice is already resolved upstream via abstract tokens in every
persisted file. This module never re-resolves a choice and never string-matches decision-theory
labels ("one-box"/"two-box"/"Newcomb"). The Part-D content tags annotate *reasoning content only*
(does the prose mention the stated accuracy / do EV arithmetic) and never emit a role.

CLI: ``python -m newcomb_eval.item_analysis [--tag base3b] [--out results/item_analysis]``
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict

import numpy as np

from .crossover import crossover_p
from .scorer import ROLE_NON_CDT  # label only; never used to resolve a choice

# Forced-choice (logprob) arms that share the same 20 items as the base.
LOGPROB_ARMS = ("base3b", "causal", "evidential", "modelpred")
# CoT-arm scaffold file is base-only; the three arms live inside it as a column.
SCAFFOLD_ARMS = ("no_cot", "free_cot", "scaffolded")
# cot_inspect traces exist for the base and two RL'd CoT models.
COT_MODELS = ("base3b", "kl002", "paired")

P_STAR = crossover_p(100.0, 60.0)  # 0.8 (global payoffs B=100, S=60)


# --------------------------------------------------------------------------------------
# Part A — per-item slope on a continuous signal (OLS with analytic SE/CI)
# --------------------------------------------------------------------------------------
@dataclass
class SlopeFit:
    slope: float
    se: float
    ci_lo: float
    ci_hi: float
    level: float       # mean(y) over the p-grid — the intercept proxy ("level")
    n_points: int
    klass: str         # "tracks" | "flat" | "anti"


def _t_crit(df: int, alpha: float = 0.05) -> float:
    """Two-sided critical value; scipy if available, else a small lookup (df>=1)."""
    try:
        from scipy.stats import t  # local import keeps the module import light
        return float(t.ppf(1 - alpha / 2, df))
    except Exception:  # pragma: no cover - fallback only
        # 97.5% t critical values for df 1..10, then ~normal.
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        return table.get(df, 1.96)


def classify(ci_lo: float, ci_hi: float) -> str:
    """Three-way label from whether the slope CI excludes 0 and its sign."""
    if ci_lo > 0:
        return "tracks"
    if ci_hi < 0:
        return "anti"
    return "flat"


def ols_slope(x, y, *, alpha: float = 0.05) -> SlopeFit:
    """OLS slope of y on x with analytic slope SE and a (1-alpha) CI.

    With n points the slope SE is ``sqrt( SSR/(n-2) / Sxx )`` and the CI uses a t critical value on
    n-2 degrees of freedom. Degenerate inputs (n<3 or no x-variance) yield a NaN-SE 'flat' fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    level = float(np.mean(y)) if n else float("nan")
    sxx = float(np.sum((x - x.mean()) ** 2)) if n else 0.0
    if n < 3 or sxx == 0.0:
        return SlopeFit(float("nan"), float("nan"), float("nan"), float("nan"), level, n, "flat")
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    intercept = y.mean() - slope * x.mean()
    resid = y - (intercept + slope * x)
    ssr = float(np.sum(resid ** 2))
    se = float(np.sqrt((ssr / (n - 2)) / sxx))
    half = _t_crit(n - 2, alpha) * se
    ci_lo, ci_hi = slope - half, slope + half
    return SlopeFit(slope, se, ci_lo, ci_hi, level, n, classify(ci_lo, ci_hi))


def bootstrap_mean_ci(values, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of per-item values (resample items w/ replacement).

    Returns ``(mean, lo, hi)``. Deterministic under ``seed``.
    """
    v = np.asarray([x for x in values if x == x], dtype=float)  # drop NaN
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if v.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def per_item_slopes(rows, *, signal: str, alpha: float = 0.05) -> dict[str, SlopeFit]:
    """Group per-sample rows by ``item_id`` and OLS-fit ``signal`` ~ ``p`` for each item.

    ``rows`` is an iterable of dicts with at least ``item_id``, ``p`` and the ``signal`` column.
    """
    by_item: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append((float(r["p"]), float(r[signal])))
    out: dict[str, SlopeFit] = {}
    for item, pts in by_item.items():
        pts.sort()
        xs = [p for p, _ in pts]
        ys = [v for _, v in pts]
        out[item] = ols_slope(xs, ys, alpha=alpha)
    return out


def slope_distribution(fits: dict[str, SlopeFit], *, seed: int = 0) -> dict:
    """Summarise a per-item slope dict: class counts + item-bootstrap mean-slope CI."""
    slopes = [f.slope for f in fits.values()]
    counts = {"tracks": 0, "flat": 0, "anti": 0}
    for f in fits.values():
        counts[f.klass] += 1
    mean, lo, hi = bootstrap_mean_ci(slopes, seed=seed)
    valid = [s for s in slopes if s == s]
    return {
        "n_items": len(fits),
        "n_tracks": counts["tracks"],
        "n_flat": counts["flat"],
        "n_anti": counts["anti"],
        "mean_slope": mean,
        "mean_slope_ci_lo": lo,
        "mean_slope_ci_hi": hi,
        "median_slope": float(np.median(valid)) if valid else float("nan"),
        "heterogeneous": counts["tracks"] > 0 and counts["anti"] > 0,
    }


# --------------------------------------------------------------------------------------
# Part C — scaffold per-item K-slope (binary, coarse hi-minus-lo across p*)
# --------------------------------------------------------------------------------------
def scaffold_item_arm_table(rows, *, p_star: float = P_STAR) -> list[dict]:
    """Per (arm,item): mean K below p* and above p*, and the coarse hi-lo slope + class.

    Binary single-repeat data, so the slope is a deliberately coarse ``mean(K|p>p*) -
    mean(K|p<p*)`` (the boundary cell ``p==p*`` is dropped). ``class`` here is sign-only (no CI):
    >0 -> tracks, <0 -> anti, ==0/NaN -> flat. Flagged coarse on purpose.
    """
    grp: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        if not _as_bool(r.get("is_valid", True)):
            continue
        key = (r["arm"], r["item_id"])
        p = float(r["p"])
        bucket = grp.setdefault(key, {"lo": [], "hi": []})
        if p < p_star:
            bucket["lo"].append(float(r["is_k"]))
        elif p > p_star:
            bucket["hi"].append(float(r["is_k"]))
    out = []
    for (arm, item), b in sorted(grp.items()):
        lo = float(np.mean(b["lo"])) if b["lo"] else float("nan")
        hi = float(np.mean(b["hi"])) if b["hi"] else float("nan")
        slope = hi - lo if (b["lo"] and b["hi"]) else float("nan")
        if slope != slope:
            klass = "flat"
        elif slope > 0:
            klass = "tracks"
        elif slope < 0:
            klass = "anti"
        else:
            klass = "flat"
        out.append({"arm": arm, "item_id": item, "k_low": lo, "k_high": hi,
                    "slope": slope, "class": klass})
    return out


# --------------------------------------------------------------------------------------
# Part D — reasoning-content tags (content only; never resolves a choice)
# --------------------------------------------------------------------------------------
_ARITH_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*[×x*]\s*\d)"          # 100 x 0.5
    r"|(=\s*\d)"                               # = 50
    r"|(expected\s+value)"
    r"|(\bE\s*\()"                             # E( ... )
    r"|(on\s+average)"
    r"|(\baverage\b)",
    re.IGNORECASE,
)


def _p_number_strings(p: float) -> list[str]:
    """Surface forms of the stated accuracy a CoT might quote (e.g. 0.5 -> '50%', '0.5')."""
    pct = p * 100
    pct_str = f"{pct:.0f}" if abs(pct - round(pct)) < 1e-9 else f"{pct:g}"
    return [f"{pct_str}%", f"{pct_str} %", f"{pct_str} percent", f"{p:g}"]


def content_tags(cot: str, p: float, answer_role: str, margin: float,
                 *, low_margin: float = 1.0) -> dict:
    """Annotate one reasoning trace. Returns tags only — never a choice/role.

    - ``mentions_p_number``: a surface form of the stated accuracy appears in the prose.
    - ``does_ev_arithmetic``: arithmetic / expected-value markers present.
    - ``committed``: |margin| >= low_margin (the forced readout was confident, not non-committal).
    - ``cot_answer_consistent``: margin sign agrees with the committed answer_role.
    """
    text = cot or ""
    mentions = any(s in text for s in _p_number_strings(p))
    arith = bool(_ARITH_RE.search(text))
    committed = abs(margin) >= low_margin
    # margin>0 favours non_cdt; consistent iff that matches the recorded answer role.
    consistent = (margin > 0) == (answer_role == ROLE_NON_CDT)
    return {
        "mentions_p_number": mentions,
        "does_ev_arithmetic": arith,
        "committed": committed,
        "cot_answer_consistent": bool(consistent),
    }


# --------------------------------------------------------------------------------------
# I/O + CLI glue (not unit-tested; exercised by the verification run)
# --------------------------------------------------------------------------------------
def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "1.0")


def _read_csv(path: str) -> list[dict]:
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _slope_rows(fits: dict[str, SlopeFit], arm: str, signal: str) -> list[dict]:
    rows = []
    for item, f in sorted(fits.items()):
        d = {"arm": arm, "item_id": item, "signal": signal}
        d.update(asdict(f))
        rows.append(d)
    return rows


def run(results_dir: str, out_dir: str, *, tag: str = "base3b", seed: int = 0) -> dict:
    """Load persisted files, run Parts A-D, write CSVs/plots, return a summary dict."""
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    lp_dir = os.path.join(results_dir, "logprob")
    summary: dict = {"p_star": P_STAR}

    # ---- Part A: per-item slopes on the continuous logprob signal, per arm ----
    all_slope_rows: list[dict] = []
    fits_by_arm: dict[str, dict[str, SlopeFit]] = {}
    dist_by_arm: dict[str, dict] = {}
    for arm in LOGPROB_ARMS:
        path = os.path.join(lp_dir, f"logprob_samples_{arm}.csv")
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        fits_margin = per_item_slopes(rows, signal="margin")
        fits_p = per_item_slopes(rows, signal="p_non_cdt")
        fits_by_arm[arm] = fits_margin
        dist_by_arm[arm] = slope_distribution(fits_margin, seed=seed)
        all_slope_rows += _slope_rows(fits_margin, arm, "margin")
        all_slope_rows += _slope_rows(fits_p, arm, "p_non_cdt")
    pd.DataFrame(all_slope_rows).to_csv(os.path.join(out_dir, "item_slopes.csv"), index=False)
    summary["slope_distribution"] = dist_by_arm

    # ---- Part B: per-item base->RL slope/level shift (paired on item_id) ----
    rl_rows: list[dict] = []
    base = fits_by_arm.get("base3b", {})
    for arm in ("causal", "evidential", "modelpred"):
        rl = fits_by_arm.get(arm)
        if not rl:
            continue
        dslopes, dlevels = [], []
        for item in sorted(set(base) & set(rl)):
            b, r = base[item], rl[item]
            rl_rows.append({"arm": arm, "item_id": item,
                            "base_slope": b.slope, "rl_slope": r.slope,
                            "base_level": b.level, "rl_level": r.level,
                            "d_slope": r.slope - b.slope, "d_level": r.level - b.level})
            dslopes.append(r.slope - b.slope)
            dlevels.append(r.level - b.level)
        ms, ls, hs = bootstrap_mean_ci(dslopes, seed=seed)
        ml, ll, hl = bootstrap_mean_ci(dlevels, seed=seed)
        summary.setdefault("rl_vs_base", {})[arm] = {
            "mean_d_slope": ms, "d_slope_ci": [ls, hs],
            "mean_d_level": ml, "d_level_ci": [ll, hl]}
    if rl_rows:
        pd.DataFrame(rl_rows).to_csv(os.path.join(out_dir, "rl_vs_base_per_item.csv"), index=False)

    # ---- Part C: scaffold per-item K-slope across the three CoT arms ----
    scaf_path = os.path.join(results_dir, "scaffold", f"scaffold_samples_{tag}.csv")
    if os.path.exists(scaf_path):
        scaf = scaffold_item_arm_table(_read_csv(scaf_path))
        pd.DataFrame(scaf).to_csv(os.path.join(out_dir, "scaffold_item_arms.csv"), index=False)
        summary["scaffold_arm_class_counts"] = {
            arm: _count_classes([r for r in scaf if r["arm"] == arm]) for arm in SCAFFOLD_ARMS}

    # ---- Part D: CoT content tags (thin: 4 items) ----
    tag_rows: list[dict] = []
    for model in COT_MODELS:
        cot_path = os.path.join(results_dir, "cot_inspect", f"cot_{model}.jsonl")
        if not os.path.exists(cot_path):
            continue
        for rec in _read_jsonl(cot_path):
            tags = content_tags(rec.get("cot", ""), float(rec["p"]),
                                rec.get("answer_role", ""), float(rec.get("margin", 0.0)))
            tag_rows.append({"model": model, "item_id": rec["item_id"], "p": rec["p"],
                             "is_optimal": rec.get("is_optimal"), **tags})
    if tag_rows:
        pd.DataFrame(tag_rows).to_csv(os.path.join(out_dir, "cot_content_tags.csv"), index=False)
        summary["cot_arith_rate"] = {
            m: round(float(np.mean([r["does_ev_arithmetic"] for r in tag_rows if r["model"] == m])), 3)
            for m in COT_MODELS if any(r["model"] == m for r in tag_rows)}

    _make_plots(out_dir, fits_by_arm, rl_rows, tag=tag)
    return summary


def _count_classes(rows) -> dict:
    c = {"tracks": 0, "flat": 0, "anti": 0}
    for r in rows:
        c[r["class"]] = c.get(r["class"], 0) + 1
    return c


def _make_plots(out_dir, fits_by_arm, rl_rows, *, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # slope histogram (strip) per arm
    if fits_by_arm:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (arm, fits) in enumerate(fits_by_arm.items()):
            slopes = [f.slope for f in fits.values() if f.slope == f.slope]
            jitter = (np.random.default_rng(i).random(len(slopes)) - 0.5) * 0.25
            ax.scatter([i + j for j in jitter], slopes, alpha=0.6, s=20)
            ax.scatter([i], [np.mean(slopes)], marker="_", s=600, color="k", zorder=5)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xticks(range(len(fits_by_arm)))
        ax.set_xticklabels(list(fits_by_arm), rotation=20)
        ax.set_ylabel("per-item slope of margin vs p")
        ax.set_title("Per-item conditioning slope by arm (margin~p); 0 = flat")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "slope_hist_by_arm.png"), dpi=150)
        plt.close(fig)

    # base->RL slope/level shift scatter
    if rl_rows:
        import pandas as pd
        df = pd.DataFrame(rl_rows)
        arms = sorted(df["arm"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for arm in arms:
            d = df[df["arm"] == arm]
            axes[0].scatter(d["base_slope"], d["rl_slope"], alpha=0.6, label=arm)
            axes[1].scatter(d["base_level"], d["rl_level"], alpha=0.6, label=arm)
        for ax, name in zip(axes, ("slope", "level")):
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, color="grey", lw=0.8, ls="--")
            ax.set_xlabel(f"base {name}")
            ax.set_ylabel(f"RL {name}")
            ax.set_title(f"per-item {name}: base vs RL (on y=x ⇒ unchanged)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "slope_level_shift.png"), dpi=150)
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-item conditioning deep-dive (CPU, persisted data)")
    ap.add_argument("--tag", default="base3b", help="scaffold-sample tag to read")
    ap.add_argument("--results", default="results", help="results dir to read from")
    ap.add_argument("--out", default=None, help="output dir (default results/item_analysis)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out_dir = args.out or os.path.join(args.results, "item_analysis")
    summary = run(args.results, out_dir, tag=args.tag, seed=args.seed)

    print(f"p* = {summary['p_star']:.3f}\n")
    print("Part A — per-item margin~p slope distribution by arm:")
    for arm, d in summary.get("slope_distribution", {}).items():
        print(f"  {arm:11s} n={d['n_items']:2d}  tracks={d['n_tracks']:2d} "
              f"flat={d['n_flat']:2d} anti={d['n_anti']:2d}  "
              f"mean_slope={d['mean_slope']:+.2f} "
              f"[{d['mean_slope_ci_lo']:+.2f},{d['mean_slope_ci_hi']:+.2f}]  "
              f"{'HETEROGENEOUS' if d['heterogeneous'] else ''}")
    if summary.get("rl_vs_base"):
        print("\nPart B — base->RL per-item shift (Δ = RL−base):")
        for arm, d in summary["rl_vs_base"].items():
            print(f"  {arm:11s} Δslope={d['mean_d_slope']:+.2f} "
                  f"[{d['d_slope_ci'][0]:+.2f},{d['d_slope_ci'][1]:+.2f}]  "
                  f"Δlevel={d['mean_d_level']:+.2f} "
                  f"[{d['d_level_ci'][0]:+.2f},{d['d_level_ci'][1]:+.2f}]")
    if summary.get("scaffold_arm_class_counts"):
        print("\nPart C — scaffold per-item K-slope class counts:")
        for arm, c in summary["scaffold_arm_class_counts"].items():
            print(f"  {arm:11s} tracks={c['tracks']:2d} flat={c['flat']:2d} anti={c['anti']:2d}")
    if summary.get("cot_arith_rate"):
        print("\nPart D — CoT EV-arithmetic rate (n=4 items, thin):")
        for m, r in summary["cot_arith_rate"].items():
            print(f"  {m:11s} {r:.0%}")
    print(f"\nWrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
