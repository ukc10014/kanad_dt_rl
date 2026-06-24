"""Cross-model credence-ladder aggregation (CPU — reads persisted probe outputs, no GPU).

Runs *after* ``credence_probe`` has been run per model (see ``results/credence/run_credence_ladder.sh``).
For each model rung it reads ``results/credence/credence_samples_<tag>.csv`` (and, if present, the
``results/logprob/p_margin_by_p_<tag>.csv`` action margin) and builds the **signature**:

  * **representation slope** — OLS of the credence gap on ``2p-1`` (≈1 = the evidential dependence is
    cleanly encoded; ≈0 = not encoded). The new thing.
  * **action slope** — the (flat) logit-margin hi-lo from ``logprob_sweep``. The disposition.
  * **divergence** = representation − action: the quantity the ladder exists to chart. If it *grows*
    with scale, comprehension and disposition are orthogonal (scale buys understanding the predictor,
    not acting on it) — direct evidence against "EDT-lean is a capability artifact".

plus the **coherence** cross-checks (so a partial slope isn't mistaken for a clean encoder, and an
EDT *answer* isn't mistaken for an EDT *world model*):

  * **variant agreement** — Pearson r between the ``outcome`` and ``prediction`` gap curves, and
    between the token gaps and the free-form ``direct`` gap. Logically-equivalent elicitations must
    agree in a coherent model.
  * **monotonicity** — fraction of adjacent p-steps where the gap is non-decreasing (should rise).
  * **resolvability** — fraction of non-degenerate cells (the saturation gate; near-0 ⇒ this rung is
    *unreadable*, not a real null — report it, don't interpret the slope).

Everything reads already-persisted files — no model load. CLI:
    python -m newcomb_eval.credence_ladder [--tags base3b base7b base14b base32b_4bit] [--out ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from .credence_probe import aggregate_credence, credence_headline

# rung tag -> (HF model id, approx params in billions for the x-axis). Sub-3B excluded by request.
DEFAULT_RUNGS = [
    ("base3b", "Qwen/Qwen2.5-3B-Instruct", 3.09),
    ("base7b", "Qwen/Qwen2.5-7B-Instruct", 7.62),
    ("base14b", "Qwen/Qwen2.5-14B-Instruct", 14.77),
    ("base32b_4bit", "Qwen/Qwen2.5-32B-Instruct", 32.5),
]


# --------------------------------------------------------------------------------------
# coherence primitives (pure — unit-test targets)
# --------------------------------------------------------------------------------------
def monotonicity(gap_by_p: dict, *, tol: float = 1e-6) -> float:
    """Fraction of adjacent p-steps (sorted) where the gap is non-decreasing. 1.0 = monotone rise."""
    ps = sorted(gap_by_p)
    if len(ps) < 2:
        return float("nan")
    ok = sum(1 for a, b in zip(ps, ps[1:]) if gap_by_p[b] >= gap_by_p[a] - tol)
    return ok / (len(ps) - 1)


def series_agreement(a: dict, b: dict) -> float:
    """Pearson r between two gap-by-p curves over their shared p keys (NaN if <3 pts or no variance)."""
    keys = sorted(set(a) & set(b))
    xs = [a[k] for k in keys if a[k] == a[k] and b[k] == b[k]]
    ys = [b[k] for k in keys if a[k] == a[k] and b[k] == b[k]]
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


# --------------------------------------------------------------------------------------
# per-model + cross-model aggregation
# --------------------------------------------------------------------------------------
def _read_samples(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            cf = r.get("credence_full", "")
            try:
                r["credence_full"] = float(cf)
            except (TypeError, ValueError):
                r["credence_full"] = float("nan")
            r["p"] = float(r["p"])
            r["is_degenerate"] = str(r.get("is_degenerate", "")).strip().lower() in ("1", "true", "1.0")
            rows.append(r)
    return rows


def _gap_by_p(rows, p_star: float) -> dict:
    df = aggregate_credence(rows, p_star=p_star)
    return {float(p): float(g) for p, g in zip(df["p"], df["gap_mean"])}


def _action_slope(results_dir: str, tag: str) -> float | None:
    path = os.path.join(results_dir, "logprob", f"p_margin_by_p_{tag}.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    d = pd.read_csv(path).sort_values("p")
    col = "margin_mean" if "margin_mean" in d.columns else "p_non_cdt_mean"
    return float(d[col].iloc[-1] - d[col].iloc[0])


def analyse_model(results_dir: str, tag: str, params: float, *, p_star: float = 0.8) -> dict | None:
    path = os.path.join(results_dir, "credence", f"credence_samples_{tag}.csv")
    if not os.path.exists(path):
        return None
    rows = _read_samples(path)
    variants = sorted({r["variant"] for r in rows})

    gaps, slopes = {}, {}
    for v in variants:
        vrows = [r for r in rows if r["variant"] == v]
        gaps[v] = _gap_by_p(vrows, p_star)
        slopes[v] = credence_headline(vrows, p_star=p_star)["fit_slope_gap_on_2pm1"]

    repr_slope = slopes.get("outcome", slopes.get("prediction", float("nan")))
    action_slope = _action_slope(results_dir, tag)
    token_gap = gaps.get("outcome") or gaps.get("prediction")
    agree_op = series_agreement(gaps.get("outcome", {}), gaps.get("prediction", {}))
    agree_td = series_agreement(token_gap or {}, gaps.get("direct", {})) if "direct" in gaps else float("nan")
    mono = [monotonicity(g) for g in gaps.values()]
    mono = [m for m in mono if m == m]
    resolvability = 1.0 - (sum(r["is_degenerate"] for r in rows) / len(rows) if rows else float("nan"))

    out = {
        "tag": tag, "params_b": params, "n_variants": len(variants),
        "repr_slope_outcome": slopes.get("outcome", float("nan")),
        "repr_slope_prediction": slopes.get("prediction", float("nan")),
        "repr_slope_direct": slopes.get("direct", float("nan")),
        "repr_slope": repr_slope,
        "action_slope": action_slope if action_slope is not None else float("nan"),
        "divergence": (repr_slope - action_slope) if action_slope is not None else float("nan"),
        "agreement_outcome_prediction": agree_op,
        "agreement_token_direct": agree_td,
        "monotonicity": (sum(mono) / len(mono)) if mono else float("nan"),
        "resolvability": resolvability,
    }
    return out


def run(results_dir: str, rungs=DEFAULT_RUNGS, *, out_dir: str | None = None) -> list[dict]:
    out_dir = os.path.abspath(out_dir or os.path.join(results_dir, "credence"))
    os.makedirs(out_dir, exist_ok=True)
    sig = [r for r in (analyse_model(results_dir, tag, params) for tag, _, params in rungs) if r]

    if sig:
        import pandas as pd
        df = pd.DataFrame(sig).sort_values("params_b")
        df.to_csv(os.path.join(out_dir, "ladder_signature.csv"), index=False)
        with open(os.path.join(out_dir, "ladder_signature.json"), "w") as fh:
            json.dump(sig, fh, indent=2)
        _plot(df, out_dir)
        print(f"{'rung':>14} {'params':>7} {'repr':>6} {'action':>7} {'diverg':>7} "
              f"{'agreeOP':>8} {'agreeTD':>8} {'mono':>5} {'resolv':>6}")
        for _, r in df.iterrows():
            print(f"{r['tag']:>14} {r['params_b']:>6.1f}B {r['repr_slope']:>+6.2f} "
                  f"{r['action_slope']:>+7.2f} {r['divergence']:>+7.2f} "
                  f"{r['agreement_outcome_prediction']:>8.2f} {r['agreement_token_direct']:>8.2f} "
                  f"{r['monotonicity']:>5.2f} {r['resolvability']:>6.2f}")
        print(f"\nwrote {out_dir}/ladder_signature.{{csv,json}} + ladder_signature.png")
    else:
        print("no per-model credence samples found — run the probe rungs first")
    return sig


def _plot(df, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df[df["params_b"] > 0].sort_values("params_b")  # drop unknown-param rungs from the plot
    if df.empty:
        return
    log_x = len(df) > 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = df["params_b"]
    ax1.plot(x, df["repr_slope"], marker="o", color="#1f77b4", label="representation slope (gap~2p−1)")
    if df["action_slope"].notna().any():
        ax1.plot(x, df["action_slope"], marker="s", color="#d62728", label="action margin slope (flat)")
    ax1.axhline(0, color="grey", lw=0.8, ls="--")
    ax1.axhline(1.0, color="#1f77b4", lw=0.8, ls=":", alpha=0.5, label="evidential ideal = 1")
    if log_x:
        ax1.set_xscale("log")
    ax1.set_xlabel("params (B, log)"); ax1.set_ylabel("slope")
    ax1.set_title("Signature vs scale\nrepresentation rising while action flat ⇒ orthogonal axes")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    for col, mk, c, lab in [("agreement_outcome_prediction", "o", "#2ca02c", "variant agreement (O↔P)"),
                            ("monotonicity", "^", "#9467bd", "monotonicity"),
                            ("resolvability", "D", "#8c564b", "resolvability")]:
        ax2.plot(x, df[col], marker=mk, color=c, label=lab)
    if log_x:
        ax2.set_xscale("log")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("params (B, log)"); ax2.set_ylabel("coherence score")
    ax2.set_title("Coherence vs scale\n(low resolvability ⇒ rung unreadable, don't interpret)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ladder_signature.png"), dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-model credence-ladder signature (CPU, persisted)")
    ap.add_argument("--results", default="results", help="results dir to read from")
    ap.add_argument("--tags", nargs="+", default=None, help="rung tags (default: 3B/7B/14B/32B)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rungs = DEFAULT_RUNGS
    if args.tags:
        by_tag = {t: (t, m, p) for t, m, p in DEFAULT_RUNGS}
        rungs = [by_tag.get(t, (t, t, float("nan"))) for t in args.tags]
    run(args.results, rungs, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
