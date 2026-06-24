"""Mechanism-credibility aggregation (CPU — reads persisted probe outputs, no GPU).

Runs after ``credence_probe`` has been run per mechanism level (see ``run_credence_mechanism.sh``)
on a FIXED model (14B). For each credibility rung m0→m3 it reads
``results/credence/credence_samples_mech_<level>_<model>.csv`` and reports, as a function of how
*credible/binding* the predictor story is:

  * **action** — one-box rate + forced-choice margin (the disposition). Does making the predictor
    credible move the *choice* toward one-boxing?
  * **credence** — the **baseline-adjusted** gap slope (`gap_adj ~ 2p-1`, the de-confounded number
    from the instrument fix) on the clean **direct** elicitation, plus `gap@0.5` (symmetry control)
    and the one-sided-saturation flag. Does a credible predictor turn the faint p-tracking we saw
    at 14B into a clean climb?

**Read:** if action one-box rate and/or credence `gap_adj` slope *rise* m0→m3 (and the m0pad placebo
stays at m0), the abstract framing's *incredible* predictor was suppressing EDT behaviour — a framing
artifact, not a fixed disposition. If flat across all rungs (even exact-copy), the dominance
disposition is robust to credibility — itself a strong, quotable result.

CLI: ``python -m newcomb_eval.credence_mechanism [--model 14b] [--out ...]``
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from .credence_probe import credence_headline
from .gen_mechanism_dataset import LEVELS


def _read(path: str) -> list[dict]:
    from .credence_probe import parse_probability
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("variant") == "direct" and r.get("raw"):  # apply the parse fix to persisted raw
                pp = parse_probability(r["raw"])
                r["credence_full"] = pp if pp is not None else float("nan")
            else:
                try:
                    r["credence_full"] = float(r.get("credence_full", "") or "nan")
                except ValueError:
                    r["credence_full"] = float("nan")
            try:
                r["margin"] = float(r.get("margin", "") or "nan")
            except ValueError:
                r["margin"] = float("nan")
            r["p"] = float(r["p"])
            rows.append(r)
    return rows


def analyse_level(results_dir: str, tag: str, *, p_star: float = 0.8) -> dict | None:
    path = os.path.join(results_dir, "credence", f"credence_samples_{tag}.csv")
    if not os.path.exists(path):
        return None
    rows = _read(path)
    out = {"tag": tag}

    # action axis (disposition)
    act = [r for r in rows if r["variant"] == "action" and r["margin"] == r["margin"]]
    if act:
        import numpy as np
        by_p = {}
        for r in act:
            by_p.setdefault(r["p"], []).append(r["margin"])
        ps = sorted(by_p)
        out["onebox_rate"] = float(np.mean([1.0 if r["margin"] > 0 else 0.0 for r in act]))
        out["margin_level"] = float(np.mean([r["margin"] for r in act]))
        out["margin_hi_lo"] = float(np.mean(by_p[ps[-1]]) - np.mean(by_p[ps[0]]))

    # credence axis (representation) — direct is the clean primary; prediction/outcome as cross-check
    for v in ("direct", "prediction", "outcome"):
        vrows = [r for r in rows if r["variant"] == v]
        if not vrows:
            continue
        hl = credence_headline(vrows, p_star=p_star)
        tagv = "" if v == "direct" else f"_{v}"
        out[f"cred_gap_adj_slope{tagv}"] = hl["fit_slope_gap_adj_on_2pm1"]
        out[f"cred_gap_adj_ci{tagv}"] = hl["fit_slope_adj_ci"]
        out[f"cred_gap_at_0.5{tagv}"] = hl["gap_at_0.5"]
        out[f"cred_saturated{tagv}"] = hl["onesided_saturation"]
    return out


def run(results_dir: str, model_tag: str = "14b", *, out_dir: str | None = None) -> list[dict]:
    out_dir = os.path.abspath(out_dir or os.path.join(results_dir, "credence"))
    os.makedirs(out_dir, exist_ok=True)
    sig = []
    for i, (level, label) in enumerate(LEVELS):
        r = analyse_level(results_dir, f"mech_{level}_{model_tag}")
        if r:
            r["level"] = level; r["label"] = label; r["order"] = i
            sig.append(r)

    if not sig:
        print("no per-mechanism credence samples found — run the probe rungs first")
        return sig

    import pandas as pd
    df = pd.DataFrame(sig).sort_values("order")
    df.to_csv(os.path.join(out_dir, "mechanism_signature.csv"), index=False)
    with open(os.path.join(out_dir, "mechanism_signature.json"), "w") as fh:
        json.dump(sig, fh, indent=2)
    _plot(df, out_dir)

    print(f"{'mechanism':>14} {'1box_rate':>9} {'margin':>7} {'cred_adj_slope':>14} "
          f"{'gap@0.5':>8} {'sat':>4}")
    for _, r in df.iterrows():
        print(f"{r['label']:>14} {r.get('onebox_rate', float('nan')):>9.2f} "
              f"{r.get('margin_level', float('nan')):>+7.2f} "
              f"{r.get('cred_gap_adj_slope', float('nan')):>+14.3f} "
              f"{r.get('cred_gap_at_0.5', float('nan')):>+8.2f} "
              f"{'Y' if r.get('cred_saturated') else '·':>4}")
    print(f"\nwrote {out_dir}/mechanism_signature.{{csv,json,png}}")
    return sig


def _plot(df, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = range(len(df))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if "onebox_rate" in df.columns:
        ax1.plot(x, df["onebox_rate"], marker="o", color="#1f77b4", label="one-box rate")
    if "margin_level" in df.columns:
        ax1b = ax1.twinx()
        ax1b.plot(x, df["margin_level"], marker="s", color="#d62728", alpha=0.6, label="margin level")
        ax1b.set_ylabel("forced-choice margin", color="#d62728")
        ax1b.axhline(0, color="#d62728", lw=0.6, ls=":", alpha=0.4)
    ax1.set_ylim(-0.05, 1.05); ax1.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(df["label"], rotation=20)
    ax1.set_ylabel("one-box rate"); ax1.set_title("Action vs predictor credibility")
    ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.3)

    if "cred_gap_adj_slope" in df.columns:
        ax2.plot(x, df["cred_gap_adj_slope"], marker="o", color="#2ca02c",
                 label="credence gap_adj slope (direct)")
    if "cred_gap_at_0.5" in df.columns:
        ax2.plot(x, df["cred_gap_at_0.5"], marker="^", color="#9467bd", alpha=0.6,
                 label="gap@0.5 (bias pedestal)")
    ax2.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax2.axhline(1.0, color="#2ca02c", lw=0.8, ls=":", alpha=0.4, label="evidential ideal = 1")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(df["label"], rotation=20)
    ax2.set_ylabel("slope"); ax2.set_title("Credence (de-biased) vs predictor credibility")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mechanism_signature.png"), dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mechanism-credibility signature (CPU, persisted)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="14b", help="model tag used in the mechanism run tags")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run(args.results, args.model, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
