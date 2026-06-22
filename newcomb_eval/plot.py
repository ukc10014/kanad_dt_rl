"""K-rate vs p plot with p* overlay (PLAN.md §2 plot.py)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def plot_krate(df, out_png: str, *, p_star: float | None = None, model_name: str = "") -> str:
    """Plot K-rate(p) with Wilson CI band and a vertical line at p*. Saves PNG; returns path."""
    p_star = p_star if p_star is not None else df.attrs.get("p_star")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    d = df.sort_values("p")

    ax.plot(d["p"], d["k_rate"], marker="o", color="#1f77b4", label="K-rate (non-CDT)")
    if {"ci_lo", "ci_hi"}.issubset(d.columns):
        ax.fill_between(d["p"], d["ci_lo"], d["ci_hi"], alpha=0.2, color="#1f77b4",
                        label="95% Wilson CI")

    if p_star is not None:
        ax.axvline(p_star, color="#d62728", linestyle="--", label=f"p* = {p_star:.3f}")

    # Mark held-out p values.
    if "is_holdout" in d.columns and d["is_holdout"].any():
        hold = d[d["is_holdout"]]
        ax.scatter(hold["p"], hold["k_rate"], s=140, facecolors="none",
                   edgecolors="#2ca02c", linewidths=2, label="held-out p", zorder=5)

    ax.axhline(0.5, color="grey", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("stated predictor accuracy  p")
    ax.set_ylabel("non-CDT selection rate (K-rate)")
    ax.set_ylim(-0.03, 1.03)
    title = "Newcomb p-sweep: K-rate vs stated accuracy"
    if model_name:
        title += f"\n{model_name}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
