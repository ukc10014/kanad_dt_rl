"""Evaluate baseline + trained arms on the p-sweep and draw the comparison (plan: eval_arms.py).

Reuses newcomb_eval.sweep.run_sweep via ModelWrapper(adapter_path=...) for each arm (no edits to
newcomb_eval). Produces a combined K-rate(p) plot: baseline vs causal vs evidential, with p*.

    python -m newcomb_rl.eval_arms                      # baseline + whatever arms have adapters
    python -m newcomb_rl.eval_arms --arms causal evidential --model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from newcomb_eval.config import EvalConfig  # noqa: E402
from newcomb_eval.crossover import crossover_for_item  # noqa: E402
from newcomb_eval.data.loader import load  # noqa: E402
from newcomb_eval.model import ModelWrapper  # noqa: E402
from newcomb_eval.sweep import run_sweep  # noqa: E402

from .rl_config import RLOOConfig  # noqa: E402

_COLORS = {"baseline": "#7f7f7f", "causal": "#d62728", "evidential": "#1f77b4"}


def _sweep_arm(cfg: EvalConfig, model_name: str, adapter_path: str | None, label: str, results_dir: str):
    cfg = dataclasses.replace(cfg, results_dir=results_dir)
    wrapper = ModelWrapper(
        model_name, adapter_path=adapter_path,
        dtype=cfg.model.dtype, device_map=cfg.model.device_map,
        use_chat_template=cfg.model.use_chat_template,
    )
    df = run_sweep(cfg, wrapper, log_dir=os.path.join(results_dir, "inspect_logs"))
    print(f"[{label}] mean K-rate={df['k_rate'].mean():.3f} invalid={1 - df['n_valid'].sum()/df['n'].sum():.3f}")
    del wrapper
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    return df


def combined_plot(curves: dict, p_star: float, out_png: str, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, df in curves.items():
        d = df.sort_values("p")
        c = _COLORS.get(label, None)
        ax.plot(d["p"], d["k_rate"], marker="o", color=c, label=label)
        if {"ci_lo", "ci_hi"}.issubset(d.columns):
            ax.fill_between(d["p"], d["ci_lo"], d["ci_hi"], alpha=0.12, color=c)
    ax.axvline(p_star, color="black", linestyle="--", alpha=0.6, label=f"p* = {p_star:.2f}")
    ax.axhline(0.5, color="grey", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("stated predictor accuracy  p")
    ax.set_ylabel("non-CDT (one-box) selection rate  K-rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"RL shifts the decision theory: K-rate(p) by reward arm\n{model_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sweep baseline + RL arms and plot the comparison")
    ap.add_argument("--arms", nargs="*", default=["causal", "evidential"])
    ap.add_argument("--model", help="HF model id (default: config)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rcfg = RLOOConfig()
    ecfg = rcfg.eval
    model_name = args.model or ecfg.model.model_name
    results_root = os.path.abspath(ecfg.results_dir)

    items = load(ecfg.dataset_path, ecfg)
    p_star = crossover_for_item(items[0], ecfg.crossover).p_star

    curves = {}
    if not args.no_baseline:
        curves["baseline"] = _sweep_arm(ecfg, model_name, None, "baseline",
                                        os.path.join(results_root, "arm_baseline"))
    for arm in args.arms:
        adapter = os.path.join(rcfg.adapter_dir, arm)
        if not os.path.isdir(adapter):
            print(f"[skip] no adapter at {adapter}")
            continue
        curves[arm] = _sweep_arm(ecfg, model_name, adapter, arm,
                                 os.path.join(results_root, f"arm_{arm}"))

    out_png = args.out or os.path.join(results_root, "k_rate_arms_comparison.png")
    combined_plot(curves, p_star, out_png, model_name)
    print(f"\ncomparison plot: {out_png}")
    print("\n=== summary (mean K-rate over grid) ===")
    for label, df in curves.items():
        print(f"  {label:12s} mean_K={df['k_rate'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
