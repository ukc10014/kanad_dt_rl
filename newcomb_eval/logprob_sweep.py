"""Logprob / logit-margin sweep (Pivot A) — the sub-argmax instrument.

K-rate binarises the decision; this sweep reads the *continuous* answer distribution instead:
for each (item, p) it measures `P(non_cdt | p)` and the logit margin
`logP(non_cdt) − logP(cdt)` at the decision point (see `ModelWrapper.answer_logprobs`). The
abstract single-token labels make this a clean two-way read, so we can see whether a "flat"
K-rate hides a *graded* margin that tracks p, and whether RL shifts the curve up (intercept) or
tilts it (slope).

    python -m newcomb_eval.logprob_sweep                       # base 3B
    python -m newcomb_eval.logprob_sweep --adapter results/adapters/evidential
    python -m newcomb_eval.logprob_sweep --model Qwen/Qwen2.5-0.5B-Instruct --limit 2
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os

from .config import EvalConfig
from .crossover import crossover_for_item
from .data.loader import load
from .prompts import build_prompt


def roles_to_tokens(token_role: dict) -> tuple[str, str]:
    """(non_cdt_token, cdt_token) from a per-sample token_role map."""
    non_cdt = next(t for t, r in token_role.items() if r == "non_cdt")
    cdt = next(t for t, r in token_role.items() if r == "cdt")
    return non_cdt, cdt


def aggregate_logprobs(rows):
    """Aggregate per-sample dicts (p, p_non_cdt, margin, is_k) into per-p stats.

    Returns a tidy DataFrame: p, p_non_cdt_mean, p_non_cdt_se, margin_mean, margin_se,
    k_rate_argmax (binarised companion), n.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    out = []
    for p, g in df.groupby("p"):
        n = len(g)
        sqn = math.sqrt(n) if n else float("nan")
        out.append({
            "p": p,
            "p_non_cdt_mean": g["p_non_cdt"].mean(),
            "p_non_cdt_se": (g["p_non_cdt"].std(ddof=1) / sqn) if n > 1 else 0.0,
            "margin_mean": g["margin"].mean(),
            "margin_se": (g["margin"].std(ddof=1) / sqn) if n > 1 else 0.0,
            "k_rate_argmax": g["is_k"].mean(),
            "n": n,
        })
    return pd.DataFrame(out).sort_values("p").reset_index(drop=True)


def run_logprob_sweep(cfg: EvalConfig, wrapper, *, prefix: str = "", out_dir: str | None = None,
                      tag: str = ""):
    """Loop items × p_grid × repeats, measure the 2-way answer distribution, aggregate."""
    out_dir = os.path.abspath(out_dir or os.path.join(cfg.results_dir, "logprob"))
    os.makedirs(out_dir, exist_ok=True)
    items = load(cfg.dataset_path, cfg)
    suffix = f"_{tag}" if tag else ""

    rows = []
    for item in items:
        for p in cfg.sweep.p_grid:
            for r in range(cfg.sweep.n_repeats):
                rp = build_prompt(item, p, dataclasses.replace(cfg.prompt, cot=False),
                                  sweep_seed=cfg.sweep.sweep_seed, repeat=r, mode=cfg.loader_mode)
                non_cdt, cdt = roles_to_tokens(rp.token_role)
                lp = wrapper.answer_logprobs(rp.text, non_cdt, cdt, prefix=prefix)
                rows.append({"item_id": item.id, "p": p, "repeat": r,
                             "p_non_cdt": lp["p_non_cdt"], "margin": lp["margin"],
                             "is_k": lp["is_k"]})

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"logprob_samples{suffix}.csv"), index=False)
    df = aggregate_logprobs(rows)
    csv_path = os.path.join(out_dir, f"p_margin_by_p{suffix}.csv")
    df.to_csv(csv_path, index=False)

    p_star = crossover_for_item(items[0], cfg.crossover).p_star
    p_slope = float(df["p_non_cdt_mean"].iloc[-1] - df["p_non_cdt_mean"].iloc[0])
    m_slope = float(df["margin_mean"].iloc[-1] - df["margin_mean"].iloc[0])
    print(f"P(non_cdt) slope (hi-lo) = {p_slope:+.3f}   margin slope = {m_slope:+.3f}   "
          f"K-rate(argmax) slope = {df['k_rate_argmax'].iloc[-1] - df['k_rate_argmax'].iloc[0]:+.2f}")
    print(f"table: {csv_path}   p* = {p_star:.2f}")
    df.attrs["p_star"] = p_star
    return df


def plot_logprob(df, p_star: float, out_png: str, model_name: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.sort_values("p")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(d["p"], d["p_non_cdt_mean"], marker="o", color="#1f77b4", label="P(non_cdt)")
    ax1.fill_between(d["p"], d["p_non_cdt_mean"] - d["p_non_cdt_se"],
                     d["p_non_cdt_mean"] + d["p_non_cdt_se"], alpha=0.15, color="#1f77b4")
    ax1.plot(d["p"], d["k_rate_argmax"], marker="s", linestyle="--", color="#7f7f7f",
             alpha=0.7, label="K-rate (argmax)")
    ax1.axvline(p_star, color="black", ls="--", alpha=0.6, label=f"p* = {p_star:.2f}")
    ax1.axhline(0.5, color="grey", lw=0.8, alpha=0.4)
    ax1.set_ylim(-0.03, 1.03); ax1.set_xlabel("stated accuracy p"); ax1.set_ylabel("P(non_cdt)")
    ax1.set_title("Continuous vs binarised one-box signal"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(d["p"], d["margin_mean"], marker="o", color="#d62728")
    ax2.fill_between(d["p"], d["margin_mean"] - d["margin_se"], d["margin_mean"] + d["margin_se"],
                     alpha=0.15, color="#d62728")
    ax2.axvline(p_star, color="black", ls="--", alpha=0.6)
    ax2.axhline(0.0, color="grey", lw=0.8, alpha=0.4)
    ax2.set_xlabel("stated accuracy p"); ax2.set_ylabel("logit margin  logP(non_cdt)−logP(cdt)")
    ax2.set_title("Logit margin vs p"); ax2.grid(alpha=0.3)

    fig.suptitle(f"Answer-logprob sweep — {model_name}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Answer-logprob / logit-margin sweep (Pivot A)")
    ap.add_argument("--model", help="HF model id (default: config Qwen2.5-3B-Instruct)")
    ap.add_argument("--adapter", help="PEFT/LoRA adapter dir to load on top of the base")
    ap.add_argument("--p-grid", dest="p_grid", nargs="+", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-n", "--repeats", type=int, default=None)
    ap.add_argument("--prefix", default="", help="teacher-forced text before the label (e.g. 'Answer: ')")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = EvalConfig()
    if args.model:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, model_name=args.model))
    if args.adapter:
        cfg = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, adapter_path=os.path.abspath(args.adapter)))
    if args.p_grid:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, p_grid=tuple(args.p_grid)))
    if args.repeats is not None:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, n_repeats=args.repeats))
    if args.limit is not None:
        cfg = dataclasses.replace(cfg, limit=args.limit)

    from .model import ModelWrapper
    wrapper = ModelWrapper(
        cfg.model.model_name, adapter_path=cfg.model.adapter_path,
        dtype=cfg.model.dtype, device_map=cfg.model.device_map,
        use_chat_template=cfg.model.use_chat_template,
    )
    df = run_logprob_sweep(cfg, wrapper, prefix=args.prefix, tag=args.tag)
    out_dir = os.path.abspath(os.path.join(cfg.results_dir, "logprob"))
    suffix = f"_{args.tag}" if args.tag else ""
    out_png = args.out or os.path.join(out_dir, f"p_margin_by_p{suffix}.png")
    plot_logprob(df, df.attrs["p_star"], out_png, cfg.model.model_name)
    print(f"plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
