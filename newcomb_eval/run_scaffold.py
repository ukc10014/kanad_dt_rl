"""Run the scaffolded-CoT three-arm experiment and plot K-rate(p) per arm (PLAN: scaffold).

    # show the three arm prompts + 5 scaffold steps, no inference:
    python -m newcomb_eval.run_scaffold --dry-run

    # smoke (tiny model, 2 items, 3 p):
    python -m newcomb_eval.run_scaffold --model Qwen/Qwen2.5-0.5B-Instruct --limit 2 \
        --p-grid 0.5 0.8 0.99

    # full base run, all three arms:
    python -m newcomb_eval.run_scaffold

    # capability-cliff probe on a bigger model (inference only):
    python -m newcomb_eval.run_scaffold --model Qwen/Qwen2.5-14B-Instruct

    # scaffold an RL'd adapter:
    python -m newcomb_eval.run_scaffold --arms scaffolded --adapter results/adapters/evidential_cot
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import EvalConfig  # noqa: E402
from .crossover import crossover_for_item  # noqa: E402
from .data.loader import load  # noqa: E402
from .prompts import build_prompt, render_scenario_block  # noqa: E402
from .scaffold import ARMS, SCAFFOLD_STEPS, run_scaffold_sweep  # noqa: E402

_COLORS = {"no_cot": "#7f7f7f", "free_cot": "#ff7f0e", "scaffolded": "#2ca02c"}


def _slope(df) -> float:
    d = df.sort_values("p")
    return float(d["k_rate"].iloc[-1] - d["k_rate"].iloc[0])


def combined_plot(curves: dict, p_star: float, out_png: str, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, df in curves.items():
        d = df.sort_values("p")
        c = _COLORS.get(label)
        ax.plot(d["p"], d["k_rate"], marker="o", color=c, label=f"{label} (slope {_slope(df):+.2f})")
        if {"ci_lo", "ci_hi"}.issubset(d.columns):
            ax.fill_between(d["p"], d["ci_lo"], d["ci_hi"], alpha=0.12, color=c)
    ax.axvline(p_star, color="black", linestyle="--", alpha=0.6, label=f"p* = {p_star:.2f}")
    ax.axhline(0.5, color="grey", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("stated predictor accuracy  p")
    ax.set_ylabel("non-CDT (one-box) selection rate  K-rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Scaffolded CoT: K-rate(p) by reasoning arm\n{model_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def _dry_run(cfg: EvalConfig) -> None:
    items = load(cfg.dataset_path, cfg)
    item = items[0]
    p = cfg.sweep.p_grid[len(cfg.sweep.p_grid) // 2]
    print(f"=== dry-run: item={item.id}  p={p}  (mode={cfg.loader_mode}) ===\n")

    no = build_prompt(item, p, dataclasses.replace(cfg.prompt, cot=False), mode=cfg.loader_mode)
    print("--- NO-COT PROMPT ---\n" + no.text + "\n")

    fc = build_prompt(item, p, dataclasses.replace(cfg.prompt, cot=True), mode=cfg.loader_mode)
    print("--- FREE-COT PROMPT ---\n" + fc.text + "\n")

    block = render_scenario_block(item, p, cfg.prompt, mode=cfg.loader_mode)
    labels = ", ".join(block.order)
    print("--- SCAFFOLDED (turn 1 = scenario + step 1) ---")
    print(block.text + "\n" + SCAFFOLD_STEPS[0]["prompt"] + "\n")
    print("--- subsequent step prompts ---")
    for step in SCAFFOLD_STEPS[1:]:
        print(f"\n[{step['id']}]\n" + step["prompt"].format(labels=labels))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scaffolded-CoT three-arm experiment")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--model", help="HF model id (default: config Qwen2.5-3B-Instruct)")
    ap.add_argument("--adapter", help="PEFT/LoRA adapter dir to load on top of the base")
    ap.add_argument("--p-grid", dest="p_grid", nargs="+", type=float, default=None)
    ap.add_argument("-n", "--repeats", type=int, default=1, help="re-renders per (item, p)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of items (smoke)")
    ap.add_argument("--free-max", dest="free_max", type=int, default=256)
    ap.add_argument("--step-max", dest="step_max", type=int, default=256)
    ap.add_argument("--decision-max", dest="decision_max", type=int, default=48)
    ap.add_argument("--tag", default="", help="suffix on output files")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = EvalConfig()
    if args.model:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, model_name=args.model))
    if args.adapter:
        cfg = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, adapter_path=os.path.abspath(args.adapter))
        )
    if args.p_grid:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, p_grid=tuple(args.p_grid)))
    if args.limit is not None:
        cfg = dataclasses.replace(cfg, limit=args.limit)

    if args.dry_run:
        _dry_run(cfg)
        return 0

    from .model import ModelWrapper

    wrapper = ModelWrapper(
        cfg.model.model_name, adapter_path=cfg.model.adapter_path,
        dtype=cfg.model.dtype, device_map=cfg.model.device_map,
        use_chat_template=cfg.model.use_chat_template,
    )
    curves, _ = run_scaffold_sweep(
        cfg, wrapper, arms=args.arms, repeats=args.repeats,
        step_max=args.step_max, free_max=args.free_max, decision_max=args.decision_max,
        tag=args.tag,
    )

    items = load(cfg.dataset_path, cfg)
    p_star = crossover_for_item(items[0], cfg.crossover).p_star
    out_dir = os.path.abspath(os.path.join(cfg.results_dir, "scaffold"))
    suffix = f"_{args.tag}" if args.tag else ""
    out_png = args.out or os.path.join(out_dir, f"k_rate_by_arm{suffix}.png")
    combined_plot(curves, p_star, out_png, cfg.model.model_name)
    print(f"plot: {out_png}")

    print("\n=== summary (mean K-rate, slope hi-lo) ===")
    for arm, df in curves.items():
        print(f"  {arm:11s} mean_K={df['k_rate'].mean():.3f}  slope={_slope(df):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
