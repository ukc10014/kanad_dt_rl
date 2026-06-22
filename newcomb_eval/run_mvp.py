"""MVP entrypoint: config -> sweep -> table + plot + invalid-rate (PLAN.md §2 run_mvp.py).

Examples
--------
    # Full run against the configured default model (Gemma-small):
    python -m newcomb_eval.run_mvp

    # Fast smoke: 2 items x 3 p-values against a tiny cached model:
    python -m newcomb_eval.run_mvp --model Qwen/Qwen2.5-0.5B-Instruct \
        --limit 2 --p-grid 0.5 0.8 0.99
"""
from __future__ import annotations

import argparse
import dataclasses

from .config import EvalConfig
from .model import ModelWrapper
from .plot import plot_krate
from .sweep import run_sweep


def build_cfg(args) -> EvalConfig:
    cfg = EvalConfig()
    if args.dataset:
        cfg.dataset_path = args.dataset
    if args.results_dir:
        cfg.results_dir = args.results_dir
    if args.limit is not None:
        cfg.limit = args.limit
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.loader_mode:
        cfg.loader_mode = args.loader_mode

    # model overrides
    cfg.model = dataclasses.replace(
        cfg.model,
        model_name=args.model or cfg.model.model_name,
        adapter_path=args.adapter_path if args.adapter_path else cfg.model.adapter_path,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else cfg.model.max_new_tokens,
        temperature=args.temperature if args.temperature is not None else cfg.model.temperature,
        device_map=args.device_map or cfg.model.device_map,
        use_chat_template=not args.no_chat_template if args.no_chat_template else cfg.model.use_chat_template,
    )

    # sweep overrides
    sweep_kw = {}
    if args.p_grid:
        sweep_kw["p_grid"] = tuple(args.p_grid)
    if args.holdout_p is not None:
        sweep_kw["holdout_p"] = tuple(args.holdout_p)
    if args.n_repeats is not None:
        sweep_kw["n_repeats"] = args.n_repeats
    if sweep_kw:
        cfg.sweep = dataclasses.replace(cfg.sweep, **sweep_kw)

    if args.cot:
        cfg.prompt = dataclasses.replace(cfg.prompt, cot=True)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Newcomb predictor-accuracy eval — MVP runner")
    ap.add_argument("--model", help="HF model id or local path (default: config Gemma-small)")
    ap.add_argument("--adapter-path", help="PEFT/LoRA adapter path (LoRA seam; unused in MVP)")
    ap.add_argument("--dataset", help="path to dataset JSON")
    ap.add_argument("--loader-mode", choices=["templated", "text_injection"])
    ap.add_argument("--limit", type=int, help="cap number of items")
    ap.add_argument("--p-grid", type=float, nargs="+", help="p values to sweep")
    ap.add_argument("--holdout-p", type=float, nargs="*", help="held-out p values")
    ap.add_argument("--n-repeats", type=int, help="re-renders per (item,p) for CI")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-samples", type=int, help="Inspect sample concurrency (default 1)")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--cot", action="store_true", help="enable CoT prompt (off by default)")
    ap.add_argument("--results-dir")
    args = ap.parse_args(argv)

    cfg = build_cfg(args)
    print(f"[run_mvp] model={cfg.model.model_name} adapter={cfg.model.adapter_path}")
    print(f"[run_mvp] dataset={cfg.dataset_path} loader_mode={cfg.loader_mode} limit={cfg.limit}")
    print(f"[run_mvp] p_grid={cfg.sweep.p_grid} n_repeats={cfg.sweep.n_repeats}")

    wrapper = ModelWrapper(
        cfg.model.model_name,
        adapter_path=cfg.model.adapter_path,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
        use_chat_template=cfg.model.use_chat_template,
    )

    df = run_sweep(cfg, wrapper)
    p_star = df.attrs.get("p_star")

    print("\n=== K-rate(p) ===")
    cols = ["p", "k_rate", "n", "n_valid", "invalid_rate", "ci_lo", "ci_hi"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\np* = {p_star:.3f}   (csv: {df.attrs.get('csv_path')})")

    png = plot_krate(df, out_png=f"{cfg.results_dir}/k_rate_vs_p.png",
                     p_star=p_star, model_name=cfg.model.model_name)
    print(f"plot: {png}")
    overall_invalid = 1.0 - (df["n_valid"].sum() / df["n"].sum())
    print(f"overall invalid rate: {overall_invalid:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
