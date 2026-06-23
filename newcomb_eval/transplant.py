"""Computation-transplant diagnostic for Newcomb-like EV bottlenecks.

This is an inference-time causal probe, not training. It asks: if we externally supply
increasing amounts of the expected-value calculation, does the model's two-way answer
distribution start tracking the EV-optimal label?

Conditions are ordered from no help to full calculation:

  none          original rendered item + a forced ``Answer:`` readout
  variables     p/B/S + which abstract label is the only-conditional vs both option
  formulas      variables + EV formulas attached to the current labels
  numeric_evs   variables + computed EVs for the current labels
  comparison    variables + a direct statement of which label has higher EV
  full          variables + formulas + computed EVs + comparison

The key interpretability readout is where the answer flips:
  - formulas/numeric_evs help -> arithmetic/formula bottleneck
  - comparison/full helps only -> comparison-to-label/inclination bottleneck
  - full still fails -> a strong inclination/format prior overrides explicit calculation

CLI:
    python -m newcomb_eval.transplant --limit 2 --tag smoke
    python -m newcomb_eval.transplant --adapter results/adapters/evidential_paired_cot --tag paired
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass
from typing import Literal

from .config import EvalConfig
from .crossover import crossover_for_item, ev_cdt, ev_non_cdt
from .data.loader import load
from .prompts import render_scenario_block

ROLE_NON_CDT = "non_cdt"
ROLE_CDT = "cdt"

CONDITIONS = ("none", "variables", "formulas", "numeric_evs", "comparison", "full")
Condition = Literal["none", "variables", "formulas", "numeric_evs", "comparison", "full"]


@dataclass
class TransplantPrompt:
    text: str
    item_id: str
    p: float
    condition: str
    legal_tokens: list[str]
    token_role: dict
    non_cdt_token: str
    cdt_token: str
    optimal_token: str
    optimal_role: str
    ev_non_cdt: float
    ev_cdt: float
    p_star: float


def role_tokens(token_role: dict) -> tuple[str, str]:
    """Return ``(non_cdt_token, cdt_token)`` from a per-sample token_role map."""
    non_cdt = next(t for t, r in token_role.items() if r == ROLE_NON_CDT)
    cdt = next(t for t, r in token_role.items() if r == ROLE_CDT)
    return non_cdt, cdt


def _fmt_num(x: float) -> str:
    """Compact, stable numeric formatting for prompts and tests."""
    return f"{x:.3f}".rstrip("0").rstrip(".")


def ev_summary(p: float, B: float, S: float) -> dict:
    """Compute EVs and the optimal role under the evidential/Newcomb calculation."""
    ev_k = ev_non_cdt(p, B, S)
    ev_c = ev_cdt(p, B, S)
    if ev_k > ev_c:
        optimal = ROLE_NON_CDT
    elif ev_c > ev_k:
        optimal = ROLE_CDT
    else:
        optimal = "tie"
    return {"ev_non_cdt": ev_k, "ev_cdt": ev_c, "optimal_role": optimal}


def transplant_block(
    condition: str,
    *,
    p: float,
    B: float,
    S: float,
    non_cdt_token: str,
    cdt_token: str,
    ev_non_cdt_value: float,
    ev_cdt_value: float,
    optimal_token: str,
) -> str:
    """Return the external calculation aid for one condition.

    The wording deliberately uses mechanically descriptive option names, not decision-theory
    labels. Abstract labels remain the only answer surface.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown transplant condition {condition!r}")
    if condition == "none":
        return ""

    # Collision-proof wording: no single-letter variable names (p/B/S collide with abstract
    # answer labels like S, B, P), and labels are backtick-quoted to set them apart.
    p_pct = f"{100 * p:.0f}%"
    variables = (
        "Helpful calculation aid:\n"
        f"- Stated accuracy = {_fmt_num(p)} ({p_pct}).\n"
        f"- conditional_amount = {_fmt_num(B)} points.\n"
        f"- guaranteed_amount = {_fmt_num(S)} points.\n"
        f"- Label `{non_cdt_token}` takes only the conditional amount.\n"
        f"- Label `{cdt_token}` takes both the conditional amount and the guaranteed amount.\n"
    )
    formulas = (
        f"- EV for label `{non_cdt_token}` = accuracy × conditional_amount.\n"
        f"- EV for label `{cdt_token}` = guaranteed_amount + (1 − accuracy) × conditional_amount.\n"
    )
    numeric = (
        f"- EV for label `{non_cdt_token}` = {_fmt_num(ev_non_cdt_value)} points.\n"
        f"- EV for label `{cdt_token}` = {_fmt_num(ev_cdt_value)} points.\n"
    )
    if optimal_token == "tie":
        comparison = "- The two labels have equal expected value.\n"
    else:
        comparison = f"- The higher expected-value label is `{optimal_token}`.\n"

    pieces = [variables]
    if condition == "variables":
        pass
    elif condition == "formulas":
        pieces.append(formulas)
    elif condition == "numeric_evs":
        pieces.append(numeric)
    elif condition == "comparison":
        pieces.append(comparison)
    elif condition == "full":
        pieces.extend([formulas, numeric, comparison])
    return "".join(pieces)


def build_transplant_prompt(
    item,
    p: float,
    cfg: EvalConfig,
    condition: str,
    *,
    sweep_seed: int = 0,
    repeat: int = 0,
) -> TransplantPrompt:
    """Render one item/p/condition prompt plus metadata for two-way answer readout."""
    block = render_scenario_block(
        item, p, cfg.prompt, sweep_seed=sweep_seed, repeat=repeat, mode=cfg.loader_mode
    )
    xo = crossover_for_item(item, cfg.crossover)
    non_cdt_token, cdt_token = role_tokens(block.token_role)
    evs = ev_summary(p, xo.payoff_big, xo.payoff_small)
    optimal_role = evs["optimal_role"]
    optimal_token = (
        non_cdt_token
        if optimal_role == ROLE_NON_CDT
        else cdt_token if optimal_role == ROLE_CDT else "tie"
    )

    aid = transplant_block(
        condition,
        p=p,
        B=xo.payoff_big,
        S=xo.payoff_small,
        non_cdt_token=non_cdt_token,
        cdt_token=cdt_token,
        ev_non_cdt_value=evs["ev_non_cdt"],
        ev_cdt_value=evs["ev_cdt"],
        optimal_token=optimal_token,
    )
    lines = [block.text.rstrip()]
    if aid:
        lines += ["", aid.rstrip()]
    lines += [
        "",
        "Using the scenario and any calculation aid above, answer with exactly one valid label.",
        f"Valid labels: {', '.join(block.order)}.",
    ]
    return TransplantPrompt(
        text="\n".join(lines),
        item_id=item.id,
        p=p,
        condition=condition,
        legal_tokens=block.legal_tokens,
        token_role=block.token_role,
        non_cdt_token=non_cdt_token,
        cdt_token=cdt_token,
        optimal_token=optimal_token,
        optimal_role=optimal_role,
        ev_non_cdt=evs["ev_non_cdt"],
        ev_cdt=evs["ev_cdt"],
        p_star=xo.p_star,
    )


def aggregate(rows):
    """Aggregate transplant readout rows by condition/p."""
    import pandas as pd

    df = pd.DataFrame(rows)
    out = []
    for (condition, p), g in df.groupby(["condition", "p"]):
        n = len(g)
        out.append(
            {
                "condition": condition,
                "p": p,
                "p_non_cdt_mean": g["p_non_cdt"].mean(),
                "p_optimal_mean": g["p_optimal"].mean(),
                "margin_mean": g["margin"].mean(),
                "k_rate_argmax": g["is_k"].mean(),
                "optimal_rate_argmax": g["is_optimal_argmax"].mean(),
                "n": n,
            }
        )
    order = {c: i for i, c in enumerate(CONDITIONS)}
    return (
        pd.DataFrame(out)
        .assign(_ord=lambda d: d["condition"].map(order))
        .sort_values(["_ord", "p"])
        .drop(columns=["_ord"])
        .reset_index(drop=True)
    )


def run_transplant_sweep(
    cfg: EvalConfig,
    wrapper,
    *,
    conditions=CONDITIONS,
    repeats: int | None = None,
    out_dir: str | None = None,
    tag: str = "",
):
    """Run the transplant diagnostic and persist per-sample + aggregate CSVs."""
    import pandas as pd

    out_dir = os.path.abspath(out_dir or os.path.join(cfg.results_dir, "transplant"))
    os.makedirs(out_dir, exist_ok=True)
    items = load(cfg.dataset_path, cfg)
    repeats = cfg.sweep.n_repeats if repeats is None else repeats
    suffix = f"_{tag}" if tag else ""

    rows = []
    for item in items:
        for p in cfg.sweep.p_grid:
            for r in range(repeats):
                for condition in conditions:
                    tp = build_transplant_prompt(
                        item, p, cfg, condition,
                        sweep_seed=cfg.sweep.sweep_seed, repeat=r,
                    )
                    read = wrapper.answer_2way(
                        wrapper._format(tp.text) + "\nAnswer:",
                        tp.non_cdt_token,
                        tp.cdt_token,
                    )
                    is_k = float(read["is_k"])
                    if tp.optimal_role == ROLE_NON_CDT:
                        p_opt = read["p_non_cdt"]
                    elif tp.optimal_role == ROLE_CDT:
                        p_opt = 1.0 - read["p_non_cdt"]
                    else:
                        p_opt = float("nan")
                    rows.append(
                        {
                            "item_id": tp.item_id,
                            "p": tp.p,
                            "repeat": r,
                            "condition": condition,
                            "non_cdt_token": tp.non_cdt_token,
                            "cdt_token": tp.cdt_token,
                            "optimal_token": tp.optimal_token,
                            "optimal_role": tp.optimal_role,
                            "ev_non_cdt": tp.ev_non_cdt,
                            "ev_cdt": tp.ev_cdt,
                            "p_star": tp.p_star,
                            "p_non_cdt": read["p_non_cdt"],
                            "p_optimal": p_opt,
                            "margin": read["margin"],
                            "is_k": is_k,
                            "is_optimal_argmax": (
                                float(
                                    (is_k == 1.0 and tp.optimal_role == ROLE_NON_CDT)
                                    or (is_k == 0.0 and tp.optimal_role == ROLE_CDT)
                                )
                                if tp.optimal_role != "tie"
                                else float("nan")
                            ),
                        }
                    )

    sample_path = os.path.join(out_dir, f"transplant_samples{suffix}.csv")
    pd.DataFrame(rows).to_csv(sample_path, index=False)
    df = aggregate(rows)
    agg_path = os.path.join(out_dir, f"transplant_by_condition_p{suffix}.csv")
    df.to_csv(agg_path, index=False)
    print(f"samples: {sample_path}\naggregate: {agg_path}")
    for condition in conditions:
        d = df[df["condition"] == condition].sort_values("p")
        if not d.empty:
            slope = float(d["p_optimal_mean"].iloc[-1] - d["p_optimal_mean"].iloc[0])
            print(
                f"{condition:12s} mean P(optimal)={d['p_optimal_mean'].mean():.3f} "
                f"hi-lo={slope:+.3f}"
            )
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Computation-transplant diagnostic")
    ap.add_argument("--model", help="HF model id (default: EvalConfig)")
    ap.add_argument("--adapter", help="PEFT/LoRA adapter dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--p-grid", dest="p_grid", nargs="+", type=float)
    ap.add_argument("-n", "--repeats", type=int)
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)

    cfg = EvalConfig()
    if args.model:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, model_name=args.model))
    if args.adapter:
        cfg = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, adapter_path=os.path.abspath(args.adapter))
        )
    if args.limit is not None:
        cfg = dataclasses.replace(cfg, limit=args.limit)
    if args.p_grid:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, p_grid=tuple(args.p_grid)))
    if args.repeats is not None:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, n_repeats=args.repeats))

    from .model import ModelWrapper

    wrapper = ModelWrapper(
        cfg.model.model_name,
        adapter_path=cfg.model.adapter_path,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
        use_chat_template=cfg.model.use_chat_template,
    )
    run_transplant_sweep(
        cfg, wrapper, conditions=tuple(args.conditions), out_dir=args.out_dir, tag=args.tag
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
