"""CoT inspector — view the reasoning ↔ forced-answer ↔ reward combos in a custom HTML viewer.

The RL reward is computed *only* from the forced-`Answer:` readout, blind to the reasoning content
(`rloo._forced_answer_roles`). This tool makes that legible: for a grid of (item, p) it generates a
CoT, reads the forced answer the same way the RL loop does (`ModelWrapper.answer_logprobs(...,
reasoning=cot)`), and records the answer, its **margin** (how strongly the reasoning implied it —
near-0 ⇒ non-committal / possible CoT↔answer mismatch), the **reward**, and whether the answer was
**EV-optimal**. Writes JSONL + a self-contained, filterable HTML table.

    python -m newcomb_eval.cot_inspect --tag base3b -n 2
    python -m newcomb_eval.cot_inspect --adapter results/adapters/evidential_cot_kl005 --tag kl005
    python -m newcomb_eval.cot_inspect --model Qwen/Qwen2.5-0.5B-Instruct --limit 2 --p-grid 0.5 0.99
"""
from __future__ import annotations

import argparse
import dataclasses
import html as _html
import json
import os

from .config import EvalConfig
from .crossover import crossover_for_item, ev_cdt, ev_non_cdt
from .data.loader import load
from .logprob_sweep import roles_to_tokens
from .prompts import build_prompt

LOW_MARGIN = 1.0  # |margin| below this ⇒ the CoT didn't commit; flag for a human read


def inspect_records(cfg: EvalConfig, wrapper, *, n_samples: int = 2, temperature: float = 0.7,
                    max_new_tokens: int = 128, p_grid=None, seed0: int = 20_000) -> list[dict]:
    """Generate CoT + forced-answer readout per (item, p, sample); score reward + EV-optimality."""
    items = load(cfg.dataset_path, cfg)
    p_grid = p_grid or cfg.sweep.p_grid
    recs = []
    for item in items:
        xo = crossover_for_item(item, cfg.crossover)
        B, S = xo.payoff_big, xo.payoff_small
        for p in p_grid:
            ev_k, ev_c = ev_non_cdt(p, B, S), ev_cdt(p, B, S)
            optimal = "non_cdt" if ev_k > ev_c else "cdt"
            for s in range(n_samples):
                rp = build_prompt(item, p, dataclasses.replace(cfg.prompt, cot=True),
                                  sweep_seed=seed0, repeat=s, mode=cfg.loader_mode)
                non_cdt, cdt = roles_to_tokens(rp.token_role)
                cot = wrapper.generate_one(rp.text, max_new_tokens=max_new_tokens,
                                           temperature=temperature)
                # Read the forced answer the RL loop's way: chat-wrapped prompt + reasoning + cue,
                # then a merge-safe 2-way next-token readout over the two labels.
                text = wrapper._format(rp.text) + cot + "\nAnswer:"
                ans = wrapper.answer_2way(text, non_cdt, cdt)
                role = "non_cdt" if ans["is_k"] else "cdt"
                recs.append({
                    "item_id": item.id, "p": p, "sample": s, "p_star": xo.p_star,
                    "cot": cot, "answer_token": non_cdt if ans["is_k"] else cdt, "answer_role": role,
                    "p_non_cdt": round(ans["p_non_cdt"], 3), "margin": round(ans["margin"], 2),
                    "reward": round(ev_k if role == "non_cdt" else ev_c, 1),
                    "optimal_role": optimal, "is_optimal": role == optimal,
                })
    return recs


def render_html(recs: list[dict], title: str) -> str:
    n = len(recs) or 1
    n_opt = sum(r["is_optimal"] for r in recs)
    n_low = sum(abs(r["margin"]) < LOW_MARGIN for r in recs)
    mean_r = sum(r["reward"] for r in recs) / n
    rows = []
    for r in sorted(recs, key=lambda r: (r["p"], r["item_id"], r["sample"])):
        cls = "opt" if r["is_optimal"] else "subopt"
        low = "low" if abs(r["margin"]) < LOW_MARGIN else ""
        rows.append(
            f'<tr class="{cls} {low}" data-opt="{int(r["is_optimal"])}" '
            f'data-low="{int(abs(r["margin"]) < LOW_MARGIN)}">'
            f'<td>{_html.escape(r["item_id"])}</td><td>{r["p"]:.2f}</td>'
            f'<td>{r["p_star"]:.2f}</td>'
            f'<td><b>{_html.escape(r["answer_token"])}</b> {r["answer_role"]}</td>'
            f'<td>{"✓" if r["is_optimal"] else "✗ ("+r["optimal_role"]+")"}</td>'
            f'<td>{r["reward"]:.0f}</td><td>{r["p_non_cdt"]:.2f}</td>'
            f'<td>{r["margin"]:+.2f}</td>'
            f'<td><details><summary>view</summary><pre>{_html.escape(r["cot"])}</pre></details></td>'
            f'</tr>'
        )
    return f"""<!doctype html><meta charset=utf-8><title>{_html.escape(title)}</title>
<style>
 body{{font:14px/1.4 system-ui,sans-serif;margin:1.5rem;color:#222}}
 h1{{font-size:1.2rem}} .sum{{margin:.5rem 0 1rem;color:#444}}
 table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}}
 th{{background:#f4f4f4;position:sticky;top:0}}
 tr.subopt{{background:#fff0f0}} tr.opt{{background:#f0fff2}} tr.low td:nth-child(8){{background:#fff3cd;font-weight:bold}}
 pre{{white-space:pre-wrap;max-width:70ch;margin:.3rem 0;color:#333}}
 button{{margin-right:.5rem;padding:3px 8px}}
</style>
<h1>{_html.escape(title)}</h1>
<div class=sum><b>{n}</b> traces · <b>{100*n_opt/n:.0f}%</b> EV-optimal · mean reward <b>{mean_r:.1f}</b>
 · <b>{n_low}</b> low-margin (|margin|&lt;{LOW_MARGIN}, possible CoT↔answer mismatch — read these).
 <br>Green = answer was EV-optimal, red = not. Yellow margin = non-committal readout.</div>
<div><button onclick="f('all')">all</button><button onclick="f('sub')">suboptimal only</button>
 <button onclick="f('low')">low-margin only</button></div>
<table><thead><tr><th>item</th><th>p</th><th>p*</th><th>answer</th><th>optimal?</th><th>reward</th>
 <th>P(non_cdt)</th><th>margin</th><th>CoT</th></tr></thead><tbody>
{chr(10).join(rows)}
</tbody></table>
<script>
 function f(m){{document.querySelectorAll('tbody tr').forEach(t=>{{
   t.style.display=(m=='all')||(m=='sub'&&t.dataset.opt=='0')||(m=='low'&&t.dataset.low=='1')?'':'none';}});}}
</script>"""


def run(cfg: EvalConfig, wrapper, *, out_dir=None, tag="", **kw):
    out_dir = os.path.abspath(out_dir or os.path.join(cfg.results_dir, "cot_inspect"))
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    recs = inspect_records(cfg, wrapper, **kw)
    with open(os.path.join(out_dir, f"cot{suffix}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    html_path = os.path.join(out_dir, f"cot{suffix}.html")
    with open(html_path, "w") as f:
        f.write(render_html(recs, f"CoT inspector — {cfg.model.model_name}{(' + '+tag) if tag else ''}"))
    n = len(recs) or 1
    print(f"{len(recs)} traces · {100*sum(r['is_optimal'] for r in recs)/n:.0f}% EV-optimal · "
          f"{sum(abs(r['margin'])<LOW_MARGIN for r in recs)} low-margin")
    print(f"viewer: {html_path}")
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CoT inspector (reasoning ↔ answer ↔ reward)")
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--p-grid", dest="p_grid", nargs="+", type=float)
    ap.add_argument("--limit", type=int)
    ap.add_argument("-n", "--n-samples", dest="n_samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    cfg = EvalConfig()
    if args.model:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, model_name=args.model))
    if args.adapter:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(
            cfg.model, adapter_path=os.path.abspath(args.adapter)))
    if args.limit is not None:
        cfg = dataclasses.replace(cfg, limit=args.limit)

    from .model import ModelWrapper
    wrapper = ModelWrapper(cfg.model.model_name, adapter_path=cfg.model.adapter_path,
                           dtype=cfg.model.dtype, device_map=cfg.model.device_map,
                           use_chat_template=cfg.model.use_chat_template)
    run(cfg, wrapper, tag=args.tag, n_samples=args.n_samples, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, p_grid=tuple(args.p_grid) if args.p_grid else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
