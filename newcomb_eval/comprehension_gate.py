"""Comprehension gate — is the flat K-rate baseline 'WON'T track', not 'CAN'T parse'?

For each (item, p) we measure three things on the SAME rendered scenario (same abstract tokens):
  1. CHOICE       — the model's actual decision (forced-choice by default; --cot to deliberate).
  2. DOMINANCE probe — "which option takes BOTH containers?"  (correct = the two-box/CDT token).
  3. ACCURACY  probe — "what % of the time does the predictor forecast correctly?" (read back p).
A sample PASSES the comprehension gate iff it gets BOTH probes right. We then report the K-rate
*conditional on gate-pass* vs overall: if the model provably parses the structure and the stated
accuracy yet its choice is still flat in p, the flat baseline is a disposition ('won't'), not a
parsing failure ('can't'). Invariant #1 preserved — probes resolve via token_role / numeric
read-back, never natural-language string-matching of 'one-box'/'two-box'.

Reuses the payoff-templated items + ModelWrapper. Default model = Qwen2.5-3B-Instruct, S=60 (p*=0.8).
"""
from __future__ import annotations
import argparse, dataclasses, json, os, re, tempfile
from collections import defaultdict

from .config import EvalConfig
from .data.loader import load
from .prompts import render_scenario_block
from .logprob_sweep import roles_to_tokens
from .payoff_ablation import make_config_dataset

TEMPLATE = os.path.join(os.path.dirname(__file__), "data", "dataset_payoff_template.json")


def parse_pct(raw: str, p: float) -> bool:
    """True if the model's numeric read-back matches the stated accuracy (accepts 70 or 0.70)."""
    m = re.search(r"\d+\.?\d*", raw)
    if not m:
        return False
    x = float(m.group(0))
    return abs(x - round(p * 100)) <= 5 or abs(x - p) <= 0.05


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Comprehension gate: K-rate among provable comprehenders")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--B", type=float, default=100.0)
    ap.add_argument("--S", type=float, default=60.0)
    ap.add_argument("--p-grid", dest="p_grid", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    ap.add_argument("-n", "--n-samples", dest="n_samples", type=int, default=4)
    ap.add_argument("--cot", action="store_true", help="deliberate before the choice (default forced)")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--load-4bit", dest="load_4bit", action="store_true",
                    help="4-bit nf4 quantisation (for 32B on limited VRAM)")
    ap.add_argument("--tag", default="comprehension_gate")
    args = ap.parse_args(argv)

    template = json.load(open(TEMPLATE))
    if args.limit:
        template = template[:args.limit]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        ds = make_config_dataset(template, args.B, args.S, tf.name)

    base = EvalConfig()
    cfg = dataclasses.replace(base, dataset_path=ds, category_filter=None,
                              model=dataclasses.replace(base.model, model_name=args.model))
    items = load(ds, cfg)
    os.unlink(ds)

    from .model import ModelWrapper
    print(f"[gate] model={args.model} S={args.S} (p*={(args.B+args.S)/(2*args.B):.2f}) "
          f"cot={args.cot} items={len(items)} p-grid={args.p_grid} n={args.n_samples}", flush=True)
    wrapper = ModelWrapper(args.model, adapter_path=args.adapter, dtype=base.model.dtype,
                           device_map=base.model.device_map,
                           use_chat_template=base.model.use_chat_template,
                           load_in_4bit=args.load_4bit)

    pcfg = dataclasses.replace(base.prompt, cot=args.cot)
    instr = pcfg.cot_instruction if args.cot else pcfg.instruction
    recs = []
    for item in items:
        for p in args.p_grid:
            for s in range(args.n_samples):
                block = render_scenario_block(item, p, pcfg, sweep_seed=20_000, repeat=s,
                                              mode=cfg.loader_mode)
                non_cdt, cdt = roles_to_tokens(block.token_role)
                labels = f" (Valid labels: {', '.join(block.order)}.)"

                # 1) CHOICE
                choice_text = block.text + "\n" + instr + labels
                cot = ""
                if args.cot:
                    cot = wrapper.generate_one(choice_text, max_new_tokens=args.max_new_tokens,
                                               temperature=args.temperature)
                    ans = wrapper.answer_2way(wrapper._format(choice_text) + cot + "\nAnswer:",
                                              non_cdt, cdt)
                else:
                    ans = wrapper.answer_2way(wrapper._format(choice_text), non_cdt, cdt)
                choice_role = "non_cdt" if ans["is_k"] else "cdt"

                # 2) DOMINANCE probe — correct answer is the take-both (cdt) token
                dom_text = (block.text + "\nWhich option corresponds to taking BOTH containers? "
                            "Respond with exactly one label." + labels)
                dom = wrapper.answer_2way(wrapper._format(dom_text), non_cdt, cdt)
                dom_pass = dom["is_k"] == 0.0

                # 3) ACCURACY probe — read back the stated predictor accuracy
                acc_text = (block.text + "\nWhat percentage of the time does the system correctly "
                            "forecast your choice? Reply with just the number.")
                acc_raw = wrapper.generate_one(acc_text, max_new_tokens=8, temperature=0.0)
                acc_pass = parse_pct(acc_raw, p)

                recs.append(dict(item_id=item.id, p=p, sample=s, choice_role=choice_role,
                                 dom_pass=bool(dom_pass), acc_pass=bool(acc_pass),
                                 gate_pass=bool(dom_pass and acc_pass), acc_raw=acc_raw.strip()[:20],
                                 cot=cot))  # full reasoning trace (empty unless --cot)

    # ---- aggregate: K-rate overall vs conditional on comprehension-pass, per p ----
    by_p_all, by_p_gate = defaultdict(list), defaultdict(list)
    for r in recs:
        k = 1 if r["choice_role"] == "non_cdt" else 0
        by_p_all[r["p"]].append(k)
        if r["gate_pass"]:
            by_p_gate[r["p"]].append(k)
    dom_rate = sum(r["dom_pass"] for r in recs) / len(recs)
    acc_rate = sum(r["acc_pass"] for r in recs) / len(recs)
    gate_rate = sum(r["gate_pass"] for r in recs) / len(recs)

    out_dir = os.path.join(base.results_dir, "comprehension_gate"); os.makedirs(out_dir, exist_ok=True)
    summary = dict(model=args.model, S=args.S, cot=args.cot,
                   dom_pass_rate=round(dom_rate, 3), acc_pass_rate=round(acc_rate, 3),
                   gate_pass_rate=round(gate_rate, 3),
                   krate_all={round(p, 2): round(sum(v) / len(v), 3) for p, v in sorted(by_p_all.items())},
                   krate_gatepass={round(p, 2): (round(sum(v) / len(v), 3) if v else None)
                                   for p, v in sorted(by_p_gate.items())})
    json.dump(summary, open(os.path.join(out_dir, f"{args.tag}_summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, f"{args.tag}_recs.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    print(f"\ndominance-probe pass: {dom_rate:.0%}   accuracy-probe pass: {acc_rate:.0%}   "
          f"BOTH (gate): {gate_rate:.0%}")
    print(f"\n{'p':>5} {'K-rate(all)':>12} {'K-rate(gate-pass)':>18} {'n_gate':>7}")
    for p in sorted(by_p_all):
        kr = sum(by_p_all[p]) / len(by_p_all[p])
        g = by_p_gate.get(p, [])
        gr = (sum(g) / len(g)) if g else None
        print(f"{p:>5.2f} {kr:>12.2f} {('%.2f' % gr) if gr is not None else 'n/a':>18} {len(g):>7}")
    print(f"\n→ if K-rate stays flat among gate-passers, the flat baseline is WON'T, not CAN'T.")
    print(f"wrote {out_dir}/{args.tag}_summary.json + _recs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
