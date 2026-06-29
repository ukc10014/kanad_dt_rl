"""Framing sweep — is the 14B's one-boxing-with-CoT a property of the MODEL or of the FRAMING?

The payoff ablation showed the 14B, under free-CoT on the **m3 (exact-copy)** framing, one-boxes at
high p and its crossover tracks p*. But the earlier "scale worsens" headline (14B two-boxes hard even
at p=0.99) was on the **abstract / statistical** framing. Those differ in framing AND reasoning mode at
once, so neither "scale fixes" nor "scale worsens" is established. This driver isolates framing: SAME
model, SAME free-CoT, SAME payoff (B=100, S=60, p*=0.8), swept across the mechanism-credibility ladder

    m0  statistical   "F correctly identifies the choices of agents like you"
    m1  indiv-model   "built a detailed predictive model of your reasoning ... simulating"
    m2  process-scan  "scanned your full decision process ... run that very process forward"
    m3  exact-copy    "executing an exact copy of your own decision procedure ... forecast is identical"

Read:
  * one-box(p) rises with p on m3 but stays flat-low on m0  ⇒  the disposition is a FRAMING artifact;
    free-CoT recruits the EV/FDT reasoning only when the dependence is made explicit.
  * one-box(p) tracks p on ALL rungs (incl. m0)             ⇒  it's the CoT, not the framing; the
    earlier abstract two-boxing was a forced-choice/transplant effect, not a free-CoT one.
  * flat-low on ALL rungs                                   ⇒  the m3 payoff-ablation crossover was a
    fluke of that run; quarantine it.

Companion to the *forced-choice* mechanism ladder (`credence_mechanism`, 2026-06-24), which found the
ACTION one-box rate only mildly framing-sensitive (m0 .59 → m3 .69). This is the free-CoT counterpart.
Invariant-safe: reuses `cot_inspect.inspect_records` (abstract tokens, forced-Answer readout).
"""
from __future__ import annotations
import argparse, dataclasses, json, os
from collections import defaultdict

from .config import EvalConfig
from .cot_inspect import inspect_records
from .crossover import crossover_p
from .payoff_ablation import empirical_crossover

LEVELS = [  # (key, dataset file, short label)
    ("m0", "dataset_mech_m0.json", "statistical"),
    ("m1", "dataset_mech_m1.json", "indiv-model"),
    ("m2", "dataset_mech_m2.json", "process-scan"),
    ("m3", "dataset_mech_m3.json", "exact-copy"),
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Framing sweep: does free-CoT one-boxing depend on framing?")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--B", type=float, default=100.0)
    ap.add_argument("--S", type=float, default=60.0)
    ap.add_argument("--levels", nargs="+", default=["m0", "m1", "m2", "m3"])
    ap.add_argument("--p-grid", dest="p_grid", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    ap.add_argument("-n", "--n-samples", dest="n_samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=5, help="items per framing (None=all)")
    ap.add_argument("--load-4bit", dest="load_4bit", action="store_true")
    ap.add_argument("--tag", default="framing_sweep")
    args = ap.parse_args(argv)

    levels = [lv for lv in LEVELS if lv[0] in args.levels]
    base = EvalConfig()
    pred = crossover_p(args.B, args.S)
    from .model import ModelWrapper
    print(f"[framing-sweep] model={args.model} B={args.B} S={args.S} (p*={pred:.2f}) "
          f"levels={[l[0] for l in levels]} p-grid={args.p_grid} n={args.n_samples}", flush=True)
    wrapper = ModelWrapper(args.model, adapter_path=args.adapter, dtype=base.model.dtype,
                           device_map=base.model.device_map,
                           use_chat_template=base.model.use_chat_template,
                           load_in_4bit=args.load_4bit)

    out_dir = os.path.join(base.results_dir, "framing_sweep"); os.makedirs(out_dir, exist_ok=True)
    recs_path = os.path.join(out_dir, f"{args.tag}_recs.jsonl")
    summ_path = os.path.join(out_dir, f"{args.tag}_summary.json")
    open(recs_path, "w").close()  # truncate; we append per-level (crash-safe — don't buffer to the end)
    summary, all_recs = [], []
    for key, fname, label in levels:
        ds = os.path.join(DATA_DIR, fname)
        cfg = dataclasses.replace(
            base, dataset_path=ds, category_filter=None,
            crossover=dataclasses.replace(base.crossover, mode="global",
                                          payoff_big=args.B, payoff_small=args.S),
            model=dataclasses.replace(base.model, model_name=args.model))
        recs = inspect_records(cfg, wrapper, n_samples=args.n_samples, temperature=args.temperature,
                               max_new_tokens=args.max_new_tokens, p_grid=tuple(args.p_grid))
        if args.limit:  # inspect_records loads all items; keep only the first N item_ids
            keep = list(dict.fromkeys(r["item_id"] for r in recs))[:args.limit]
            recs = [r for r in recs if r["item_id"] in keep]
        for r in recs:
            r["framing"] = key; r["framing_label"] = label
        all_recs += recs

        agg = defaultdict(list)
        for r in recs:
            agg[r["p"]].append(1 if r["answer_role"] == "non_cdt" else 0)
        krate = {p: sum(v) / len(v) for p, v in agg.items()}
        emp = empirical_crossover(krate)
        summary.append(dict(framing=key, label=label,
                            krate={round(p, 2): round(k, 3) for p, k in sorted(krate.items())},
                            p_star_emp=(round(emp, 3) if emp is not None else None), n=len(recs)))
        ks = "  ".join(f"{p:.2f}:{krate[p]:.2f}" for p in sorted(krate))
        print(f"[framing {key} {label}]  K-rate[{ks}]  "
              + (f"emp_xover={emp:.3f}" if emp is not None else "emp_xover=NONE"), flush=True)
        # crash-safe: persist this level's CoT recs + the running summary immediately
        with open(recs_path, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        json.dump(dict(model=args.model, B=args.B, S=args.S, p_star_pred=round(pred, 3),
                       summary=summary), open(summ_path, "w"), indent=1)

    res = dict(model=args.model, B=args.B, S=args.S, p_star_pred=round(pred, 3), summary=summary)
    json.dump(res, open(summ_path, "w"), indent=1)
    with open(recs_path, "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"\n=== SUMMARY: does free-CoT one-boxing depend on framing? (p*_pred={pred:.2f}) ===")
    print(f"{'framing':>12} {'one-box @ .5/.6/.7/.8/.9/.99':>32} {'emp_xover':>10}")
    for s in summary:
        ks = " ".join(f"{s['krate'].get(round(p,2), float('nan')):.2f}" for p in args.p_grid)
        print(f"{s['framing']+' '+s['label']:>12} {ks:>32} {str(s['p_star_emp']):>10}")
    print(f"\nwrote {out_dir}/{args.tag}_summary.json + _recs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
