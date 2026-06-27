"""Payoff ablation — does the empirical K-rate crossover FOLLOW p*?

Hold B=100, vary S so the theoretical crossover p* = (1+S/B)/2 moves across {0.6,0.7,0.8,0.9}
(S ∈ {20,40,60,80}). For each payoff config, sweep stated p and measure CoT K-rate(p); locate the
empirical crossover (where K-rate steps through 0.5) and compare to the predicted p*.

  empirical crossover tracks p*  ⇒  the model genuinely computes EV from the stated numbers.
  empirical crossover stuck ~0.8 ⇒  it pattern-matched a memorised threshold, not real reasoning.

Reuses newcomb_eval.cot_inspect.inspect_records (CoT + forced-answer readout, abstract tokens,
invariant-safe) on ONE loaded model across all configs. Default model = Qwen2.5-3B-Instruct.
The payoffs are injected via the prompt (dataset_payoff_template.json markers) — never baked.
"""
from __future__ import annotations
import argparse, dataclasses, json, os, re, tempfile
from collections import defaultdict

from .config import EvalConfig
from .cot_inspect import inspect_records
from .crossover import crossover_p

TEMPLATE = os.path.join(os.path.dirname(__file__), "data", "dataset_payoff_template.json")


def make_config_dataset(template_items, B: float, S: float, path: str):
    """Write a temp dataset with {{PAYOFF_*}} markers replaced by concrete B/S (p kept as marker)."""
    out = []
    for it in template_items:
        q = (it["question"].replace("{{PAYOFF_BIG}}", str(int(B)))
                            .replace("{{PAYOFF_SMALL}}", str(int(S))))
        it2 = dict(it); it2["question"] = q
        it2["payoff_big"] = B; it2["payoff_small"] = S
        out.append(it2)
    json.dump(out, open(path, "w"))
    return path


def empirical_crossover(krate_by_p):
    """First upward crossing of K-rate=0.5, linearly interpolated. None if it never crosses."""
    ps = sorted(krate_by_p)
    for i in range(len(ps) - 1):
        p0, p1, k0, k1 = ps[i], ps[i + 1], krate_by_p[ps[i]], krate_by_p[ps[i + 1]]
        if k0 < 0.5 <= k1 and k1 != k0:
            return p0 + (0.5 - k0) * (p1 - p0) / (k1 - k0)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Payoff ablation: does K-rate crossover follow p*?")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--B", type=float, default=100.0)
    ap.add_argument("--p-stars", dest="p_stars", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--p-grid", dest="p_grid", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    ap.add_argument("-n", "--n-samples", dest="n_samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=5, help="number of items (None=all 20)")
    ap.add_argument("--tag", default="payoff_ablation")
    args = ap.parse_args(argv)

    template = json.load(open(TEMPLATE))
    if args.limit:
        template = template[:args.limit]

    from .model import ModelWrapper
    base = EvalConfig()
    print(f"[payoff-ablation] model={args.model} p*={args.p_stars} p-grid={args.p_grid} "
          f"items={len(template)} n={args.n_samples} mnt={args.max_new_tokens}", flush=True)
    wrapper = ModelWrapper(args.model, adapter_path=args.adapter, dtype=base.model.dtype,
                           device_map=base.model.device_map,
                           use_chat_template=base.model.use_chat_template)

    out_dir = os.path.join(base.results_dir, "payoff_ablation"); os.makedirs(out_dir, exist_ok=True)
    summary, all_recs = [], []
    for p_star in args.p_stars:
        S = round(args.B * (2 * p_star - 1))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            ds = make_config_dataset(template, args.B, S, tf.name)
        cfg = dataclasses.replace(
            base, dataset_path=ds, category_filter=None,
            crossover=dataclasses.replace(base.crossover, mode="global",
                                          payoff_big=args.B, payoff_small=S),
            model=dataclasses.replace(base.model, model_name=args.model))
        recs = inspect_records(cfg, wrapper, n_samples=args.n_samples, temperature=args.temperature,
                               max_new_tokens=args.max_new_tokens, p_grid=tuple(args.p_grid))
        os.unlink(ds)
        for r in recs:
            r["config_p_star"] = p_star; r["S"] = S
        all_recs += recs

        agg = defaultdict(list)
        for r in recs:
            agg[r["p"]].append(1 if r["answer_role"] == "non_cdt" else 0)
        krate = {p: sum(v) / len(v) for p, v in agg.items()}
        emp = empirical_crossover(krate)
        pred = crossover_p(args.B, S)
        summary.append(dict(p_star_pred=round(pred, 3), S=int(S),
                            p_star_emp=(round(emp, 3) if emp is not None else None),
                            krate={round(p, 2): round(k, 3) for p, k in sorted(krate.items())},
                            n=len(recs)))
        ks = "  ".join(f"{p:.2f}:{krate[p]:.2f}" for p in sorted(krate))
        print(f"[ablation] p*_pred={pred:.2f} (S={int(S)})  K-rate[{ks}]  "
              f"emp_xover={emp:.3f}" if emp is not None else
              f"[ablation] p*_pred={pred:.2f} (S={int(S)})  K-rate[{ks}]  emp_xover=NONE", flush=True)

    res = dict(model=args.model, B=args.B, summary=summary)
    json.dump(res, open(os.path.join(out_dir, f"{args.tag}_summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, f"{args.tag}_recs.jsonl"), "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print("\n=== SUMMARY: does empirical crossover follow p*? ===")
    print(f"{'p*_pred':>8} {'S':>4} {'p*_emp':>8}  follows?")
    for s in summary:
        emp = s["p_star_emp"]
        flag = "—" if emp is None else ("✓" if abs(emp - s["p_star_pred"]) <= 0.1 else "✗")
        print(f"{s['p_star_pred']:>8} {s['S']:>4} {str(emp):>8}  {flag}")
    print(f"\nwrote {out_dir}/{args.tag}_summary.json + _recs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
