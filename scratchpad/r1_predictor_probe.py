"""Stage 0b — does a REASONING R1 predictor carry a p-conditional signal? (go/no-go for the loop)
              + does a neutral BREVITY nudge cut tokens without killing that signal?

Premise (R1_ITERATED_PLAN.md): if the predictor (a lagged copy of the policy) REASONS, its predicted
one-box probability P_pred(one-box | prompt, stated p) tracks stated p — making the evidential reward
conditional on p from step 0 (the lever the p-blind 3B lacked). This probe measures that directly, for
base R1 as the predictor, with GREEDY reasoning to match the trainer's `_predictor_p` (selfplay_cot).

reason-then-read: per (item, p): greedy `<think>` chain, append forced `Answer:`, read BOTH
  - argmax one-box rate  (robust; mirrors rloo._forced_answer_roles), and
  - soft P(non_cdt)      (2-way answer-token softmax = the exact p_eff the trainer uses).

Two prompt variants compared in ONE model load:
  - default : "Think step by step. ..." (the config CoT instruction)
  - brevity : "Reason concisely — a few sentences is enough. ..." (content-neutral length nudge)
DECISION (per "Prompt framing" temperature check): keep brevity ONLY if it cuts gen_len while the
p-slope survives (R1 needs *enough* reasoning to track p). Reasoning *content* is never hinted.

GO/NO-GO (default variant): P_pred + onebox both rise with p => build the loop.
Invariant #1 preserved: answer resolved over per-sample abstract tokens; no NL string-matching.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os

import torch

from newcomb_eval.scorer import ROLE_CDT, ROLE_INVALID, ROLE_NON_CDT, resolve_choice

CUE = "\nAnswer:"
BREVITY_INSTRUCTION = (
    "Reason concisely — a few sentences is enough. On the final line write "
    "'Answer: <LABEL>' with your choice."
)


def first_ids(tok, label):
    """Candidate first-token ids of `label` after 'Answer:' (bare or space-led)."""
    ids = []
    for s in (label, " " + label):
        t = tok(s, add_special_tokens=False).input_ids
        if t:
            ids.append(t[0])
    return ids


def build_grid_prompts(cfg, limit):
    from newcomb_eval.data.loader import load
    from newcomb_eval.prompts import build_prompt
    items = load(cfg.dataset_path, cfg)
    if limit:
        items = items[:limit]
    grid = list(cfg.sweep.p_grid)
    return grid, {p: [build_prompt(it, p, cfg.prompt, sweep_seed=0, repeat=0, mode=cfg.loader_mode)
                      for it in items] for p in grid}


@torch.no_grad()
def reason_then_read(model, tok, rps, mnt, micro, wrap, device):
    """Per prompt -> (P_pred_non_cdt soft, role argmax, closed_think, gen_len). GREEDY (trainer-matched)."""
    soft, roles, closed, glen = [], [], [], []
    for i in range(0, len(rps), micro):
        chunk = rps[i:i + micro]
        enc = tok([wrap(r.text) for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        Lp = enc.input_ids.shape[1]
        gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.pad_token_id)
        comp = tok.batch_decode(gen[:, Lp:], skip_special_tokens=True)
        for c in comp:
            closed.append("</think>" in c)
            glen.append(len(tok(c, add_special_tokens=False).input_ids))
        texts2 = [wrap(r.text) + c + CUE for r, c in zip(chunk, comp)]
        enc2 = tok(texts2, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc2, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
        s0 = out.scores[0].float()
        ans = tok.batch_decode(out.sequences[:, enc2.input_ids.shape[1]:], skip_special_tokens=True)
        for j, r in enumerate(chunk):
            non = next(t for t, role in r.token_role.items() if role == ROLE_NON_CDT)
            cdt = next(t for t, role in r.token_role.items() if role == ROLE_CDT)
            ln = max(s0[j, k].item() for k in first_ids(tok, non))
            lc = max(s0[j, k].item() for k in first_ids(tok, cdt))
            soft.append(torch.softmax(torch.tensor([ln, lc]), 0)[0].item())
            role, _t, valid = resolve_choice(ans[j], r.legal_tokens, r.token_role, cot=False)
            roles.append(role if valid else ROLE_INVALID)
    return soft, roles, closed, glen


def run_variant(model, tok, cfg, limit, mnt, micro, wrap, device):
    grid, by_p = build_grid_prompts(cfg, limit)
    rows = []
    print(f"\n{'p':>5} {'P_pred(soft)':>13} {'onebox':>8} {'invalid':>8} {'closed':>7} {'gen_len':>8}",
          flush=True)
    for p in grid:
        soft, roles, closed, glen = reason_then_read(model, tok, by_p[p], mnt, micro, wrap, device)
        n = len(roles)
        nk = sum(r == ROLE_NON_CDT for r in roles)
        ninv = sum(r == ROLE_INVALID for r in roles)
        rows.append(dict(p=p, p_pred_soft=sum(soft) / n, onebox=nk / max(1, n - ninv),
                         invalid=ninv / n, closed=sum(closed) / n, gen_len=sum(glen) / n, n=n))
        r = rows[-1]
        print(f"{p:>5.2f} {r['p_pred_soft']:>13.3f} {r['onebox']:>8.3f} {r['invalid']:>8.2f} "
              f"{r['closed']:>7.2f} {r['gen_len']:>8.0f}", flush=True)
    lo, hi = rows[0], rows[-1]
    summ = dict(slope_soft=hi["p_pred_soft"] - lo["p_pred_soft"], slope_ob=hi["onebox"] - lo["onebox"],
                gen_len=sum(r["gen_len"] for r in rows) / len(rows),
                closed=sum(r["closed"] for r in rows) / len(rows))
    print(f"  slope(p={hi['p']}-{lo['p']}): P_pred {summ['slope_soft']:+.3f}  onebox {summ['slope_ob']:+.3f}"
          f"   mean gen_len {summ['gen_len']:.0f}  closed {summ['closed']:.0%}")
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dataset", default="newcomb_eval/data/dataset_mech_m3.json",
                    help="BINDING m3 exact-copy framing (honest for self-snapshot; lifted EDT). NOT m0.")
    ap.add_argument("--max-new-tokens", dest="mnt", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=6, help="items per p")
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--variants", default="default,brevity", help="comma-sep: default,brevity")
    ap.add_argument("--tag", default="r1d")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from newcomb_eval.config import EvalConfig
    base = EvalConfig()
    instr = {"default": base.prompt.cot_instruction, "brevity": BREVITY_INSTRUCTION}

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    device = "cuda"

    def wrap(text):
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)

    print(f"[0b] model={args.model} dataset={args.dataset} items/p={args.limit} mnt={args.mnt} "
          f"variants={args.variants} (GREEDY reasoning, trainer-matched)", flush=True)

    results = {}
    for v in args.variants.split(","):
        print(f"\n========== variant: {v} ==========\n  instruction: {instr[v]!r}")
        cfg = dataclasses.replace(base, dataset_path=args.dataset, category_filter=None,
                                  prompt=dataclasses.replace(base.prompt, cot=True,
                                                             cot_instruction=instr[v]))
        rows, summ = run_variant(model, tok, cfg, args.limit, args.mnt, args.micro, wrap, device)
        results[v] = (rows, summ)
        os.makedirs("results/credence", exist_ok=True)
        with open(f"results/credence/r1_predictor_probe_{args.tag}_{v}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n========== SUMMARY ==========")
    print(f"{'variant':>9} {'P_pred slope':>13} {'onebox slope':>13} {'mean gen_len':>13} {'closed':>7}")
    for v, (_, s) in results.items():
        print(f"{v:>9} {s['slope_soft']:>+13.3f} {s['slope_ob']:>+13.3f} {s['gen_len']:>13.0f} "
              f"{s['closed']:>7.0%}")
    print("\nGO/NO-GO (default): build the loop iff P_pred & onebox slopes are clearly positive.")
    if "default" in results and "brevity" in results:
        d, b = results["default"][1], results["brevity"][1]
        cut = 1 - b["gen_len"] / max(1.0, d["gen_len"])
        kept = b["slope_soft"] >= 0.5 * d["slope_soft"] and b["slope_soft"] > 0.2
        verdict = ("KEEP brevity" if (cut > 0.25 and kept) else
                   "DROP brevity (slope not preserved)" if not kept else
                   "marginal — brevity barely cuts tokens; default is fine")
        print(f"BREVITY: gen_len cut {cut:+.0%}, P_pred slope {b['slope_soft']:+.3f} vs "
              f"{d['slope_soft']:+.3f} (default) => {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
