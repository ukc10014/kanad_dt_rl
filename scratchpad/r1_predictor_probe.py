"""Stage 0b — does a REASONING R1 predictor carry a p-conditional signal? (go/no-go for the loop)

The R1 iterated-game premise (R1_ITERATED_PLAN.md): if the predictor (a lagged copy of the policy)
REASONS, its predicted one-box probability P_pred(one-box | prompt with stated p) tracks the stated
p — making the evidential reward conditional on p from step 0 (the lever the p-blind 3B predictor
lacked). This probe measures that signal directly, for base R1 as the predictor.

reason-then-read: per (item, p): generate a <think> chain, append a forced `Answer:` continuation,
then read BOTH
  - argmax one-box rate  (robust; mirrors rloo._forced_answer_roles), and
  - soft P(non_cdt)      (2-way answer-token softmax = the exact p_eff the trainer will use).
Single forward via generate(output_scores=True): scores[0] -> soft prob; decoded -> argmax role.

GO/NO-GO: both rise with p => build the loop (reward is conditional from step 0).
          flat        => even the reasoning predictor is p-blind in the scoring path; rethink
                         (e.g. third-person "what will an agent like you do?" framing).
Invariant #1 preserved: answer resolved over per-sample abstract tokens; no NL string-matching.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses

import torch

from newcomb_eval.scorer import ROLE_CDT, ROLE_INVALID, ROLE_NON_CDT, resolve_choice

CUE = "\nAnswer:"


def first_ids(tok, label):
    """Candidate first-token ids of `label` as it may appear after 'Answer:' (bare or space-led)."""
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
    by_p = {}
    for p in grid:
        rps = [build_prompt(it, p, cfg.prompt, sweep_seed=0, repeat=0, mode=cfg.loader_mode)
               for it in items]
        by_p[p] = rps
    return grid, by_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dataset", default="newcomb_eval/data/dataset_mech_m3.json",
                    help="BINDING predictor framing (m3 exact-copy) — honest for a self-snapshot "
                         "predictor and the framing that lifted EDT (gap_adj 0.21->0.31). NOT abstract m0.")
    ap.add_argument("--max-new-tokens", dest="mnt", type=int, default=4096, help="reasoning budget")
    ap.add_argument("--limit", type=int, default=6, help="items per p (None=all)")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--micro", type=int, default=8, help="reasoning-gen micro-batch")
    ap.add_argument("--tag", default="r1d")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from newcomb_eval.config import EvalConfig
    cfg = EvalConfig()
    # BINDING predictor framing (m3 exact-copy): the mech datasets carry category=None, so the default
    # category_filter would match 0 items — disable it. This is the honest framing for a self-snapshot
    # predictor (predictor IS a copy of the policy) and the one that lifted EDT in the credence runs.
    cfg = dataclasses.replace(cfg, dataset_path=args.dataset, category_filter=None,
                              prompt=dataclasses.replace(cfg.prompt, cot=True))
    grid, by_p = build_grid_prompts(cfg, args.limit)
    n_items = len(by_p[grid[0]])
    print(f"[0b] model={args.model} dataset={args.dataset} items/p={n_items} grid={grid} "
          f"mnt={args.mnt} temp={args.temp}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.config.pad_token_id = tok.pad_token_id

    def wrap(text):
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def reason_then_read(rps):
        """Per prompt -> (P_pred_non_cdt soft, role argmax, closed_think, gen_len)."""
        soft, roles, closed, glen = [], [], [], []
        for i in range(0, len(rps), args.micro):
            chunk = rps[i:i + args.micro]
            enc = tok([wrap(r.text) for r in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            Lp = enc.input_ids.shape[1]
            gen = model.generate(**enc, max_new_tokens=args.mnt, do_sample=True,
                                 temperature=args.temp, top_p=args.top_p, pad_token_id=tok.pad_token_id)
            comp = tok.batch_decode(gen[:, Lp:], skip_special_tokens=True)
            for c in comp:
                closed.append("</think>" in c)
                glen.append(len(tok(c, add_special_tokens=False).input_ids))
            # forced-answer pass: read first-answer-token logits (soft) + greedy label (argmax)
            texts2 = [wrap(r.text) + c + CUE for r, c in zip(chunk, comp)]
            enc2 = tok(texts2, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            out = model.generate(**enc2, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id, output_scores=True,
                                 return_dict_in_generate=True)
            scores0 = out.scores[0].float()  # [b, V] logits of the first answer token
            ans = tok.batch_decode(out.sequences[:, enc2.input_ids.shape[1]:], skip_special_tokens=True)
            for j, r in enumerate(chunk):
                non = next(t for t, role in r.token_role.items() if role == ROLE_NON_CDT)
                cdt = next(t for t, role in r.token_role.items() if role == ROLE_CDT)
                ln = max(scores0[j, k].item() for k in first_ids(tok, non))
                lc = max(scores0[j, k].item() for k in first_ids(tok, cdt))
                soft.append(torch.softmax(torch.tensor([ln, lc]), 0)[0].item())
                role, _t, valid = resolve_choice(ans[j], r.legal_tokens, r.token_role, cot=False)
                roles.append(role if valid else ROLE_INVALID)
        return soft, roles, closed, glen

    rows = []
    print(f"\n{'p':>5} {'P_pred(soft)':>13} {'onebox(argmax)':>15} {'invalid':>8} "
          f"{'closed':>7} {'gen_len':>8}", flush=True)
    for p in grid:
        soft, roles, closed, glen = reason_then_read(by_p[p])
        n = len(roles)
        nk = sum(r == ROLE_NON_CDT for r in roles)
        ninv = sum(r == ROLE_INVALID for r in roles)
        soft_mean = sum(soft) / n
        ob = nk / max(1, n - ninv)
        crate = sum(closed) / n
        glen_mean = sum(glen) / n
        rows.append(dict(p=p, p_pred_soft=soft_mean, onebox_argmax=ob,
                         invalid=ninv / n, closed=crate, gen_len=glen_mean, n=n))
        print(f"{p:>5.2f} {soft_mean:>13.3f} {ob:>15.3f} {ninv/n:>8.2f} "
              f"{crate:>7.2f} {glen_mean:>8.0f}", flush=True)

    lo, hi = rows[0], rows[-1]
    print(f"\nSLOPE (p={hi['p']} - p={lo['p']}): "
          f"P_pred(soft) {hi['p_pred_soft']-lo['p_pred_soft']:+.3f}   "
          f"onebox(argmax) {hi['onebox_argmax']-lo['onebox_argmax']:+.3f}")
    print("GO if both clearly positive (predictor's one-box tendency tracks stated p).")

    out_csv = f"results/credence/r1_predictor_probe_{args.tag}.csv"
    import os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
