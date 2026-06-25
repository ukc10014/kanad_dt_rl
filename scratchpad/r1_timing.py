"""R1-Distill-8B generation-timing benchmark — feasibility/cost estimate for the iterated game.

The iterated (self-snapshot) game on R1 is dominated by ONE cost: batched long-reasoning
generation. R1 only tracks p when it reasons (no-think => flat one-box), so both the policy
rollout AND a p-conditional predictor must emit full <think> chains (~up to 4096 tok). This
measures the dominant primitive — batched CoT generation throughput — and extrapolates a per-step
and full-run wall-clock so we can judge tractability on the single A40 (and how much more GPU a
real run would want).

Sweeps batch sizes in a SINGLE model load. Per config reports: wall, seqs/min, aggregate tok/s,
gen_len mean/median, % closing </think> (truncation gate — the Run-3/4 trap), peak mem, then
extrapolates to a 150-step RLOO run at K*P=64 rollout samples/step (+ P=8 predictor samples).
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import torch


def build_prompts(n):
    from newcomb_eval.config import EvalConfig
    from newcomb_eval.data.loader import load
    from newcomb_eval.prompts import build_prompt
    cfg = EvalConfig()
    cfg = dataclasses.replace(cfg, prompt=dataclasses.replace(cfg.prompt, cot=True))
    items = load(cfg.dataset_path, cfg)
    grid = list(cfg.sweep.p_grid)
    out = []
    i = 0
    while len(out) < n:
        rp = build_prompt(items[i % len(items)], grid[i % len(grid)], cfg.prompt,
                          sweep_seed=0, repeat=i, mode=cfg.loader_mode)
        out.append(rp.text)
        i += 1
    return out


def run_config(model, tok, batch, mnt, temp, top_p, K, P, steps):
    torch.cuda.reset_peak_memory_stats()
    texts = [tok.apply_chat_template([{"role": "user", "content": t}],
                                     tokenize=False, add_generation_prompt=True)
             for t in build_prompts(batch)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    Lp = enc.input_ids.shape[1]
    with torch.no_grad():  # warm up
        _ = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=mnt, do_sample=True, temperature=temp,
                             top_p=top_p, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    out = gen[:, Lp:]
    comps = tok.batch_decode(out, skip_special_tokens=True)
    lens = [(row != tok.pad_token_id).sum().item() for row in out]
    n = len(comps)
    closed = sum("</think>" in c for c in comps)
    spm = n / dt * 60
    peak = torch.cuda.max_memory_allocated() / 1e9
    roll = K * P
    gen_policy = roll / spm * 60
    gen_full = (roll + P) / spm * 60
    OVH = 1.4  # logprob(old+ref)+backward over the gen tokens; rough multiplier (flagged guess)
    print(f"\n===== batch={n} max_new_tokens={mnt} =====")
    print(f"wall={dt:.1f}s  gen_len mean={sum(lens)/n:.0f} median={sorted(lens)[n//2]} max={max(lens)}")
    print(f"</think> closed: {closed}/{n} ({closed/n:.0%})   throughput: {spm:.1f} seqs/min  "
          f"{sum(lens)/dt:.0f} tok/s  peak={peak:.1f}GB")
    print(f"  extrapolate (K*P={roll}/step, +overhead x{OVH}):")
    for label, s in (("policy-only", gen_policy*OVH), ("full+predictor", gen_full*OVH)):
        tot = s * steps
        print(f"    {label:15s}: {s:.0f}s/step -> {steps}-step run {tot/3600:.1f}h ({tot/60:.0f}min)")
    return dict(batch=n, mnt=mnt, seqs_per_min=spm, gen_len_mean=sum(lens)/n,
                closed=closed/n, peak_gb=peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--batches", default="16,32", help="comma-sep micro-batch sizes")
    ap.add_argument("--mnt", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    batches = [int(b) for b in args.batches.split(",")]
    print(f"[r1-timing] model={args.model} batches={batches} mnt={args.mnt} temp={args.temp}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.config.pad_token_id = tok.pad_token_id

    for b in batches:
        run_config(model, tok, b, args.mnt, args.temp, args.top_p, args.K, args.P, args.steps)
    print("\n[r1-timing] done", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
