# R1-Distill iterated-game plan (self-snapshot, reasoning predictor)

*Durable plan (survives context summarization). Owner-facing. Created 2026-06-25.*

## The one-paragraph why
The 3B self-snapshot game gave only **flat bistability** because the 3B predictor is read via the
**forced-token reflex**, which is **p-blind** — so the self-fulfilling loop can only reinforce the
overall *lean*, never a *p-conditional* rule (results.md Day-5). **R1-Distill tracks p only when it
reasons.** So if we make the **predictor reason too**, its prediction `P_pred(one-box | prompt with
stated p)` becomes **conditional on the stated p** → the evidential reward's fill-accuracy `p_eff`
tracks stated p → the reward gradient is **conditional from step 0**. That is the entire lever the 3B
lacked. The reasoning predictor is the experiment.

## Prompt framing — REQUIRED (use the BINDING m3 predictor, NOT abstract m0)
All R1 runs (probe 0b + trainer + evals) use the **m3 "exact-copy" mechanism dataset**
(`newcomb_eval/data/dataset_mech_m3.json`), NOT the abstract m0 "agents like you X% of the time"
reference-class statistic. Two reasons: (1) m3 is the **honest** description of a self-snapshot
predictor — it literally *is* a copy of the policy's decision procedure; (2) m3 was the framing that
**lifted EDT** in the credence/mechanism-credibility runs (one-box 0.59→0.69, credence `gap_adj`
0.21→0.31 on 14B, length-placebo-controlled) — i.e. the abstract m0 framing was *suppressing* the very
action↔box dependence we want the loop to exploit. Wire via `dataset_path=dataset_mech_m3.json` +
`category_filter=None` (mech items carry `category=None`, so the default filter would match 0 items).
m2 "process-scan" is the nearest alternative if "exact copy on *this very problem*" overstates the
snapshot *lag*. Probe `scratchpad/r1_predictor_probe.py` already defaults to m3.

## Reframe vs the 3B (important — changes what "success" means)
On the 3B the question was "can RL *install* the conditional rule?" (answer: no). R1 **already has**
the rule (free-CoT K-rate 0.08→0.83 across p). So the R1 question is **not** installation — it is:
> **Is the conditional EDT rule a self-consistent fixed point under self-prediction?**
i.e. put a model that one-boxes-high / two-boxes-low into a world where the predictor *is itself,
reasoning* — does "one-box above p\*, two-box below" **survive as an equilibrium**, or does the loop
collapse to a degenerate one (all-one-box / all-two-box / indifference) the way outcome-RL eroded the
3B's CoT slope toward a uniform lean? Plus **hysteresis**: do different seeds → different fixed points?
That is a genuinely novel decision-theoretic-self-consistency result, win or lose.

## Stages & decision gates

### Stage 0 — cheap precursors (inference only; do BEFORE building the loop)
- **0a. Timing benchmark** *(RUNNING — `scratchpad/r1_timing.py` → `results/r1_timing.log`)*.
  **Gate:** tractable on 1×A40 / must-trim / need-more-GPU. Sets K, P, steps, max_new_tokens, p-grid.
- **0b. Predictor-tracks-p probe** *(≈30–45 min inference, tiny code)*. Measure the *reward's init
  signal* directly: for **base R1 as predictor**, read `P_pred(one-box | prompt, stated p)` across the
  grid via **reason-then-read-2way** (generate `<think>`, force `Answer:`, read 2-way softmax).
  **Gate (go/no-go for the whole idea):** does `P_pred` rise with stated p? **Yes →** the loop carries
  a conditional signal from step 0, build it. **Flat →** even the reasoning predictor is p-blind in the
  scoring path; rethink (maybe the predictor must be prompted *third-person* "what will an agent like
  you do?"). This is the single most decision-relevant cheap check; can reuse most of the timing
  harness + `rloo._forced_answer_roles` logic.

### Stage 1 — build the reasoning self-snapshot trainer (main code; see "Code" below)
- **Gate:** CPU unit-smoke (adapter plumbing, cache) + **2-step GPU smoke on R1** — no OOM, sane
  gen_len, `</think>` closes, `p_eff` varies across p, invalid-rate low. (CLAUDE.md smoke-before-batch.)

### Stage 2 — short calibration run (attended)
- Seed **base R1**, **swept p**, ~20–40 steps, small K×P. Watch **K-rate(p) slope vs step**, `p_eff(p)`,
  invalid, gen_len. **Gate:** rule stable/sharpening vs eroding vs collapsing; tune cadence/budget; OOM
  or too-slow → trim (levers below) or escalate GPU.

### Stage 3 — the experiments
- **3a. Stability:** base-R1 seed → run to convergence → final K-rate(p) slope. Stable conditional
  fixed point, or collapse? (The headline.)
- **3b. Hysteresis/bistability of the *conditional rule*:** seed a committed **two-boxer** and/or
  **one-boxer** R1 (make via a few oracle-anchor / causal-RL steps on R1, reusing
  `oracle_kl_control.py`-style seeding) → do seeds land in different fixed points? Does the conditional
  rule have a *basin*, or is there a single attractor?
- **Gate:** write up either way (stable equilibrium = positive; collapse = "self-reference destroys the
  rule", also publishable).

## Code (focused; reuse maximised)
1. **`newcomb_rl/selfplay_cot.py` — ✅ BUILT + CPU-validated (GPU smoke pending).** Subclass
   `SnapshotRLOO` (`SnapshotRLOOCoT`): CoT on; **`_predictor_p` overridden** to reason-then-read
   (greedy `<think>` with the frozen `predictor` adapter → forced `Answer:` → 2-way answer-token
   softmax, invariant #1 preserved); **cache** `P_pred[(item_id,p)]` cleared on snapshot refresh
   (dedups repeated draws within a window — predictor cost ≈ #distinct `(item,p)` per window, *not*
   literally zero); chat template forced on (R1's name lacks "instruct"); m3 dataset + `--drop-pstar`
   defaults. No edits to `rloo.py`/`reward.py`/`selfplay.py`.
2. **`scratchpad/r1_predictor_probe.py` — ✅ BUILT (Stage 0b; m3-wired).** Same reason-then-read; the
   go/no-go *and* the de-risk for #1's predictor read (run 0b before trusting the trainer's `p_eff`).
3. **Eval/plot reuse:** `logprob_sweep`/`cot_inspect` (`--model`) for endpoint K-rate(p);
   `plot_3b_dynamics.py`-style step-trajectory plot for training dynamics.
4. **No new env / no rloo.py edits** — the modelpred reward path + CoT forced-answer already exist; we
   only changed *where the predictor probability comes from* (reflex → reasoning) + the cache.

## Run commands (when GPU is back — after the machine switch)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 0b — go/no-go: does base-R1's REASONED prediction track stated p? (m3, ~30-45 min)
python -m scratchpad.r1_predictor_probe --tag r1d_m3            # add --limit 20 for the full set

# Stage-1 GPU SMOKE — 2 steps, tiny, catch OOM / sane gen / p_eff varies / low invalid
python -m newcomb_rl.selfplay_cot --tag r1_smoke --steps 2 --eval-every 2 \
  --K 2 --P 2 --micro 4 --max-new-tokens 1024 --drop-pstar --snapshot-every 1

# Stage-2/3a — trimmed calibration from base R1 (~3-4 h), then plot the trajectory
python -m newcomb_rl.selfplay_cot --tag r1_calib --steps 30 --eval-every 10 --eval-items 4 \
  --K 4 --P 4 --micro 1 --max-new-tokens 2048 --p-grid 0.5 0.6 0.9 0.99 --snapshot-every 10 --kl-ref seed
# micro=1 REQUIRED: micro≥4 OOMs the learn-step backward on R1-8B (validated 2026-06-25). eval-items=4
# (eval-items=1 → nan when a single truncated item fails to parse). If K=4/P=4 OOMs in *generation*,
# drop to --K 4 --P 2 or --max-new-tokens 1536.
# coarse grid: 2 clean low (two-box optimal) + 2 clean high (one-box optimal); drops the noisy,
# longest-reasoning near-p* mid band (0b: mid cells ~2700-3400 tok & only 33-67% </think>-closed).
```
**Smoke watch-list (the things CPU couldn't check):** no OOM (lower `--micro` to 4/2 if so); predictor
`</think>`-closed rate high (printed at end; raise `--max-new-tokens` if low); `p_model` varies across
the grid (the conditional signal — if flat, the reason-then-read is p-blind → revisit 0b); invalid-rate
low; `gen_len` well under the cap. **Memory note:** training adds gradient/activation cost on top of 0a's
generation peaks, so start at `--micro 8` and drop if it OOMs.

## Cost-trimming levers (apply per the Stage-0a gate)
- `max_new_tokens` 4096 → **2048** (R1 closes `</think>` under 2048 except near p\*).
- **Drop p=0.8 (=p\*)** from the grid — it ties (EVs equal) AND burns the full budget; biggest single
  speedup, minimal info loss.
- **K×P** 8×8=64 → **4×4=16** (noisier RLOO advantage, much cheaper).
- **Fewer steps** — R1 starts *with* the rule, so convergence/erosion may show in ≤30 steps.
- **Predictor cache** (above) — already removes the predictor from the hot path.
- If still intractable on 1×A40 → that *is* the "ask for more GPU" finding (escalate with the benchmark
  number in hand).

## Timings — MEASURED (0a, 2026-06-25, 1×A40, HF `generate`, mech-m3 CoT, temp 0.6, 4096-tok budget)
- **batch=16:** 539s wall, gen_len mean **1528** (median 1516, max 4096), **94%** `</think>`-closed,
  **1.8 seqs/min**, 26.1 GB peak.
- **batch=32:** 917s wall, gen_len mean 1527 (median 1282, max 4096), 97% closed, **2.1 seqs/min**,
  36.2 GB peak.
- **Straggler-bound, NOT throughput-bound:** batch wall is set by the *longest* sequence (≥1 hits the
  full 4096 — the p\*=0.8 tie), so 16→32 barely moved seqs/min (1.8→2.1). HF `generate` runs ~4.5
  tok/s/seq, ~10× under the A40 bandwidth floor (no flash-attn/vLLM). ⇒ **capping `max_new_tokens` and
  dropping p\*=0.8 are the highest-leverage cheap wins**, more than bigger batches.

**Full spec (K×P=64, 4096 tok, 150 steps, policy-only): ≈ 107–126 h ≈ 4.5–5 DAYS/run on 1×A40 —
intractable as specified.** (Naive +12% for the predictor, but the predictor CACHE makes it ≈ policy-only.)

**Trimmed calibration (K×P=16, cap 2048 tok, DROP p\*=0.8, ~30 steps, predictor cached): ≈ 3–4 h —
tractable.** Trim math: K×P 64→16 (÷4) · 4096→2048 + no-p\* straggler cut (÷~1.7) · 150→30 steps (÷5) ·
predictor amortized. Good enough for Stage 2 and likely Stage 3a.

**For full fidelity:** swap the ROLLOUT generation to a fast backend (vLLM / flash-attn) — a ~5–20×
lever — and/or more GPUs. Decide AFTER the trimmed calibration shows the dynamics are worth scaling.

## Status / pointers
- Timing benchmark: `results/r1_timing.log` (running). 3B context + Day-5 synthesis: `results.md`.
- R1 facts: tracks p only when reasoning; no-think = flat one-box 1.0; p=0.8 burns full budget
  (results.md "R1-Distill" section). Existing self-snapshot code: `newcomb_rl/selfplay.py`.
