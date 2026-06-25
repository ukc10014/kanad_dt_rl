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

## Code to write (focused; reuse maximised)
1. **`newcomb_rl/selfplay_cot.py`** *(main new file; subclass `SnapshotRLOO`)*:
   - set `cfg.eval.prompt.cot = True` (policy rollout CoT + forced-answer already handled by
     `rloo.rollout`/`_forced_answer_roles`).
   - **override `_predictor_p`** → **reason-then-read**: activate the frozen `predictor` adapter,
     `generate` a `<think>` chain per prompt, append `Answer:` cue, read the 2-way softmax over the
     legal abstract tokens (invariant #1 preserved). Returns per-prompt `P_pred(non_cdt)`.
   - **CACHE** `P_pred[(item_id, p)]` and only recompute on snapshot refresh (predictor frozen between
     refreshes; dataset is ≤20 items × grid ≤160 prompts). **This makes predictor cost ~amortized to
     zero** — dominant cost stays the policy rollout. (Key feasibility lever.)
   - memory-safe: micro-batched generate for predictor too; `expandable_segments`, cap `max_new_tokens`.
2. **`scratchpad/r1_predictor_probe.py`** *(Stage 0b; tiny)* — base-R1 `P_pred(one-box|p)` across grid,
   reason-then-read; print/plot the curve. (Can be a thin wrapper over the Stage-1 method.)
3. **Eval/plot reuse:** `logprob_sweep`/`cot_inspect` (already `--model`-parametrised) for endpoint
   K-rate(p); `plot_3b_dynamics.py`-style step-trajectory plot for training dynamics.
4. **No new env / no rloo.py edits** expected — the modelpred reward path + CoT forced-answer already
   exist; we only change *where the predictor probability comes from* (reflex → reasoning) and add a cache.

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
