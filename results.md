# Results log — Newcomb p-sweep

Running log of baseline + experiment runs. Each entry records the config, headline numbers,
and interpretation. Raw per-sample transcripts are kept under `results/inspect_logs/` (Inspect
`.eval` logs) — always retained, not just the aggregates.

> **How to read K-rate(p):** K-rate = fraction of *valid* samples choosing the **non-CDT**
> (one-box / EDT+FDT) option at each stated predictor accuracy `p`. A model that *reasons from
> the structure* is **low below `p*` and steps up above it** (crossover at `p*`). A flat line =
> the choice is independent of `p` → persona/recitation, not structural reasoning.

---

## Run 1 — Baseline: Qwen2.5-3B-Instruct  (2026-06-22)

**Headline:** flat-high K-rate (~0.70–0.95) across the whole `p` range, **no tracking of `p`**,
slight dip near `p*`. The model one-boxes ~83% of the time *regardless* of the stated accuracy —
a clean **persona/recitation baseline** (strong prior toward the canonical one-box answer),
exactly the contamination-relevant signal the harness was built to detect. This is the baseline
RL is meant to move (flat-high → a curve that switches at `p*`).

**Config**
- Model: `Qwen/Qwen2.5-3B-Instruct` (instruct; chat template on), greedy (`temperature=0`), `max_new_tokens=6`
- Dataset: `newcomb_eval/data/dataset_opaque_quant.json` — 20 abstract opaque-Newcomb items
- Payoffs (global): B=100, S=60 → **p\* = 0.8**
- `p_grid = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99]`, `n_repeats=1` → **160 samples**
- Held-out (marked only, MVP): `p ∈ {0.75, 0.85}`
- **Invalid rate: 0.000** (instruct model emits clean labels)

**K-rate(p)** (95% Wilson CI)

| p | K-rate | n | invalid | CI |
|------|--------|----|---------|-----------|
| 0.50 | 0.900 | 20 | 0.000 | [0.699, 0.972] |
| 0.60 | 0.950 | 20 | 0.000 | [0.764, 0.991] |
| 0.70 | 0.900 | 20 | 0.000 | [0.699, 0.972] |
| 0.75 | 0.800 | 20 | 0.000 | [0.584, 0.919] |
| 0.80 | 0.700 | 20 | 0.000 | [0.481, 0.855] |
| 0.85 | 0.800 | 20 | 0.000 | [0.584, 0.919] |
| 0.90 | 0.700 | 20 | 0.000 | [0.481, 0.855] |
| 0.99 | 0.900 | 20 | 0.000 | [0.699, 0.972] |

Mean K-rate 0.831. Artifacts: `results/k_rate_by_p.csv`, `results/k_rate_vs_p.png`.

**Reproduce**
```bash
python -m newcomb_eval.run_mvp --max-new-tokens 6 --max-samples 1
```

---

## Run 0 — Smoke: Qwen2.5-0.5B-Instruct  (2026-06-22)

Secondary data point (smaller model). Also **no tracking** — flatter and lower, U-shaped around
~0.35–0.65 (hovering near chance), 0% invalid, 120 samples on `p_grid=[0.5,0.6,0.7,0.8,0.9,0.99]`.
Useful contrast: the 0.5B sits near 0.5 (weak/indifferent), the 3B sits high (strong one-box
prior) — neither tracks `p`. (Base `Qwen2.5-0.5B` without chat template emitted empty/echo text →
100% invalid; the harness correctly recorded invalid rather than scoring a silent CDT.)

**Reproduce**
```bash
python -m newcomb_eval.run_mvp --model Qwen/Qwen2.5-0.5B-Instruct \
  --p-grid 0.5 0.6 0.7 0.8 0.9 0.99 --max-new-tokens 6
```

---

## Run 2 — RLOO: causal vs evidential reward arms (Qwen2.5-3B-Instruct, 2026-06-22)

**Setup.** RLOO LoRA (r=16, α=32), K=8 samples/prompt, P=8 prompts/rollout, 150 steps/arm,
forced-choice, deterministic-EV reward, KL-to-base via `disable_adapter`. Train on
`p_grid \ {0.75,0.85}`, eval on the full grid (n=20/p). **Same base + data; only the reward's
causal structure differs** (causal = fill independent of action; evidential = fill correlates
with action at accuracy p). Plot: `results/k_rate_arms_comparison.png` (local; gitignored).

**Result (mean K-rate over the grid):**

| arm | mean K-rate | shape |
|---|---|---|
| baseline | 0.831 | flat-high (EDT-leaning one-boxer) |
| **causal** | **0.000** | flat-zero — **clean CDT** (always two-box) |
| **evidential** | **0.456** | flat ~0.45–0.50, slight **negative** slope — **did NOT form the EDT step** |

Evidential per-p (n=20): 0.50, 0.45, 0.50, 0.40, 0.50, 0.50, 0.45, 0.35 (p=0.5→0.99) — flat,
if anything *lower* at high p. Training trajectory (in-loop greedy): 0.856 → 0.037 → 0.481 →
0.531 (overshoot to two-box, then drift back; **not converged to a step**).

**Interpretation — an asymmetry, not a symmetric success.**
- **Causal reward → clean CDT.** Naive RL (two-boxing dominates for any fill) drove the
  one-boxing baseline to always-two-box across all p, by step 50. Confirms "ordinary RL is
  CDT-biased": default causal credit-assignment makes the model grab both boxes.
- **Evidential reward → did NOT induce EDT-tracking.** The EDT-favorable reward failed to
  produce the step at p*=0.8; it landed at indifferent ~0.5, mildly anti-tracking.
- **Why:** CDT here is a *uniform disposition* (two-box optimal at every p → gradient points the
  same way everywhere → easy to install). Correct EDT is a *conditional policy* (one-box iff
  p>p*) that requires the model to **represent and condition on the stated p** — exactly the
  thing the baseline showed it does NOT do. So the evidential reward asks for a capability the
  base model lacks, and RLOO muddles toward ~0.5 instead of the step. This is the
  capability-vs-disposition asymmetry (PLAN.md §8) appearing *inside* the RL: you can cheaply
  RL a disposition, but not cheaply RL a conditional competence the base can't execute.

**Caveats (do not over-claim from one run):** single seed, 150 steps, forced-choice single
token (no room to compute EV); evidential run was unstable (overshoot then drift), LR/KL/steps
untuned; train-p mix is two-box-weighted (0.5/0.6/0.7 favor two-box). The decisive test of
capability-vs-disposition is **CoT** — if a CoT-evidential run forms the step, the failure was
format/capability, not fundamental.

**Reproduce**
```bash
python -m newcomb_rl.train_rl --arm causal     --steps 150 --K 8 --P 8
python -m newcomb_rl.train_rl --arm evidential --steps 150 --K 8 --P 8
python -m newcomb_rl.eval_arms --arms causal evidential
```

## Pending / to-run
- **Evidential follow-up** (the open question): CoT reward so the model can reason about p;
  + LR/KL sweep, more steps, balanced/curriculum p-sampling, dense-EV reward.
- Gemma-2-2b-it baseline (comparability to Tennant et al.; see PLAN.md §2 model-choice note).
- Oesterheld 2024 capabilities-dataset sanity check (PLAN.md §5a) — confirm nothing is "sus"
  about our hand-built dataset before trusting the baseline/RL.
