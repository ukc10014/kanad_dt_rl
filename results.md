# Results log — Newcomb p-sweep

Running log of baseline + experiment runs. Each entry records the config, headline numbers,
and interpretation. Raw per-sample transcripts are kept under `results/inspect_logs/` (Inspect
`.eval` logs) — always retained, not just the aggregates.

> **How to read K-rate(p):** K-rate = fraction of *valid* samples choosing the **non-CDT**
> (one-box / EDT+FDT) option at each stated predictor accuracy `p`. A model that *reasons from
> the structure* is **low below `p*` and steps up above it** (crossover at `p*`). A flat line =
> the choice is independent of `p` → persona/recitation, not structural reasoning.

---

## Day-1 plain-language summary (2026-06-22)

The basic question we're chasing is whether a small language model actually *reasons* about a
Newcomb-style problem, or just falls back on a habit. The setup gives the model a "predictor" with
some stated accuracy, and a genuinely thoughtful agent should change its choice depending on how
accurate that predictor is said to be — cooperate when the predictor is reliable, defect when it
isn't. So we sweep the stated accuracy from low to high and watch whether the model's choice shifts
at the point where the math says it should.

The headline is that, left to answer in a single word, the model doesn't shift at all. It just
gives its favorite answer (cooperate/one-box) regardless of the stated accuracy — a flat line.
That's recitation, not reasoning. But the moment you let it think out loud, it *does* start to
respond to the accuracy. So the ability is in there; it just can't be expressed in a snap one-word
answer. That distinction — between a capability the model has latently and whether it can actually
deploy it — turned out to be the spine of the whole day.

Then we asked what reinforcement learning actually does to this. The clean finding is that RL can
easily move the model's overall *lean* — we can reliably turn it into a confident defector, or
nudge it to cooperate more — but it cannot teach the model the *conditional* rule (cooperate only
when the predictor is accurate enough). In our shorthand: RL moves the intercept, not the slope. It
changes the model's disposition or "personality," but it doesn't install the underlying competence.
We confirmed this four different ways, including building a more sensitive instrument that reads the
model's underlying probabilities rather than just its final answer — and even down at that level,
every RL'd version is flat, just sitting at a different height.

Two cleverer attempts to break the pattern both came up short in instructive ways. We walked the
model through a careful five-step analysis (a "scaffold"), and it correctly *names* the key fact —
that its choice is correlated with the prediction — but then it applies a blanket rule
("correlated, so cooperate") without ever doing the actual expected-value arithmetic, so it still
doesn't track accuracy. The scaffold made the model tidier and more consistent, but it didn't
unlock the reasoning. And we tried making the predictor "real" — an actual model predicting the
choice, rather than a stipulated number — which is a nicer, more honest version of the problem.
That didn't help either, for a satisfying reason: the predictor we used is itself blind to the
accuracy, so there was simply no signal there to learn from. A model-based predictor is only as
smart as the model you build it from.

General impression: a genuinely productive day with an unusually coherent story. Four independent
experiments all point the same direction. The take-home is that a model's decision-theoretic
*disposition* is cheap to move with RL, but its decision-theoretic *competence* is not — RL elicits
and reshapes what's already there rather than teaching something new, and the real reasoning only
surfaces when the model is given room to compute. That's a small, crisp instance of a pattern seen
in RL for language models more broadly.

Honest caveats: small model, mostly single runs, and — importantly — we haven't yet given RL its
very best shot at learning the conditional rule (the current setup lets RL "win" by just shifting
the overall lean). The most interesting next moves are the ones that would actually *try to break*
this conclusion: teaching the model the reasoning directly first and then doing RL, or making the
predictor a copy of the model itself so the problem becomes genuinely self-referential. If one of
those finally produces the conditional behavior, it flips the result from "RL only changes
disposition" to "here's exactly what it takes to install competence" — the bigger, more exciting
finding. Either way, today built the measurement scaffolding to tell the difference cleanly.

---

## Tomorrow — concrete todos (pick up in the morning)

Goal for tomorrow: **try to break the intercept-vs-slope result** (or pin down exactly why it
holds). Ordered by value; detail/rationale in the "Day-1 synthesis" section below.

1. **Give RL a fair shot at the slope (cheapest; do first).** Add an **EV-balanced / symmetric
   curriculum** so the net "lean" gradient ≈ 0 and the *only* way to earn reward is to condition on
   `p`; plus a **paired-across-`p`** reward that pays the policy for *flipping* its answer as `p`
   crosses `p*`. If a step forms → RL *can* learn the conditional with the right objective; if not →
   the intercept-only result is robust, not just under-specified. (`sampler.py` + `reward.py`.)
2. **SFT/STaR competence install → then RL (the clean 2×2).** Rejection-sample the 3B's own correct
   CoT, SFT a *competent* base, verify it truly computes via the **payoff ablation** (below), then
   run causal vs evidential RL on it: does RL *override* a genuine EV computation, or *preserve* it?
   (`newcomb_rl/sft.py`.)
3. **Payoff ablation (validity control — run regardless).** Move `p*` by changing S/B
   (`p*∈{0.6,0.7,0.8,0.9}`) and check the empirical crossover *follows* `p*`. Discriminates real
   EV-reasoning from a memorized ~0.8 threshold. Run on the CoT arm (the only one that tracks) and
   any SFT'd model. (reuse `crossover.crossover_p` / `CrossoverConfig`.)
4. **C2 — self-snapshot predictor (the one that could actually move).** Predictor = a lagging
   *snapshot of the policy* (not the frozen base) → genuine self-prediction fixed-point dynamics.
   Watch for bistability / the policy learning to be unpredictable to itself. (`rloo.py` snapshot.)
5. **Comprehension gate (cheap rigor).** Per-item "which option takes both?" / "what accuracy was
   stated?" probe; report K-rate conditional on comprehension-pass. Confirms the flat baseline is
   "won't track," not "can't parse."
6. **Capability-cliff probe (optional, decides whether scale matters).** Run the scaffold + logprob
   sweep on Qwen2.5-14B (fits the A40 for inference) — does a bigger model form the step where the
   3B can't? Tells us whether to scale RL to ~7–14B.

Also still queued from the earlier rigor pass: harden the CoT parser (#5 — free-CoT hit 38%
invalid), bump `n_repeats` for tighter CIs, per-sample CSV/strata, soften any remaining
over-claims. Probing/causal-patching (is `p` represented-but-unused?) is the deeper diagnostic.

---

## Run 1 — Baseline: Qwen2.5-3B-Instruct  (2026-06-22)

**Headline:** flat-high K-rate (~0.70–0.95) across the whole `p` range, **no tracking of `p`**,
slight dip near `p*`. The model one-boxes ~83% of the time *regardless* of the stated accuracy:
**no evidence of `p`-tracking in this (forced-choice) setup** — a strong prior toward the
canonical one-box answer. This is the baseline RL is meant to move (flat-high → a curve that
switches at `p*`).

> **Updated by Run 3:** do *not* read this flat line as a fixed "persona/disposition." Under CoT
> the *same base model* tracks `p` (slope +0.50). The forced-choice flatness is a
> **format/capability artifact** (no room to compute EV in one token), not an inability or
> unwillingness to condition on `p`. Mechanism here is inferred, not measured.

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

## Run 3 — RLOO evidential **under CoT** (Qwen2.5-3B-Instruct, 2026-06-22)

**Setup.** Same evidential reward as Run 2, but the policy now reasons before answering
(`--cot`, 128–256 tok), so it *can* compute EV over the stated `p`. K=5/P=5, 60 steps, in-loop
greedy eval. Adapter: `results/adapters/evidential_cot` (standalone, gitignored). This is the
decisive capability-vs-disposition test flagged in Run 2's caveats.

**Trajectory (in-loop greedy eval):**

| step | eval mean_K | K@p=0.5 | K@p=0.99 | slope (hi−lo) | train invalid |
|---|---|---|---|---|---|
| 0 (CoT baseline) | 0.440 | 0.00 | 0.50 | **+0.50** | — |
| 1 | — | — | — | — | **0.48** |
| 20 | 0.400 | 0.40 | 0.20 | −0.20 | 0.00 |
| 40 | 0.650 | 0.70 | 0.80 | +0.10 | 0.00 |
| 60 (final) | 0.662 | 0.70 | 0.80 | **+0.10** | 0.00 |

**Result — the same asymmetry, now under CoT.** The CoT *baseline* (step 0) already tracks `p`
(slope **+0.50**): the 3B can partly do the EV reasoning when given room — so the flat
forced-choice baseline (Run 1) was a **format/capability artifact, not a fixed disposition**.
But evidential RL did **not sharpen** that step. It raised overall one-boxing (mean_K 0.44→0.66)
by lifting the **floor** (K@p=0.5: 0.00→0.70) far more than the **ceiling** (K@p=0.99: 0.50→0.80),
**eroding the conditional tracking** (slope +0.50→+0.10). RL installed a *more-uniform one-box
disposition*, not the conditional EV-competence — the exact Run-2 asymmetry, reproduced in a
second format. (Step-1 `invalid=0.48` flags a fragile CoT parser mid-training — a data-quality
caveat, see harness note below.)

**Reproduce**
```bash
python -m newcomb_rl.train_rl --arm evidential --cot --steps 60 --K 5 --P 5 \
  --eval-every 20 --eval-items 10
```

---

## Scaffolded-CoT three-arm harness (built 2026-06-22; base 3B run in progress)

New `newcomb_eval/scaffold.py` + `run_scaffold.py`: a port of cosmichost_mp's scaffolded-CoT
experiment into our stack. Three arms over the same p-sweep, all scored by the existing
abstract-token `resolve_choice`: **no_cot** (≡ Run-1 forced choice), **free_cot** (≡ Run-3
baseline path), **scaffolded** (5 neutral sub-questions: parties→options→outcomes→relationship→
decision, each its own turn). Purpose: separate **capability** (K-rate(p) slope per arm),
**extraction** (the Step-4 "relationship" response, persisted verbatim), and **inclination**
(decision conditional on a correct Step-4). Model-agnostic (`--model`/`--adapter`) so the same
tool probes the capability cliff on a bigger model (Qwen2.5-14B bf16 fits the A40 for inference;
32B-4bit for the direct cosmichost comparison) and re-scores the RL'd adapters. Smoke on 0.5B:
clean pipeline, 0% invalid after the decision-step parser fix (16→48 tok, label-first prompt).
Base-3B three-arm result: **Run 4** below. 59 tests green.

## Run 4 — Scaffolded-CoT three arms (Qwen2.5-3B-Instruct, 2026-06-22)

| p | no_cot | free_cot | scaffolded |
|---|---|---|---|
| 0.50 | 0.90 | 0.31 | 0.80 |
| 0.60 | 0.95 | 0.46 | 0.60 |
| 0.70 | 0.90 | 0.31 | 0.60 |
| 0.75 | 0.80 | 0.46 | 0.79 |
| 0.80 | 0.70 | 0.67 | 0.68 |
| 0.85 | 0.80 | 0.56 | 0.80 |
| 0.90 | 0.70 | 0.33 | 0.55 |
| 0.99 | 0.90 | 0.82 | 0.75 |
| **mean** | **0.831** | **0.490** | **0.697** |
| **slope (hi−lo)** | **+0.00** | **+0.51** | **−0.05** |
| **invalid** | 0.000 | **0.381** | **0.012** |

**Result — scaffolding fixes structure but *suppresses* p-tracking.**
- **no_cot** reproduces Run 1 exactly (flat-high 0.831).
- **free_cot tracks p** (slope **+0.51**, K@0.5=0.31→K@0.99=0.82) — replicating Run 3's CoT
  baseline — but at the cost of (a) **dropping overall one-boxing** 0.83→0.49 (the cosmichost
  "free-CoT pulls toward CDT on prediction problems" effect, reproduced cross-model) and (b)
  **38% parse failures** (the fragile CoT parser, reviewer #5).
- **scaffolded is FLAT** (slope **−0.05**, mean 0.70). The neutral 5-step decomposition *fixed the
  parse failures* (invalid 0.38→0.01) and *prevented the CDT cascade* (mean 0.49→0.70) — but it
  **did NOT produce the p-tracking step**; it actively flattened the +0.51 free-CoT slope.

**Why (the Step-4 transcripts localize it).** At **both** p=0.5 and p=0.99 the model's Step-4
("relationship") gives the *identical* answer — "**correlational**" — and one-boxes at both. So:
- **Extraction succeeds (qualitatively):** the scaffold reliably surfaces the EDT-relevant feature
  (the choice↔predictor link is *correlational, not causal*).
- **The decision is a *qualitative heuristic*, not a *quantitative EV computation*:** the model
  applies "correlational ⇒ one-box" **independent of the stated accuracy**, so it one-boxes even at
  p=0.5 where EV favours two-box. It never converts the numeric `p` into `EV(one-box)=p·B` vs
  `EV(two-box)=S+(1−p)·B`. The failure is in the **extraction→decision step**, not perception.

**Capability / extraction / inclination — decomposed (the point of the experiment):**
- **Capability** (compute the EV crossover): *present but only elicited by free-form CoT*, which
  occasionally stumbles into the arithmetic; the neutral structural scaffold does **not** route
  through it.
- **Extraction** (surface the structure): *succeeds* — Step-4 correctly classifies "correlational".
- **Inclination** (given correct extraction): "correlational ⇒ one-box" applied as a **p-blind
  qualitative rule**, which is why scaffolded is flat. A scaffold that explicitly demanded "compute
  the expected value of each option at the stated accuracy" would be the test of whether the
  quantitative capability can be *routed to* — current neutral scaffold does not reach it.

**Reproduce**
```bash
python -m newcomb_eval.run_scaffold --tag base3b
```
Artifacts: `results/scaffold/{k_rate_by_arm_p_base3b.csv, k_rate_by_arm_base3b.png}`,
per-step transcripts in `results/scaffold/scaffold_logs/transcripts_*_base3b.jsonl`.

---

## Run 5 — Pivot A: logprob / logit-margin analysis (Qwen2.5-3B-Instruct, 2026-06-22)

**What.** The sub-argmax instrument (`newcomb_eval/logprob_sweep.py` + `ModelWrapper.answer_logprobs`):
instead of binarising the decision, read the renormalised 2-way `P(non_cdt | p)` and the logit
margin `logP(non_cdt) − logP(cdt)` at the answer position. Forced-choice; base + each RL'd adapter.

**Result 1 — the flat baseline is real below the argmax (not a binarisation artifact).** On the
base 3B, forced-choice, the *continuous* margin is **also flat** — `P(non_cdt)` slope (p=0.5→0.99)
= **−0.02**, margin slope = **−1.31**, with a **dip near p\*** (`P(non_cdt)`: 0.88 @0.5 → 0.71 @0.8
→ 0.86 @0.99). So the flat-high K-rate is **not** masking a graded EV signal — there is no hidden
p-tracking in the single-token distribution. Combined with Run 3 (CoT tracks p at +0.50), this
sharpens the claim: **the conditional EV competence is a *sequential-compute* capability (it needs
reasoning tokens), not something present-but-thresholded in the immediate next-token logits.**

**Result 2 — intercept-vs-slope, confirmed at the logit level (and richer than K-rate showed).**

| arm | mean `P(non_cdt)` | mean logit margin | K-rate(argmax) | shape |
|---|---|---|---|---|
| base | 0.805 | **+3.46** | 0.838 | flat-high |
| causal | 0.000 | **−18.16** | 0.000 | flat, **deeply saturated** two-box |
| evidential | 0.457 | **−0.51** | 0.456 | flat at **indifference** |

All three curves are **flat in p**; RL moves only the **level** (intercept), never the slope —
exactly the day-1 claim, now *below the argmax*. The margin adds what K-rate cannot: causal RL did
not merely flip the answer (K=0), it drove the margin to **−18** (extreme, saturated confidence),
while evidential parked it at **−0.51 ≈ 0** (true indifference, not a weak step). The disposition
shift is a confident, uniform intercept move.

**Feeds Pivot C.** The base `P(non_cdt|prompt)` measured here (flat ~0.8, dip at p\*) *is* the
emergent accuracy the model-based-predictor reward sources — so the predictor's "accuracy" is
roughly the base's flat one-box prior (with the p\* dip), set up to be tested next.

**Reproduce**
```bash
python -m newcomb_eval.logprob_sweep --tag base3b
python -m newcomb_eval.logprob_sweep --adapter results/adapters/causal     --tag causal
python -m newcomb_eval.logprob_sweep --adapter results/adapters/evidential --tag evidential
```
Artifacts: `results/logprob/p_margin_by_p_{base3b,causal,evidential}.csv` + `.png`.

---

## Run 6 — Pivot C: model-based predictor, frozen base (C1) (Qwen2.5-3B-Instruct, 2026-06-22)

**What.** Replace the *stipulated* accuracy `p` with a **real predictor**: the box-fill is decided
by what a model would have predicted. C1 uses the **frozen base** as predictor — per rollout prompt,
`P_pred(non_cdt | prompt)` is read from the existing `disable_adapter()` forward
(`rloo._predictor_p`) and used as the evidential reward's accuracy (the stated grid `p` is ignored
for this arm). 150 steps, K=8/P=8 — matched to Run 2's evidential arm.

**Trajectory (in-loop greedy eval; `p_model` = measured predictor accuracy):**

| step | eval mean_K | slope (hi−lo) | train `p_model` |
|---|---|---|---|
| 0 | 0.856 | +0.05 | — |
| 25 | 0.806 | +0.20 | 0.46 |
| 50 | 0.550 | −0.25 | 0.86 |
| 75 | 0.531 | −0.20 | 0.96 |
| 100 | 0.556 | −0.05 | 0.83 |
| 125 | 0.750 | −0.05 | 0.84 |
| 150 (final) | **0.719** | **+0.05** | 0.59 |

Logprob re-score of the final adapter: **P(non_cdt)=0.718, margin +8.36, slope(P) −0.24**.

**Result — a *real* correlation did NOT change the verdict; still intercept, not slope.** The
model-based-predictor arm lands **flat** (slope +0.05 eval / −0.24 logit) at mean_K≈0.72 — no
p-tracking step, same as the stipulated evidential arm. Two specifics:
- **It only inherits the base's p-awareness — and the base has none.** Run 5 showed the base
  `P(non_cdt|prompt)` is flat in `p`; grounding the reward in it therefore supplies **no p-signal**,
  so it cannot teach p-tracking. A model-based predictor is only as conditional as the predictor
  model.
- **Self-reinforcement (a taste of the fixed point).** Because the base predicts the canonical
  one-box (`p_model`≈0.7–0.9 ≳ p\*), the reward *rewards* one-boxing → RL **sharpened** the one-box
  prior: margin **+8.36 > base +3.46**. The base's own belief became the reward target and got
  amplified. (This is *not* a true fixed point: the predictor is frozen, so per-prompt `p_model`
  is constant; the step-to-step swings 0.46→0.96 are 8-prompt sampling noise, not dynamics.)

**Four-arm intercept-vs-slope (logit level) — the consolidated result:**

| arm | P(non_cdt) | logit margin | slope(P) | what RL did |
|---|---|---|---|---|
| base | 0.805 | +3.46 | −0.02 | — |
| causal | 0.000 | −18.16 | −0.00 | saturated two-box |
| evidential | 0.457 | −0.51 | −0.15 | pushed to indifference |
| modelpred | 0.718 | +8.36 | −0.24 | sharpened one-box |

**All four flat.** RL moves the intercept to four different levels; none produce the slope. The
model-based predictor (C1) confirms the day-1 claim a fourth time and adds *why a truer construct
doesn't rescue it*: the conditional signal has to exist in the predictor, and a p-blind base can't
provide it. **Next (C2):** a frozen *snapshot* predictor (lagging the policy) for genuine
self-prediction fixed-point dynamics — the version that could actually move.

**Reproduce**
```bash
python -m newcomb_rl.train_rl --arm evidential_modelpred --steps 150 --K 8 --P 8 --eval-every 25
python -m newcomb_eval.logprob_sweep --adapter results/adapters/evidential_modelpred --tag modelpred
```

---

## Day-1 synthesis — what is RL actually adding, and where next (2026-06-22)

> **Partly executed since:** the logprob diagnostic (#1 below) is **Run 5** and the model-based
> predictor pivot is **Run 6** — both confirmed the intercept-vs-slope claim (now a four-arm,
> logit-level result). Still open from this list: SFT/STaR competence install, EV-balanced /
> paired-across-`p` reward-shaping, per-`p` ablations, the self-snapshot predictor (C2), and
> probing/patching.

### The finding: RL moves the *intercept*, not the *slope*
Across every RL run, the thing that moves is the **level** of one-boxing, never the
**p-conditional shape**:

| arm | what moved | K(p) shape |
|---|---|---|
| causal (forced) | intercept → 0 | flat, slope 0 |
| evidential (forced) | intercept → ~0.5 | flat, slope ~0 |
| evidential (CoT) | intercept ↑ 0.44→0.66 | slope **collapsed** +0.50→+0.10 |

RL reweights the model's **marginal propensity** to emit one-box vs two-box; it does **not**
teach the *function* `p → optimal action`. The one time the base actually had that function (the
+0.50 CoT-baseline slope), RL **eroded** it (raised the floor faster than the ceiling, averaging
the step away).

**Claim (the paper-shaped takeaway):** *outcome RL with a low-rank update installs a
decision-theoretic **disposition** (an intercept shift) but not a p-**conditional competence**
(a slope). The competence must pre-exist in the base — CoT shows it does — and outcome reward
doesn't sharpen it, it averages over it.* This is a crisp micro-instance of the broader
"RL elicits/reweights latent capability rather than installing new capability" result
(RLVR-doesn't-expand-the-reasoning-boundary; Qwen spurious-reward findings). It's unusually clean
here because the capability axis (slope) and the disposition axis (intercept) are **orthogonal
and separately measurable**.

**Why (mechanistically):**
1. **Net gradient is dominated by the marginal, not the conditional.** One shared LoRA serves all
   `p`; if the training-`p` grid isn't EV-balanced (ours isn't — 0.5/0.6/0.7 favour two-box,
   0.8/0.9/0.99 favour one-box, magnitudes asymmetric), the net pull is toward the on-average
   winner → a uniform shift. Carving a *step* needs per-`p` advantages to fight and the policy to
   already condition on `p` enough to resolve them locally; a weak-`p` base + small LoRA collapses
   to the average.
2. **Credit assignment too coarse to reward the *reason*.** Outcome reward lands on the whole
   trajectory; the model satisfies it by nudging its prior, never having to compute EV. RL takes
   the cheapest path: change the bias, not the algorithm.
3. **KL + low rank = nudge, not restructure.** Anchored near the base's `p→action` shape; reward
   only tilts the level.

**Honest caveat (open, not yet settled):** we have **not yet given RL a fair shot at the slope.**
The current objective lets RL win via the marginal, so "RL only shifts disposition" is so far
**underdetermined** vs "we under-specified the objective." Before concluding RL *can't* add the
slope, it owes an objective that *requires* the slope (balanced/contrastive — see below). That is
also the most likely route to a positive result.

### Next steps — deeper search (understand what's happening)
Cheap → deep:
1. **Instrument the loop (cheap, decisive).** Log per-step **per-`p` advantage** and **per-`p`
   policy shift**; add a **logit-margin probe** `logit(non_cdt) − logit(cdt)` vs `p` per
   checkpoint (continuous, pre-binarization view of whether RL flattens the representational
   slope, not just the K-rate).
2. **Use the scaffold (near-zero new code).** Grade **Step-4 extraction** (states the
   choice↔predictor correlation? mentions `p`? computes EV?) and correlate with the decision:
   extraction-good-but-decision-uniform → bottleneck is **extraction→action wiring** (inclination);
   extraction-`p`-blind → **perception/representation**.
3. **Per-`p` RL ablations (clean controls).** Train evidential on *only* high-`p`, then *only*
   low-`p`. Uniform-one-box and uniform-two-box respectively = direct proof RL learns the
   **marginal of the training distribution**, not a conditional.
4. **Capacity sweep** (LoRA rank 4→64, KL ↓). Step forms at higher rank/lower KL → bottleneck was
   optimization/anchoring; not → representational.
5. **Probing + causal patching (deep).** Is the stated `p` **linearly decodable** at the decision
   point, and does **ablating/patching** it change the action? `p`-represented-but-unused → RL's
   failure is *wiring*, not perception. (Heed the cosmichost lesson: probe for "is `p` represented
   and used," not for a DT-disposition vector — theirs encoded *content, not disposition*.)

### Next steps — use RL differently (the multi-step instinct is right)
1. **Process reward over the scaffolded trajectory (highest value; scaffold is the substrate).**
   Reward correct **Step-3 payoff enumeration** + **Step-4 correlation/EV identification**, not
   just the final token → forces the model to *use* the structure instead of shifting its prior.
   (Risk: format reward-hacking; mitigate with abstract-token discipline + a Step-4 judge.)
2. **Multi-turn scaffolded RL even with only a *terminal* reward should beat single-round** — the
   action is now downstream of self-generated reasoning *in the gradient path*, so terminal reward
   can reinforce "compute EV → act on it" as a reusable, cross-`p` behaviour. Testable prediction:
   scaffolded-trajectory RL preserves/sharpens the slope where single-round eroded it.
3. **High-ceiling: a sequential / iterated Newcomb env.** Make `p` **instrumentally necessary** —
   the model plays a sequence, accuracy is revealed only through realised outcomes, and it must
   *infer* `p` and condition on it to score. Can't score without the conditional → RL is forced to
   install it (or fail visibly). Biggest build; likeliest to actually *create* competence.
4. **Two cheap reward-shaping levers (no new env) that attack marginal-collapse directly:**
   - **EV-balanced / symmetric curriculum** so net marginal gradient ≈ 0 and the *only* way to
     earn reward is to condition on `p`. Cheapest test of mechanism #1.
   - **Paired-across-`p` rollouts:** sample the same item below *and* above `p*` in a group and
     reward the policy for **changing its answer when `p` crosses `p*`** → optimizes the *slope*,
     not the intercept. The single most targeted intervention on top of existing RLOO.

### Prioritization
- **Now (cheap; sharpens the claim):** EV-balanced curriculum + paired-across-`p` reward + loop
  instrumentation + per-`p` ablations; plus **Step-4 extraction grading** from the finishing
  scaffold run (perception vs wiring, ~free). These either *produce a step* (RL **can** add the
  slope with the right objective) or *confirm marginal-collapse* (clean negative result).
- **Next:** process reward over the scaffold trajectory.
- **Deep/ambitious:** `p`-representation probing + causal patching; iterated-Newcomb env.

The most valuable single move is to **try to break the intercept-vs-slope asymmetry** with a
balanced/contrastive objective — right now we've only shown RL *defaults* to disposition, not that
it's *confined* to it.

### Still-queued housekeeping (orthogonal to the above)
- **Scaffold base-3B result** (running) → write up as **Run 4**: does `scaffolded` form a clean
  step at `p*` that `no_cot`/`free_cot` don't? (capability) + what does Step-4 reveal? (extraction)
- **Capability-cliff probe**: scaffold sweep on Qwen2.5-14B (and/or 32B-4bit) vs 3B — is *scale*
  the missing ingredient? Decide whether to scale RL to ~7–14B **after** this (RL on 32B doesn't
  fit the single A40; 32B bf16 ≈ 64 GB > 46 GB).
- Gemma-2-2b-it baseline (comparability to Tennant et al.; PLAN.md §2 model-choice note).
- Oesterheld 2024 capabilities-dataset sanity check (PLAN.md §5a) — confirm our hand-built dataset
  isn't "sus" before over-trusting the baseline/RL.
