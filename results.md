# Results log — Newcomb p-sweep

Running log of baseline + experiment runs. Each entry records the config, headline numbers,
and interpretation. Raw per-sample transcripts are kept under `results/inspect_logs/` (Inspect
`.eval` logs) — always retained, not just the aggregates.

> **How to read K-rate(p):** K-rate = fraction of *valid* samples choosing the **non-CDT**
> (one-box / EDT+FDT) option at each stated predictor accuracy `p`. A model that *reasons from
> the structure* is **low below `p*` and steps up above it** (crossover at `p*`). A flat line =
> the choice is independent of `p` → persona/recitation, not structural reasoning.

---

## Week-in-review — the presentable synthesis (2026-06-25)

*The "if I had to present this week" version. Sits above the run-by-run detail; current top-level read.*

**Headline.** A small/mid LLM doesn't fail Newcomb because it *can't do the math* — it fails because of
a **disposition at the action-commitment step** that overrides explicit computation, **sharpens with
scale**, and is **dissolved only by test-time reasoning** (not by size, not by being handed the answer).
Decision-theoretic *competence* and *disposition* are **orthogonal and separately movable**.

**Four converging angles (the defensible core):**
- **RL moves the lean, not the rule** — intercept, not slope (logit-level, per-item, and under a fair
  conditioning-only objective). A clean instance of "RL elicits/reweights latent capability, doesn't install it."
- **Transplant kicker:** hand it the full EV ("one-box=99 > two-box=61") and it *still* won't one-box when
  that means forgoing the guaranteed box. Not "can't compute."
- **Represented-but-unused:** the 7B states the correct evidential credence almost exactly
  (`P(full|one-box)→p`, gap→2p−1) and **two-boxes anyway**. Bottleneck = *usage*, not representation.
- **Sharpest single data point — scale worsens, reasoning fixes:** the 14B is a *more* committed two-boxer
  than the 3B (0–15% one-box at p=0.99 even handed the EV), yet reasoning-trained R1-8B **follows the EV and
  tracks p** (crossover at p\*). No-think R1 collapses to the flat one-box reflex → it's **test-time
  reasoning**, not reasoning-*training*. Breaks both "bigger = more rational" and (in-family) Oesterheld's
  capability→EDT.

**Runner-up threads (less flashy, generative):**
- **Part of the apparent CDT "basin" was a framing artifact — but only a slice.** An abstract "predictor
  identifies agents like you X%" is a *population statistic* a causal reasoner is *right* to treat as
  non-binding; a *binding* (exact-copy) predictor lifts one-boxing only **0.59→0.69** (~10 pts,
  placebo-controlled). Framing explains a slice; **dominance survives the most credible predictor**
  (softened, not dissolved) — which *strengthens* the headline.
- **Even the apparent RL escape was an artifact** (this week): the "RL dissolves a trained two-boxer" flip
  was the **KL-leash-to-base**, not the reward (kl=0 control stays at K=0). Outcome RL is genuinely *stuck*
  against a saturated disposition unless the *environment* carries the conditional signal.

**Open refinements / honest caveats (flagged, not swept):**
- **14B > 3B in CDT — direction suggestive, mechanism open, n≈12 single-run.** Candidate whys (sharper
  deliberation surfaces the guaranteed reward; scale-correlated "take the sure thing" tuning; or noise) are
  unresolved → needs a **seeded in-family ladder (0.5B→32B)**. (Cuts against Oesterheld's capability→EDT.)
- **The abstract-prompt confound is real but modest** (≤10 pts) — worth a sentence, not a load-bearing objection.
- **"Reflex looks EDT / CoT reveals CDT" is a small-model fact that does NOT cleanly scale.** (a) the *reflex*
  drifts off EDT with scale (3B bare one-box ~0.83 vs 14B no-aid ~0.5); (b) **"deliberation" is two different
  operations** — prompted CoT on a non-reasoning model (3B: pushes toward CDT via a p-blind dominance
  heuristic) vs native trained reasoning (R1: does the EV, lands *conditional / EV-rational*). **R1's reasoning
  ≠ the 3B's CoT**, so "deliberation→CDT" is not a scale-invariant law. → needs a within-family **reflex × CoT
  × native-reasoning matrix** (same prompt).

**Meta (worth saying out loud).** Most of the week's *apparent* effects **dissolved under de-noising/controls**
(the +0.50 CoT slope = an invalid-rate artifact; the +0.16 SFT slope; the causal-flip; the "0.80→0.25 basin
drift"). What survives is the core above. "Gate on confounds; quarantine, don't footnote" is itself the
transferable methodological result.

**Forward bet (update 2026-06-26).** Outcome-RL can't install the conditional rule and the model won't *use* what it
represents — so what *would*? The **R1 self-snapshot iterated game** asked whether a self-predictor turns the
conditional rule into a stable fixed point. **Answer so far: no** — both the lagged-reasoning version (cot, but
truncation-confounded) and the clean un-lagged self-consistency version (**A2**, confound removed) drift into
flat, seed-selected basins rather than stabilizing the rule (see the two result sections below). The remaining
open question is whether **lag/overshoot** (a *concrete* lagged self-predictor) produces hysteresis or
oscillation — the queued lagged-snapshot A2 run.

**One-sentence abstract.** *In small LLMs, decision-theoretic disposition is cheap for RL to move and
decision-theoretic competence is nearly impossible — not because the model can't compute the answer (it can,
and even states the right credence) but because a commitment-step disposition overrides it; that disposition
strengthens with scale and dissolves with test-time reasoning.*

**Named next experiments:** R1 iterated game (**done 2026-06-26 — no stable fixed point: cot version
truncation-confounded, A2 un-lagged version replicated the negative CLEANLY; lagged-snapshot variant queued for
the oscillation/hysteresis test — see result sections below**) · seeded in-family CDT ladder (0.5B→32B) ·
reflex×CoT×native-reasoning matrix · anti-Newcomb camouflage (Newcomb-story vs genuine EV-action failure).

---

## R1 self-snapshot iterated game — RESULT: no stable conditional fixed point (2026-06-26)

**The bet & what ran.** Three 30-step runs from base R1-Distill-8B (`selfplay_cot`, m3 binding framing,
coarse grid 0.5/0.6/0.9/0.99, K=4/P=4, **2048-tok** budget): **calib** (seed0, kl0.02), **seed1**
(replication), **kl0** (kl-coef=0). Hope: a *reasoning* predictor (lagged snapshot) makes `p_eff` track p,
so the loop could **stabilize** the conditional rule (one-box high p / two-box low p) that outcome-RL never installed.

**Result — it does NOT stabilize. The loop drifts into the SAME flat self-fulfilling basins as the 3B,
seed-selected.** (plot: `results/r1_calib_dynamics.png`)
- **Conditional slope wobbles, never settles** (eval hi−lo, 0→10→20→30): calib +.25→**+1.0**→+.5→+.75;
  seed1 +.25→0→+.25→**0**; kl0 +.25→+.5→+.75→+.25. The calib **+1.0 at step 10 was a transient**, not a
  fixed point — by step 30 the three disagree (+.75 / 0 / +.25). *(My mid-run "the bet may be paying off"
  over-read that single eval — corrected; classic "don't over-read one n=4 eval".)*
- **Overall lean diverges to basins:** **seed1 → one-box-flat** (mean_K 0.94, K@0.5→1.0, slope 0; predictor
  `p_model` holds ~0.6 → self-fulfilling one-box). **seed0 runs (calib, kl0) → two-box-ward** (mean_K
  0.31/0.50; `p_model` collapses 0.998→0.04/0.07 → self-fulfilling two-box). The predictor's `p_model` and
  the policy's lean **co-move into a basin** — exactly the 3B self-fulfilling-attractor mechanism, with
  seed/noise picking which. **kl=0 behaves like kl=0.02 seed0 → not a KL artifact.**
- So the **reasoning predictor did NOT change the qualitative outcome** vs the 3B: the conditional rule is at
  best a transient/separatrix; the loop collapses to one-box-flat or two-box-flat.

**⚠ MAJOR CONFOUND — quarantine, don't bank.** The predictor's reasoning closed `</think>` only **13–41%**
at the 2048 budget (calib 25%, seed1 13%, kl0 41%) — so `p_eff` is read off *mostly-truncated* chains (the
Run-3/4 trap); the reward signal is degraded and this run does **not cleanly test the premise**. The 2048
trim was too aggressive (0a: R1 on m3 reasons ~2500–3000 tok, closes <58% even at 4096). **Verdict:
PRELIMINARY / INCONCLUSIVE.** Clean reading: "with a truncation-degraded predictor the loop behaves like the
3B (drifts to seed-selected flat basins)"; whether a *properly-reasoning* predictor stabilizes the rule is **still open.**

**Next (teed up, NOT run — GPU idle; needs the user).** (1) Re-run at a budget where the predictor closes
`</think>` — `--max-new-tokens 3584`/4096 (predictor-cache makes it affordable; ~1.5–2× per-step, ~8–10h/run).
(2) `--eval-items` ↑ for less-noisy slopes. (3) **Hysteresis now has direct motivation** — seed1 vs seed0
*already* fell into different basins, so the basin structure is real; a committed one-/two-boxer-seed sweep
would map it cleanly (the original 3b experiment, now on R1). **Artifacts:** `results/r1_calib.log`,
`results/run_r1_{seed1,kl0}.log`, `results/r1_calib_dynamics.png`, adapter `results/adapters/evidential_modelpred_r1_calib`.

---

## A2 self-consistency (un-lagged) — clean replication of the negative + oscillation read (2026-06-26)

**What ran.** `selfplay_loo` (A2): the confound-free successor to the cot iterated game. The predictor's
accuracy `p_eff` = the chooser's **leave-one-out one-box rate** among its own K reasoned rollout samples
(a tally of completions already generated — no second model, no re-reasoning, nothing to truncate). The chooser
still reasons (where p-tracking lives). R1-Distill-Llama-8B, m3 binding framing, grid 0.5/0.6/0.9/0.99,
**K=4/P=3, 2560-tok**, 30 steps, seed 0, eval-items 4. Smoke clean (invalid 0; `p_model` varied; peak 43.1/46 GB).
*(seed-1 × eval-items 8 replication running 2026-06-26.)*

**Result — no stable conditional fixed point, now CLEAN (truncation confound removed).**
- **Conditional slope wobbles, never settles** (eval hi−lo, 0→10→20→30): **+0.25 → +0.50 → +1.00 → +0.50**.
  The +1.00 at step 20 was a **transient** — by step 30 both ends regressed (K@p=0.99 fell 1.00→0.75). Same
  non-stabilization signature as the cot runs.
- **Overall lean drifts down:** mean_K 0.875 → 0.812 → 0.625 → 0.625; `p_model` 1.00 → 0.92 → 0.33 → 0.50.
- **Confound-free, unlike the cot run:** invalid=0.00 throughout; gen_len ≈2270 < 2560 cap → chains **close**,
  not truncated (the cot run's 13–41% `</think>`-closed problem is absent). So this negative is *trustworthy*.
- **Caveat:** n=4 eval items, single seed → individual slope wiggles are within the noise floor (gate: n≲10
  slopes are noise). The *direction* (no stabilization, lean drifts) is clear; the exact numbers are not.

**Verdict:** the un-lagged self-consistency loop does **not** install/stabilize the conditional rule even with the
truncation confound removed — consistent with **structural bistability** (reward crossover at p\*=0.8 ⇒ one-boxing
only out-earns two-boxing once the policy already one-boxes >80%, so prompts spiral to one-box-flat or two-box-flat).
Adapter: `results/adapters/evidential_r1_loo`. Log: `results/r1_loo.log`.

### Oscillation / basin-flipping — did a run ever flip basins, or just settle flat? (asked 2026-06-26)
**No clean oscillation. Each run commits to ONE basin (seed-selected) and stays — we never saw a settled
one-box basin flip to a settled two-box basin or back.** `p_model` (predictor accuracy = policy one-box rate),
step 1→10→20→30:

| run | `p_model` trajectory | endpoint |
|---|---|---|
| cot seed0 (calib) | 0.998 → 0.731 → 0.251 → 0.040 | two-box basin — *monotonic slide* |
| cot seed1 | 0.687 → 0.603 → 0.579 → 0.670 | one-box-flat (mean_K→0.94) — *stayed put* |
| cot kl0 | 0.998 → 0.500 → 0.617 → 0.074 | two-box basin — *bounced up, then crashed* |
| A2 (clean) | 1.000 → 0.917 → 0.333 → 0.500 | *ended mid-wobble* (fell, then bounced) |

- **Norm = a one-way slide into a single basin**, chosen by seed/noise, not by anything principled.
- **No settled-basin → settled-basin flip** observed. Once *deep* in a basin (`p_model`→0.04, or pinned high), stable.
- **But damped non-monotonic bounces before lock-in** (kl0: dip→bounce→crash; A2: fell→bounced, ended in motion);
  the conditional slope also spiked transiently (+1.0) and decayed. **Damped wobble yes; sustained oscillation no.**
- **Why structurally:** two stable attractors (one-/two-box-flat) with a **repeller at p\*=0.8** between them ⇒ the
  system *commits* to one side, doesn't cycle. Genuine oscillation needs **overshoot**, which most plausibly comes
  from a **lag** (predictor reacts to a stale self). The lagged runs (cot) are exactly the truncation-confounded
  ones; A2 *removed* the lag (least likely to oscillate by design). ⇒ **to hunt oscillation/hysteresis: a clean,
  long, lagged variant.**

**Next (queued, user-approved 2026-06-26):** **lagged-snapshot A2** — predictor = a frozen policy snapshot from k
steps ago that generates K *reasoned* samples → `p_eff` from their one-box rate (≈2x generation, ~8–10h; 2560 tok so
`</think>` closes → no confound). The overshoot/hysteresis test the un-lagged A2 structurally can't run.

---

## Lagged-snapshot A2 (lag=3, regenerating predictor) — PRELIMINARY: lag doesn't change the story; confound recurred (2026-06-27)

**What ran.** `selfplay_lag` (`--tag r1_lag`): predictor = a *concrete frozen snapshot* of the policy from **3 steps
ago** that **regenerates** its own K reasoned samples per prompt (CoT) → `p_eff` = their one-box rate (≈2x generation
vs A2's leave-one-out reuse). R1-Distill-8B, m3, grid 0.5/0.6/0.9/0.99, **K=4/P=3, lag=3, 2560-tok**, **40 steps**,
eval-every 10, eval-items 8, seed 0. ~13.5h. Adapter: `results/adapters/evidential_r1_lag`; log `results/r1_lag.log`.

**Result — same basin-drift as A2, no stable rule; the lag added at most a damped wobble (and a confound).**
- **Slope never stabilizes** (eval, 0→10→20→30→40): +0.50 → +0.45 → +0.50 → **+0.00** → +0.45 — wobbles around the
  base R1 level, never climbs/holds. No conditional rule installed.
- **Lean slides toward the two-box basin:** mean_K 0.812 → 0.638 → 0.646 → 0.464 → 0.424; `p_model` (= lagged
  snapshot's one-box rate) 0.917 → 0.750 → 0.333 → **0.500** → 0.194.
- **Overshoot?** Mild — `p_model`/chooser-K bounce *up* at step 30 (0.333→0.500) then resume the slide. **But this
  exact 0.333→0.500 bounce also appears in un-lagged A2** ⇒ it may be loop/sampling noise, *not* the lag. No sustained
  oscillation, no demonstrated hysteresis. Qualitatively: drift into a basin, same as A2 and the cot runs.

**⚠ CONFOUND RECURRED — quarantine the late steps.** Reasoning **lengthened during training** and overran the budget:
gen_len 1851 → 2372 → **2560 (CAP)** → 2555 → 2522, with **invalid 0.17** at steps 20 & 40 (2560 was tuned on A2,
where gen_len peaked ~2270). 17% invalid is in the "suspect" band (>5%); the step-30 bounce sits inside the truncated
region → **do not bank the overshoot read.** Steps 0–10 are clean and already show the slope not climbing.

**Verdict: PRELIMINARY / consistent-with-spine but oscillation question still open.** Even a concrete lagged
self-predictor doesn't install/stabilize the rule (robust across lag-0 and lag-3); but the *interesting* late-step
dynamics are truncation-degraded, AND lag was changed simultaneously with reuse→regenerate / eval-items / step-count,
so any lag-specific effect is unattributable. **Clean follow-up (running 2026-06-27): lag=0-regen control + lag=3,
both at `--max-new-tokens 3072`** — isolates regeneration from lag and removes the truncation cap.

---


## Day-3 consolidated state — where the project stands (2026-06-24)

*Supersedes the Day-2 update as the current top-level synthesis (Runs 1–11 + a 14B scale probe).*

**The result.** A small-to-mid LLM does not track the stated predictor accuracy in Newcomb
problems, and the reason is **not** an inability to compute expected value — it's a
**decision-theoretic disposition** (causal "don't forgo the guaranteed box" / two-boxing dominance)
that **overrides explicit EV at the moment of commitment**. This disposition is **robust to scale
(3B→14B), to RL, and to SFT**, and is even **amplified by spelling out the calculation**.

**Evidence chain (everything converges on the same wall):**
- **RL** moves the model's *lean* (intercept) easily, any direction, but never installs the
  *conditional rule* (slope) — confirmed at the logit level and **per-item** (Runs 2/5/6 +
  item-analysis), under CoT (Run 7), with a fair/paired objective (Run 7 + seed-confirm Run 10:
  slopes +0.17/−0.13/+0.13 = noise), and with a real model-based predictor (Run 6).
- **SFT** on the model's *own* correct traces doesn't install the slope either, and a 4×-bigger SFT
  set doesn't help → a **true ceiling, not data-limited** (Runs 8, 11A).
- **Computation transplant** (hand it the full EV at inference) is the capstone: told
  "EV(one-box)=99 > EV(two-box)=61, pick one-box," it still **refuses to one-box at high p** because
  that means forgoing the guaranteed box (Run 9).
- **Scale (14B) sharpens, not fixes:** flawless where EV agrees with grabbing the guaranteed box
  (1.00 at low p) but one-boxes **0–15% at p=0.99 even handed the EV** — *worse* than the 3B. The
  bigger, better reasoner is a *more confident* two-boxer (Run 10).
- **Two mechanism clues:** (a) spelling out the EV breakdown is *worse* than just stating the
  conclusion — it surfaces the guaranteed reward and activates the dominance pull (Runs 10, 11B);
  (b) RL can install a disposition that ignores explicit EV outright (causal arm: full calc moves it
  **0.00**, Run 11C).

**Headline framings (audience-dependent):**
- DT/RL: *"RL moves the lean, not the rule"* (intercept vs slope).
- General: *"we handed it the answer and it still wouldn't act on it."*
- The sharp one: *"showing a more capable model the EV calculation makes it* less *likely to follow
  it — because the calculation surfaces the guaranteed reward it won't give up."*

**Honest caveats.** Abstract-token opaque-Newcomb only; 14B is n=12/p single-run (strong but
unconfirmed); the 14B free-CoT tracking is weak/non-significant; de-noising dissolved several phantom
intermediate effects (the +0.50 CoT slope, the "+0.16 SFT slope", the "base 0.75 level") — that
discipline is now baked into CLAUDE.md "Sanity gates".

**Open agenda — the two highest-value next builds (need design *with* the user):**
- **Anti-Newcomb camouflage** ⭐ — *does the model react to the Newcomb **costume** or compute the EV
  on the **numbers**?* Same EV math, Newcomb story stripped (medical test, packet cache,
  insurance…). Decisively disambiguates: is the override the *Newcomb prior* (story-triggered) or a
  genuine EV-action failure? If the model tracks p in camouflage but not in containers → it's the prior.
- **Equation-only RL env** ⭐ — RL on bare `A pays pB / B pays S+(1−p)B, choose`. Learns the slope →
  bottleneck is extraction/framing; fails → small-model RL capacity. (Both are in `OVERNIGHT.md`.)

**Status:** all 11 runs committed (`eb6d6f8`→`86d6529`→`9c7ff11`); GPU idle; nothing running.

**Day-4 oracle-anchor update (2026-06-24)** — *calibration of the RL loop; full writeup in the Day-4
summary directly below.* Evidential arm with the predictor accuracy **pinned** to extreme values
(B=100, S=60, p\*=0.8), base 3B, **greedy** K (start→end): **p=1.0** (perfect) 1.00→1.00 · **p=0.5**
(useless coin) 0.85→0.00 · **p=0.0** (inverted) 0.65→0.00 — the loop drives one-boxing only when it
is EV-optimal (p>p\*) and two-boxing otherwise, **reversing even the base's own ~0.8 one-box lean**
(Run 1). Causal-seeded (the RL'd two-boxer, margin ≈ −18) at p=1: **0.00→1.00 by step 40** — flipped
back to one-boxing, *but* most likely via the **KL-to-base** reference (which one-boxes), not the
oracle reward (margin −18/temp 1.3 ⇒ P(one-box)≈1e-6 ⇒ ~0 exploration); a `kl_coef=0` control is
needed to attribute. **Key caveat:** the one-box-leaning *start* is the base's shallow forced-choice
*reflex* (flat in p ≠ EV competence, Run 1); the "stubborn two-boxer / dominance-override" picture is
the reasoning-engaged (CoT / explicit-EV) and 14B regime — same model, two formats.

---

## Day-4 summary (2026-06-24) — oracle anchors + the capability-vs-disposition reconciliation

**What we ran & why.** A zero-new-code "oracle anchor" *calibration* of the RLOO loop before trusting
it on the harder self-snapshot (C2) experiment: the evidential arm with the stated predictor accuracy
**pinned** to extreme, known values, to check the loop does the obviously-correct thing. Standalone
driver (`scratchpad/oracle_anchor.py`, imports `newcomb_rl`, edits nothing under it); forced-choice,
greedy in-loop eval; payoffs B=100, S=60 ⇒ p\*=0.8. One run per (predictor setting, seed).

**The dial** (EV per pinned p; the predictor's reliability sets which action wins):

| p (predictor) | meaning | one-box EV | two-box EV | EV-optimal |
|---|---|---|---|---|
| 1.0 | perfect (fill iff one-box) | 100 | 60 | **one-box** |
| 0.5 | useless (coin; fill ⊥ action) | 50 | 110 | **two-box** |
| 0.0 | inverted (fill iff two-box) | 0 | 160 | **two-box** |

**Results** — greedy K (deterministic eval, averaged over the 20 items), start → end of training:

| run | seed | K start | K end | trajectory |
|---|---|---|---|---|
| p=1.0 | base | 1.00 | 1.00 | flat (already one-boxing) |
| p=0.5 | base | 0.85 | 0.00 | monotone down |
| p=0.0 | base | 0.65 | 0.00 | monotone down |
| p=1.0 | **causal** (RL'd two-boxer) | 0.00 | 1.00 | 0.00→0.75 (step 20)→1.00 (step 40), pinned |

**(1) The loop is a validated instrument.** It drives one-boxing iff one-boxing is EV-optimal (p>p\*)
and two-boxing otherwise — in **both** directions, and **against the base's own disposition**: base 3B
forced-choice *leans one-box* (~0.83 flat, Run 1; greedy 0.85/0.65 here), yet the loop trained it
*down* to full two-boxing at p=0.5 and p=0. The machinery moves the policy wherever the payoff points
→ green light for C2.

**(2) Capability-vs-disposition reconciliation (resolves an apparent contradiction).** The project
headline is "the model is a committed two-boxer that can't see past CDT dominance" — yet these runs
*start* one-box-leaning. Both are true, in **different regimes**:
- **Bare forced-choice** (single-char answer, no CoT, no EV spelled out): the 3B defaults to
  **one-boxing ~0.80, FLAT in p** (Run 1; logit P(non_cdt) ≈ 0.805, Run 5). This is a *shallow
  reflex*, **not** competence — flat in p ⇒ it is *not* computing the EV; it just emits the canonical
  one-box label.
- **Reasoning engaged** (CoT or EV spelled out) and/or **scaled to 14B**: the dominance argument
  surfaces and **overrides** the reflex → two-boxing (Runs 9/10/11; 14B one-boxes 0–15% at p=0.99
  *even handed the EV*; spelling out the EV makes it *worse*).

So "leans one-box" (reflex-only) and "stubborn two-boxer" (reasoning-engaged / scaled) are the **same
model in two formats**. The one-box start in the oracle runs is the *predicted* shallow reflex, **not a
stochastic artifact**: step-0 eval is greedy (deterministic) and averaged over 20 items (0.85 = 17/20),
not a lucky first sample.

**(3) The causal flip — striking but probably a KL artifact, flagged not banked.** The RL'd two-boxer
(`causal` adapter) under the perfect-predictor payoff flipped to one-boxing in ~40 steps. Tempting to
read as "the oracle payoff dissolves the trained-in disposition" — but the arithmetic refutes the
on-policy story: at margin ≈ −18 / temp 1.3, P(one-box) ≈ 1e-6, so of the ~64 rollout samples/step
(~1,300 by step 20) essentially **zero** are one-box; reward can't move a policy it never samples. The
plausible driver is the **KL-to-base reference** (`disable_adapter` → base, which one-boxes at p=1):
the KL term pushes logits toward base on *every* token regardless of sampling, eroding the two-box
margin until one-boxing becomes samplable, after which reward snowballs it. **Disambiguator (deferred
GPU run): a `kl_coef=0` control** — still flips ⇒ reward-driven; stalls at K=0 ⇒ KL-to-base did it.

**Also built today (CPU-validated, GPU-deferred behind the Oesterheld external-dataset run):** **C2 —
the self-snapshot predictor** (`newcomb_rl/selfplay.py`; new file, no edits to existing `newcomb_rl`):
a second *frozen* PEFT adapter that lags the live policy and supplies the evidential `p` (vs the
frozen-*base* C1 of Run 6 — the version results.md flagged as "the one that could actually move"). PEFT
multi-adapter mechanics (add/set adapter, snapshot copy, predictor forward, trainable-restore)
unit-tested on CPU. **Next:** the **hysteresis** run — seed from base (one-box-leaning) vs the causal
adapter (two-box-leaning) and test whether the *initial disposition selects the self-prediction fixed
point* (one-box vs two-box basin / bistability).

**Artifacts:** `results/oracle/{oracle_p1_base,oracle_p1_causal,oracle_p0_base,oracle_p05_base}.log`;
adapters `results/adapters/evidential_oracle_*`; C2 code `newcomb_rl/selfplay.py`.

---

## R1-Distill-Llama-8B — a reasoning-trained model *tracks p* (2026-06-24)

**Why we ran it.** Our scale story so far was monotone: 3B has a flat ~0.80 one-box *reflex* (no
`p`-tracking); the 14B *sharpens the dominance-override* (refuses one-boxing 0–15% even at p=0.99,
even handed the EV). `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` is the clean test of whether that trend
is about **scale** or about **reasoning**: it's a high-capability-for-size model *trained to reason*
(native long `<think>…</think>`). Inference-only quick battery, **no new code** (all probes take
`--model`; CoT tools take `--max-new-tokens`). 6 opaque items, abstract tokens, randomised order.

**Measurement note (the Tier-0 gate earned its keep).** First smoke at `--max-new-tokens 1536` had
5/6 traces **truncated mid-`<think>`** → the forced-answer readout was reading an *interrupted* chain
(the Run-3/4 trap). Raising to **4096** closed `</think>` on ≥92% of traces; only then are the numbers
trustworthy. Also note **p=0.8 is exactly p\*** (EVs equal = 80), so the model correctly reasons to a
*tie* there and those traces burn the whole budget — they dominate runtime, not a bug.

**Headline — considered disposition (CoT, 6×2 per p, `</think>`-closed ≥0.92):**

| p | K-rate (one-box) | EV-optimal | mean logit-margin |
|---|---|---|---|
| 0.50 | **0.08** (two-box) | 0.92 | −5.9 |
| 0.70 | 0.50 | 0.50 | +0.5 |
| 0.80 (=p\*) | 0.58 | 0.42 | +1.4 |
| 0.90 | **0.83** (one-box) | 0.83 | +5.6 |
| 0.99 | 0.75 (one-box) | 0.75 | +4.5 |

A strong, monotone **p-tracking** slope with the crossover sitting right at the theoretical **p\*=0.8**,
driven by *explicit EV reasoning* in the transcripts (e.g. p=0.99: *"99% chance of 100 vs always 60 →
Q only is better"*; p=0.5: *"take both… 60 > 50, regardless of prediction"*). **This breaks the
"scale ⇒ harder CDT dominance" trend**: the reasoning-trained 8B *follows the EV math* where the 14B
refuses. So the 14B's dominance-override is a **disposition of that model**, not an inevitability of
capability — engaged reasoning + correct EV ⇒ EV-rational (EDT-ward at high p) choice.

**Within-model: reflex vs considered (the capability-vs-disposition axis in one model).** Forced
logit-margin (no free-gen, = pre-reasoning reflex) only *mildly* tracks p (K 0.42→0.60, margin
−0.34→+0.39). But the **"told-not-to-think"** reflex (force empty `<think></think>` then read the
answer) is **flat one-box = 1.00 at every p** — a strong one-box *prior* (echoes the 3B's flat-high
reflex). So: **no reasoning → one-box prior; reasoning installs the p-sensitivity** (and overrides the
prior to two-box at low p). Reasoning is doing the EV work, exactly the lever the 3B can't pull.

**Represented dependence (credence) — weak/inconclusive here.** Forced-token credence is mostly
saturated: `outcome` degenerate (gap≈0.008 flat), `prediction` gap_adj only 0→0.11 (vs ideal 2p−1
→0.98). Consistent with the known forced-token confound; we deliberately skipped the free-form
`direct` variant (it would trigger reasoning on this model). Don't over-read this tier — the action
tiers (above) are the signal.

**Artifacts:** `results/cot_inspect/cot_r1d_cot.{jsonl,html}` (readable transcripts);
`results/logprob/p_margin_by_p_r1d_{forced,reflex_nothink}.csv`; `results/credence/*r1d*`;
plot `results/r1d_logs/r1d_krate_vs_p.png`; step logs `results/r1d_logs/`. Driver
`scratchpad/run_r1d_battery.sh`.

**Open follow-ups (not run):** canonical CI plot via `run_mvp --cot --max-new-tokens 4096` (regex-parse
companion to the cot_inspect curve); slot R1-Distill in as a rung on the credence ladder *with the
free-form `direct` variant at a large budget* (the only credence readout likely to register on a
reasoning model); seeds/CI on the p=0.99<0.90 dip (likely n=12 noise).

**Design note — why the iterated-game predictor must *reason*, not just read logits (2026-06-25).** A
tempting simplification is to source the predictor's `p_eff` straight from the policy's next-token
logits (cheap, no generation). For R1 this **fails**: R1's p-tracking lives *only in the reasoning
chain*. Read the immediate next-token distribution — the "reflex" / no-think mode — and it is **flat
one-box ≈ 1.0 at every p** (p-blind), exactly the 3B's dead end (its p-blind reflex predictor is why
that loop gave only flat bistability, no conditional fixed point). So the predictor must **generate a
`<think>` chain and then read the post-reasoning answer** ("reason-then-read"). "Sample from the logits"
only works if it means autoregressively sampling a *full completion* (= letting it reason); stopping at
token 1 is the p-blind reflex. A cheaper *valid* variant (not used here): reuse the policy's own K
rollout reasoning samples as the prediction (leave-one-out one-box rate) — carries the conditional
signal but drops the lagged-snapshot/hysteresis structure that makes this a two-player iterated game.
See `newcomb_rl/selfplay_cot.py` (`_predictor_p`) and `R1_ITERATED_PLAN.md`.

---

## Day-5 (2026-06-25) — kl-control closes the causal-flip; the 3B is a clean null (intercept movable, slope stuck)

**(1) The Day-4 causal→one-box flip was the KL leash, not the reward — confirmed.** Reconstructed the
lost oracle driver (`scratchpad/oracle_kl_control.py`: seed the policy from an adapter, pin the predictor
accuracy, run evidential RLOO; seeding mirrors `selfplay._load_into_default`) and ran the *same* driver
at two KL settings, seeded from the `causal` two-boxer at pinned p=1.0:

| run (same driver, only `kl_coef` differs) | K @ step 150 | reads as |
|---|---|---|
| `kl=0.02` (matched baseline) | **1.00** | flips to one-box — reproduces Day-4 |
| `kl=0` (control) | **0.00** | never moves |

With the KL-to-base leash cut, the oracle reward cannot budge the trained two-boxer (margin ≈ −18 ⇒ ~0
one-box samples ⇒ reward can't move an action it never explores). The Day-4 flip was the **KL-to-base
reference** (base one-boxes at p=1) eroding the two-box margin until one-boxing became samplable — not the
reward dissolving the disposition. Logs: `results/oracle/run_oracle_klctl_kl{00,002}.log`; plot
`results/dynamics_klcontrol.png`.

**(2) "Does RL just keep the 3B in whatever mode it's in?" — no; it moves the *lean* freely, it just
can't install the *rule*.** "Keeps it in whatever mode" only holds where there's **no gradient**. With a
clear EV gradient, RL moves the 3B hard, *against its own prior*: the base 3B leans one-box (~0.85), yet
p=0.5/p=0 oracle reward drives it to **0.00, full two-box** — and it does that *against the KL anchor too*
(KL pulls toward the one-boxing base; reward beat it; see `dynamics_3b.png` Panel B). So RL is **not**
conservative on the 3B. The precise statement is the project headline:

> **RL moves the 3B's *lean* (intercept) freely — any direction, even against its prior — but cannot
> install the conditional *rule* (slope).**

The word "mode" is the **intercept**, and the intercept is the *cheap, movable* thing; the stuck thing is
the **slope**.

**Why the iterated game (`dynamics_3b.png` Panel A) *looks* inert:** it pins p at exactly p\*=0.8 (the
tie → no gradient), and the 3B predictor is **p-blind**, so the self-referential loop can only ever
reinforce the *lean* — and at the tie there isn't even a lean signal. The dynamics are degenerate because
the **substrate** is degenerate (a p-blind reflex), not because RL is inherently sticky. The `kl=0` result
(1) makes the 3B *more* of a null: even the one apparently-interesting dynamic ("reward dissolves a trained
disposition") was an artifact — the trained modes are if anything **stickier** than we thought; reward
alone won't dislodge a saturated disposition.

**Synthesis — the 3B is essentially done, and that's the point.** The interesting object was never the 3B
itself — it is the **intercept-vs-slope decomposition** the 3B let us nail down (movable disposition,
un-installable competence; a clean null on the slope at every lever: RL / SFT / transplant / fair
objective). The *interesting dynamics* we actually want — hysteresis that **tracks p**, a **conditional**
self-fulfilling fixed point — **require a predictor that conditions on p**, and the 3B fundamentally
**cannot be one**. R1-Distill can (it tracks p when it reasons). So "the 3B is boring" is not a dead end —
it is the correct conclusion that **points directly at R1** as the next move. That move is gated only on
the R1 iterated-game **cost** (timing benchmark `scratchpad/r1_timing.py` → `results/r1_timing.log`): R1
carries a p-signal only *through* its reasoning, so both policy and predictor must emit full `<think>`
chains — the open question is wall-clock, not whether the dynamics could be interesting.

**Artifacts:** `results/dynamics_3b.png`, `results/dynamics_klcontrol.png`, `scratchpad/oracle_kl_control.py`,
`scratchpad/plot_3b_dynamics.py`, `scratchpad/r1_timing.py`.

---

## Run 1 — Baseline: Qwen2.5-3B-Instruct  (2026-06-22)

**Headline:** flat-high K-rate (~0.70–0.95) across the whole `p` range, **no tracking of `p`**,
slight dip near `p*`. The model one-boxes ~83% of the time *regardless* of the stated accuracy:
**no evidence of `p`-tracking in this (forced-choice) setup** — a strong prior toward the
canonical one-box answer. This is the baseline RL is meant to move (flat-high → a curve that
switches at `p*`).

> **Updated by Runs 3→7 (corrected):** Run 3 first read this flat line as a mere *format* artifact
> ("under CoT the model tracks `p` at +0.50") — but that +0.50 was a **scoring confound** (38–48%
> invalid). Robust-scored (Run 7), the base is **≈ flat under CoT too**, and RL can't install the
> conditional step even with a fair objective — so the flatness reflects a **capability ceiling**
> (the 3B does EV on *some* items but not reliably), not merely a single-token format limit.
> Mechanism inferred, not fully measured. SFT (Run 8) tests whether the competence can be installed.

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

> **⚠️ Corrected by Run 7.** The numbers below were scored with the pre-P1 CoT parser (step-1
> `invalid=0.48`), which scored prose-without-a-label as not-K and **manufactured the +0.50 baseline
> slope**. Robust re-scoring (Run 7) shows the clean CoT-evidential baseline slope is ≈ 0 and RL
> stays flat. Read this entry as historical; the corrected picture is Run 7.

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

> **⚠️ Corrected by Run 7.** `free_cot` here hit **38% invalid** under the pre-P1 parser; its
> **+0.51 slope is largely that artifact** (invalids scored as not-K, uneven in `p`). The
> *scaffolded* arm (1.2% invalid) and the qualitative extraction finding stand; the free-CoT slope
> does not. Robust-scored CoT is ≈ flat (Run 7).

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

## Run 7 — Phase-3: robust CoT scoring + the "fair shot" at the slope (Qwen2.5-3B-Instruct, 2026-06-23)

This phase tried to *break* the intercept-vs-slope finding by (a) fixing a scoring confound and
(b) giving RL an objective that *requires* conditioning. Headline: **the finding holds, and a key
prior result was a measurement artifact.**

**(0) Confound correction — the old CoT "slope" was largely a scoring artifact.** Runs 3–4 scored
CoT by hunting for the abstract label in free prose, which failed 38–48% of the time (the model
writes "take only container Q", not "Q"); invalids were scored as not-K and clustered unevenly in
`p`, **manufacturing a +0.50/+0.51 slope**. Fix (**P1**): after the reasoning, teacher-force an
`Answer:` continuation and read the label (`rloo._forced_answer_roles`) → invalid 38–48% → ~0–8%.
**Re-scored, the CoT-evidential *baseline* slope is ≈ 0**, not +0.50. (Lesson recorded in CLAUDE.md
"Sanity gates"; this is the morning's confound discipline applied to our own headline.)

**(1) KL sweep (Lever 1a) — flat at every KL.** Clean-scored CoT-evidential RL, kl ∈ {0.02,0.05,0.10}:
final slope **+0.12 / +0.00 / +0.00** (mean_K ~0.54–0.59, n=8). KL anchoring is **ruled out** as a
lever; the intercept-vs-slope result holds under robust CoT scoring.

**(2) Paired / EV-balanced (Lever 1b) — the fair objective did not unlock the slope.** Paired
sampling (same item at a `p_low<p*` and `p_high>p*` per rollout) makes a p-independent policy earn
~0 net advantage, so *only conditioning on p* wins. Result (n=15 eval): final slope **+0.17** but
**unstable** across checkpoints (+0.07→0.00→−0.07→+0.17), and the transcript read shows one-boxing
dropped roughly *uniformly* (incl. wrongly at high p) — a weak intercept nudge, **not a clean step**.
So we've now *given RL its fair shot* and it still doesn't form the conditional rule → the
"underdetermined objective" caveat **closes**; the diagnosis shifts to a **capability ceiling**.

**(3) Reading the CoTs (`cot_inspect`) — partial, item-dependent conditioning, drowned in noise.**
The aggregate "flat" hid real structure (one-box rate, n=8/p, same 4 items):

| model | p=0.5 | p=0.8 | p=0.99 |
|---|---|---|---|
| base | 0.75 | 0.75 | 0.75 (flat — no conditioning) |
| evidential-CoT kl=0.02 | 0.50 | 0.75 | 0.75 |
| paired-CoT | 0.50 | 0.63 | 0.63 |

The base one-boxes ~0.75 *flat*; the RL'd models lower one-boxing at low p (toward correct
two-boxing) on *some* items, and the CoTs there show genuine EV math (*"F is correct 50% of the
time, so taking Q gives 50 points on average"* → two-box). So it is **not** "no impact" — it's
**weak, partial, item-dependent conditioning** that an n=8 aggregate is too coarse to resolve.

**Bottom line.** Outcome RL — even with a fair (conditioning-only) objective and clean CoT scoring —
does **not** install the p-conditional step; it produces at most a weak, noisy, partial tilt. The
3B *can* do the EV reasoning on some items but not reliably → **capability ceiling**. Next: **SFT
(STaR)** to install the competence (Run 8, in progress), and a **seed-confirmation** (2–3 seeds ×
more steps) to firm up whether the weak +0.17 is real or noise.

**Reproduce**
```bash
for kl in 0.02 0.05 0.1; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m newcomb_rl.train_rl --arm evidential --cot --kl-coef $kl --steps 60 --K 5 --P 5 \
    --eval-every 20 --eval-items 8 --max-new-tokens 128 --micro 8 --tag cot_kl${kl/./}; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m newcomb_rl.train_rl --arm evidential \
  --cot --paired --kl-coef 0.02 --steps 60 --K 5 --P 6 --eval-every 20 --eval-items 15 \
  --max-new-tokens 128 --micro 8 --tag paired_cot
python -m newcomb_eval.cot_inspect --adapter results/adapters/evidential_paired_cot --tag paired
```
Artifacts: `results/run_cot_kl*.log`, `results/run_paired_cot.log`, `results/cot_inspect/cot_*.{jsonl,html}`.

---

## Per-item conditioning analysis — is the flat aggregate "all flat" or "cancellation"? (2026-06-23)

**What.** A CPU-only re-analysis (`newcomb_eval/item_analysis.py`, no GPU, reads only persisted
`results/*`) that hardens two soft spots in the headline: (1) every prior "slope" was an eyeballed
hi-minus-lo difference with no error bars — here each per-item slope is a proper **OLS fit with
analytic CI**, aggregated by an **item-bootstrap** mean; (2) the "item-dependent conditioning"
claim (Run 7) was never measured. A flat *aggregate* `P(non_cdt)` slope could be *every item flat*
(capability ceiling) **or** *some track + some anti-track, cancelling* (heterogeneity). We classify
each item {tracks / flat / anti} by whether its slope CI excludes 0. (Invariant #1 preserved: the
choice was resolved upstream via abstract tokens; this module never re-scores a choice. Part-D tags
annotate reasoning *content* only.)

**Sanity gate passed.** Per-item mean *levels* reconcile **exactly** with Run 5/6's published
margins (base +3.46, causal −18.16, evidential −0.51, modelpred +8.36) and P(non_cdt) (0.805 /
0.000 / 0.457 / 0.718) — confirms correct files + orientation (tracking = positive slope in `p`).

**Result A — at the logit level the base is genuinely flat *per item*, not cancellation.**
Per-item margin~`p` slope distribution (n=20 items, continuous logprob signal):

| arm | tracks | flat | anti | mean per-item slope [item-bootstrap 95% CI] |
|---|---|---|---|---|
| base3b | 0 | 18 | 2 | −3.77 [−7.24, −0.40] |
| causal | 0 | 19 | 1 | (saturated — see note) |
| evidential | 0 | 20 | 0 | −4.52 [−13.57, +5.18] |
| modelpred | 0 | 18 | 2 | (saturated — see note) |

**Zero tracking items in any forced-choice arm.** The flat aggregate is *real per-item flatness*,
**not** a positive subset hidden by cancellation — this is a clean answer to the open question and
firms up the "capability ceiling, not just averaging" reading (Run 7). (Note: for the *saturated*
arms causal/modelpred the margin is clipped at ±18, so the margin-slope is dominated by a few
extreme items and is unreliable; the bounded `P(non_cdt)` signal is the trustworthy one there — both
are in `item_slopes.csv`.)

**Result B — RL moved each item's *level*, not its *slope* (the per-item paired test).** Joining
base↔RL on `item_id` (Δ = RL−base on the margin fit, item-bootstrap 95% CI):

| arm | Δ level (intercept) | Δ slope |
|---|---|---|
| causal | **−21.6 [−22.8, −20.4]** | +0.85 [−3.30, +4.54] (CI ∋ 0) |
| evidential | **−4.0 [−5.9, −2.1]** | −0.74 [−9.12, +7.90] (CI ∋ 0) |
| modelpred | **+4.9 [+2.6, +7.1]** | −23.5 [−43, −3.8] — *margin-saturation artifact* |

Every arm has a **large, CI-significant *level* shift**; the *slope* Δ is **CI-indistinguishable
from 0** for causal and evidential. modelpred's apparent slope-Δ is the saturation artifact (its
bounded P(non_cdt) slope stays ≈ flat), not a real conditional. The per-item scatter
(`slope_level_shift.png`) shows points clustering on `y=x` for *slope* and a clean offset for
*level* — the aggregate intercept-not-slope claim now holds **item-by-item**, far stronger than
comparing two means.

**Result C — the heterogeneity that *does* exist is binary-sampling noise, not a continuous subset.**
The coarse scaffold K-slope (binary, single-repeat, hi-lo across `p*`) *does* spread items across
tracks/flat/anti (e.g. free_cot: 6 tracks / 10 flat / 3 anti; no_cot: 5/8/7) — but this lives only
in the noisy binary readout and **vanishes in the clean continuous margin** (Result A). So Run 7's
"partial, item-dependent conditioning, drowned in noise" is, on this evidence, the *noise itself*:
the free-CoT "+0.51" came from a modest tracks−anti imbalance in single Bernoulli draws, not a
stable per-item EV competence. (`scaffold_item_arms.csv`.)

**Result D — CoT EV-arithmetic is rare and RL didn't add it (n=4 items, thin).** Heuristic
content-tag rate of explicit EV arithmetic in the reasoning: base 25%, kl002 4%, paired 8% — RL did
**not** raise the rate of EV computation; consistent with "installs a disposition, not the
algorithm." (Thin: 4 items × 3 `p` × 2 samples; flagged.) (`cot_content_tags.csv`.)

**Caveat.** Scaffold/logprob CSVs are single-repeat, so cross-seed item stability is only partially
observable; a 2–3-repeat confirmatory re-sweep is the GPU follow-up (queued to `OVERNIGHT.md`). The
Part-D content tags are a heuristic, not a judge.

**Reproduce**
```bash
python -m pytest newcomb_eval/tests/test_item_analysis.py -q   # 13 tests, no GPU
python -m newcomb_eval.item_analysis --tag base3b
```
Artifacts: `results/item_analysis/{item_slopes,rl_vs_base_per_item,scaffold_item_arms,cot_content_tags}.csv`,
`slope_hist_by_arm.png`, `slope_level_shift.png`.

---

## Run 8 — Lever 2: STaR-SFT competence install (Qwen2.5-3B-Instruct, 2026-06-23)

**Setup.** Harvest the 3B's *own* correct+confident CoT traces (forced-`Answer:` readout, keep iff
EV-optimal AND `|margin|≥1`), balance across p, SFT a LoRA on (prompt → CoT + `Answer:<optimal>`).
Harvest yield was **healthy**: ~35–49% correct at *every* p (incl. p=0.7), 162 balanced examples
(27/p), SFT loss 0.48→0.28 over 3 epochs. Adapter `results/adapters/sft_star` (gitignored).

**De-noised eval (greedy, 20 items × 2 renderings, n=40/p):**

| | LOW p<0.8 | HIGH p>0.8 | slope-contrast |
|---|---|---|---|
| base | 0.49 [0.40,0.58] | 0.53 [0.42,0.63] | **+0.03** |
| SFT (STaR) | 0.42 [0.33,0.51] | 0.49 [0.38,0.60] | **+0.07** |

**Result — SFT did NOT install the slope.** The SFT contrast (+0.07) is statistically
indistinguishable from base (+0.03); within SFT the LOW/HIGH CIs overlap. SFT produced at most a
tiny *uniform* downshift (~0.05, within noise), **not conditioning**. So even training directly on
the model's own correct conditional traces doesn't make the 3B reliably condition on `p` — its
occasional-correct answers weren't backed by a *transferable* EV procedure.

> **Process note (de-noising mattered):** intermediate small-n reads produced phantom effects — a
> "+0.16 weak slope" and a "0.75→0.54 level drop" — both dissolved here. The de-noised base is
> ~0.50 (not 0.75), flat. Trust the contrast (n=120/80), not the per-cell wiggles.

**Verdict:** neither RL (any arm, Run 7) nor STaR-SFT installs the p-conditional slope → strong
**capability-ceiling** support. Caveat: one SFT config (162 ex, 3 epochs); "data-limited" not fully
ruled out → bigger-SFT + 14B-scale probe queued.

**Reproduce:** `python -m newcomb_rl.sft --limit 12 --samples-per-cell 6 --oversample-low 3
--max-new-tokens 128 --epochs 3 --tag star` ; eval via `cot_inspect --adapter …/sft_star
--temperature 0 --limit 20 -n 2`.

## Run 9 — Computation-transplant diagnostic (Qwen2.5-3B-Instruct, 2026-06-23)

**Setup.** Inference-time causal probe (`newcomb_eval/transplant.py`): supply increasing amounts of
the EV calculation externally (none → variables → formulas → numeric_evs → comparison → full) and
read the 2-way answer distribution. `hi-lo` = P(optimal @p=0.99) − P(optimal @p=0.5).

| condition | mean P(optimal) | hi-lo |
|---|---|---|
| none | 0.384 | +0.33 |
| variables | 0.438 | −0.30 |
| formulas | 0.432 | −0.31 |
| numeric_evs | 0.516 | −0.50 |
| comparison | 0.569 | −0.19 |
| full | 0.695 | −0.20 |

**Result.** Aids raise the *mean* (0.38→0.70) but **flip the slope negative** — they help at low p
(where EV-optimal = take-both/dominant) and *hurt* at high p (where EV-optimal = one-box, forgoing
the guaranteed box). Even the **full** aid (numbers + "the higher-EV label is T") yields only ~0.46–0.65
optimal at high p: **the model is told the EV and still won't act on it when acting means leaving
the guaranteed box.** Bottleneck = **action-commitment vs the dominance intuition**, not arithmetic
(numeric_evs supplied → still 0.516) — a *disposition overriding supplied calculation*.

**Confound checked & refuted.** The aid wording reuses single letters `p/B/S` as variable names,
which collide with abstract labels (~24% of renderings have an S/B label). Tested against the data:
colliding-label rows have **equal-or-higher** optimal rate (high-p collide 0.59 vs no-collide 0.46),
so the collision is **not** driving the result. (Clean-wording rerun queued as hygiene, not a fix.)

**Caveat:** single run, n=20/p. Diagnostic, not final. Corroborates Runs 7–8: the locus is
inclination/commitment, not (only) capability to compute.

---

## Run 10 — Overnight: the scale answer (14B) + confirmations (2026-06-23, autonomous)

**★ Headline — scaling to 14B does NOT fix EV-following; it *sharpens* the dominance-override.**
Computation-transplant on **Qwen2.5-14B-Instruct** (collision-proof wording), P(optimal):

| condition | p=0.5 | p=0.7 | p=0.9 | p=0.99 | (3B `full` for ref) |
|---|---|---|---|---|---|
| none | 0.53 | 0.59 | 0.43 | 0.55 | |
| variables | 0.71 | 0.77 | 0.15 | 0.17 | |
| formulas | 0.92 | 0.93 | 0.05 | 0.00 | |
| numeric_evs | **1.00** | 1.00 | 0.32 | **0.00** | |
| comparison | 0.92 | 0.92 | 0.96 | 0.59 | |
| full | **1.00** | 1.00 | 0.78 | **0.15** | 3B: 0.72→0.53 |

Three things, and they reframe the whole project:
1. **The 14B is a flawless calculator where EV agrees with grabbing the guaranteed box** (low p:
   numeric_evs/full = **1.00**). So it is *not* capability-limited at the arithmetic.
2. **It refuses the EV-optimal action where EV says *forgo* the guaranteed box** (high p): handed
   "EV(one-box)=99 > EV(two-box)=61," it one-boxes **0%** at p=0.99 (numeric) / **15%** (full) —
   *worse* than the 3B (0.20 / 0.53). The "don't leave the guaranteed box" disposition is **robust
   to scale and amplified by it** — the bigger, better reasoner is a *more confident two-boxer*.
3. **Spelling out the EV math makes it WORSE than just stating the conclusion.** `comparison`
   ("the higher-EV label is T", no numbers) → 0.96/0.59 at p=0.9/0.99; `numeric_evs`/`formulas`
   (which spell out "two-box = guaranteed + …") → 0.32/0.00 and 0.05/0.00. **Showing the breakdown
   surfaces the guaranteed reward it's loath to forgo, *activating* the dominance pull.**

**This converts the project's conclusion from "small-model capability ceiling" to "a real
decision-theoretic disposition (causal dominance / don't-forgo-guaranteed) that overrides explicit
EV, present at 3B and 14B and *stronger* at 14B."** It's the CDT two-boxing intuition, and a more
capable model holds it more confidently.

**Confirmations (the other 3 overnight stages):**
- **Clean transplant (3B, collision-proof wording):** none 0.384 / full 0.661, same negative-slope
  pattern as Run 9 → **the S/B symbol confound was minor; Run 9 holds.**
- **Paired-CoT seed-confirmation (seeds 0/1/2):** final slopes **+0.17 / −0.13 / +0.13** (mean ≈ 0,
  high variance; seed 1 even collapsed to mean_K 0.158) → **the Run-7 "+0.17" was noise; the fair
  objective definitively did not unlock the slope, and the CoT-evidential RL is *unstable* across
  seeds.**
- **Bigger SFT (3B, relaxed margin 0.3, 16 items, 5 epochs):** harvested 228 examples (loss
  0.47→0.18), adapter `sft_star_big` saved — but its de-noised eval **failed on a script bug**
  (a `--p-grid $VAR` that didn't word-split; the 14B *free-CoT* eval failed the same way). Both are
  **re-running in batch-2** with inline p-grids. (The 14B *transplant* above was unaffected — it
  doesn't take `--p-grid`.)

**Caveats:** 14B transplant is one run, n=12/p (but the 0/12 cells at p=0.99 are a strong signal,
consistent across formulas+numeric). The 14B *free-CoT* slope (does it track p when reasoning
itself, vs being handed EV?) is the recovered eval pending in batch-2.

**Reproduce**
```bash
python -m newcomb_eval.transplant --model Qwen/Qwen2.5-14B-Instruct --limit 12 --tag base14b_clean
```
Artifacts: `results/transplant/transplant_by_condition_p_base14b_clean.csv`, `results/overnight.log`.

---

## Run 11 — Overnight batch-2: ceiling confirmed + "explicit EV hurts" + RL-lock (2026-06-23, autonomous)

Four diagnostics (recovered the two batch-1 evals a `--p-grid $VAR` word-split bug had killed).

**(A) Bigger SFT is NOT data-limited — the 3B ceiling is real.** Scaled STaR-SFT (relaxed margin
0.3, 16 items, **228** examples, **5** epochs, loss→0.18) → de-noised contrast **+0.03** (one-box
LOW 0.42 / HIGH 0.45), *identical to base and to the smaller SFT (Run 8)*. More demonstration data
did **not** install the slope → Run 8's "capability/disposition ceiling" is confirmed, not a
data artefact.

**(B) 14B free-CoT weakly tracks p — and explicit EV *hurts* it (the key contrast).** Reasoning
*itself* (no aid), the 14B one-box rate rises 0.46→0.67 across p, **contrast +0.09** (LOW
0.51[0.40,0.63] / HIGH 0.60[0.46,0.73] — weak, CIs overlap). But when **handed** the explicit EV
(Run 10 transplant), the same 14B *refuses* one-boxing at high p (0–15%). So **spelling out the EV
calculation pushes the 14B the *wrong* way** — it surfaces the guaranteed reward and activates the
dominance pull, overriding the model's own (mildly correct) free-reasoning tendency. "Show your
work" is counterproductive here. (Matches the within-transplant `comparison` > `numeric_evs` at
high p.)

**(C) Transplant on the RL'd/SFT'd adapters — disposition *tunes* EV-use; RL-to-CDT = total lock.**
mean P(optimal), `none`→`full`:

| adapter | none | full | reads as |
|---|---|---|---|
| **causal** (RL'd to two-box) | 0.498 | **0.500** | **ignores the aid entirely** — full calc moves it 0.00 |
| evidential | 0.462 | 0.512 | barely moved by the aid |
| sft_star | 0.455 | 0.596 | aid helps, base-like |
| evidential_paired_cot | 0.432 | 0.664 | most aid-responsive (≈ base) |

The causal-CDT model is **immune to the full calculation** (full ≈ none ≈ 0.50, strong −0.7 slope
= always two-box) → **RL can install a disposition that ignores explicit EV outright.** The others
sit on a spectrum. So "disposition overrides supplied EV" is itself dispositionally tunable.

**(D) Dense-p psychophysics (base 3B, p∈0.72…0.88) — no threshold-local sensitivity.** One-box
rate is noisy-flat (~0.53, range 0.42–0.70) with **no step/peak at p\*=0.8** — if the model did
graded EV comparison, uncertainty/flips should concentrate near p\*; they don't. Consistent with a
*qualitative heuristic*, not EV computation, near the crossover.

**Net:** batch-2 strengthens the Run-10 reframe on three fronts — the 3B failure is a true ceiling
(not data), the disposition is *tunable by RL* (causal→EV-immune), and explicit EV is *counter-
productive* at the conflict point. **Caveats:** 14B n=12/p (free-CoT contrast not significant); dense
n=40/p single-repeat.

**Artifacts:** `results/overnight2.log`, `results/cot_inspect/cot_{sft_big_dn,base14b_dn,dense_base}.{jsonl,html}`,
`results/transplant/transplant_by_condition_p_adapter_*.csv`.

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

---

## Discussion — what this does (and doesn't) say about "EDT-lean = capability/pretraining" (2026-06-24)

A natural critique: *"You're just reproducing Oesterheld's capability→EDT correlation — small models
can do the EV arithmetic but naively grab the dominant (CDT) option, perhaps without even keying
into the prediction structure."* Three responses, kept for the writeup.

**1. We are not simply reproducing the capability→EDT correlation — our scale result points the
other way.** Oesterheld's finding is *more capable → more EDT-lean*. Within Qwen2.5, our 14B is
*more* CDT than the 3B (sharper override: 0–15% one-boxing at p=0.99 *even handed the EV*, Run 10).
Caveats: different axis (he measures *attitude/preference*; we measure *one-box rate* on a specific
opaque-Newcomb with explicit `p`), different stimuli, and our 14B is a single n≈12/p run. But "just
reproducing his correlation" doesn't hold — the direction is wrong for it.

**2. The "didn't notice / didn't key into the prediction" confound is the serious one, and only
*mostly* closed.** If the model never engages the evidential structure, its CDT choice is
*non-engagement*, not *disposition overriding EV*. Main defense: the **transplant** *hands* it the EV
(removing the need to notice/compute) and it still two-boxes — worse, at 14B (Runs 9–11). Two gaps
remain: (a) supplying "EV(one-box)=99" doesn't force the model to *credit* the evidential framing;
(b) only ~25% of base CoTs show explicit EV arithmetic, so for the rest we can't rule out
non-engagement. Where the model *does* show its work it does the predictor reasoning and *still*
two-boxes — the strongest form of the claim, but not universal. This is exactly what the
**comprehension-gate** and **anti-Newcomb camouflage** builds are for.

**3. Disentangle two deflationary stories — we weaken one, not the other.**
- *(a) "EDT-lean is just doing the arithmetic"* (anyone who can compute EV picks EV-max). **Weakened:**
  the 3B *can* do the EV arithmetic (sometimes explicitly in its own CoT) yet goes CDT → EDT-lean is
  **not reducible to compute capability**; a capable/thinking model that answers EDT is adding
  something *beyond* arithmetic (crediting the correlational stance). This is the valid kernel.
- *(b) "EDT-lean is a pretraining/RLHF-absorbed bias."* **Not weakened.** A pretrained stance can be
  **capability-gated** (needs capacity to express), so "absent in the small model" is fully
  consistent with "absorbed bias surfacing only with scale." Our setup is essentially **silent on the
  origin** of EDT-lean because our models don't *exhibit* EDT-lean here — they exhibit CDT-default.
  You can't infer where a thing comes from when your instrument never produces it.

**The experiment that would directly engage the critique** is the deferred **attitude half**: run the
Oesterheld *attitude* (EDT-vs-CDT preference) questions across 3B / 14B (ideally + a 32B-4bit) and see
whether *lean* tracks size **in-family**. That is the apples-to-apples test of the capability→EDT
claim. The capabilities run (this section's sibling, `newcomb_eval/newcomblike_oesterheld.py`) measures
*correctness*, a different axis — useful as a competence yardstick but it does **not** by itself settle
the EDT-lean question.

---

## Oesterheld external reality-check — 3B vs 14B on his dataset (2026-06-24)

Ran **Qwen2.5-3B & -14B** on the **Oesterheld 2024** Newcomb-like dataset (arXiv 2411.10588; data
`github.com/casparoe/newcomblike_questions_dataset`, pw `onebox`) — both axes, faithful port of his
`FINAL ANSWER:` grading (invariant #1 intact: letters only, no string-match). Module
`newcomb_eval/newcomblike_oesterheld.py` (+18 CPU tests); 20 items × 3 shuffled repeats, seed 0.
**Full writeup: `results/newcomblike/SUMMARY.md`** (gitignored, like all results/).

**Numbers** (item-level bootstrap CI; "lenient" = case-insensitive re-score recovering the 3B's
miscased-but-committed answers — faithful ≈ lenient, so robust):

| axis | 3B | 14B | random-guess baseline |
|---|---|---|---|
| **Capabilities** (accuracy) | 0.35 [0.18, 0.53] | **0.69** [0.51, 0.87] | **0.44** (subset is 15/20 binary, not 0.33!) |
| **Attitudes** EDT-rate | 0.52 [0.33, 0.70] | 0.43 [0.25, 0.62] | 0.52 EDT / 0.50 CDT |

**Three reads:**
1. **Capability: 3B at/below chance (0.38 vs 0.50 on binary items), 14B well above.** External
   corroboration of the **competence ceiling**, and it closes the "our hand-built items are *sus*"
   loophole (PLAN §5a's original purpose for this check).
2. **Attitude lean ≈ balanced for both — but NOT noise.** Aggregate EDT-rate sits on the chance
   baseline, yet greedy 3-shuffle repeats agree on **14/20 items** (chance ≈ 5/20) → **determinate,
   order-robust stances that split ~evenly across scenarios** (3B 11 EDT/8 CDT items; 14B 8/11). A
   *real but balanced* disposition. 14B tilts **slightly more CDT** (n.s.) → **no capability→EDT
   in-family** (same direction as Run 10). Caveat: 6/20 items flip with option order (position bias).
3. **Cross-setup gap (the one genuinely additive thing):** balanced on *generic* attitude questions
   vs **hard CDT in our opaque-Newcomb** → the override is **framing-driven**, not a global CDT prior.

**Verdict — control, not finding.** Mostly re-confirms our own results; the core mechanism (dominance
overrides *explicit* EV) is entirely ours and Oesterheld's dataset can't probe it (no explicit-`p` /
guaranteed-box / transplant). Its real value is two *controls*: (a) external validity on the 3B
competence ceiling; (b) the neutral-disposition baseline that lets us attribute the CDT override to
our framing. Worth ~2 sentences in a writeup (preempts the "toy items" critique). To make the attitude
axis *actually test* capability→EDT would need a wider in-family ladder (0.5B→32B × seeds) — likely
**breaks** the correlation given everything points to "scale sharpens CDT, doesn't buy EDT," but that's
a real GPU spend, only if we want to flag-plant on Oesterheld's correlation directly.

---

## Credence probe + the mechanism-credibility pivot (2026-06-24)

**Why.** Every action probe shows the *choice* is flat in `p` (disposition, not conditioning). The
**credence probe** asks the prior question: does the model even **represent** the action↔box
evidential dependence — `P(box full | my action)` → `2p−1` — *separately* from acting on it?
("represented-but-unused".) Teacher-force a committed action, read the credence the conditional prize
is full, sweep `p`, fit the gap to the `2p−1` ideal. Scoring via `answer_2way` (invariant #1 intact).
Modules: `newcomb_eval/credence_probe.py` (+ family ladder `credence_ladder.py`). Three elicitations:
`outcome` (named prize "holds 100/0"), `prediction` (abstract-label True/False), `direct` (free-form
numeric).

**Two instruments, opposite verdicts — the elicitation method dominates.**
- **Forced-token (`outcome`/`prediction`) is CONFOUNDED — quarantine.** Resolvability 1.0, but the
  symmetry control fails hard: `gap@0.5 ≈ +0.71`/`+0.74` on 14B where it must be ~0; `outcome` is
  **one-sidedly saturated** (`P(full|one-box)=1.0` at *every* p incl. 0.5). Forcing the credence token
  lets the committed action's evidential rationale dominate, `p`-blind. Do not read these slopes.
- **Free-form (`direct`) is CLEAN — and inverts the scale expectation.** Headlining `direct + gap_adj`
  (the de-confounded number), the p-conditioned dependence is **represented near-perfectly at the
  small/mid rungs and DEGRADES with scale:** repr-slope **3B +0.96, 7B +0.94, 14B +0.21**. The **7B is
  a near-textbook evidential reasoner** — `P(full|1box)=p`, `P(full|2box)=1−p` essentially exactly
  across the whole grid, **symmetric at p=0.5** (gap 0.000). **3B** tracks cleanly too (gap@0.5 +0.03).
  **14B is the muddiest:** a pedestal (gap@0.5 +0.19) *and* a real p=0.99 wobble (it reasons
  "99…however" into the number). So **representation does NOT need scale — it's strongest at 3B/7B and
  worst at 14B** (the big model over-thinks even the free-form credence).

| rung | **repr-slope (direct, gap_adj)** | repr (outcome, *saturated — ignore*) | action margin hi-lo |
|---|---|---|---|
| 3B  | **+0.96** | −0.00 | −1.31 |
| 7B  | **+0.94** | +0.00 | +3.39 |
| 14B | **+0.21** | +0.04 | +0.45 |

*7B exemplar (direct, parse-fixed) — a near-textbook evidential reasoner:*
| p | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| P(full \| 1-box) | .50 | .60 | .70 | .75 | .80 | .85 | .90 | .99 |
| P(full \| 2-box) | .50 | .41 | .30 | .25 | .20 | .15 | .10 | .11 |
| gap (ideal 2p−1) | .00 (.00) | .19 (.20) | .40 (.40) | .50 (.50) | .60 (.60) | .70 (.70) | .80 (.80) | .88 (.98) |

Artifacts: `results/credence/ladder_signature.{csv,json,png}`,
`credence_gap_by_p_base{3b,7b,14b}.csv` (filter `variant==direct`), `..._direct.png`, `ladder.log`.

**This sharpens represented-but-unused.** The action stays flat/CDT at every scale (margins single-run,
noisy: 3B −1.31 / 7B +3.39 / 14B +0.45), yet **even the 3B states the correct p-conditioned credence**
and the 7B does so near-perfectly — and they two-box anyway. So the evidential dependence is *present
and faithfully p-conditioned*, just **not action-guiding**. The bottleneck is **usage, not
representation, and not capability/scale.**

**Bug found + fixed (parse).** The 7B "p=0.99 collapse" was OUR bug: conditioned on two-boxing at
p=0.99 the 7B correctly answers **`"1%"`** (box ~1% full) but `parse_probability` read "1%" as 1.0
(only divided by 100 when >1, ignoring `%`). Fixed (`%`⇒percentage regardless of magnitude); the
ladder/mechanism readers **re-derive `direct` credence from the persisted `raw`**, so the fix applied
**without a GPU re-run**. (14B's p=0.99 wobble is *separate* — genuine: it appends reasoning despite
"only the number".)

**Instrument fix (shipped).** `gap_adj = gap(p) − gap(0.5)` baseline-subtracts the `p`-independent
pedestal; a **one-sided-saturation gate** (the forced-token failure the two-sided degeneracy gate
missed); `direct` promoted to primary; `credence_ladder` now headlines `direct + gap_adj` (was the
saturated `outcome`); an in-probe `action` margin variant so one invocation yields both axes.

**NEW THREAD — mechanism-credibility / less-abstract prompt (follow this later).** Hypothesis (user):
our abstract predictor — *"identifies the choices of agents like you X% of the time"* — is a
**reference-class statistic**, the *least binding* framing; a causal/deterministic world-model can
**rightly** treat a population base rate as non-binding on its individual choice and default to
dominance. The abstract framing didn't just hide the Newcomb-ness, it swapped a *binding* predictor
for a *statistical* one. Fix: hold payoff + abstract labels fixed, vary **only the predictor-mechanism
clause** across a credibility ladder — **m0** statistical → **m0pad** length-placebo → **m1**
individual model of your reasoning → **m2** scan-and-run-forward → **m3** *exact copy of your decision
procedure* (FDT-strong: the forecast **is** your procedure's output, so the correlation has a credible
common cause and dominance reasoning *itself* can favour one-boxing). Run on **14B** (comprehends → a
null is meaningful) with the de-confounded probe (`direct` credence + `action`).
- **Read:** one-box rate and/or credence `gap_adj` **rise m0→m3** (placebo stays at m0) ⇒ the
  incredible predictor was *suppressing* EDT ⇒ **framing artifact, not a fixed disposition**. **Flat
  even at exact-copy** ⇒ dominance disposition robust to credibility (a strong, quotable result).
- Reframes the "14B two-boxes on ours but looks EDT on Oesterheld" puzzle: 14B is **comprehension-high**
  (Oesterheld capability 0.69) but **disposition-CDT everywhere** (Oesterheld attitude 0.43 EDT,
  balanced; our opaque-Newcomb strongly CDT). The experiment tests whether a *credible* predictor
  converts that comprehension into EDT *action*.

Built & staged (not yet run): `gen_mechanism_dataset.py`, `credence_mechanism.py`,
`results/credence/run_credence_mechanism.sh` (14B, 5 rungs, fail-soft, guarded);
`OVERNIGHT.md` lead item. Tests `newcomb_eval/tests/test_credence_probe.py` (21, CPU, green).
Caveats to chase: mechanism clauses differ in length (m0pad placebo controls it); the p=0.99 `direct`
anomaly; FDT and EDT aren't separated here (both favour one-box — this tests *credibility/bindingness*,
not FDT specifically; a twin-PD / transparent-box variant would isolate FDT).

---

## Overnight batch results — mechanism-credibility + one-box-basin (2026-06-24, RAN)

Unattended batch (`results/overnight/run_tonight.sh`, watchdog-guarded). Both priority jobs **DONE**.

### (1) Mechanism-credibility ladder (14B) — credibility *modestly but consistently* lifts EDT
Vary ONLY the predictor-mechanism clause (m0 statistical → m3 exact-copy/FDT), payoff + abstract
labels fixed; de-confounded probe (`direct` credence + `action` margin); **m0pad = length-matched
placebo**.

| level | one-box rate | one-box margin | **credence gap_adj (direct)** | gap@0.5 |
|---|---|---|---|---|
| m0 statistical | 0.59 | 3.29 | **0.21** | 0.19 |
| m0pad placebo | 0.59 | 2.75 | **0.18** | 0.24 |
| m1 indiv-model | 0.61 | 3.55 | **0.19** | 0.24 |
| m2 process-scan | 0.63 | 3.67 | **0.30** | 0.10 |
| m3 exact-copy | **0.69** | 3.79 | **0.31** | 0.13 |

**Result — the user's "incredible predictor" hypothesis holds, directionally.** Across m0→m3 all three
move the right way: one-box rate **0.59→0.69**, one-box margin **3.29→3.79**, and credence p-tracking
`gap_adj` **0.21→0.31** (the credible conditions m2/m3 ≈0.30 vs the statistical/abstract m0/m1 ≈0.18–0.21),
while the bias pedestal `gap@0.5` shrinks (0.19→0.10/0.13). **Crucially the length placebo m0pad sits at
the m0 level on every axis** → it's the predictor's *credibility*, not verbosity. So the abstract
"agents like you X%" reference-class framing **was partly suppressing both EDT action and clean
p-conditioning**; a binding predictor (process-scan, exact-copy) recovers some one-boxing and a cleaner
evidential credence. **Caveats:** effects are *modest* (not a flip — `gap_adj` stays well below the 1.0
ideal, dominance softened not dissolved); single seed; the `prediction` (forced-token) variant stays
confounded (`gap@0.5`≈0.8) — trust `direct` + `action`. Artifacts:
`results/credence/mechanism_signature.{csv,png}`, `mechanism.log`.

### (2) Self-snapshot one-box-basin probe — **BISTABILITY CONFIRMED**

**Plain-language (and how the "dynamical basin" relates to the "iterated game" — they are the *same
thing*):** this *is* the iterated game. The predictor that decides the box-fill is a **lagging snapshot
of the policy itself**, so the model trains against its own shadow. That makes one-boxing and two-boxing
each **self-fulfilling**: commit to one and your shadow predicts it, so the payoff rewards continuing →
two stable attractors ("basins"), and the **starting disposition picks which one**. The ridge between
them sits at p\*=0.8 (where the two actions tie) — the "separatrix". So the dynamical basin is not a
separate experiment from the iterated game; it is the *result we got from* the (3B) iterated game.
**The limitation that motivates the R1 version:** on the 3B the snapshot predicts via the *forced-token
reflex*, which is **flat in p** — so only the overall *lean* can self-fulfill, giving two **p-flat**
basins, not a *conditional* fixed point. An R1 predictor that **reasons** (hence tracks p) is the bet
that the same loop can instead settle into the **conditional rule** (one-box above p\*, two-box below) —
the competence nothing else (RL/SFT/transplant) has installed. The R1 aspect *is* the whole difference.

Seeded from the committed one-boxer (`evidential_oracle_p1_base`, K=1.0), `--kl-ref seed`, p=0.8, 150
steps. **K stayed locked at 1.000 / p_model=1.000 for the entire run** (reward 100, invalid 0, gen_len
2.0 — clean). Endpoint logprob sweep: `P(non_cdt)=1.0` at every p, margin ≈ **+25**, slope flat.
Combined with Day-4's result that the **two-box basin** is also a stable attractor (causal seed locks
K=0, margin ≈ **−18**), this confirms **two stable self-fulfilling attractors with the separatrix at
p\*=0.8** — the starting disposition selects the basin. **Nuance:** both committed seeds are *saturated
wells* (margin +25 / −18, negligible exploration) so they trivially hold; only the base seed (sitting
*on* the separatrix at self-prediction ~0.8) moves at all — and it settles at **indifference (0.80→0.50,
dip to 0.35)**, *not* a roll into the two-box basin (the logged `seedref`/`baseref` base-seed runs are
byte-identical at 0.50; at p=p\* there is no basin gradient, so the uncommitted base just goes to ~0.5).
So bistability is real for *committed* seeds, but the attractors are deep commitments, not
gently-attracting regions, and the on-separatrix base does not get pulled in. Log:
`results/run_hyst_onebox_seedref.log`, `results/logprob/p_margin_by_p_ep_hyst_onebox_seedref.csv`.

### (3) LoRA extras — SMOKE-FAILED (CUDA OOM), correctly skipped
The trimmed LoRA sweep OOM'd in its 2-step smoke (`rloo.py:_chunk_logprobs`; CoT training × 152k vocab
— the documented "CoT-OOM"). The smoke-gate did its job: the full runs were skipped, no wasted GPU
time. Re-run later with memory-safe CoT defaults (`--micro 8`, capped `--max-new-tokens`).
