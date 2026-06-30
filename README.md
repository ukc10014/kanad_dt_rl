# READ FIRST — orientation to `results.md`

*Analysis snapshot by Claude, 2026-06-30. This is a **reading guide**, not a spec. It exists so you
(or a future session) can re-orient quickly and not get whiplash from the inconsistencies in
`results.md` before we rewrite it. Source-of-truth hierarchy is unchanged: **`PLAN.md`** = spec,
**`CLAUDE.md`** = operating contract + sanity gates, **`results.md`** = the chronological run log, **this
file** = a temporary reading guide over that log.*

---

## Why this file exists

`results.md` was accreted chronologically (Jun 22 → Jun 29). The **early entries (Jun 22–25)** and the
**later ones (Jun 27–29)** reach partly different conclusions, and the **top-of-file synthesis blocks**
(Week-in-review, Day-3, the LessWrong outline) were written mid-stream and **never updated** after the
later runs landed. So reading top-to-bottom is confusing. The run-by-run detail lower down is the honest
record and should stay; it's the *headline synthesis at the top* that's stale.

## The one inconsistency that matters

- **Early headline (still sitting at the top of `results.md`):** *"scale worsens CDT — the 14B is a
  more committed two-boxer than the 3B; sharpest single data point."* This came from the **transplant +
  forced-choice** regime on the **abstract** framing (Runs 9–11).
- **Later work (Jun 27–28) substantially walked it back:** under **free CoT**, the 14B tracks `p*`
  cleanly — the empirical crossover **follows the payoffs** (payoff ablation) and is **flat across the
  framing ladder m0→m3** (framing sweep). The "hard two-boxer" was a **regime artifact**, not a property
  of scale or framing.
- The reconciliation **is already in `results.md`** (the 6/28 *"Reconciling…"* section) but was never
  propagated up to the headline blocks. **→ Re-center the takeaway on: "test-time reasoning is the
  lever; scale and framing are second-order."**

## Current defensible takeaways (what survives the de-noising)

1. **Competence ≠ disposition, and they're separately movable.** The model *can* compute the EV (14B
   crossover follows `p*` under CoT; 7B states the credence `2p−1` near-perfectly; 3B/14B comprehend the
   structure ~99–100%). Whether it *acts* on that is a separate axis.
2. **The lever is test-time reasoning — not scale, not framing.** Cleanest result in the project: the
   within-gate 14B A/B (same items, comprehension pinned 100% both arms) — forced-choice → flat
   ~coin-flip; +CoT → crossover lands on `p*`. *This should be the headline.*
3. **RL moves the lean (intercept), not the rule (slope).** Robust from the 3B (RL/SFT/transplant/fair
   objective all fail to install the slope) up to R1-8B.
4. **Self-referential reward ⇒ structural bistability** (double-well, repeller at `p*`); removing the
   feedback loop (exogenous-EV reward) removes the well. A property of the *environment*, robust to model
   capability (it bites even the reasoning model R1).
5. **Methodology is itself a result:** *gate on confounds; quarantine, don't footnote.* Most of the
   week's apparent effects (+0.50 CoT slope, +0.16 SFT slope, the causal-flip, "scale worsens") dissolved
   under controls.

## The two recent threads (Jun 26–29) — what they actually established

- **14B prompt-robustness:** two-boxing is **not** robust across prompts. Under free CoT, all framings
  m0→m3 track `p*`; what *is* robust is the **flat, p-blind reflex at the bare forced-choice step**
  (~0.55), regardless of framing. (Note: the 3B's forced-choice reflex actually leans *one-box* ~0.83 —
  "two-boxing in the 3B" is regime-specific, not its default.)
- **R1 under RL:** R1 **tracks `p` at test time** (with reasoning) but RL does **not** install/stabilize
  the conditional rule. Self-referential loops **collapse into bistable basins** (clean negative across
  lag-0 and lag-3; the lag is a non-factor). The exogenous-EV paired run (6/29) **avoids the collapse**
  (slope stays positive — mechanistic win) but does **not** demonstrably **steepen** the slope
  (noise-limited null, n=8, single seed). I.e. R1-under-RL is the natural *extension* of "RL moves the
  lean, not the rule," now confirmed up to a reasoning model.

## One genuinely interesting finding to revisit (not yet clean)

On the 14B: **handing it the spelled-out EV makes it *more* CDT at high p** (a monotone dose-response in
the transplant — the more of the breakdown you spell out, the more it two-boxes), while **letting it
derive the EV via CoT makes it EDT-consistent.** Verified against the raw artifacts: the CoT gate is
**0% invalid**, the traces show **real EV arithmetic**, and the transplant dose-response is real.
**Caveat:** the headline contrast confounds *reasoning-mode* with *EV-source* — the clean disambiguating
cell (**CoT + handed-EV**) was never run. The experiment that would nail it: a 2×2 of
**reasoning {on, off} × EV {self-derived, handed}** on the same items, ≥3 seeds.

## Self-play RL — the predictor-design question (asked & answered)

A recurring question about the iterated-game setup: *the predictor is just some version of the policy
(its own current samples in A2, a lagged snapshot in the cot/lag variants) — is there any value in
training the predictor as a **separate, independent model** that uses information about the chooser, or
is that no different from sampling the chooser's own rollouts?*

**Short answer: not informatively different — at the fixed point.** The predictor's accuracy is
maximized by outputting `P(one-box | prompt)`, i.e. the chooser's own action marginal. A separately
trained predictor is just a *function approximator* for that quantity; sampling the chooser's K rollouts
is a *direct Monte-Carlo estimate* of the same quantity. **Same target.** So a well-trained separate
predictor *converges to* "sample the chooser," and the core finding (RL doesn't install the conditional
rule; self-referential reward → bistable double-well) is unchanged. The intuition "it should feel
different but probably isn't" is correct **for the existing result**.

**Where "separate" genuinely buys something — all of it dynamics/control, not the fixed point:**
1. **A tunable, continuous feedback delay (the strongest reason).** A frozen snapshot is a crude
   *discrete dead-time* delay of exactly *k* steps; a separately-trained predictor with its own learning
   rate is a *first-order lag* with time-constant τ ≈ 1/(predictor LR). Oscillation/hysteresis is
   governed by the **gain × delay** product, and a learned predictor lets you dial both — so it's the
   *right* instrument for the still-open "can lag induce hysteresis?" question, not a redundant one.
2. **Independent control of the predictor's accuracy ceiling / calibration / information set.** The
   sampling estimator is *always perfectly self-calibrated* to the current chooser. A separate model can
   be capped (e.g. 70%-max), miscalibrated, or denied information (predict from the prompt only, without
   seeing the chooser's reasoning chain). That *moves the attractors* and tests whether the bistability
   survives an imperfect predictor — a genuinely new question.
3. **Variance reduction.** A trained predictor pools across prompts → smoother `p_eff` than a noisy K=4
   tally (which can only be 0/.25/.5/.75/1). Better measurement, same finding.

**The decomposition worth holding onto:** weight-sharing-vs-separateness is **invisible to the chooser** —
the chooser only reads the *framing clause* (m0→m3). So "make the predictor a separate model" changes the
**reward dynamics**, while "make the predictor an exact copy of you" (FDT licensing) is a **text/framing**
move the chooser reasons about. **Orthogonal knobs.** (A separate *correlating* model is, if anything,
*less* binding than an exact copy — but only if you tell the chooser so.)

**Practical recommendation:** don't build a second trained model *now* — everything here is already
noise-floor-limited at n=4–8, and a second moving part adds reward-hacking/collapse failure modes before
the n/seed problem is fixed. The 80/20 that captures buy #1 at near-zero cost is an **EMA / low-pass
filter on `p_eff`** — mathematically a one-parameter first-order-lag "predictor," giving you the tunable
delay for the oscillation hunt without a second network. (An `_ema` artifact already exists in the tree —
`results/logprob/p_margin_by_p_ep_hyst_causal_seedref_ema.csv` — so that knob was at least prototyped;
worth checking what it showed.) Build the full separate predictor only if the *specific* target is
oscillation or predictor-imperfection — **not** to re-confirm the result.

## Self-play dynamics — collapse vs oscillation vs hysteresis (a dynamical-systems read)

*Refines the `results.md` open-question block (results.md:226, "why monotonic slide-and-stick, never
oscillation"), which guessed the lag was "too small / confounded / wells too deep." The sharper answer
is structural — and it **corrects** an over-optimistic earlier framing that "lag is the missing
ingredient for oscillation." It isn't, in this loop.*

**The loop as a 1-D system.** Let π = the policy's one-box propensity. The chooser's best response to a
predictor of accuracy `p_eff` is a **step**: one-box iff `p_eff > p*`. And `p_eff` = the (possibly
lagged) policy one-box rate π. So π-above-p\* → rewards one-boxing → π **rises**; π-below-p\* → π
**falls**. That's **positive (self-fulfilling) feedback**, giving **attractors at the saturating corners
π=0 / π=1** and a **repeller at p\***. Gradient descent on that double-well = roll into the nearest well
⇒ the **default is monotone collapse into a seed-selected basin** (what every self-referential run showed).

**Why lag alone does NOT buy oscillation.** A sustained oscillation (limit cycle) needs
**negative/restoring** feedback toward a **stable interior** point that a delay then destabilizes
(overshoot → correct → overshoot) — the thermostat-with-lag / matching-pennies picture, where onset is
governed by **gain × delay** crossing a threshold (a Hopf bifurcation; gain ≈ LR × reward-response slope,
delay ≈ lag). Our loop has the **wrong feedback sign** (positive/coordination) and its equilibria sit at
**saturating walls**, with no interior point to overshoot. So adding lag gives **delayed** collapse plus
at most a **damped transient wobble** — never a sustained cycle. (This is why lag=3 still collapsed
monotonically; it refines the "wells too deep" guess into "wrong feedback sign / corner attractors" — a
**structural**, not quantitative, obstruction.)

**When oscillation WOULD appear.** Only if you engineer a **stable interior equilibrium** for a delay to
destabilize, then push **gain × delay** (i.e. **LR × lag**) past threshold. Routes, cheapest first:
(a) **cap/squash the predictor's accuracy** (`p_eff = clip(π_lag, 0.5, 0.9)` or a logistic) so the
attractor moves off the corner into the interior; (b) a small **anti-coordination** term (reward the
chooser for *deviating* from the prediction) → genuine negative feedback / matching-pennies cycles (this
changes the game); (c) a **strong restoring regularizer** (KL toward a p\*-mixed reference) making the
interior point stable. So the "weird convergence of LR and lag" instinct is the right *mechanism* — it
just needs an interior equilibrium to act on, which the vanilla loop lacks.

**Why this ties to the predictor-design question above.** Sampling-the-chooser is **zero-delay and
perfectly self-calibrated** → attractors pinned at the saturating corners → no oscillation, fast collapse.
A **truly independent** predictor buys the two things a cycle needs — a **tunable delay** (≈1/LR_pred; the
EMA-on-`p_eff` knob is the cheap proxy) *and* the ability to be **imperfect/capped** (interior attractor).
But a *perfectly accurate* independent predictor just recreates the corner attractors and still collapses
— so independence is **necessary-ish, not sufficient**; the real enabler is the **interior equilibrium**,
not the separate weights.

**The distinction to carry (and to fix in any rewrite): hysteresis ≠ oscillation.** *Hysteresis* (the
basin/state depends on history / initial condition) is **generic** to bistable systems and is **already
half-shown** — seed selects the basin; the "barrier-kick" perturbation experiment would confirm the full
loop. *Oscillation* is a **separate, harder** thing this reward geometry **structurally cannot** produce
without the interior-equilibrium engineering above. The `results.md` open-question block treats them
together — they should be split: **"can lag induce hysteresis?" ≈ yes; "can lag induce sustained
oscillation?" ≈ no**, not in the self-fulfilling loop.

## Next experiment queued — does CoT generalise beyond Newcomb? (the DT fingerprint)

Both the 14B and R1 reach EV-rational / EDT-consistent behaviour *on Newcomb* under CoT. Open
question: is that **general decision-theory competence**, or are they just good at **Newcomb
specifically** (persona / pattern-match)? **Built and CPU-tested 2026-06-30, ready to run on the GPU:**
`newcomb_eval/signature.py` runs a model over the four DT-zoo problems already in the repo
(opaque/transparent Newcomb, counterfactual mugging, XOR blackmail — `dataset_raw.json`, 5 items each,
already DT-labelled) and reads off a **CDT / EDT / FDT / incoherent** signature. A coherent signature
under CoT ⇒ general competence; one-boxes Newcomb but an **incoherent** tuple ⇒ Newcomb-specific. Bonus:
**XOR blackmail separates EDT from FDT**, so it also settles the "EDT(+FDT)-consistent, can't separate"
ambiguity flagged at `results.md:1665`. The classification core is pure + unit-tested
(`newcomb_eval/tests/test_signature.py`, 4 green); only the driver needs a GPU. **Full run plan + guards
(comprehension gate, forced-choice/CoT split, memorisation caveat, the missing smoking-lesion/hitchhiker
items) are the top item in `OVERNIGHT.md`.** This is the deferred PLAN §8 "fingerprint" eval, now
restricted to the high-capability models where its comprehension confound is weakest.

## Confidence / caveats to carry in your head

- Many headline numbers are **single-seed, n=4–12 per cell** = at/below the noise floor by the project's
  own rule. **Trust patterns that replicate across conditions, not single cells / single-run slope signs.**
- The **early synthesis reads more confident than the late runs support.** When in doubt, the later,
  more-hedged entries win.

## Provenance gotchas (important if you want to re-analyze)

- `results/` is tracked for transcripts/summaries/plots, **but `*.log` and `results/adapters/` are
  gitignored.** The self-play **raw per-step logs** and **all trained LoRA adapters** are **not in git**
  and **not on a fresh clone** — they only ever lived on the original GPU box. **Consequence:** you
  **cannot re-run a fine-grained `p*` on the final post-RL policies** without recovering those weights or
  re-training. (The one worth recovering is `results/adapters/evidential_r1_paired` — the exogenous run
  that didn't collapse.)
- The over-time **dynamics are preserved** as `results.md` checkpoint tables + the figure
  `results/selfplay_dynamics_combined.png` (added 2026-06-30; built by
  `scratchpad/plot_selfplay_combined.py`).

## When we rewrite `results.md`

- **Demote "scale worsens"** to a regime caveat; **lead with "reasoning is the lever."**
- State **R1-under-RL** as the extension of "RL moves the lean, not the rule" (+ the bistability mechanism).
- **Keep the run-by-run detail** (honest record); only fix the **top synthesis blocks** so they match
  the 6/27–29 results.
- Optionally run the clean **reasoning × EV-source 2×2** before building a writeup around the
  "handed-EV backfires" finding.
