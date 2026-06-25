# PRESENTATION.md — how to tell the story (graphics + slide flow)

*Planning doc for the week's results talk. Narrative + the visuals that carry each beat. "Build"
status flags what exists vs. what renders over data we already hold (don't pre-build; decide later).*
Headline synthesis lives in `results.md` → "Week-in-review". Numbers/runs are in `results.md`.

## The one-line thesis (the spine every visual serves)
**A small/mid LLM doesn't fail Newcomb because it can't do the math — it fails because of a disposition
at the action-commitment step that overrides explicit computation, sharpens with scale, and is
dissolved only by test-time reasoning.** Competence (slope) and disposition (intercept) are orthogonal
and separately movable.

## The organizing diagram (pick ONE; A = process, B = concept)
- **(A) Hypothesis-elimination flowchart — "everything we did."** Vertical tree; each node = a suspected
  cause, edge = the experiment (with run #), leaf = verdict:
  `flat K-rate(p)` → **format?** (CoT/scaffold, R3/4/7) → *no* → **can't compute?** (transplant+SFT,
  R8/9/11) → *no* → **scale?** (14B, R10) → *no — worse* → **framing?** (mechanism ladder) → *partly,
  modest* → **reasoning?** (R1) → **YES** ⟹ *reasoning-dissolvable commitment-step disposition* →
  **what installs it?** (R1 iterated game, running). Best "whole program" slide — systematic elimination.
- **(B) Intercept-vs-slope concept space — the claim in one frame.** Axes: intercept (disposition) ×
  slope (competence). RL arms scatter *horizontally* at slope≈0 (causal→0, evidential→0.5, base→0.8,
  modelpred→0.7); R1 sits *up* at slope>0. Arrows: "RL moves you sideways; only reasoning moves you up."
  Best *conceptual* slide.

## Money-plots (ranked by punch; all render over data we hold)
1. **K-rate(p) overlay — THE headline.** All RL arms + base as *flat lines*; R1 the *one rising line*;
   vertical marker at p\*=0.8. "Everything flat except the reasoning model." Data: R1 (Run 5/6 logprob) +
   R1 battery/0b. *Build: render.*
2. **Transplant heatmap, 3B vs 14B side-by-side.** rows = aid (none→full), cols = p, cell = P(optimal);
   aid raises low-p but flips high-p negative; 14B → **0.00** at p=0.99 even with full EV. "Handed the
   answer, won't act — scale sharpens." Data: Run 9 (3B) + Run 10 (14B). *Build: render.*
3. **Represented-but-unused.** Ideal **2p−1 diagonal** + the 7B credence gap sitting on it + the **flat
   action margin** along the bottom. "Knows it, doesn't use it." Data: credence ladder. *Build: render.*
4. **Scale-vs-reasoning bar.** one-box rate @ high p for {3B, 14B, R1-reasoning, R1-no-think}: 14B lowest,
   R1-reasoning tracks, R1-no-think flat-high. The sharpest single data point. Data: scattered, assemble.
5. **Training dynamics** — `results/dynamics_3b.png` (bistability + oracle anchors). *Build: HAVE IT.*
6. **kl-control** — `results/dynamics_klcontrol.png` (the flip was the KL leash). *Build: HAVE IT.*
7. **Mechanism-credibility ladder** — one-box rate + credence `gap_adj` over m0→m3, m0pad placebo flat at
   m0. "Framing explains a slice (modest), placebo-controlled." Data: mechanism_signature.csv. *Build: render.*
8. **(forthcoming) Iterated-game trajectory** — K-rate(p) slope over training: does the conditional rule
   persist / sharpen / collapse? The forward payoff plot. Data: the running calibration.

## Tables
- **4-arm intercept-vs-slope:** arm × {P(non_cdt), logit margin, slope}. All flat-slope, four levels. The
  "RL moves level not shape" receipt. (Run 5/6.)
- **Capability/disposition matrix:** rows {3B, 14B, R1} × cols {reflex one-box, reasoning tracks p?,
  transplant @ high-p, represents credence?}. Model comparison at a glance.
- **(backup) Phantoms killed by de-noising:** effect | naive value | after controls (+0.50 CoT slope →
  flat; +0.16 SFT slope → 0; causal-flip → KL artifact; 0.80→0.25 basin drift → 0.50). Sells the rigor.

## Conceptual schematics (small, hand-drawn-style)
- **Setup schematic (slide 1):** two boxes + predictor + EV crossover at p\*, with the K-rate(p) reading
  guide ("flat = recitation; step at p\* = reasoning").
- **Mechanism money-shot (the thesis in one picture):**
  `prompt → [represents EV / states 2p−1 correctly] → [commitment step: dominance pull OVERRIDES] → two-box`.
  Override happens *after* correct representation — that's the whole claim.

## Slide flow (how they chain)
setup schematic → K-rate(p) overlay (flat) → intercept-vs-slope scatter ("level not shape") →
**is it capability?** transplant heatmap + represented-but-unused → **is it scale?** scale bar (worse) →
**is it reasoning?** R1 reveal (the turn) → **is it framing?** mechanism ladder (modest) →
dynamics (bistability + kl-control) → **what installs it?** iterated-game schematic + trajectory →
mechanism money-shot + one-sentence abstract → (backup) phantoms-killed table.

## Honest caveats to keep on the slides
Small models; several single-run n≈12–20 cells (14B especially); 14B>3B-CDT direction suggestive,
mechanism open; abstract-prompt confound real but modest (≤10 pts); "reflex-EDT/CoT-CDT" is a
small-model fact that doesn't cleanly scale (reflex drifts off EDT with size; R1 *reasoning* ≠ 3B *CoT*).

## Asset status
HAVE: `dynamics_3b.png`, `dynamics_klcontrol.png`. RENDER-OVER-EXISTING-DATA: plots 1,2,3,4,7 + all
tables. PENDING DATA: plot 8 (iterated-game trajectory, from the running calibration).
