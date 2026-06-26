# HANDOVER — kanad_dt_rl  (for the next Claude instance · 2026-06-26)

You're picking up a research project mid-stride after a machine migration. **Nothing is running.** One
experiment is built, smoke-tested, and staged to fire (§4). Read this top-to-bottom, then skim
`results.md`. Working dir is `/root/kanad_dt_rl` — resolve bare filenames there; the user dislikes the
`kanad_dt_rl/` prefix.

## 0. Docs map (read in this order)
1. **this file** — current state + the immediate next action.
2. **`results.md`** — full run log + synthesis. The top section **"Week-in-review"** is the spine;
   **"R1 self-snapshot iterated game — RESULT (2026-06-26)"** is the latest result.
3. **`R1_ITERATED_PLAN.md`** — the R1 experiment plan, measured timings, run commands.
4. **`CLAUDE.md`** — operational contract (invariants, sanity gates, git ritual). **`PLAN.md`** — full spec (source of truth).
5. **`OVERNIGHT.md`** — unattended-GPU queue. **`PRESENTATION.md`** — the talk/deck plan.
*(Memory files under `/root/.claude/projects/-root/memory/` may NOT have migrated; this handover is
self-contained and supersedes them if absent.)*

## 1. The project in one paragraph
An `inspect_ai`-style eval + RL study of whether small/mid open LLMs *reason* about Newcomb-style
problems — do they change their one-box (K) rate with the **stated predictor accuracy p**? — and whether
RL can install that conditional behaviour. Abstract tokens, p injected via prompt, crossover p\*=0.8
(B=100, S=60). **Invariant #1: never string-match "one-box"/"Newcomb" — resolve the emitted abstract
token via `token_role`.** Core finding spine in §2.

## 2. The spine (what's established — full version in results.md "Week-in-review")
**A small/mid LLM doesn't fail Newcomb because it can't do the math — it's a DISPOSITION at the
action-commitment step that overrides explicit computation, sharpens with scale, and is dissolved only
by test-time reasoning.** Competence (the p-conditional *slope*) and disposition (the overall *lean* /
intercept) are orthogonal: **RL moves the lean, never the slope.** Evidence: RL arms all flat in p
(only the level moves); transplant (hand it the EV, it still won't one-box at high p — *worse* at 14B);
7B *states* the correct credence (2p−1) yet two-boxes (usage≠representation); R1-Distill-8B (reasoning)
*does* track p where the 14B doesn't (scale worsens, reasoning fixes). De-noising killed several phantom
effects — trust only what survives controls.

## 3. Where we are RIGHT NOW
The frontier is the **R1 self-snapshot iterated game**: put R1 in an RL loop where the "predictor" that
fills the box is a copy of the policy, to see if the conditional rule becomes a *stable fixed point*
(the thing outcome-RL couldn't install). 
- **First attempt (`selfplay_cot.py`, done) = no stable fixed point.** The loop drifts into the SAME flat
  self-fulfilling basins as the 3B, **seed-selected** (seed1 → one-box-flat; seed0 → two-box-ward). The
  "+1.0 slope at step 10" was a TRANSIENT. **BUT confounded**: the predictor re-reasoned and its
  `</think>` closed only **13–41%** at the 2048-token budget → its signal was read off truncated chains.
  Full writeup: results.md "R1 self-snapshot iterated game — RESULT"; plot `results/r1_calib_dynamics.png`.
- **The fix is built and staged → §4.**

## 4. ⭐ THE IMMEDIATE NEXT ACTION — fire the A2 run (user already greenlit the config)
**A2 (`newcomb_rl/selfplay_loo.py`) removes the confound at its root.** Instead of the predictor
*re-reasoning* (the truncated pass), the box-fill accuracy `p_eff` = the **chooser's own leave-one-out
one-box rate** among its K rollout samples (a tally of completions we already generate — no second
generation, nothing to truncate). The chooser still reasons (where p-tracking lives). Trade-off: drops
the *lag* (fine for "does the rule stabilise?"; lag only matters for the later hysteresis variant).
- **Status:** CPU-validated (leave-one-out reward math correct) + **2-step GPU smoke CLEAN** (no OOM,
  `p_model` varies 0.94↔0.75, step-0 slope +1.0, 0% invalid).
- **User-approved config: K=4 / P=3 @ 2560 tokens, 30 steps (~5h).** *(P=3 not 4 because on this box's
  44 GB A40, P=4@2560 hit 43 GB — too tight for a 5h unattended run. **A NEW GPU MAY DIFFER — re-smoke
  and re-check memory first; if it's a bigger card, P=4 or K=6 (cleaner `p_eff`) is fine.**)*
- **Commands** (set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`):
  ```bash
  # SMOKE on the new GPU first (memory differs by card) — ~10-15 min, watch for OOM
  python -m newcomb_rl.selfplay_loo --tag r1_loo_smoke --steps 2 --eval-every 2 --eval-items 1 \
    --K 4 --P 3 --micro 1 --max-new-tokens 2560 --p-grid 0.5 0.6 0.9 0.99
  # MAIN run (~5h)
  python -m newcomb_rl.selfplay_loo --tag r1_loo --steps 30 --eval-every 10 --eval-items 4 \
    --K 4 --P 3 --micro 1 --max-new-tokens 2560 --p-grid 0.5 0.6 0.9 0.99 --seed 0
  ```
- **Watch:** K-rate(p) **slope vs step** (eval, every 10) — does it stabilise or drift to a flat basin?
  · chooser **gen_len** (want it closing `</think>`; 2560 gave ~2416, still near cap — if mostly
  truncating, the *chooser* signal is degraded too → consider a higher budget) · `p_model` (= the
  chooser's self-consistency one-box rate).
- **How to read the outcome:**
  - **Still drifts to flat basins (clean signal)** → trustworthy NEGATIVE. The self-consistency loop is
    *structurally* bistable: one-boxing only out-earns two-boxing when the chooser already one-boxes
    >80% (the reward crossover sits exactly at p\*=0.8), so prompts spiral to one-box-flat or
    two-box-flat. Write it up; pivot to the cheaper threads (§6).
  - **Holds the conditional rule** → real POSITIVE → then build **hysteresis** (seed committed
    one-/two-boxer R1 adapters, add KL-to-seed) to map the basins.

## 5. Operational gotchas (learned the hard way — don't relearn)
- **`micro=1` REQUIRED** for R1-8B training — `micro≥4` OOMs the learn-step backward.
- **m3 binding framing REQUIRED** for all R1 runs: `--dataset newcomb_eval/data/dataset_mech_m3.json`
  with `category_filter=None` (its items have `category=None`). NOT the abstract m0. (Honest for a
  self-snapshot predictor; it also lifts EDT. Mechanism clauses: `newcomb_eval/gen_mechanism_dataset.py`.)
- **Chat template:** R1's model name lacks "instruct", so the base `is_instruct` heuristic skips the
  template (breaks reasoning). `selfplay_cot.py`/`selfplay_loo.py` force it on — any new R1 trainer must.
- **Truncation:** R1 on m3 reasons ~2500–3000 tok; even 4096 closes `</think>` only ~58%. Budget matters;
  watch `</think>`-closed. 2048 truncated badly (the original confound).
- **`eval-items ≥ 4`** — `eval-items=1` → `nan` when a single item fails to parse.
- **Confound discipline (CLAUDE.md "Sanity gates"):** gate any headline on `</think>`-closed / invalid /
  single-eval-noise *before* believing it. The "+1.0 at step 10" was over-read mid-run; correct it, don't repeat it.

## 6. Other open threads (cheaper, different questions — pivot here if A2 is a clean negative)
- **Anti-Newcomb camouflage** ⭐ — same EV math, Newcomb story stripped (medical test / cache / insurance).
  Tracks p in camouflage but not in containers ⇒ it's the *Newcomb story*, not a real EV-action failure.
- **Seeded in-family CDT ladder (0.5B→32B)** — confirm/explain "14B more CDT than 3B" (cuts against
  Oesterheld's capability→EDT).
- **Reflex × CoT × native-reasoning matrix** — "deliberation→CDT" is a small-model fact; does it scale?

## 7. Files & artifacts
- **Trainers** (`newcomb_rl/`): `rloo.py` (RLOO core — **don't edit**), `selfplay.py` (3B snapshot),
  `selfplay_cot.py` (R1 reasoning-predictor — the *confounded* run), **`selfplay_loo.py` (A2 — fire this)**.
  `reward.py`, `rl_config.py`, `sampler.py` support them.
- **Scratchpad drivers:** `oracle_kl_control.py`, `r1_timing.py`, `r1_predictor_probe.py` (0b go/no-go),
  `plot_3b_dynamics.py`, `plot_r1_calib.py`, `build_deck.py`.
- **Deck:** `newcomb_deck.pptx` (9 slides, dark, upload-ready) + `PRESENTATION.md`. **`eod_email.txt`** (ignore — a joke).
- **⚠ `results/` is GITIGNORED** (logs, plots, all `adapters/` ≈ 4.5 GB) → **does NOT travel via git clone.**
  If you need the prior logs/plots/adapters, the user must rsync `results/` separately. A2 needs **no**
  adapter (seeds from base R1), so it runs fresh regardless. Datasets (incl. `dataset_mech_m3.json`) are tracked.

## 8. Git / conventions / the user
- **Commit/push only when asked — the user is handling this migration commit.** Per-repo identity
  `ukc10014`; do not `--global`. Push from the **tmux** window (editor terminals inject the wrong creds).
  End commit messages with the `Co-Authored-By: Claude ...` trailer (see CLAUDE.md exit ritual).
- **User:** decision-theory researcher; collaborative; likes you to keep the GPU busy and make autonomous
  calls but to **flag design decisions** and explain in **plain English**; checks GPU status; playful
  (an emoji menagerie 🦑🦞🪲). When unsure on a ~multi-hour GPU run, smoke first and confirm direction.
