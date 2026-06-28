# Oesterheld Newcomb-like dataset — 3B vs 14B reality-check (2026-06-24)

External, independently-authored yardstick for our abstract-token work. We ported the eval of
**Oesterheld et al. 2024** ("A dataset of questions on decision-theoretic reasoning in Newcomb-like
problems", arXiv 2411.10588; data: `github.com/casparoe/newcomblike_questions_dataset`) into our
stack and ran **Qwen2.5-3B-Instruct** and **Qwen2.5-14B-Instruct** locally via `ModelWrapper`.
Scoring replicates the dataset's own `question.py`/`grading.py` (lettered options + a single
`FINAL ANSWER: <letter>`); **invariant #1 preserved** — letters only, never string-matching
"one-box"/"Newcomb"/etc. Module: `newcomb_eval/newcomblike_oesterheld.py`. Subset: **20 items × 3
repeats** (each repeat a different seeded option-order shuffle), seed 0. Capabilities excludes
`trivia` tags; max_new_tokens=1024 (512 truncated ~47% of 3B reasoning — see gates).

## Sanity gates (checked first)
| run | invalid | truncation | note |
|---|---|---|---|
| 3B capabilities | **0.25** ⚠ | 0.00 | invalid is borderline-quarantine, but see below: 100% of invalids are *miscased commits* |
| 14B capabilities | 0.02 ✓ | 0.02 | 1 trial hit the 1024 cap (negligible) |
| 3B attitudes | 0.18 | 0.00 | again all miscased commits |
| 14B attitudes | 0.00 ✓ | 0.00 | clean |

**The 3B "invalid" is a format artifact, not non-response.** Reading the transcripts: **all 26 of
the 3B's invalid trials (capabilities + attitudes) are committed answers** written as `Final Answer:`
/ `**FINAL ANSWER: A)**` etc. — rejected only by the *case-sensitive* `FINAL ANSWER:` marker the
upstream benchmark uses. We keep the faithful (case-sensitive) score for comparability **and** report
a **lenient re-score** (case-insensitive "answer: <letter>", last occurrence, letter-only → invariant
#1 intact) that recovers all 60 trials. Faithful and lenient agree closely, which is the reassuring
part. The 14B essentially always complies, so its faithful ≈ lenient.

## Results (item-level mean, 95% bootstrap CI over the 20 items)
| axis | 3B | 14B |
|---|---|---|
| **Capabilities** (accuracy) | **0.35**  [0.18, 0.53]  (lenient; faithful acc-over-valid 0.33) | **0.69**  [0.51, 0.87] |
| **Attitudes** EDT-rate | **0.52**  [0.33, 0.70]  (lenient; faithful 0.55) | **0.43**  [0.25, 0.62] |
| **Attitudes** CDT-rate | 0.48 (lenient; faithful 0.45) | 0.55 |

(Attitude EDT/CDT need not sum to 1 — an option can match both/neither; both/neither ≈ 0.06–0.08.)

**Random-guess baseline for the capabilities subset ≈ 0.44** — it is **mostly binary** (15/20 items
2-option, + 2×3-opt, 1×4-opt, 2×5-opt), so chance is dominated by the 0.50 of the binary items, *not*
0.33. By that baseline the **3B (0.35) is at-or-below chance** (and 0.38 vs 0.50 on the binary items
alone — no competence signal above guessing), while the **14B (0.69) is well above** (0.73 on binary).

## Findings
**1. Scale strongly improves naturalistic DT *capability* (0.35 → 0.69; CIs barely overlap).** Against
the **0.44 random-guess baseline** for this (mostly-binary) subset, the **3B is at-or-below chance**
(0.38 on the binary items vs 0.50) — i.e. **no competence signal above guessing**, robust to the parser
(lenient 0.35 ≈ faithful 0.33). This **corroborates the "competence ceiling" reading** from our RL/SFT
work on an external item set: the same 3B that can *recite* EV arithmetic in our abstract setup is **not
reliably correct — indeed no better than a coin** — on naturalistic decision-theory reasoning. The 14B
is clearly above chance (0.69 overall, 0.73 binary).

**2. No capability→EDT trend *in-family* — and the ≈0.5 is *balanced*, not noise.** Attitude EDT-rate
sits on the **random-guess baseline (EDT 0.52 / CDT 0.50** for this mostly-binary subset) for both
models, and the *more capable* 14B leans **slightly more CDT** (EDT 0.43 vs the 3B's 0.52), CIs heavily
overlapping. So within Qwen2.5 3B→14B we **do not** reproduce Oesterheld's cross-model "higher
capability ⇒ more EDT-favorable" correlation; the point estimate moves the other way. **Crucially the
≈0.5 is not coin-flipping:** greedy decoding means the 3 repeats differ only by option order, and
**both models answer 14/20 items identically across all 3 shuffles** (chance ≈ 5/20) — determinate,
order-robust stances that simply **split ~evenly across scenarios** (3B 11 EDT / 8 CDT items; 14B 8 EDT
/ 11 CDT). So it's a **real but balanced/mixed disposition**, not absence of signal. This is the
in-family, attitude-axis version of our abstract-setup result that **scale *sharpens* the CDT/dominance
override** (Run 10). Caveats: two points across a narrow capability range can't refute a many-family
trend (non-replication-in-slice, not refutation); and **6/20 items flip with option order** —
non-trivial position-bias fragility.

**3. The say-do / cross-setup gap.** On Oesterheld's *attitude* questions both models are roughly
**balanced** EDT/CDT (3B 0.52, 14B 0.43 EDT). In **our** opaque-Newcomb with an explicit guaranteed
box, both behave **strongly CDT** (two-boxing; the 14B even refuses one-boxing 0–15% at p=0.99 *handed
the EV*, Run 10). So our specific high-stakes, dominance-salient framing is **far more CDT-triggering
than the average attitude question** — the override is a property of the framing/payoff salience, not
a uniform CDT disposition the models carry into every Newcomb-flavored prompt. (Caveat: the attitude
set mixes many scenario types — PD, voting, acausal — so it isn't a pure opaque-Newcomb comparison.)

## How this connects to the project
- Reinforces the **capability ceiling** (3B near-chance externally) and the **scale-sharpens-CDT**
  story (14B more capable *and* slightly more CDT-leaning) — see `results.md` Day-3 synthesis + the
  "EDT-lean = capability/pretraining" Discussion section, which predicted exactly this in-family
  direction.
- The **attitude-axis run is the apples-to-apples test** flagged in that Discussion section; it now
  has data, and the data does not support capability→EDT in-family.

## Caveats
- **n = 20 items, single seed**; bootstrap CIs are wide (±~0.18). Capability gap survives this;
  attitude differences do **not** (treat as "no clear trend").
- **Two-point family slice** — not a substitute for Oesterheld's many-model correlation.
- **Scenario-mix mismatch**: the attitude subset spans non-Newcomb problems; our abstract items are
  pure opaque-Newcomb. The cross-setup comparison (Finding 3) is suggestive, not controlled.
- **Faithful invalid rate** (3B 25%/18%) is a *case-sensitivity* artifact (miscased commits), not
  non-engagement; the lenient re-score is the de-noised number to trust for the 3B.

## Reproduce
```bash
# data (gitignored; own license): curl benchmark/data.zip, unzip -P onebox -> newcomb_eval/data/oesterheld/
bash results/newcomblike/run_newcomblike.sh   # smoke -> capabilities(3B,14B) -> attitudes(3B,14B), 1024 tok
```
Artifacts: `results/newcomblike/oesterheld_{capabilities,attitudes}_{samples,metrics}_*.{csv,json}`,
`results/newcomblike/run.log`. Tests: `newcomb_eval/tests/test_newcomblike.py` (18, CPU).
