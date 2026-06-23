# OVERNIGHT.md — unattended / batch GPU queue

A rough, living list of GPU work that's safe to run **unattended** (overnight or while away).
The A40 is a single GPU, so "batch" means *queue independent runs back-to-back in a for-loop*, not
literally concurrent. A good overnight candidate is a run whose results **aggregate** and that
**doesn't need a human to decide the next step mid-batch** — sweeps, ablations, self-contained
baselines/probes. At this stage **most work is serial** (each result informs the next move), so the
overnight bucket is mainly the ablations we'd run regardless.

## Operating rules (learned the hard way)
- **Smoke before batch.** Run each command ~2 steps first — today's CoT OOM would've been caught by
  a 2-step smoke. (CLAUDE.md "Sanity gates".)
- **Memory-safe CoT defaults:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--micro 8`,
  cap `--max-new-tokens` (~128). Long sequences × 152k vocab OOM otherwise.
- **Distinct `--tag` per run** so adapters/logs don't clobber (the `--cot` default tag is just
  `cot`; always override it in a sweep).
- **Fail-soft loop:** `... > results/run_$tag.log 2>&1 || true; echo "done $tag"` so one crash
  doesn't abort the batch. Logs → `results/run_<tag>.log`; adapters → `results/adapters/<arm>_<tag>`
  (gitignored).
- Don't launch a batch onto a GPU already running a primary experiment — check `pgrep -f train_rl`.

## Overnight-friendly (fire-and-forget; read in the morning)
- [ ] **Seed-confirmation of the paired-CoT slope** *(runnable now; queued after STaR)* — the Lever-1b
      `+0.17` slope is from one noisy 60-step run. Re-run paired-CoT at **2–3 seeds × more steps**
      (e.g. `--steps 120 --eval-items 15`) and check whether the slope stabilizes, grows, or vanishes.
      Settles whether "fair objective ≈ plain objective" is real. (`--seed 0/1/2 --paired --cot`.)
- [ ] **LoRA capacity sweep** *(runnable now)* — rank ∈ {4,8,16,32,64} × kl ∈ {0.005,0.02,0.05},
      CoT-evidential. Localizes "optimization/anchoring vs representational" bottleneck. Flags
      `--lora-rank --kl-coef` already exist.
- [ ] **cot_inspect generation** *(runnable now)* — `python -m newcomb_eval.cot_inspect --adapter
      <dir> --tag <name>` on base + each trained adapter → HTML viewers to read in the morning.
- [ ] **Capability-cliff probe** *(runnable now; slow)* — `run_scaffold` + `logprob_sweep` on
      `Qwen2.5-14B-Instruct` (inference, fits the A40). Does a bigger model form the step the 3B can't?
- [ ] **Gemma-2-2b-it baseline** *(runnable now)* — `run_mvp` + a CoT variant, for Tennant-et-al.
      comparability (PLAN §2).
- [ ] **Seed / CI runs** *(runnable now)* — re-run key configs at higher `n_repeats` / multiple
      `--seed` for tighter CIs (cheap variance reduction; n=8 slopes are noisy).
- [ ] **Per-p ablations** *(needs tiny build: a train-p restriction flag)* — CoT-evidential trained
      on only-high-p then only-low-p. Uniform-one-box / uniform-two-box ⇒ RL learns the training
      marginal.
- [ ] **Attn-only vs MLP-only LoRA** *(needs tiny build: expose `target_modules` in rl_config)* —
      two runs. Attn helps slope ⇒ routing/access-to-p; MLP ⇒ computation.

## Serial / needs-eyes (run attended; each informs the next move)
- **Lever 1b paired-CoT** — primary experiment; read it and decide next.
- **SFT/STaR (Lever 2)** — multi-stage (sample→filter→SFT→eval), decisions between stages.
- **Payoff ablation** — only validates a slope *if one appears*; needs the dataset build below.
- **C2 self-snapshot predictor** — watch for instability / bistability.
- **Process-reward over the scaffold** — needs reward-design decisions.

## Batch-2 — auto-chains after batch-1 tonight (existing tools, low autonomous risk)
Launched by the assistant when batch-1 (`bz5e6ccds`) frees the GPU. No new code — extends Run 9.
- [ ] **Transplant on the RL'd / SFT'd adapters** — `transplant --adapter results/adapters/{causal,
      evidential,evidential_paired_cot,sft_star}`. Does the *disposition* change how the model uses
      supplied EV (e.g. does causal-CDT resist the aid harder at high p)? Directly extends Run 9.
- [ ] **Threshold-local psychophysics** — `cot_inspect --temperature 0 --limit 20 -n 2
      --p-grid 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88` on base (+ sft_star). Local sensitivity
      near p\* — does anything peak at the crossover, or is it flat through the threshold?

## New-code ideas (morning, WITH user — need a build + design choices; do not build solo overnight)
Highest-value first (the other model's "second line", reordered for our current dominance-override finding):
- **Anti-Newcomb camouflage (Idea 9)** ⭐ — isomorphic items (same `A pays pB, B pays S+(1-p)B` math)
  with the Newcomb/container/predictor story stripped (medical test, packet cache, insurance…). If
  the model tracks `p` in camouflage but not in containers → it's the **Newcomb prior/story**, not
  EV competence. The single most informative new build given Run 9.
- **Equation-only RL environment (Idea 7)** ⭐ — RL on bare `A pays pB / B pays S+(1-p)B, choose`,
  no semantics. Learns the slope → bottleneck is extraction/framing; fails → small-model RL capacity.
- **Subskill unit-tests (Idea 6)** — tiny arithmetic/formula/mapping battery, base vs RL adapters →
  did RL damage general arithmetic? Cheap once built.
- **CoT counterfactual swap (Idea 2) / delayed-answer commitment (Idea 8)** — does the answer
  causally track the numeric `p`, or a pre-formed story / a pre-anchored label? Both cheap inference.

## Not-ready (need a code/data build before they can be queued)
- payoff-parametrised (templated-payoff) dataset → payoff ablation
- C2 snapshot-predictor mechanism in `rloo`
- Oesterheld dataset acquisition → external-dataset sanity check

## Now running (do not disturb)
- **Batch-1** (`bz5e6ccds`, ~5–6 h) → `results/overnight.log`: [1] clean transplant ✓ →
  [2] bigger SFT (`sft_star_big`) + de-noised eval → [3] 14B scale probe (cot_inspect + transplant) →
  [4] paired-CoT seeds 1,2. **Batch-2 auto-chains when this frees the GPU.**
- Done today: Lever 1a KL sweep, Lever 1b paired (Run 7), STaR-SFT (Run 8), transplant (Run 9).
