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

## Not-ready (need a code/data build before they can be queued)
- payoff-parametrised (templated-payoff) dataset → payoff ablation
- `newcomb_rl/sft.py` → STaR/SFT
- `target_modules` flag in `rl_config` → attn/MLP ablation
- train-p restriction flag → per-p ablations
- C2 snapshot-predictor mechanism in `rloo`
- Oesterheld dataset acquisition → external-dataset sanity check

## Now running (do not disturb)
- **Lever 2 STaR SFT** (3B, harvest 12 items + 3 SFT epochs) — `results/run_sft_star.log`,
  adapter → `results/adapters/sft_star`. (Lever 1a KL sweep + Lever 1b paired-CoT: done, see Run 7.)
