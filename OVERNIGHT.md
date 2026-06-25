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
- [ ] **Mechanism-credibility ladder on 14B — is the abstract predictor *incredible*?** ⭐⭐ *(STAGED;
      runnable now when GPU free)* — `bash results/credence/run_credence_mechanism.sh`. Our abstract
      predictor ("identifies agents like you X% of the time") is a **reference-class statistic** — the
      least binding framing; a causal world-model can rightly treat it as non-binding → dominance.
      Vary ONLY the predictor-mechanism clause across **m0 statistical → m0pad length-placebo → m1
      individual-model → m2 process-scan → m3 exact-copy (FDT-strong)**, payoff + abstract labels
      fixed. 14B only (comprehends → null is meaningful), de-confounded probe (`direct` free-form
      credence + `action` margin). **Read:** one-box rate and/or credence `gap_adj` **rise m0→m3**
      (placebo stays at m0) ⇒ incredible predictor was *suppressing* EDT ⇒ framing artifact, not fixed
      disposition; **flat even at exact-copy** ⇒ dominance robust to credibility (quotable). Outputs
      `results/credence/mechanism_signature.{csv,json,png}`. Built+CPU-smoked; `gen_mechanism_dataset.py`,
      `credence_mechanism.py`; tests green (21). Caveats: clause-length confound (m0pad controls it);
      FDT≠EDT not isolated. See results.md "mechanism-credibility pivot (2026-06-24)".
- [x] **Credence ladder — represented-but-unused, across the Qwen2.5 family** ⭐ *(RAN 2026-06-24 —
      forced-token probe CONFOUNDED: `gap@0.5≈0.7`, `outcome` one-sided-saturated; free-form `direct`
      partially tracks p at 14B; coherence scales (14B agree 0.68). Instrument fixed (gap_adj +
      saturation gate). See results.md.)* — `bash results/credence/run_credence_ladder.sh`. Does the model *represent*
      the action↔box evidential dependence (credence gap → `2p−1`) even where its *action* stays flat
      CDT? Rungs **3B / 7B / 14B (bf16) + 32B (4-bit)**, each a separate process (clean GPU release);
      runs `logprob_sweep` (action margin) + `credence_probe --variant outcome prediction direct`
      per rung, then `credence_ladder` (CPU) → `results/credence/ladder_signature.{csv,json,png}`.
      **Headline to look for:** representation slope *rising* with scale while action slope stays ~0
      (divergence grows) ⇒ comprehension orthogonal to disposition ⇒ EDT-lean isn't a capability
      artifact. **Gates:** read `resolvability` per rung first — low ⇒ that rung is *unreadable*
      (saturation), not a real null; and `gap@0.5≈0` (symmetry control). Coherence cols (variant
      agreement, monotonicity) flag an EDT *answer* without an EDT *world model*.
      Caveats logged: 7B (~15 GB) + 32B (~65 GB) **download** on first run (disk+time); 32B has no
      action-margin (logprob_sweep has no quant path — action trend comes from the bf16 rungs);
      32B 4-bit injects a quantization confound. CPU-smoked (logic + plumbing); GPU-gated behind
      the other instance's `selfplay`. Module: `newcomb_eval/credence_probe.py` + `credence_ladder.py`;
      tests `newcomb_eval/tests/test_credence_probe.py` (18, CPU, green).
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
- [ ] **Self-snapshot one-box-basin probe — completes the C2 hysteresis** ⭐ *(runnable now; reuses
      `newcomb_rl/selfplay.py`, which now has `--kl-ref/--kl-coef`)* — Day-4 C2 found the **two-box
      basin is a strong self-fulfilling attractor** (causal-seed locks K=0 / margin ≈ −18, robust to
      KL-ref *and* EMA; `p_model`↔K move in lockstep), but **base-seed settles at indifference** (base
      self-prediction ~0.69–0.80 sits on the `p*`=0.8 separatrix ⇒ drifts 0.80→0.50, dip to 0.35 — *not*
      a roll into the two-box basin; at p=p\* there is no basin gradient). The
      one-box basin is **untested** — no seed started clearly above `p*`. Fix: **seed from a committed
      one-boxer** (oracle p=1 adapter, K=1.0 ⇒ self-prediction ~1.0 ≫ 0.8) and check it **stays**:
      `python -m newcomb_rl.selfplay --seed-adapter results/adapters/evidential_oracle_p1_base
      --kl-ref seed --p 0.8 --snapshot-every 10 --steps 150 --K 8 --P 8 --eval-every 15
      --tag hyst_onebox_seedref` (optionally also `evidential_oracle_p1_causal`), then
      `python -m newcomb_eval.logprob_sweep --adapter
      results/adapters/evidential_modelpred_hyst_onebox_seedref --tag ep_hyst_onebox_seedref`.
      **Headline:** K stays ~1 & `p_model` stays high ⇒ one-box basin is *also* self-fulfilling ⇒
      asymmetric **bistability confirmed** (two attractors, separatrix at `p*`). If it *also* decays
      to two-box ⇒ two-box is the *only* attractor at these payoffs (report honestly). **Smoke** 0.5B
      first; **memory** one 3B job (~38 GB), serial; watch invalid-rate (collapse) + `p_model`↔K lockstep.

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

## Nothing running — both overnight batches complete (GPU idle)
- **Batch-1 → Run 10**, **Batch-2 → Run 11** (both in `results.md`). Done: Lever 1a KL sweep,
  Lever 1b paired (Run 7), STaR-SFT (Run 8), transplant (Run 9), 14B scale + confirmations (Run 10),
  bigger-SFT/14B-free-CoT/adapter-transplant/dense-p (Run 11).
- **Overnight headline:** scale *sharpens* the dominance-override (14B refuses one-boxing 0–15% at
  p=0.99 even handed the EV); bigger SFT didn't help (true ceiling, not data); explicit EV is
  counterproductive (surfaces the guaranteed reward); RL-to-CDT installs an EV-immune disposition.
- **Morning agenda (new builds, WITH user):** anti-Newcomb camouflage ⭐ (is it the Newcomb prior?)
  + equation-only RL env ⭐ (can RL learn the EV primitive stripped of story?). Both need design.
