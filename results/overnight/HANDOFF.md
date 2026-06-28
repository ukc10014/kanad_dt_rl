# Overnight batch — HANDOFF (2026-06-24, in case the assistant session is terminated)

An **unattended overnight GPU batch is running detached** and will continue even if the Claude
session that launched it dies.

## What's running
- `results/overnight/run_tonight.sh` (orchestrator, nohup-detached) — waits for the GPU to be idle
  (R1-Distill job clears), then runs **serially, smoke-first, fail-soft, watchdog-guarded**:
  1. **Mechanism-credibility ladder (14B)** → `results/credence/mechanism_signature.{csv,json,png}`
  2. **Self-snapshot one-box-basin probe (3B)** + endpoint `logprob_sweep`
     → `results/run_hyst_onebox_seedref.log`, adapter `evidential_modelpred_hyst_onebox_seedref`
  3. **Trimmed LoRA extras** (rank 8/32 × kl 0.02) — only if reached
- `results/overnight/finalize.sh` (nohup-detached) — waits for the orchestrator to finish, then
  auto-writes **`results/overnight/FINDINGS.md`** (raw numbers, no interpretation needed).

## Watchdog
Kills a job on TIMEOUT / STALL / OOM / invalid-rate-climb and continues to the next. It does **NOT**
kill a clean two-box decay (K pinned at 0 is a *valid* result for the hysteresis probe). All 5
kill-paths were trip-tested green before launch.

## Live status
- `results/overnight/SUMMARY.md` — per-job status table (updates as jobs finish)
- `results/overnight/orchestrator.log` — orchestrator timeline
- `results/run_<tag>.log` — per-job logs

## To fold into results.md (when done)
Read `results/overnight/FINDINGS.md`, then **append** (don't overwrite — two instances write
results.md) a dated section interpreting:
- **Mechanism:** does one-box rate / credence `gap_adj` rise m0 (statistical) → m3 (exact-copy)?
  Rise ⇒ the incredible predictor was suppressing EDT (framing artifact). Flat even at exact-copy ⇒
  dominance robust to credibility.
- **Hysteresis:** did K stay ~1 (⇒ **bistability**, two attractors, separatrix at p\*=0.8) or decay to
  0 (⇒ **two-box is the only attractor**)?
- Gate the mechanism read on `gap@0.5≈0` + saturation flags (see results.md "mechanism-credibility
  pivot 2026-06-24"); the forced-token `outcome` variant is confounded — trust `direct`.

## Do NOT
Start any "New-code ideas" (anti-Newcomb camouflage, equation-only RL) — those need design with the
user. R1-Distill outputs (`*r1d*`, `cot_inspect/cot_r1d*`) belong to the other instance — leave alone.
