# results/logs — recovered RL training dynamics

This directory holds the **per-step self-play / RLOO training dynamics** that the blanket
`*.log` / `logs/` gitignore rules had been silently excluding. A prior session, working from
a GitHub checkout that lacked these, had to rebuild the dynamics curves from only ~4
eval-checkpoints per run transcribed by hand into `results.md`. This snapshot is a full copy
of the old GPU box, so the raw logs were available again and are parsed/preserved here.

## Files
| file | what |
|---|---|
| `rl_dynamics_steps.csv` | one row per `(run_id, step)`, train + eval fields merged: `reward, K_train, invalid, gen_len, p_model` (train rows) and `mean_K, slope, k_at_p_lo, k_at_p_hi, p_lo, p_hi` (eval rows). |
| `rl_dynamics_kbyp.csv` | long form: one row per `(run_id, step, p_position)` giving each logged `K@p` value. |
| `rl_dynamics_runs.csv` | one row per run: parsed header metadata (model, arm, dataset, train_p grid, p\*, kl, kl_ref, seed_adapter, lag, cot, …). |
| `adapter_manifest.csv` | sha256 + size + structure of every LoRA adapter under `results/adapters/` (which stays gitignored, ~7 GB). Integrity/provenance record. |
| `raw/` | verbatim copies of the source `*.log` files the CSVs were parsed from (preserves subdir layout). |

Regenerate the CSVs: `python3 scratchpad/parse_rl_logs.py .`
Redraw the dynamics figure (`results/selfplay_dynamics_fullres.png`): `python scratchpad/plot_selfplay_fullres.py`

## What these recover
The numeric per-step trajectories (`mean_K`, conditional `slope = K@p_hi − K@p_lo`, and the
self-predictor accuracy `p_model`) for the R1 self-play family (`r1_calib`, `r1_seed1`,
`r1_kl0`), the clean A2 self-consistency run (`r1_loo`), the lagged variants (`r1_lag` lag=3,
`r1_lag0` lag=0), the 3B self-snapshot hysteresis sweep (`selfsnapshot/hyst_*`), the oracle
anchors, and the Qwen CoT/paired-CoT KL sweep. These are the data behind
"**self-referential RL collapses into bistable basins; RL moves the lean, not the rule**."

## What is NOT here (gaps — surfaced honestly)
- **No embedded sample reasoning traces.** None of the recovered logs contain
  `sample=(prompt, completion)` / `<think>` dumps. On-disk reasoning traces live instead in the
  already-tracked `results/cot_inspect/*.jsonl` and `results/scaffold/scaffold_logs/*.jsonl`.
- **The exogenous-EV paired run is absent from this snapshot.** Its adapter
  `results/adapters/evidential_r1_paired` and its log `scratchpad/r1_full.log` — both referenced
  by `results.md` on `origin/main` — do **not** exist on this box (only a smoke,
  `evidential_r1_paired_smoke2`, is present). The full paired run was evidently executed on the
  newer box that produced branch `claude/confident-shannon-e1f3uc`, after this disk snapshot.
  Its 5-checkpoint trajectory survives only in the `results.md` table and is carried into the
  figure as the single table-sourced ("[table]") series.

## Note on aggregate logs
`raw/` includes some concatenated multi-run logs (`oracle/all.log`, `overnight*.log`,
`orchestrator.log`, `klctl_batch.log`). The parser **skips** these to avoid corrupting run
identity — every run they aggregate has its own per-run log that is parsed instead.
