#!/usr/bin/env bash
# Oesterheld Newcomb-like reality-check: capabilities + attitudes, 3B + 14B.
# Fail-soft, memory-safe, 1024-token budget (512 truncated ~47% of 3B reasoning).
# Run from repo root ONLY when the GPU is free (oracle/train held). See plan + module docstring.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/newcomblike/run.log
MNT=1024

# guard: never stomp the other instance's primary run
if pgrep -af 'oracle_anchor|train_rl|selfplay\.py' | grep -qv 'run_newcomblike'; then
  echo "ABORT: oracle/train/selfplay running — not contending for GPU" | tee -a "$LOG"; exit 1
fi

echo "=== newcomblike START $(date) (max_new_tokens=$MNT) ===" | tee -a "$LOG"
run () { echo -e "\n### $* ###" | tee -a "$LOG"; \
  python -m newcomb_eval.newcomblike_oesterheld "$@" --max-new-tokens $MNT >>"$LOG" 2>&1 \
  || echo "FAILED: $*" | tee -a "$LOG"; }

# 0) smoke (3 items, 1 repeat) — verify pipeline end-to-end before committing
run --mode capabilities --model Qwen/Qwen2.5-3B-Instruct --limit 3 --n 1 --exclude-tags trivia --tag smoke_3b

# 1) CAPABILITIES (objectively scored) — 20 items x 3, trivia excluded
run --mode capabilities --model Qwen/Qwen2.5-3B-Instruct  --limit 20 --n 3 --exclude-tags trivia --tag qwen25-3b-instruct
run --mode capabilities --model Qwen/Qwen2.5-14B-Instruct --limit 20 --n 3 --exclude-tags trivia --tag qwen25-14b-instruct

# 2) ATTITUDES (EDT-vs-CDT lean) — 20 items x 3; same seed => same 20 items across models
run --mode attitudes --model Qwen/Qwen2.5-3B-Instruct  --limit 20 --n 3 --tag qwen25-3b-instruct
run --mode attitudes --model Qwen/Qwen2.5-14B-Instruct --limit 20 --n 3 --tag qwen25-14b-instruct

echo "=== newcomblike DONE $(date) ===" | tee -a "$LOG"
echo "--- metrics ---" | tee -a "$LOG"
for f in results/newcomblike/oesterheld_*_metrics_*.json; do echo "$f:"; cat "$f"; echo; done | tee -a "$LOG"
