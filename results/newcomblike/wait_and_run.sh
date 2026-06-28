#!/usr/bin/env bash
# Wait for the GPU to free (other instance's oracle/train run), then fire the Oesterheld pipeline.
# Polls every 30s; gates on "no oracle/train/selfplay proc AND <8GB used". Caps at ~6h.
set -u
cd "$(dirname "$0")/../.." || exit 1
WLOG=results/newcomblike/wait.log
echo "=== wait_and_run START $(date) ===" | tee -a "$WLOG"
for i in $(seq 1 720); do
  busy=$(pgrep -af 'oracle_anchor|train_rl|selfplay\.py' | grep -v 'wait_and_run' || true)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$busy" ] && [ "${used:-99999}" -lt 8000 ]; then
    echo "GPU free (used=${used}MiB) at $(date) after ${i} polls — launching pipeline" | tee -a "$WLOG"
    bash results/newcomblike/run_newcomblike.sh
    echo "=== wait_and_run DONE $(date) ===" | tee -a "$WLOG"
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT: GPU still busy after ~6h (used=${used:-?}MiB, busy=[$busy])" | tee -a "$WLOG"
exit 1
