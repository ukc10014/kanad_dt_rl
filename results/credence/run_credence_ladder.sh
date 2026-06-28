#!/usr/bin/env bash
# Credence-ladder: does the action<->box dependence get *represented* (and stay action-unused)
# across the Qwen2.5 family? Runs the credence probe (+ logprob action-margin) per rung, each in
# its OWN process for clean GPU release, then the CPU cross-model aggregation.
#
# Rungs: 3B / 7B / 14B in bf16 (~28 GB max), 32B in 4-bit (~20 GB) — all fit a 46 GB A40 one at a
# time. 7B (~15 GB) and 32B (~65 GB) DOWNLOAD on first run; ensure disk + time.
# Fail-soft (one rung dying doesn't sink the rest). Run ONLY when the GPU is free.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/credence/ladder.log
mkdir -p results/credence

# guard: never contend with the other instance's training run
if pgrep -af 'oracle_anchor|train_rl|selfplay\.py' | grep -qv 'run_credence_ladder'; then
  echo "ABORT $(date): training run active (selfplay/oracle/train) — not contending for GPU" | tee -a "$LOG"
  exit 1
fi

echo "=== credence-ladder START $(date) ===" | tee -a "$LOG"
run () { echo -e "\n### $* ###" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1 || echo "FAILED: $*" | tee -a "$LOG"; }

PROBE_VARIANTS="--variant outcome prediction direct"

# bf16 rungs: action margin (logprob) + credence probe with the flat-action overlay.
for spec in \
  "Qwen/Qwen2.5-3B-Instruct base3b" \
  "Qwen/Qwen2.5-7B-Instruct base7b" \
  "Qwen/Qwen2.5-14B-Instruct base14b"; do
  set -- $spec; M=$1; T=$2
  run python -m newcomb_eval.logprob_sweep --model "$M" --tag "$T"
  run python -m newcomb_eval.credence_probe --model "$M" --tag "$T" $PROBE_VARIANTS \
      --action-margin "results/logprob/p_margin_by_p_${T}.csv"
done

# 32B in 4-bit: credence only (logprob_sweep has no quant path; the action-slope trend comes from
# the bf16 rungs). bitsandbytes quantizes at load time, so the FULL ~65 GB bf16 must download first.
# Disk guard: skip unless the 32B is already cached OR there is comfortable free space — never risk
# filling the disk under an unattended run (would also threaten the other instance's checkpoints).
CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen2.5-32B-Instruct"
FREE_GB=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -d "$CACHE" ] || [ "${FREE_GB:-0}" -ge 75 ]; then
  run python -m newcomb_eval.credence_probe --model Qwen/Qwen2.5-32B-Instruct --tag base32b_4bit \
      $PROBE_VARIANTS --quantize 4bit
else
  echo "SKIP 32B: only ${FREE_GB}G free, need ~75G for the bf16 download (4-bit quantizes at load). \
See OVERNIGHT.md for the AWQ/free-disk options." | tee -a "$LOG"
fi

# cross-model signature + coherence (CPU, reads the persisted files above)
run python -m newcomb_eval.credence_ladder

echo "=== credence-ladder DONE $(date) ===" | tee -a "$LOG"
echo "--- signature ---" | tee -a "$LOG"
[ -f results/credence/ladder_signature.csv ] && cat results/credence/ladder_signature.csv | tee -a "$LOG"
