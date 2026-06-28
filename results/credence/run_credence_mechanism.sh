#!/usr/bin/env bash
# Mechanism-credibility ladder on the 14B: does making the predictor *credible/binding* (statistical
# -> exact-copy/FDT) move the action toward one-boxing and turn the faint p-tracking in the clean
# (free-form) credence read into a real climb? Holds payoff + abstract labels fixed; varies only the
# predictor-mechanism clause. Uses the de-confounded probe (gap_adj baseline-subtraction; free-form
# `direct` primary; forced-token `outcome` was shown saturated, so it's dropped here).
# 14B only (the model that demonstrably comprehends -> a null is meaningful). Run when GPU is free.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/credence/mechanism.log
mkdir -p results/credence

if pgrep -af 'oracle_anchor|train_rl|selfplay\.py' | grep -qv 'run_credence_mechanism'; then
  echo "ABORT $(date): training run active — not contending for GPU" | tee -a "$LOG"; exit 1
fi

echo "=== mechanism-ladder START $(date) ===" | tee -a "$LOG"
run () { echo -e "\n### $* ###" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1 || echo "FAILED: $*" | tee -a "$LOG"; }

# 0) (re)generate the mechanism datasets (CPU, idempotent)
run python -m newcomb_eval.gen_mechanism_dataset

M=Qwen/Qwen2.5-14B-Instruct
# clean axes only: direct (free-form credence) + prediction (abstract-label cross-check) + action.
VARIANTS="--variant direct prediction action"
for level in m0 m0pad m1 m2 m3; do
  run python -m newcomb_eval.credence_probe --model "$M" \
      --data "newcomb_eval/data/dataset_mech_${level}.json" $VARIANTS --tag "mech_${level}_14b"
done

# cross-mechanism signature (CPU)
run python -m newcomb_eval.credence_mechanism --model 14b

echo "=== mechanism-ladder DONE $(date) ===" | tee -a "$LOG"
[ -f results/credence/mechanism_signature.csv ] && cat results/credence/mechanism_signature.csv | tee -a "$LOG"
