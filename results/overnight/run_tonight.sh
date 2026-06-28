#!/usr/bin/env bash
# Unattended serial overnight batch (single A40, shared box).
# Order: (1) mechanism-credibility ladder 14B  (2) self-snapshot one-box-basin + endpoint
#        (3) trimmed LoRA extras (if reached). Each: idle-gate -> smoke -> watchdogged full -> summary.
# Watchdog kills on TIMEOUT / STALL / OOM / invalid-rate-climb; does NOT kill a clean two-box decay
# (K pinned at 0/1 is a real result). Env: DRYRUN=1 prints queue; SKIP_IDLE=1 bypasses the GPU gate.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/overnight; mkdir -p "$OUT"
LOG="$OUT/orchestrator.log"; SUMMARY="$OUT/SUMMARY.md"
: "${DRYRUN:=0}"; : "${SKIP_IDLE:=0}"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
# foreign GPU users (other instance's r1d/cot_inspect, or any stray newcomb proc not us)
foreign(){ pgrep -af 'newcomb_eval|newcomb_rl|cot_inspect|selfplay|train_rl|DeepSeek|r1d' \
           | grep -vE 'run_tonight|pgrep'; }

wait_idle(){
  [ "$SKIP_IDLE" = 1 ] && return 0
  log "waiting for GPU idle (<2GB x3 reads, no foreign procs)..."
  local n=0 waited=0
  while :; do
    local u; u=$(gpu_used); u=${u:-99999}
    if [ -z "$(foreign)" ] && [ "$u" -lt 2000 ]; then n=$((n+1)); else n=0; fi
    [ "$n" -ge 3 ] && { log "GPU idle confirmed (${u}MiB)"; return 0; }
    sleep 20; waited=$((waited+20))
    [ "$waited" -ge 21600 ] && { log "ABORT: GPU not idle after 6h"; return 1; }
  done
}

row(){ echo "| $1 | $2 | $3 | $4 | $5 |" >> "$SUMMARY"; }

# run_job NAME TAG TIMEOUT_MIN STALL_MIN DEGEN(0/1) EXPECTED_OUT SMOKE_CMD FULL_CMD
run_job(){
  local name="$1" tag="$2" tmo="$3" stall="$4" degen="$5" outp="$6" smoke="$7" full="$8"
  local jlog="results/run_${tag}.log"; local t0; t0=$(date +%s)
  if [ "$DRYRUN" = 1 ]; then
    echo "JOB $name [tag=$tag tmo=${tmo}m stall=${stall}m degen=$degen]"
    echo "   smoke: $smoke"; echo "   full : $full"; echo "   out  : $outp"; return 0
  fi
  if [ -n "$outp" ] && [ -e "$outp" ]; then
    log "SKIP $name (exists: $outp)"; row "$name" SKIPPED "output exists" - "$outp"; return 0
  fi
  wait_idle || { row "$name" SKIPPED "gpu-busy 6h" - -; return 0; }
  if [ -n "$smoke" ]; then
    log "SMOKE $name"
    if ! timeout 25m bash -c "$smoke" > "${jlog}.smoke" 2>&1; then
      log "SMOKE-FAIL $name (see ${jlog}.smoke)"; row "$name" SMOKE-FAIL "see ${jlog}.smoke" - -; return 0
    fi
    wait_idle || { row "$name" SKIPPED "gpu-busy" - -; return 0; }
  fi
  log "RUN $name (timeout ${tmo}m)"
  setsid bash -c "$full" > "$jlog" 2>&1 &
  local pg=$!; local started; started=$(date +%s); local prog=0 status="" reason=""
  while kill -0 "$pg" 2>/dev/null; do
    sleep 30; local now; now=$(date +%s)
    if [ $((now-started)) -ge $((tmo*60)) ]; then status=TIMEOUT; reason="exceeded ${tmo}m"; break; fi
    if grep -qiE 'CUDA out of memory|OutOfMemoryError|torch\..*[Oo]utOf[Mm]emory' "$jlog" 2>/dev/null; then
      status=OOM; reason="cuda oom"; break; fi
    grep -qE 'step=|gap fit|gap_adj|rung|### |\[(direct|action|prediction|outcome|m[0-9])' "$jlog" 2>/dev/null && prog=1
    if [ "$prog" = 1 ]; then
      local age=$(( (now - $(stat -c %Y "$jlog" 2>/dev/null || echo "$now")) / 60 ))
      [ "$age" -ge "$stall" ] && { status=STALLED; reason="no output ${age}m"; break; }
    fi
    # degeneracy (RL): invalid-rate climbing — NOT K-pinning (that is a valid collapse)
    if [ "$degen" = 1 ] && tail -25 "$jlog" 2>/dev/null | grep -qE 'invalid=0\.[5-9]|invalid=1\.0'; then
      status=DEGENERATE; reason="invalid-rate>0.5"; break; fi
  done
  if [ -z "$status" ]; then
    wait "$pg"; local rc=$?
    [ "$rc" = 0 ] && status=DONE || { status=FAILED; reason="exit $rc"; }
  else
    log "KILL $name ($status: $reason)"; kill -TERM -- -"$pg" 2>/dev/null; sleep 5; kill -9 -- -"$pg" 2>/dev/null
  fi
  if [ "$SKIP_IDLE" != 1 ]; then
    local w=0; while [ "$(gpu_used || echo 0)" -ge 2000 ] && [ "$w" -lt 8 ]; do sleep 20; w=$((w+1)); done
  fi
  local wall=$(( ($(date +%s)-t0)/60 ))m
  log "$status $name ($reason) $wall"; row "$name" "$status" "${reason:-ok}" "$wall" "$jlog"
}

# ---------------------------------------------------------------------------------------------
echo "# Overnight batch — started $(date)" > "$SUMMARY"
echo "" >> "$SUMMARY"; echo "| job | status | reason | wall | log |" >> "$SUMMARY"; echo "|---|---|---|---|---|" >> "$SUMMARY"
log "=== overnight batch START $(date) ==="

# Watchdog trip-tests (TESTWD=1): fake jobs proving each kill path before trusting it overnight.
if [ "${TESTWD:-0}" = 1 ]; then
  SKIP_IDLE=1; log "WATCHDOG TRIP-TESTS"
  run_job "t-done"    t_done  5 1 0 "" "" "echo step=1; sleep 2; echo ok"
  run_job "t-oom"     t_oom   5 1 0 "" "" "echo step=1; echo 'CUDA out of memory'; sleep 999"
  run_job "t-degen"   t_degen 5 1 1 "" "" "for i in 1 2 3; do echo invalid=0.9; done; sleep 999"
  run_job "t-timeout" t_tmo   1 1 0 "" "" "echo step=1; sleep 999"
  run_job "t-stall"   t_stl   5 1 0 "" "" "echo step=1; sleep 999"
  log "=== trip-tests DONE ==="; exit 0
fi

M14=Qwen/Qwen2.5-14B-Instruct

# (1) Mechanism-credibility ladder — TOP PRIORITY (14B inference, ~40m)
run_job "mechanism-ladder" "mech_14b" 90 20 0 "results/credence/mechanism_signature.csv" \
  "python -m newcomb_eval.gen_mechanism_dataset && python -m newcomb_eval.credence_probe --model $M14 --data newcomb_eval/data/dataset_mech_m0.json --variant direct action --limit 2 --p-grid 0.5 0.99 --tag mech_smoke14b" \
  "bash results/credence/run_credence_mechanism.sh"

# (2) Self-snapshot one-box-basin probe + endpoint logprob_sweep (3B RL, ~15m)
run_job "self-snapshot-hysteresis" "hyst_onebox_seedref" 50 8 1 "results/adapters/evidential_modelpred_hyst_onebox_seedref" \
  "python -m newcomb_rl.selfplay --model Qwen/Qwen2.5-0.5B-Instruct --kl-ref seed --p 0.8 --steps 3 --K 4 --P 4 --eval-every 2 --tag hyst_smoke05b" \
  "python -m newcomb_rl.selfplay --seed-adapter results/adapters/evidential_oracle_p1_base --kl-ref seed --p 0.8 --snapshot-every 10 --steps 150 --K 8 --P 8 --eval-every 15 --tag hyst_onebox_seedref && python -m newcomb_eval.logprob_sweep --adapter results/adapters/evidential_modelpred_hyst_onebox_seedref --tag ep_hyst_onebox_seedref"

# (3) EXTRAS — trimmed LoRA-capacity sweep (only if reached; cheap-ish RL; smoke-gated)
for spec in "8 0.02" "32 0.02"; do
  set -- $spec; R=$1; KL=$2
  run_job "lora-r${R}-kl${KL}" "lora_r${R}_kl${KL}" 120 8 1 "results/adapters/evidential_lora_r${R}_kl${KL}" \
    "python -m newcomb_rl.train_rl --arm evidential --cot --lora-rank $R --kl-coef $KL --steps 4 --tag lora_smoke_r${R}" \
    "python -m newcomb_rl.train_rl --arm evidential --cot --lora-rank $R --kl-coef $KL --steps 100 --tag lora_r${R}_kl${KL}"
done

log "=== overnight batch DONE $(date) ==="
echo "" >> "$SUMMARY"; echo "_done $(date)_" >> "$SUMMARY"
