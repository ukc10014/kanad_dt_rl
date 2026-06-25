#!/usr/bin/env bash
# Guarded overnight fanout for the R1 self-snapshot iterated game.
# Waits for the base calibration (r1_calib) to finish, HEALTH-GATES it ("no major hiccup"), then
# fires two fanouts unless the base run broke. Reuses selfplay_cot; no new code, no new adapters.
#   A  r1_seed1 : seed-confirmation (--seed 1)        -> is the fixed point seed-robust?
#   B  r1_kl0   : KL-independence control (--kl-coef 0)-> does the conditional rule form w/o KL pull?
# Each ~4-5h; sequential; fail-soft. Read results/r1_overnight.log + results/run_r1_{seed1,kl0}.log AM.
set -u
cd /root/kanad_dt_rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/r1_overnight.log
CALIB=results/r1_calib.log
: > "$LOG"
echo "[overnight] armed $(date); waiting for r1_calib to finish" >> "$LOG"

# 1) wait for the base calibration to exit (max ~8h so we never hang forever)
for _ in $(seq 1 96); do
  pgrep -f 'selfplay_cot --tag r1_calib' >/dev/null || break
  sleep 300
done
if pgrep -f 'selfplay_cot --tag r1_calib' >/dev/null; then
  echo "[overnight] ABORT: r1_calib still running after ~8h — not firing." >> "$LOG"; exit 0
fi
echo "[overnight] r1_calib process gone $(date)" >> "$LOG"

# 2) health gate — fire only if the base run had no major hiccup
ok=1
grep -q '\[r1_calib\] EXIT 0' "$CALIB" || { echo "[overnight] gate: no clean EXIT 0" >> "$LOG"; ok=0; }
last_eval=$(grep -oE 'eval mean_K=(nan|[0-9.]+)' "$CALIB" | tail -1)
echo "[overnight] gate: last $last_eval" >> "$LOG"
echo "$last_eval" | grep -qi nan && { echo "[overnight] gate: final eval nan" >> "$LOG"; ok=0; }
last_inv=$(grep -oE 'invalid=[0-9.]+' "$CALIB" | tail -1 | cut -d= -f2)
echo "[overnight] gate: last invalid=$last_inv" >> "$LOG"
[ -n "$last_inv" ] && awk "BEGIN{exit !($last_inv < 0.5)}" || { echo "[overnight] gate: invalid>=0.5 or missing" >> "$LOG"; ok=0; }
if [ "$ok" != "1" ]; then
  echo "[overnight] GATE FAILED — major hiccup in r1_calib; firing nothing. Inspect $CALIB." >> "$LOG"; exit 0
fi
echo "[overnight] GATE PASSED — firing fanouts $(date)" >> "$LOG"

# 3) fanouts (sequential, fail-soft). Same validated config as the base calib.
run() {
  local tag=$1; shift
  echo "[overnight] START $tag $(date)" >> "$LOG"
  python -m newcomb_rl.selfplay_cot --tag "$tag" --steps 30 --eval-every 10 --eval-items 4 \
    --K 4 --P 4 --micro 1 --max-new-tokens 2048 --p-grid 0.5 0.6 0.9 0.99 --snapshot-every 10 "$@" \
    > "results/run_$tag.log" 2>&1
  echo "[overnight] DONE $tag EXIT $? $(date)" >> "$LOG"
}
run r1_seed1 --kl-ref seed --seed 1               # A: seed-confirmation
run r1_kl0   --kl-ref seed --kl-coef 0.0 --seed 0 # B: KL-independence control
echo "[overnight] ALL DONE $(date)" >> "$LOG"
