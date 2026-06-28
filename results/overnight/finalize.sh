#!/usr/bin/env bash
# Durable, no-LLM findings extractor. Waits for the overnight orchestrator to finish, then dumps the
# RAW results into results/overnight/FINDINGS.md so the night is captured even if the assistant
# session is terminated. A human or a revived session folds FINDINGS.md into results.md.
set -u
cd "$(dirname "$0")/../.." || exit 1
F=results/overnight/FINDINGS.md

# wait for the orchestrator to exit (poll its presence)
while pgrep -f 'run_tonight.sh' >/dev/null 2>&1; do sleep 60; done

{
  echo "# Overnight batch — FINDINGS (auto-extracted $(date))"
  echo
  echo "Raw dump; fold into results.md. Job status table follows, then per-job results."
  echo
  echo "## Status table"; cat results/overnight/SUMMARY.md 2>/dev/null || echo "(no SUMMARY)"
  echo

  echo "## (1) Mechanism-credibility ladder (14B)"
  if [ -f results/credence/mechanism_signature.csv ]; then
    echo '```'; cat results/credence/mechanism_signature.csv; echo '```'
    echo "Read: does one-box rate / credence gap_adj rise m0(statistical)->m3(exact-copy)?"
    echo "Plots: results/credence/mechanism_signature.png"
  else
    echo "MISSING — check results/run_mech_14b.log"
    tail -15 results/run_mech_14b.log 2>/dev/null | sed 's/^/    /'
  fi
  echo

  echo "## (2) Self-snapshot one-box-basin probe (3B)"
  if [ -f results/run_hyst_onebox_seedref.log ]; then
    echo "First + last eval lines (did K stay ~1 [bistable] or decay to 0 [two-box only]?):"
    echo '```'
    grep -E 'eval|p_model' results/run_hyst_onebox_seedref.log 2>/dev/null | head -3
    echo "..."
    grep -E 'eval|p_model' results/run_hyst_onebox_seedref.log 2>/dev/null | tail -3
    echo '```'
  else
    echo "MISSING — check results/run_hyst_onebox_seedref.log"
  fi
  echo "Endpoint logprob slope (ep_hyst_onebox_seedref):"
  if [ -f results/logprob/p_margin_by_p_ep_hyst_onebox_seedref.csv ]; then
    echo '```'; cat results/logprob/p_margin_by_p_ep_hyst_onebox_seedref.csv; echo '```'
  else
    grep -iE 'slope|p\* =' results/run_hyst_onebox_seedref.log 2>/dev/null | tail -3 | sed 's/^/    /'
  fi
  echo

  echo "## (3) LoRA extras (if reached)"
  for r in 8 32; do
    L=results/run_lora_r${r}_kl0.02.log
    [ -f "$L" ] && { echo "### rank $r kl 0.02 (last eval):"; echo '```'; grep -E 'eval|slope' "$L" 2>/dev/null | tail -3; echo '```'; }
  done
  echo
  echo "_auto-extracted $(date) — interpretation pending (assistant or human)._"
} > "$F" 2>&1

echo "wrote $F"
