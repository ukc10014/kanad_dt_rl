"""Full-resolution redraw of the R1 self-play / exogenous RLOO dynamics figure.

SUPERSEDES scratchpad/plot_selfplay_combined.py, which self-documents that it was
"reconstructed from results.md eval-checkpoint tables [because] raw per-step logs (*.log)
and final adapters were NOT retained in git" -- and consequently left gaps (e.g. the lag=0
control had `mean=None`, "endpoints only in results.md").

Those raw logs have now been recovered and parsed into results/logs/rl_dynamics_steps.csv.
This script reads that CSV directly, so:
  * every eval checkpoint actually logged is plotted (no hand-transcription),
  * the lag=0 mean-K trajectory is restored (was missing in the reconstructed version),
  * truncation-confounded points (logged invalid >= 0.15) are auto-flagged, not guessed.

Only the exogenous-EV paired run remains table-sourced: its log is `scratchpad/r1_full.log`,
which is absent from this snapshot (the full run was executed on the newer box; the adapter
`results/adapters/evidential_r1_paired` is likewise absent here -- only a smoke exists).

Three panels vs training step:
  A. overall lean      = eval mean K-rate (greedy)
  B. conditional slope = K@p_hi - K@p_lo  (the p*-tracking signal)
  C. predictor accuracy p_model = policy's OWN one-box rate (self-referential runs only)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
STEPS = REPO / "results" / "logs" / "rl_dynamics_steps.csv"

# run_id -> (legend label, colour). Order = legend/draw order.
RUNS = [
    ("evidential_modelpred_r1_calib", "cot calib (seed0, kl.02)", "tab:blue"),
    ("evidential_modelpred_r1_seed1", "cot seed1 (kl.02)",        "tab:cyan"),
    ("evidential_modelpred_r1_kl0",   "cot kl0",                  "tab:purple"),
    ("evidential_r1_loo",             "A2 self-consist (clean)",  "tab:green"),
    ("evidential_r1_lag",             "lag=3 regen",              "tab:orange"),
    ("evidential_r1_lag0",            "lag=0 regen (control)",    "tab:red"),
]

# exogenous-EV paired (reward = EV at the STATED p; no self-prediction) -> does NOT collapse.
# Log absent (scratchpad/r1_full.log); values transcribed from results.md (origin/main).
EXO = dict(label="exogenous-EV paired (no self-pred) [table]", c="black",
           step=[0, 10, 20, 30, 40], mean=[0.696, 0.703, 0.797, 0.656, 0.562],
           slope=[0.43, 0.62, 0.62, 0.62, 0.25])

CONFOUND_INVALID = 0.15  # logged invalid >= this -> truncation-suspect, ring it


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    rows = defaultdict(list)
    with STEPS.open() as f:
        for r in csv.DictReader(f):
            rows[r["run_id"]].append(r)
    series = {}
    for rid, rs in rows.items():
        rs.sort(key=lambda r: int(r["step"]))
        ev = [(int(r["step"]), fl(r["mean_K"]), fl(r["slope"])) for r in rs if fl(r["mean_K"]) is not None]
        pm = [(int(r["step"]), fl(r["p_model"])) for r in rs if fl(r["p_model"]) is not None]
        inv = {int(r["step"]): fl(r["invalid"]) for r in rs if fl(r["invalid"]) is not None}
        series[rid] = dict(ev=ev, pm=pm, inv=inv)
    return series


def ring_confounds(ax, xs, ys, inv):
    for x, y in zip(xs, ys):
        if inv.get(x, 0.0) >= CONFOUND_INVALID:
            ax.plot(x, y, marker="o", ms=12, mfc="none", mec="crimson", mew=1.8, zorder=7)


def main():
    s = load()
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(11.0, 13.2), sharex=True)

    # Panel A — overall lean (mean K)
    for rid, label, c in RUNS:
        ev = s[rid]["ev"]
        xs, ys = [e[0] for e in ev], [e[1] for e in ev]
        axA.plot(xs, ys, marker="o", ms=6, lw=1.9, color=c, label=label, alpha=0.92)
        ring_confounds(axA, xs, ys, s[rid]["inv"])
    axA.plot(EXO["step"], EXO["mean"], marker="o", ms=6, lw=3.2, color=EXO["c"],
             label=EXO["label"], zorder=6)
    axA.axhline(0.5, ls=":", color="grey", lw=1)
    axA.text(40.3, 0.5, "coin-flip", va="center", fontsize=8, color="grey")
    axA.set_ylim(0, 1.02)
    axA.set_ylabel("mean one-box rate (greedy eval)")
    axA.set_title("A · overall lean vs step  —  self-referential runs drift into flat (≈0/≈1) basins; "
                  "exogenous stays mid", fontsize=10.5, loc="left")
    axA.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)

    # Panel B — conditional slope
    for rid, label, c in RUNS:
        ev = s[rid]["ev"]
        xs, ys = [e[0] for e in ev], [e[2] for e in ev]
        axB.plot(xs, ys, marker="o", ms=6, lw=1.9, color=c, label=label, alpha=0.92)
        ring_confounds(axB, xs, ys, s[rid]["inv"])
    axB.plot(EXO["step"], EXO["slope"], marker="o", ms=6, lw=3.2, color=EXO["c"], zorder=6)
    axB.axhline(0.0, color="black", lw=1)
    axB.fill_between([-2, 42], 0, 1.15, color="green", alpha=0.04)
    axB.text(0.4, 1.09, "above 0  ⇒  tracks p (one-boxes more at high p)", fontsize=8.5, color="green")
    axB.text(0.4, -0.18, "≈ 0  ⇒  flat in p (no conditional rule)", fontsize=8.5, color="dimgrey")
    axB.text(45.5, -0.27, "○ = truncation-confounded (logged invalid ≥ 0.15)",
             fontsize=7.5, color="crimson", ha="right")
    axB.set_ylim(-0.32, 1.18)
    axB.set_ylabel("conditional slope  (K@p_hi − K@p_lo)")
    axB.set_title("B · conditional slope vs step  —  wobbles, never stabilises "
                  "(the +1.0 spikes are transients)", fontsize=10.5, loc="left")

    # Panel C — predictor accuracy p_model (self-referential only; exogenous has none)
    for rid, label, c in RUNS:
        pm = s[rid]["pm"]
        xs, ys = [e[0] for e in pm], [e[1] for e in pm]
        axC.plot(xs, ys, marker="o", ms=6, lw=1.9, color=c, label=label, alpha=0.92)
        ring_confounds(axC, xs, ys, s[rid]["inv"])
    for y, txt in [(1.0, "one-box basin (attractor)"), (0.0, "two-box basin (attractor)")]:
        axC.axhline(y, ls=":", color="grey", lw=1)
        axC.text(40.3, y, txt, va="center", fontsize=8, color="grey")
    axC.axhline(0.8, ls="--", color="firebrick", lw=1.2)
    axC.text(40.3, 0.8, "p*=0.8 (repeller)", va="center", fontsize=8, color="firebrick")
    axC.set_ylim(-0.03, 1.05)
    axC.set_ylabel("predictor accuracy  p_model\n(= policy one-box rate)")
    axC.set_xlabel("training step")
    axC.set_title("C · predictor accuracy vs step  —  co-moves with the lean into a seed-selected basin "
                  "(exogenous excluded: no self-predictor)", fontsize=10.5, loc="left")
    axC.set_xlim(-1.5, 46)

    fig.suptitle("R1 self-play & exogenous RLOO — dynamics over training\n"
                 "(rebuilt from recovered per-step logs → results/logs/rl_dynamics_steps.csv; "
                 "exogenous run table-sourced, log absent)", fontsize=12.5, y=0.997)
    fig.text(0.5, 0.005,
             "Eval n=4 items (cot/A2/lag) or n=8 (exogenous), single seed → individual wiggles are within "
             "the noise floor; read trajectory shape, not point values.  Self-referential reward "
             "(predictor=own one-box rate) ⇒ double-well ⇒ monotone slide into a flat basin.  "
             "Exogenous-EV reward (stated p) ⇒ no well ⇒ slope stays positive (no collapse).",
             ha="center", fontsize=8, color="dimgrey", wrap=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    out = REPO / "results" / "selfplay_dynamics_fullres.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
