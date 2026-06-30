"""Combined self-play / exogenous RLOO dynamics — reconstructed from results.md checkpoint tables.

Raw per-step logs (*.log) and final adapters (results/adapters/) were NOT retained in git, so this
re-draws the over-time view from the eval-checkpoint numbers transcribed into results.md (cross-checked
against scratchpad/plot_r1_calib.py for the three cot runs). Eval n=4 items (cot/A2/lag) or n=8
(exogenous), single seed -> individual wiggles are within the noise floor; read the trajectory shape.

Three panels vs training step:
  A. overall lean      = eval mean K-rate (greedy)
  B. conditional slope = K@p_hi - K@p_lo   (the p*-tracking / "one-box preference vs p" signal)
  C. predictor accuracy p_model = policy's OWN one-box rate (self-referential runs only; exogenous has
     no self-predictor -- that's the whole point of including it as the non-collapsing contrast).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- data (from results.md checkpoint tables) -------------------------------------------------
# self-referential family (predictor accuracy = policy one-box rate); all drift into flat basins
SELF = {
    "cot calib (seed0, kl.02)": dict(
        c="tab:blue",   ms=[0,10,20,30],     mean=[0.625,0.375,0.375,0.312],
        ss=[0,10,20,30], slope=[0.25,1.00,0.50,0.75],
        ps=[1,10,20,30], pmod=[0.998,0.731,0.251,0.040]),
    "cot seed1 (kl.02)": dict(
        c="tab:cyan",   ms=[0,10,20,30],     mean=[0.625,0.938,0.875,0.938],
        ss=[0,10,20,30], slope=[0.25,0.00,0.25,0.00],
        ps=[1,10,20,30], pmod=[0.687,0.603,0.579,0.670]),
    "cot kl0": dict(
        c="tab:purple", ms=[0,10,20,30],     mean=[0.625,0.625,0.500,0.500],
        ss=[0,10,20,30], slope=[0.25,0.50,0.75,0.25],
        ps=[1,10,20,30], pmod=[0.998,0.500,0.617,0.074]),
    "A2 self-consist (clean)": dict(
        c="tab:green",  ms=[0,10,20,30],     mean=[0.875,0.812,0.625,0.625],
        ss=[0,10,20,30], slope=[0.25,0.50,1.00,0.50],
        ps=[1,10,20,30], pmod=[1.000,0.917,0.333,0.500]),
    "lag=3 regen": dict(
        c="tab:orange", ms=[0,10,20,30,40],  mean=[0.812,0.638,0.646,0.464,0.424],
        ss=[0,10,20,30,40], slope=[0.50,0.45,0.50,0.00,0.45],
        ps=[1,10,20,30,40], pmod=[0.917,0.750,0.333,0.500,0.194],
        confound_steps=[20,40]),                       # invalid 0.17 (truncation) -> quarantine
    "lag=0 regen (control)": dict(
        c="tab:red",    ms=None, mean=None,            # mean_K endpoints only in results.md
        ss=[0,10,20,30,40], slope=[0.50,0.36,0.34,-0.12,0.12],
        ps=[1,10,20,30,40], pmod=[0.917,0.750,0.111,0.083,0.000]),
}
# exogenous-EV paired (reward = EV at the STATED p; no self-prediction) -> does NOT collapse
EXO = dict(name="exogenous-EV paired (no self-pred)", c="black",
           ms=[0,10,20,30,40], mean=[0.696,0.703,0.797,0.656,0.562],
           ss=[0,10,20,30,40], slope=[0.43,0.62,0.62,0.62,0.25])

# ---- figure -----------------------------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(11.0, 13.2), sharex=True)

def plot_run(ax, xs, ys, d, label=None, lw=1.9, z=3):
    ax.plot(xs, ys, marker="o", ms=6, lw=lw, color=d["c"], label=label, alpha=0.92, zorder=z)

# Panel A — overall lean
for name, d in SELF.items():
    if d["mean"] is not None:
        plot_run(axA, d["ms"], d["mean"], d, label=name)
plot_run(axA, EXO["ms"], EXO["mean"], EXO, label=EXO["name"], lw=3.2, z=6)
axA.axhline(0.5, ls=":", color="grey", lw=1)
axA.text(40.3, 0.5, "coin-flip", va="center", fontsize=8, color="grey")
axA.set_ylim(0, 1.02); axA.set_ylabel("mean one-box rate (greedy eval)")
axA.set_title("A · overall lean vs step  —  self-referential runs drift into flat (≈0 / ≈1) basins; "
              "exogenous stays mid", fontsize=10.5, loc="left")
# proxy handle so lag=0 (which has no mean_K series) still appears in the shared legend
from matplotlib.lines import Line2D
handles, labels = axA.get_legend_handles_labels()
handles.append(Line2D([0], [0], color=SELF["lag=0 regen (control)"]["c"], marker="o", lw=1.9))
labels.append("lag=0 regen (control)  [slope/p_model only]")
axA.legend(handles, labels, fontsize=8, ncol=2, loc="lower left", framealpha=0.9)

# Panel B — conditional slope (the p*-tracking signal)
for name, d in SELF.items():
    plot_run(axB, d["ss"], d["slope"], d, label=name)
    for cs in d.get("confound_steps", []):
        i = d["ss"].index(cs)
        axB.plot(cs, d["slope"][i], marker="o", ms=12, mfc="none", mec="crimson", mew=1.8, zorder=7)
plot_run(axB, EXO["ss"], EXO["slope"], EXO, label=EXO["name"], lw=3.2, z=6)
axB.axhline(0.0, color="black", lw=1)
axB.fill_between([-2, 42], 0, 1.15, color="green", alpha=0.04)
axB.text(0.4, 1.09, "above 0  ⇒  tracks p  (one-boxes more at high p)", fontsize=8.5, color="green")
axB.text(0.4, -0.18, "≈ 0  ⇒  flat in p  (no conditional rule)", fontsize=8.5, color="dimgrey")
axB.text(45.5, -0.27, "○ = truncation-confounded (lag=3, invalid 0.17)",
         fontsize=7.5, color="crimson", ha="right")
axB.set_ylim(-0.32, 1.18); axB.set_ylabel("conditional slope  (K@p_hi − K@p_lo)")
axB.set_title("B · conditional slope vs step  —  wobbles, never stabilises "
              "(the +1.0 spikes are transients)", fontsize=10.5, loc="left")

# Panel C — predictor accuracy p_model (self-referential only)
for name, d in SELF.items():
    plot_run(axC, d["ps"], d["pmod"], d, label=name)
    for cs in d.get("confound_steps", []):
        i = d["ps"].index(cs)
        axC.plot(cs, d["pmod"][i], marker="o", ms=12, mfc="none", mec="crimson", mew=1.8, zorder=7)
for y, txt in [(1.0, "one-box basin (attractor)"), (0.0, "two-box basin (attractor)")]:
    axC.axhline(y, ls=":", color="grey", lw=1)
    axC.text(40.3, y, txt, va="center", fontsize=8, color="grey")
axC.axhline(0.8, ls="--", color="firebrick", lw=1.2)
axC.text(40.3, 0.8, "p*=0.8 (repeller)", va="center", fontsize=8, color="firebrick")
axC.set_ylim(-0.03, 1.05); axC.set_ylabel("predictor accuracy  p_model\n(= policy one-box rate)")
axC.set_xlabel("training step")
axC.set_title("C · predictor accuracy vs step  —  co-moves with the lean into a seed-selected basin "
              "(exogenous excluded: no self-predictor)", fontsize=10.5, loc="left")
axC.set_xlim(-1.5, 46)

fig.suptitle("R1 self-play & exogenous RLOO — dynamics over training\n"
             "(reconstructed from results.md eval-checkpoint tables; raw logs & adapters not retained)",
             fontsize=12.5, y=0.997)
fig.text(0.5, 0.005,
         "Eval n=4 items (cot/A2/lag) or n=8 (exogenous), single seed → individual wiggles are within "
         "the noise floor; read trajectory shape, not point values.  "
         "Self-referential reward (predictor=own one-box rate) ⇒ double-well ⇒ monotone slide into a "
         "flat basin.  Exogenous-EV reward (stated p) ⇒ no well ⇒ slope stays positive (no collapse).",
         ha="center", fontsize=8, color="dimgrey", wrap=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.975])
out = "results/selfplay_dynamics_combined.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
