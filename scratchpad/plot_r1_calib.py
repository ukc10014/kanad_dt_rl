"""Plot the R1 self-snapshot iterated-game result (calib + 2 fanouts). CPU-only.

Headline: the conditional rule does NOT stabilize — the loop drifts into the SAME flat
self-fulfilling basins as the 3B, seed-selected (seed1 -> one-box-flat; seed0 -> two-box-ward).
Heavily confounded by predictor truncation (</think> closed only 13-41% at the 2048 budget).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps = [0, 10, 20, 30]
RUNS = {
    "calib (seed0, kl .02)": dict(c="tab:blue", mk="o",
        mean=[0.625, 0.375, 0.375, 0.312], slope=[0.25, 1.00, 0.50, 0.75]),
    "seed1 (seed1, kl .02)": dict(c="tab:green", mk="s",
        mean=[0.625, 0.938, 0.875, 0.938], slope=[0.25, 0.00, 0.25, 0.00]),
    "kl0 (seed0, kl 0)": dict(c="tab:orange", mk="^",
        mean=[0.625, 0.625, 0.500, 0.500], slope=[0.25, 0.50, 0.75, 0.25]),
}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

for name, d in RUNS.items():
    axA.plot(steps, d["mean"], marker=d["mk"], color=d["c"], lw=2, ms=7, label=name)
axA.axhline(0.5, ls=":", color="gray", lw=1)
axA.annotate("one-box basin", (2, 0.96), color="tab:green", fontsize=9)
axA.annotate("two-box-ward", (2, 0.27), color="tab:blue", fontsize=9)
axA.set_title("A · overall lean (eval mean K-rate) vs step\n→ drifts into seed-selected flat basins", fontsize=11)
axA.set_xlabel("training step"); axA.set_ylabel("mean one-box rate (greedy eval)")
axA.set_ylim(-0.05, 1.05); axA.legend(fontsize=9, loc="center left"); axA.grid(alpha=0.3)

for name, d in RUNS.items():
    axB.plot(steps, d["slope"], marker=d["mk"], color=d["c"], lw=2, ms=7, label=name)
axB.axhline(0.0, ls=":", color="gray", lw=1)
axB.set_title("B · conditional slope (K@p0.99 − K@p0.5) vs step\n→ wobbles, never stabilizes (the +1.0 was a transient)", fontsize=11)
axB.set_xlabel("training step"); axB.set_ylabel("eval slope (hi − lo)")
axB.set_ylim(-0.15, 1.1); axB.legend(fontsize=9, loc="upper right"); axB.grid(alpha=0.3)

fig.suptitle("R1 self-snapshot iterated game — no stable conditional fixed point "
             "(CONFOUNDED: predictor </think>-closed only 13–41% at 2048-tok budget)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("results/r1_calib_dynamics.png", dpi=130, bbox_inches="tight")
print("wrote results/r1_calib_dynamics.png")
