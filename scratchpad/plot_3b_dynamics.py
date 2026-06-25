"""Plot 3B RL training dynamics (mean_K vs step) from the in-loop eval logs. CPU-only.

Two panels:
  A — self-snapshot ITERATED GAME (predictor = lagging copy of policy), p pinned at p*=0.8:
      bistability — committed seeds hold, the on-the-fence base settles at indifference.
  B — ORACLE ANCHORS (predictor accuracy pinned to extremes), the loop as a validated instrument:
      drives one-boxing iff one-boxing is EV-optimal, in both directions.
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"step=(\d+) eval mean_K=([0-9.]+)")


def traj(path):
    seen, xs, ys = set(), [], []
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                s = int(m.group(1))
                if s in seen:
                    continue
                seen.add(s)
                xs.append(s)
                ys.append(float(m.group(2)))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


PANEL_A = [
    ("one-box seed → locks at 1.0", "results/run_hyst_onebox_seedref.log", "tab:green", "o"),
    ("base seed (on separatrix) → indifference", "results/selfsnapshot/hyst_base_seedref.log", "tab:blue", "s"),
    ("causal/two-box seed → locks at 0.0", "results/selfsnapshot/hyst_causal_seedref.log", "tab:red", "^"),
]
PANEL_B = [
    ("p=1.0 base — perfect ⇒ one-box optimal", "results/oracle/oracle_p1_base.log", "tab:green", "o"),
    ("p=0.5 base — coin ⇒ two-box optimal", "results/oracle/oracle_p05_base.log", "tab:orange", "s"),
    ("p=0.0 base — inverted ⇒ two-box optimal", "results/oracle/oracle_p0_base.log", "tab:red", "^"),
    ("p=1.0 causal-seed — the flip (KL-suspect)", "results/oracle/oracle_p1_causal.log", "tab:purple", "D"),
]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

for label, path, color, mk in PANEL_A:
    xs, ys = traj(path)
    axA.plot(xs, ys, marker=mk, color=color, label=label, lw=2, ms=6)
axA.axhline(0.5, ls=":", color="gray", lw=1)
axA.text(2, 0.52, "separatrix / indifference (p=p*)", color="gray", fontsize=8)
axA.set_title("A · 3B iterated game (self-snapshot predictor), p pinned at p*=0.8\nbistability: two self-fulfilling basins", fontsize=10)
axA.set_xlabel("training step")
axA.set_ylabel("mean K-rate (one-box rate), greedy in-loop eval")
axA.set_ylim(-0.05, 1.05)
axA.legend(fontsize=8, loc="center right")
axA.grid(alpha=0.3)

for label, path, color, mk in PANEL_B:
    xs, ys = traj(path)
    axB.plot(xs, ys, marker=mk, color=color, label=label, lw=2, ms=6)
axB.set_title("B · 3B oracle anchors (predictor accuracy pinned)\nthe loop drives the policy to the EV-optimal action", fontsize=10)
axB.set_xlabel("training step")
axB.legend(fontsize=8, loc="center right")
axB.grid(alpha=0.3)

fig.suptitle("3B Newcomb-RL training dynamics (mean one-box rate vs step)", fontsize=12, y=1.0)
fig.tight_layout()
out = "results/dynamics_3b.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")

# ---- companion: kl-control (the causal flip is the KL leash, not the reward) ----
KL = [
    ("kl=0.02 (KL leash on) → flips to one-box", "results/oracle/run_oracle_klctl_kl002.log", "tab:purple", "D"),
    ("kl=0 (control) → stays two-box", "results/oracle/run_oracle_klctl_kl00.log", "tab:brown", "x"),
]
fig2, ax = plt.subplots(figsize=(7, 5))
for label, path, color, mk in KL:
    xs, ys = traj(path)
    ax.plot(xs, ys, marker=mk, color=color, label=label, lw=2, ms=8)
ax.set_title("kl-control: the causal→one-box flip is the KL-to-base leash, not the reward\n"
             "(seed = causal two-boxer, predictor pinned p=1.0 ⇒ one-box is EV-optimal)", fontsize=9)
ax.set_xlabel("training step")
ax.set_ylabel("mean K-rate (one-box rate)")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9, loc="center right")
ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("results/dynamics_klcontrol.png", dpi=130, bbox_inches="tight")
print("wrote results/dynamics_klcontrol.png")
