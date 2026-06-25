"""Build the week's results deck -> newcomb_deck.pptx (upload to Google Slides).

Figures (matplotlib) + native pptx tables/text. Slide map (per user):
 1 headline claim (spine) + mechanism money-shot
 2 hypothesis-elimination tree + intercept-vs-slope focus box
 3 capability/disposition matrix + 4-arm intercept table
 4 K-rate(p) overlay
 5 transplant heatmap (14B)
 6 R1 iterated-game: setup + preliminary (run in progress)
 7 mechanism-credibility: abstract vs binding prompt + m0->m3 table
 8 caveats / confounders / further work
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

FIG = "results/deck"; os.makedirs(FIG, exist_ok=True)
NAVY = RGBColor(0x1F, 0x2D, 0x3D); ACC = RGBColor(0x2E, 0x5C, 0x8A); GREY = RGBColor(0x55, 0x55, 0x55)

# ---------------------------------------------------------------- figures
def _box(ax, x, y, w, h, text, fc, fs=11, ec="#444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

def fig_moneyshot(path):
    fig, ax = plt.subplots(figsize=(12.4, 2.5)); ax.axis("off"); ax.set_xlim(0, 12.4); ax.set_ylim(0, 2.5)
    xs = [0.1, 3.35, 6.75, 10.3]; w = [3.0, 3.15, 3.3, 2.0]; h = 1.7; y = 0.45
    txt = ["Newcomb prompt\n(stated accuracy p)",
           "Model REPRESENTS the EV /\nstates P(full|action)=2p−1\n**correctly**",
           "Commitment step:\nDOMINANCE pull\n(“don't forgo the\nguaranteed box”)\nOVERRIDES",
           "→ two-box\n(CDT)"]
    fc = ["#eaeef5", "#e7f5e9", "#fdeaea", "#fff2e0"]
    for x, ww, t, c in zip(xs, w, txt, fc):
        _box(ax, x, y, ww, h, t.replace("**", ""), c, fs=11.5)
    for i in range(3):
        x0 = xs[i] + w[i]; x1 = xs[i + 1]
        ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2),
                     arrowstyle="-|>", mutation_scale=18, lw=1.6, color="#333"))
    ax.text(6.2, 2.35, "the override happens AFTER correct representation — that is the whole claim",
            ha="center", fontsize=10.5, style="italic", color="#555")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def fig_tree(path):
    fig, ax = plt.subplots(figsize=(9.6, 6.6)); ax.axis("off"); ax.set_xlim(0, 9.6); ax.set_ylim(0, 6.6)
    nodes = [
        ("Flat K-rate(p): the model doesn't track p", "#eaeef5", ""),
        ("Just a format limit? (single-token)", "#fff", "✗  flat under CoT too (de-noised)"),
        ("Can't compute the EV?", "#fff", "✗  refuses even when handed the EV"),
        ("Fixed by scale? (14B)", "#fff", "✗  worse — more committed CDT"),
        ("A framing artifact? (predictor credibility)", "#fff", "~  partly (modest, ≤10 pts)"),
        ("Dissolved by reasoning? (R1-Distill)", "#e7f5e9", "✓  tracks p, crossover at p*"),
    ]
    y = 6.0; bh = 0.62; gap = 0.34; bx = 0.2; bw = 5.4
    cy = []
    for t, c, v in nodes:
        _box(ax, bx, y, bw, bh, t, c, fs=11)
        if v:
            col = "#1b7a2f" if v.startswith("✓") else ("#b06a00" if v.startswith("~") else "#9c2a2a")
            ax.text(bx + bw + 0.25, y + bh / 2, v, ha="left", va="center", fontsize=10.3, color=col)
        cy.append(y); y -= (bh + gap)
    for i in range(len(nodes) - 1):
        ax.add_patch(FancyArrowPatch((bx + bw / 2, cy[i]), (bx + bw / 2, cy[i + 1] + bh),
                     arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#555"))
    _box(ax, 0.2, y - 0.15, 9.0, 0.95,
         "⇒ a reasoning-dissolvable DISPOSITION at the commitment step\n"
         "(not a capability gap) — next: can an environment install the rule?  R1 iterated game (running)",
         "#eef3fb", fs=11, ec="#2E5C8A")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def fig_krate(path):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    base_p = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99]
    base = [0.90, 0.95, 0.90, 0.80, 0.70, 0.80, 0.70, 0.90]
    caus = [0.0] * 8; evid = [0.50, 0.45, 0.50, 0.40, 0.50, 0.50, 0.45, 0.35]
    r1_p = [0.5, 0.7, 0.8, 0.9, 0.99]; r1 = [0.08, 0.50, 0.58, 0.83, 0.75]
    ax.plot(base_p, base, "o--", color="tab:blue", lw=2, label="3B base — flat-high (reflex)")
    ax.plot(base_p, evid, "s--", color="tab:orange", lw=2, label="3B evidential-RL — flat ~0.5")
    ax.plot(base_p, caus, "v--", color="tab:red", lw=2, label="3B causal-RL — flat-zero")
    ax.plot(r1_p, r1, "D-", color="tab:green", lw=2.6, ms=8, label="R1-8B (reasoning) — TRACKS p")
    ax.axvline(0.8, ls=":", color="grey"); ax.text(0.805, 0.04, "p*=0.8", color="grey", fontsize=9)
    ax.set_xlabel("stated predictor accuracy p"); ax.set_ylabel("K-rate (one-box / non-CDT rate)")
    ax.set_ylim(-0.05, 1.05); ax.set_title("Everything is flat in p — except the reasoning model")
    ax.legend(fontsize=9, loc="center left"); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def fig_heatmap(path):
    import numpy as np
    conds = ["none", "variables", "formulas", "numeric_evs", "comparison", "full"]
    ps = [0.5, 0.6, 0.7, 0.75, 0.85, 0.9, 0.99]
    M = {
        "none": [.527, .382, .590, .471, .431, .427, .550],
        "variables": [.711, .652, .774, .747, .179, .146, .174],
        "formulas": [.924, .895, .927, .920, .176, .047, .002],
        "numeric_evs": [1.0, .629, 1.0, 1.0, .373, .316, .000],
        "comparison": [.923, .833, .924, .917, .667, .956, .585],
        "full": [1.0, 1.0, 1.0, 1.0, .666, .779, .146],
    }
    A = np.array([M[c] for c in conds])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    im = ax.imshow(A, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ps))); ax.set_xticklabels(ps); ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds)
    for i in range(len(conds)):
        for j in range(len(ps)):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if 0.25 < A[i, j] < 0.8 else "white")
    ax.set_xlabel("stated p   (EV-optimal: two-box ← | → one-box)"); ax.set_ylabel("EV aid supplied")
    ax.set_title("14B handed the EV: P(EV-optimal action)\nrefuses to one-box at high p even with the full calc")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(optimal)")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def fig_r1game(path):
    fig, ax = plt.subplots(figsize=(9.4, 4.0)); ax.axis("off"); ax.set_xlim(0, 9.4); ax.set_ylim(0, 4.0)
    _box(ax, 0.3, 1.4, 3.0, 1.3, "POLICY (R1)\nreasons → acts\n(one-box / two-box)", "#e7f5e9", 11)
    _box(ax, 6.1, 1.4, 3.0, 1.3, "PREDICTOR =\nlagged snapshot of policy\nREASONS → predicts", "#eaeef5", 11)
    ax.add_patch(FancyArrowPatch((3.3, 2.3), (6.1, 2.3), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333"))
    ax.text(4.7, 2.55, "action", ha="center", fontsize=9.5)
    ax.add_patch(FancyArrowPatch((6.1, 1.7), (3.3, 1.7), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333"))
    ax.text(4.7, 1.35, "fills box w/ accuracy p_eff = P_pred(one-box)", ha="center", fontsize=9.5)
    ax.text(4.7, 3.6, "Because the predictor REASONS, p_eff tracks stated p → reward is conditional from step 0",
            ha="center", fontsize=10.5, style="italic", color="#555")
    ax.text(4.7, 0.5, "Q: is the conditional rule (one-box high p / two-box low p) a self-consistent fixed point?",
            ha="center", fontsize=10.5, color="#2E5C8A")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

for fn, p in [(fig_moneyshot, "moneyshot"), (fig_tree, "tree"), (fig_krate, "krate"),
              (fig_heatmap, "heatmap"), (fig_r1game, "r1game")]:
    fn(f"{FIG}/{p}.png")
print("figures done")

# ---------------------------------------------------------------- pptx
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide(title, sub=None):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.4), Inches(0.95)).text_frame
    tb.word_wrap = True; p = tb.paragraphs[0]; p.text = title
    p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = NAVY
    if sub:
        sp = tb.add_paragraph(); sp.text = sub; sp.font.size = Pt(13); sp.font.italic = True; sp.font.color.rgb = GREY
    return s

def pic(s, path, x, y, w):
    s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))

def bullets(s, items, x, y, w, h, size=15):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame; tf.word_wrap = True
    for i, (lvl, t) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "    – ") + t
        p.font.size = Pt(size if lvl == 0 else size - 2); p.font.color.rgb = NAVY if lvl == 0 else GREY
        p.space_after = Pt(4)

def table(s, rows, x, y, w, h, header=True, fs=12, colw=None):
    nr = len(rows); nc = len(rows[0])
    t = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if colw:
        for j, cw in enumerate(colw):
            t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = str(val); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = c.text_frame.paragraphs[0]; para.font.size = Pt(fs)
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            if header and i == 0:
                para.font.bold = True; para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                c.fill.solid(); c.fill.fore_color.rgb = ACC
            else:
                c.text_frame.paragraphs[0].font.color.rgb = NAVY
    return t

def focusbox(s, text, x, y, w, h):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = sh.text_frame; tf.word_wrap = True
    sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xEE, 0xF3, 0xFB)
    sh.line.color.rgb = ACC; sh.line.width = Pt(1.5)
    p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(13.5); p.font.color.rgb = NAVY

# S1 — headline + money-shot
s = slide("The claim", "Newcomb decision-theory in small/mid LLMs — week-in-review")
fb = s.shapes.add_textbox(Inches(0.5), Inches(1.25), Inches(12.3), Inches(1.7)).text_frame; fb.word_wrap = True
p = fb.paragraphs[0]
p.text = ("A small/mid LLM doesn't fail Newcomb because it can't do the math — it fails because of a "
          "DISPOSITION at the action-commitment step that overrides explicit computation, sharpens with "
          "scale, and is dissolved only by test-time reasoning.")
p.font.size = Pt(19); p.font.bold = True; p.font.color.rgb = NAVY
p2 = fb.add_paragraph(); p2.text = ("Competence (slope) and disposition (intercept) are orthogonal and "
          "separately movable: RL moves the lean, not the rule."); p2.font.size = Pt(14); p2.font.color.rgb = GREY
pic(s, f"{FIG}/moneyshot.png", 0.5, 3.4, 12.3)

# S2 — hypothesis tree + focus box
s = slide("How we got there — every alternative explanation, tested and killed")
pic(s, f"{FIG}/tree.png", 0.3, 1.2, 8.4)
focusbox(s, ("INTERCEPT vs SLOPE.\n\nAcross every RL arm the model's overall LEAN (intercept) moves "
             "freely — any direction, even against its prior — but the p-conditional RULE (slope) never "
             "forms.\n\nRL elicits/reweights latent disposition; it does not install competence.\n\n"
             "→ confirmed at the logit level, per-item, and under a fair conditioning-only objective."),
         8.95, 1.4, 4.05, 5.4)

# S3 — capability matrix + 4-arm
s = slide("Disposition ≠ capability — two receipts")
bullets(s, [(0, "Capability / disposition matrix (the model can; it won't):")], 0.5, 1.15, 12, 0.4, size=14)
table(s, [
    ["model", "no-think reflex", "tracks p when reasoning?", "1-box @ p=.99 given full EV", "verdict"],
    ["3B", "~0.83 one-box", "no (flat, de-noised)", "~0.53", "baseline"],
    ["14B", "~0.50", "weak (+0.09, n.s.)", "0–15%  (worse!)", "scale ⇒ more CDT"],
    ["R1-8B", "1.00 one-box", "YES (slope, crossover p*)", "— follows EV in CoT", "reasoning ⇒ EV-rational"],
], 0.5, 1.6, 12.3, 1.6, fs=12.5, colw=[1.2, 2.2, 3.0, 3.2, 2.7])
bullets(s, [(0, "RL moves only the intercept (logit level) — four arms, four levels, slope ≈ 0:")], 0.5, 3.7, 12, 0.4, size=14)
table(s, [
    ["RL arm", "P(non-CDT)", "logit margin", "slope in p", "what RL did"],
    ["base", "0.805", "+3.46", "≈ 0", "—"],
    ["causal", "0.000", "−18.16", "≈ 0", "saturated two-box"],
    ["evidential", "0.457", "−0.51", "≈ 0", "pushed to indifference"],
    ["modelpred", "0.718", "+8.36", "≈ 0", "sharpened one-box"],
], 0.5, 4.15, 12.3, 2.0, fs=12.5, colw=[2.0, 2.2, 2.2, 1.8, 4.1])

# S4 — K-rate overlay
s = slide("RL moves the level, never the shape — only reasoning bends the curve")
pic(s, f"{FIG}/krate.png", 2.4, 1.35, 8.5)
bullets(s, [(0, "Flat in p = recitation/disposition; a step at p* = structural EV reasoning. "
             "Only R1 (reasoning) tracks p, with the crossover at the theoretical p*=0.8.")], 0.6, 6.7, 12, 0.6, size=13)

# S5 — transplant heatmap
s = slide("“We handed it the answer and it still wouldn't act on it”")
pic(s, f"{FIG}/heatmap.png", 2.5, 1.35, 8.3)
bullets(s, [(0, "Supplying the EV raises P(optimal) at LOW p (where EV agrees with grabbing the guaranteed box) "
             "but FLIPS it negative at HIGH p — the 14B one-boxes 0–15% at p=0.99 even told “EV(one-box) "
             "> EV(two-box)”. Spelling out the math is counter-productive: it surfaces the guaranteed reward.")],
        0.6, 6.7, 12.1, 0.6, size=12.5)

# S6 — R1 iterated game
s = slide("What would INSTALL the rule? — the R1 self-snapshot game", "running now; preliminary")
pic(s, f"{FIG}/r1game.png", 0.4, 1.3, 8.0)
bullets(s, [
    (0, "Predictor = a lagging copy of the policy that ALSO reasons → p_eff tracks p (the lever the p-blind 3B lacked)."),
    (0, "0b confirmed it: R1's reasoned prediction tracks p, slope +0.50."),
    (0, "Preliminary (step 0): base R1 already conditional — K@p0.5=0.50, K@p0.99=0.75, slope +0.25."),
    (0, "Full 30-step trajectory in progress → does the conditional rule PERSIST (positive) or COLLAPSE to a uniform lean?"),
    (1, "3B analogue (built): two self-fulfilling basins; the R1 bet is a *conditional* fixed point."),
], 8.6, 1.4, 4.4, 5.0, size=12.5)

# S7 — mechanism credibility
s = slide("Was the CDT lean just a prompt artifact? — partly, but modest")
bullets(s, [
    (0, "Abstract (m0): “the process correctly identifies the choices of agents like you X% of the time” "
        "— a population statistic a causal reasoner is RIGHT to treat as non-binding."),
    (0, "Binding (m3): “…executes an exact copy of your own decision procedure on this very problem…” "
        "— the honest framing for a self-snapshot predictor."),
], 0.5, 1.2, 12.3, 1.6, size=13.5)
table(s, [
    ["predictor framing", "one-box rate", "credence gap_adj (→p-tracking)"],
    ["m0  statistical (abstract)", "0.59", "0.21"],
    ["m0pad  length placebo", "0.59", "0.17"],
    ["m1  individual model", "0.61", "0.19"],
    ["m2  process-scan", "0.63", "0.30"],
    ["m3  exact-copy (binding)", "0.69", "0.31"],
], 0.5, 3.05, 9.5, 2.4, fs=13, colw=[4.5, 2.5, 2.5])
focusbox(s, ("Credibility lifts EDT m0→m3 (placebo-controlled) — but only ~10 pts.\n\nDominance "
             "SURVIVES the most binding predictor (softened, not dissolved). Framing explains a slice; "
             "the disposition is robust to credibility."), 10.2, 3.05, 2.8, 2.4)

# S8 — caveats / further work
s = slide("Caveats, confounders & next experiments")
bullets(s, [
    (0, "CAVEATS / CONFOUNDERS"),
    (1, "Small/mid models; several single-run n≈12–20 cells (14B especially)."),
    (1, "14B>3B-CDT direction suggestive, mechanism open (sharper deliberation? tuning? noise?)."),
    (1, "Abstract-prompt confound real but modest (≤10 pts)."),
    (1, "“reflex-EDT / CoT-CDT” is a small-model fact: reflex drifts off EDT with scale, and R1 "
        "*reasoning* ≠ 3B *CoT* (different operations) — “deliberation→CDT” is not scale-invariant."),
    (1, "Discipline: many apparent effects dissolved under de-noising (+0.50 CoT slope, +0.16 SFT slope, "
        "causal-flip = KL artifact). Reported claims are the survivors."),
    (0, "FURTHER WORK"),
    (1, "R1 self-snapshot iterated game (running) — does self-reference install/stabilize the conditional rule?"),
    (1, "Seeded in-family CDT ladder (0.5B→32B) — confirm/explain 14B>3B; cuts against capability→EDT."),
    (1, "Reflex × CoT × native-reasoning matrix; anti-Newcomb camouflage (Newcomb-story vs genuine EV-action failure)."),
], 0.5, 1.2, 12.4, 5.8, size=14)

out = "newcomb_deck.pptx"; prs.save(out)
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
