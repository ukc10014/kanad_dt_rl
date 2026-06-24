"""Credence probe — is the action↔box-state dependence *represented but unused*?

Sibling to ``logprob_sweep.py``. Where that instrument reads the two **action** tokens at the
decision point (and finds them flat in ``p``), this one teacher-forces a *committed action* and then
reads the model's **credence about the box state** — does the conditional prize end up full?

For a perfect evidential reasoner, committing to one-box is evidence the predictor (accuracy ``p``)
forecast one-box, so::

    P(conditional prize full | I one-box) = p
    P(conditional prize full | I two-box) = 1 - p
    credence gap  g(p) = P(full|one-box) - P(full|two-box) = 2p - 1   (0 @ p=.5 -> ~.98 @ p=.99)

A causal / non-representing reasoner treats the box as a fixed unknown independent of the action, so
``g(p) ≈ 0``, flat. Overlaying the (flat) action margin from ``logprob_sweep`` then distinguishes:

  * gap rises with p (tracks 2p-1) while action stays flat  -> **represented-but-unused**
    (the belief is there; the decision step ignores it -> lever is steering, not more outcome RL).
  * gap also flat                                            -> **not represented**
    (must install the credence first; fine-tuning the answer to EDT is ill-posed).

Two methodologically-distinct elicitations, both with the same ``2p-1`` reference:
  * ``outcome``     — the named conditional prize "will hold exactly 100 / 0" (value tokens literal
                      in every prompt). The prize entity differs per item; we derive it.
  * ``prediction``  — abstract-label proposition "the system predicted I would take only <LABEL>";
                      True / False. Fully abstract-token-pure.

Scoring uses ``ModelWrapper.answer_2way`` (merge-safe first-divergent-token), so unequal-length
strings like "100" vs "0" are compared as single logits — no summed-logprob length bias.

**Invariant #1 (CLAUDE.md):** the scorer compares two chosen credence tokens; it never string-matches
"one-box"/"two-box"/"Newcomb"/"predictor". The committed-action surface text is the dataset's own
``item.non_cdt_text`` / ``item.cdt_text``; this module authors none of the forbidden strings. Pure,
measurement-only — forward passes, no weight updates.

    python -m newcomb_eval.credence_probe --model Qwen/Qwen2.5-0.5B-Instruct --limit 2 \
        --p-grid 0.5 0.8 0.99 --variant outcome
    python -m newcomb_eval.credence_probe --model Qwen/Qwen2.5-14B-Instruct --tag base14b \
        --variant outcome --action-margin results/logprob/p_margin_by_p_base14b.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re

from .config import EvalConfig
from .crossover import crossover_for_item
from .data.loader import load
from .item_analysis import bootstrap_mean_ci, ols_slope
from .logprob_sweep import roles_to_tokens
from .prompts import build_prompt, render_scenario_block

# action keys -> (which seeded label, which dataset option text)
ACTIONS = ("major_alone", "both")
# Committed-action statement. Surface text comes from the dataset option, so this module never
# authors "one-box"/"two-box"/etc. (invariant #1). {label} is the seeded abstract token.
COMMIT_TEMPLATE = "I have decided. My final and irrevocable choice is option {label}: {option_text}"

# Saturation guard: a credence pinned this close to 0/1 regardless of action/p is a broken
# measurement (flat-because-degenerate), not a real null.
_SAT_EPS = 0.02


# --------------------------------------------------------------------------------------
# conditional-prize entity extraction (for the ``outcome`` variant)
# --------------------------------------------------------------------------------------
_PRIZE_STOP = {"is", "are", "was", "were", "holds", "hold", "contains", "contain",
               "worth", "holding", "has", "have"}


def conditional_prize_name(scenario: str) -> str | None:
    """Name of the conditional (big) prize — the entity that 'holds 100 points if ...'.

    Differs per item ('Container Q', 'Box K', 'the MAJOR prize', 'Account HI', ...). We take the
    sentence fragment immediately before '100 points if' and strip the trailing copula/verb.
    Returns None if the marker phrase is absent (caller should skip the outcome variant then).
    """
    key = "100 points if"
    idx = scenario.find(key)
    if idx == -1:
        return None
    prefix = scenario[:idx].rstrip()
    for sep in (". ", "? ", "! "):
        if sep in prefix:
            prefix = prefix.rsplit(sep, 1)[1]
    words = prefix.split()
    while words and words[-1].lower() in _PRIZE_STOP:
        words.pop()
    return " ".join(words) if words else None


# --------------------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------------------
def _action_label_text(block, item, action: str) -> tuple[str, str]:
    """(seeded abstract label, dataset option surface text) for a committed action."""
    non_cdt, cdt = roles_to_tokens(block.token_role)
    if action == "major_alone":
        return non_cdt, item.non_cdt_text
    return cdt, item.cdt_text


def credence_cue(variant: str, phrasing: str, *, onebox_label: str,
                 entity: str | None) -> tuple[str, str, str]:
    """Return (cue_text, tok_full, tok_empty) for a (variant, phrasing).

    ``tok_full`` is the token whose probability is the model's credence that the conditional prize
    is full (== predictor forecast one-box). Credence is read as the 2-way softmax of
    (logit(tok_full), logit(tok_empty)).
    """
    if variant == "outcome":
        ent = entity or "the conditional prize"
        cues = {
            "report":   f"Given that choice, {ent} will turn out to hold exactly ",
            "revealed": f"The contents are now revealed: {ent} holds exactly ",
            "observer": f"An observer who can see inside would say {ent} holds exactly ",
        }
        return cues[phrasing], "100", "0"
    if variant == "prediction":
        # Fixed proposition (independent of the conditioned action): the system forecast one-box.
        cues = {
            "report":   (f"Given that choice, the system's earlier prediction was most likely that "
                         f"I would take only {onebox_label}. True or False? Answer: "),
            "revealed": (f"The system's stored prediction is now revealed; it had predicted I would "
                         f"take only {onebox_label}. True or False? Answer: "),
            "observer": (f"An observer with access to the stored prediction confirms it was "
                         f"'take only {onebox_label}'. True or False? Answer: "),
        }
        return cues[phrasing], "True", "False"
    raise ValueError(f"unknown variant {variant!r}")


def build_assistant_prefix(label: str, option_text: str, cue_text: str) -> str:
    """Teacher-forced assistant turn: committed action + credence cue (no trailing token)."""
    return COMMIT_TEMPLATE.format(label=label, option_text=option_text) + " " + cue_text


def build_credence_base(wrapper, block, item, *, action: str, phrasing: str,
                        variant: str, entity: str | None) -> tuple[str, str, str]:
    """Full chat-templated text up to (but excluding) the credence token + (tok_full, tok_empty)."""
    label, option_text = _action_label_text(block, item, action)
    onebox_label, _ = roles_to_tokens(block.token_role)
    cue_text, tok_full, tok_empty = credence_cue(
        variant, phrasing, onebox_label=onebox_label, entity=entity)
    prefix = build_assistant_prefix(label, option_text, cue_text)
    base = wrapper._format(block.text) + prefix
    return base, tok_full, tok_empty


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def score_credence(wrapper, base: str, tok_full: str, tok_empty: str) -> dict:
    """2-way credence that the conditional prize is full, via merge-safe answer_2way.

    Returns ``credence_full`` (= softmax over the two credence-token logits at the cue position),
    the raw ``margin`` logit(full)-logit(empty), and ``is_degenerate`` (the read collapsed / both
    tokens unresolved).
    """
    out = wrapper.answer_2way(base, tok_full, tok_empty)
    cf = out["p_non_cdt"]  # answer_2way labels its first token "non_cdt"; here that is tok_full
    is_degenerate = (cf != cf)  # NaN => a label didn't resolve to a token
    return {"credence_full": cf, "margin": out["margin"], "is_degenerate": bool(is_degenerate)}


# --------------------------------------------------------------------------------------
# direct (free-form numeric) elicitation — the coherence cross-check
# --------------------------------------------------------------------------------------
# A third, methodologically-distinct read: ask the model to *state* the probability as a number,
# then parse it. A coherent model's stated number should agree with its implicit token credence;
# disagreement = "gives the answer without a coherent world model". Uses the abstract-label
# proposition (no entity dependence) so it is always available.
DIRECT_CUE = (
    "Now, as a percentage from 0 to 100, what is the probability that the system had predicted "
    "I would take only {label}? Respond with only the number.\nProbability: "
)
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_probability(text: str) -> float | None:
    """First number in ``text`` -> probability in [0,1]. '85'/'85%'->0.85; '0.85'->0.85; '1%'->0.01.

    A trailing '%' means percentage REGARDLESS of magnitude (so '1%' -> 0.01, not 1.0 — the bug that
    corrupted the p=0.99 two-box credence). Without '%', >1 is read as a percentage ('99'->0.99) and
    <=1 as an already-formed probability ('0.8'->0.8).
    """
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    v = float(m.group(1))
    is_pct = text[m.end():m.end() + 1] == "%"
    if is_pct or v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def build_direct_base(wrapper, block, item, *, action: str) -> str:
    """Chat-templated base ending at 'Probability: ' (model is asked to emit a number)."""
    label, option_text = _action_label_text(block, item, action)
    onebox_label, _ = roles_to_tokens(block.token_role)
    cue = DIRECT_CUE.format(label=onebox_label)
    return wrapper._format(block.text) + build_assistant_prefix(label, option_text, cue)


def _generate_continuation(wrapper, base: str, max_new_tokens: int = 8) -> str:
    """Greedy continuation from an already-templated base (mirrors ModelWrapper.generate, but the
    base already carries the chat template + a teacher-forced assistant prefix)."""
    import torch
    with wrapper._lock:
        enc = wrapper.tokenizer(base, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(wrapper.model.device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = wrapper.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=wrapper.tokenizer.pad_token_id)
        new = gen[0, enc["input_ids"].shape[1]:]
        return wrapper.tokenizer.decode(new, skip_special_tokens=True)


def score_credence_direct(wrapper, base: str, *, max_new_tokens: int = 8) -> dict:
    raw = _generate_continuation(wrapper, base, max_new_tokens=max_new_tokens)
    p = parse_probability(raw)
    return {"credence_full": p if p is not None else float("nan"),
            "raw": raw, "is_degenerate": bool(p is None)}


# --------------------------------------------------------------------------------------
# aggregation (pure / CPU — the unit-test target)
# --------------------------------------------------------------------------------------
def _collapse_phrasings(rows):
    """(item_id, p, action) -> {mean credence over phrasings, phrasing SD, n}. Drops NaN credence."""
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        cf = float(r["credence_full"])
        if cf != cf:
            continue
        buckets.setdefault((r["item_id"], float(r["p"]), r["action"]), []).append(cf)
    import statistics
    out = {}
    for key, vals in buckets.items():
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[key] = {"mean": sum(vals) / len(vals), "sd": sd, "n": len(vals)}
    return out


def aggregate_credence(rows, *, p_star: float, action_margin_by_p: dict | None = None):
    """Per-p credence gap table + the 2p-1 reference. ``rows`` are per-sample dicts with at least
    item_id, p, action ('major_alone'|'both'), phrasing, credence_full.

    Columns: cred_full_major_alone_mean, cred_full_both_mean, gap_mean, gap_se, ref_2pm1,
    phrasing_sd_mean, sum_check_mean, is_degenerate, action_margin_mean, n_items.
    """
    import numpy as np
    import pandas as pd

    collapsed = _collapse_phrasings(rows)
    ps = sorted({float(r["p"]) for r in rows})
    out = []
    for p in ps:
        items = sorted({k[0] for k in collapsed if k[1] == p})
        gaps, ma, bo, sums, sds = [], [], [], [], []
        for it in items:
            cma = collapsed.get((it, p, "major_alone"))
            cbo = collapsed.get((it, p, "both"))
            if cma is None or cbo is None:
                continue
            ma.append(cma["mean"]); bo.append(cbo["mean"])
            gaps.append(cma["mean"] - cbo["mean"])
            sums.append(cma["mean"] + cbo["mean"])
            sds.append((cma["sd"] + cbo["sd"]) / 2.0)
        n = len(gaps)
        sqn = math.sqrt(n) if n else float("nan")
        gap_mean = float(np.mean(gaps)) if n else float("nan")
        ma_mean = float(np.mean(ma)) if n else float("nan")
        bo_mean = float(np.mean(bo)) if n else float("nan")
        # degenerate: both conditioned credences pinned at the same saturated value -> no signal
        degen = (n > 0 and abs(ma_mean - bo_mean) < _SAT_EPS
                 and (min(ma_mean, bo_mean) > 1 - _SAT_EPS or max(ma_mean, bo_mean) < _SAT_EPS))
        row = {
            "p": p,
            "cred_full_major_alone_mean": ma_mean,
            "cred_full_both_mean": bo_mean,
            "gap_mean": gap_mean,
            "gap_se": (float(np.std(gaps, ddof=1)) / sqn) if n > 1 else 0.0,
            "ref_2pm1": 2 * p - 1,
            "phrasing_sd_mean": float(np.mean(sds)) if n else float("nan"),
            "sum_check_mean": float(np.mean(sums)) if n else float("nan"),
            "is_degenerate": bool(degen),
            "n_items": n,
        }
        if action_margin_by_p is not None:
            row["action_margin_mean"] = action_margin_by_p.get(p, float("nan"))
        out.append(row)
    df = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
    # Baseline-subtract the p=0.5 pedestal: the committed action leaks a p-INDEPENDENT evidential
    # bias into the credence (gap@0.5 should be 0 but isn't). gap_adj isolates the p-MODULATION,
    # which is what "tracks 2p-1" actually requires. ref_2pm1 is unchanged (it is 0 at p=0.5).
    if len(df):
        base_p = 0.5 if (df["p"] == 0.5).any() else float(df["p"].iloc[0])
        base_gap = float(df.loc[df["p"] == base_p, "gap_mean"].iloc[0])
        df["gap_adj_mean"] = df["gap_mean"] - base_gap
    return df


def credence_headline(rows, *, p_star: float, action_margin_by_p: dict | None = None) -> dict:
    """Headline stats. Per item, OLS-fit the gap on x=(2p-1): slope ~1 (EDT) vs ~0 (CDT); bootstrap
    the per-item slope mean. Also gap_at_0.5 (symmetry control), hi-lo gap, and RMSE vs 2p-1.
    """
    import numpy as np

    collapsed = _collapse_phrasings(rows)
    ps = sorted({float(r["p"]) for r in rows})
    items = sorted({k[0] for k in collapsed})

    base_p = 0.5 if 0.5 in ps else ps[0]
    per_item_slope, per_item_slope_adj = [], []
    gap_by_p_accum = {p: [] for p in ps}
    ma_by_p, bo_by_p = {p: [] for p in ps}, {p: [] for p in ps}
    for it in items:
        xs, ys = [], []
        base_gap = None
        series = {}
        for p in ps:
            cma = collapsed.get((it, p, "major_alone"))
            cbo = collapsed.get((it, p, "both"))
            if cma is None or cbo is None:
                continue
            gap = cma["mean"] - cbo["mean"]
            series[p] = gap
            xs.append(2 * p - 1); ys.append(gap)
            gap_by_p_accum[p].append(gap)
            ma_by_p[p].append(cma["mean"]); bo_by_p[p].append(cbo["mean"])
            if p == base_p:
                base_gap = gap
        if len(xs) >= 3:
            per_item_slope.append(ols_slope(xs, ys).slope)
            if base_gap is not None:  # baseline-subtracted: isolate the p-modulation
                ys_adj = [series[p] - base_gap for p in sorted(series)]
                per_item_slope_adj.append(ols_slope([2 * p - 1 for p in sorted(series)], ys_adj).slope)

    mean_slope, lo, hi = bootstrap_mean_ci(per_item_slope)
    mean_slope_adj, lo_a, hi_a = bootstrap_mean_ci(per_item_slope_adj)
    gap_mean_by_p = {p: (float(np.mean(v)) if v else float("nan")) for p, v in gap_by_p_accum.items()}
    rmse = float(np.sqrt(np.mean([(gap_mean_by_p[p] - (2 * p - 1)) ** 2
                                  for p in ps if gap_mean_by_p[p] == gap_mean_by_p[p]])))
    lo_p, hi_p = ps[0], ps[-1]
    # one-sided saturation: a conditioned credence pinned near 0/1 across ALL p (p-blind) — the
    # forced-token failure the two-sided degeneracy gate misses. ma pinned high OR bo pinned low.
    ma_means = [float(np.mean(ma_by_p[p])) for p in ps if ma_by_p[p]]
    bo_means = [float(np.mean(bo_by_p[p])) for p in ps if bo_by_p[p]]
    ma_sat = bool(ma_means and min(ma_means) > 1 - _SAT_EPS)
    bo_sat = bool(bo_means and max(bo_means) < _SAT_EPS)
    out = {
        "fit_slope_gap_on_2pm1": mean_slope,
        "fit_slope_ci": [lo, hi],
        "fit_slope_gap_adj_on_2pm1": mean_slope_adj,   # pedestal-removed: the trustworthy number
        "fit_slope_adj_ci": [lo_a, hi_a],
        "gap_hi_lo": gap_mean_by_p[hi_p] - gap_mean_by_p[lo_p],
        "gap_at_0.5": gap_mean_by_p.get(base_p, float("nan")),
        "bias_pedestal": gap_mean_by_p.get(base_p, float("nan")),  # = gap@baseline (should be ~0)
        "onesided_saturation": ma_sat or bo_sat,
        "rmse_vs_2pm1": rmse,
        "n_items": len(per_item_slope),
    }
    if action_margin_by_p is not None and lo_p in action_margin_by_p and hi_p in action_margin_by_p:
        out["action_margin_hi_lo"] = action_margin_by_p[hi_p] - action_margin_by_p[lo_p]
    return out


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------
def _read_action_margin(path: str | None) -> dict | None:
    """Load p -> margin_mean from a logprob_sweep per-p CSV (for the flat-action overlay)."""
    if not path or not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    col = "margin_mean" if "margin_mean" in df.columns else "p_non_cdt_mean"
    return {float(p): float(m) for p, m in zip(df["p"], df[col])}


def run_credence_probe(cfg: EvalConfig, wrapper, *, variants=("outcome",),
                       phrasings=("report", "revealed", "observer"),
                       out_dir: str | None = None, tag: str = "", action_margin_csv: str | None = None):
    """Loop items × p_grid × actions × phrasings × variants × repeats; persist raw samples;
    aggregate per-p; print the headline; return the per-p DataFrame (df.attrs['p_star'])."""
    import pandas as pd

    out_dir = os.path.abspath(out_dir or os.path.join(cfg.results_dir, "credence"))
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    items = load(cfg.dataset_path, cfg)
    action_margin = _read_action_margin(action_margin_csv)

    rows = []
    for item in items:
        entity = conditional_prize_name(item.scenario)
        for p in cfg.sweep.p_grid:
            for r in range(cfg.sweep.n_repeats):
                block = render_scenario_block(
                    item, p, dataclasses.replace(cfg.prompt, cot=False),
                    sweep_seed=cfg.sweep.sweep_seed, repeat=r, mode=cfg.loader_mode)
                onebox_label = roles_to_tokens(block.token_role)[0]
                for variant in variants:
                    if variant == "outcome" and entity is None:
                        continue
                    if variant == "action":
                        # forced-choice action margin over the two action labels (the disposition
                        # axis) — mirrors logprob_sweep but kept here so a single invocation gives
                        # both axes (no quant-less logprob_sweep dependency for mechanism runs).
                        rp = build_prompt(item, p, dataclasses.replace(cfg.prompt, cot=False),
                                          sweep_seed=cfg.sweep.sweep_seed, repeat=r, mode=cfg.loader_mode)
                        non_cdt, cdt = roles_to_tokens(rp.token_role)
                        lp = wrapper.answer_logprobs(rp.text, non_cdt, cdt)
                        rows.append({
                            "item_id": item.id, "p": p, "repeat": r, "variant": "action",
                            "action": "forced", "phrasing": "-", "onebox_label": non_cdt,
                            "credence_full": lp["p_non_cdt"], "margin": lp["margin"],
                            "is_degenerate": False, "raw": "",
                        })
                        continue
                    if variant == "direct":
                        for action in ACTIONS:
                            base = build_direct_base(wrapper, block, item, action=action)
                            sc = score_credence_direct(wrapper, base)
                            rows.append({
                                "item_id": item.id, "p": p, "repeat": r, "variant": "direct",
                                "action": action, "phrasing": "direct", "onebox_label": onebox_label,
                                "credence_full": sc["credence_full"], "margin": float("nan"),
                                "is_degenerate": sc["is_degenerate"], "raw": sc["raw"],
                            })
                        continue
                    for action in ACTIONS:
                        for phrasing in phrasings:
                            base, tok_f, tok_e = build_credence_base(
                                wrapper, block, item, action=action, phrasing=phrasing,
                                variant=variant, entity=entity)
                            sc = score_credence(wrapper, base, tok_f, tok_e)
                            rows.append({
                                "item_id": item.id, "p": p, "repeat": r, "variant": variant,
                                "action": action, "phrasing": phrasing, "onebox_label": onebox_label,
                                "credence_full": sc["credence_full"], "margin": sc["margin"],
                                "is_degenerate": sc["is_degenerate"], "raw": "",
                            })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"credence_samples{suffix}.csv"), index=False)
    p_star = crossover_for_item(items[0], cfg.crossover).p_star

    # aggregate + headline per variant (so an outcome/prediction disagreement is visible)
    all_aggs = []
    for variant in variants:
        vrows = [r for r in rows if r["variant"] == variant]
        if not vrows:
            continue
        if variant == "action":  # disposition axis: just the margin trend over p
            import numpy as np
            by_p = {}
            for r in vrows:
                by_p.setdefault(float(r["p"]), []).append(float(r["margin"]))
            ps_a = sorted(by_p)
            slope = float(np.mean(by_p[ps_a[-1]]) - np.mean(by_p[ps_a[0]]))
            print(f"[action] one-box margin hi-lo = {slope:+.3f}   "
                  f"margin@0.5 = {np.mean(by_p[ps_a[0]]):+.2f}")
            continue
        df_v = aggregate_credence(vrows, p_star=p_star, action_margin_by_p=action_margin)
        df_v.insert(0, "variant", variant)
        all_aggs.append(df_v)
        hl = credence_headline(vrows, p_star=p_star, action_margin_by_p=action_margin)
        sat = " SATURATED!" if hl["onesided_saturation"] else ""
        gate = " ⚠gap@0.5≠0" if abs(hl["gap_at_0.5"]) > 0.1 else ""
        print(f"[{variant}] gap_adj fit(~2p-1) = {hl['fit_slope_gap_adj_on_2pm1']:+.3f} "
              f"[{hl['fit_slope_adj_ci'][0]:+.3f},{hl['fit_slope_adj_ci'][1]:+.3f}]   "
              f"raw fit = {hl['fit_slope_gap_on_2pm1']:+.3f}   "
              f"gap@0.5 = {hl['gap_at_0.5']:+.3f}{gate}   "
              f"rmse(2p-1) = {hl['rmse_vs_2pm1']:.3f}{sat}")

    df = pd.concat(all_aggs, ignore_index=True) if all_aggs else pd.DataFrame()
    csv_path = os.path.join(out_dir, f"credence_gap_by_p{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"p* = {p_star:.2f}   table: {csv_path}")
    df.attrs["p_star"] = p_star
    return df


def plot_credence(df, p_star: float, out_png: str, model_name: str, *, variant: str = "outcome") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["variant"] == variant].sort_values("p") if "variant" in df.columns else df.sort_values("p")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(d["p"], d["gap_mean"], marker="o", color="#1f77b4", label="credence gap  P(full|1box)−P(full|2box)")
    ax.fill_between(d["p"], d["gap_mean"] - d["gap_se"], d["gap_mean"] + d["gap_se"],
                    alpha=0.15, color="#1f77b4")
    ax.plot(d["p"], d["ref_2pm1"], ls="--", color="black", alpha=0.7, label="evidential ideal  2p−1")
    if "action_margin_mean" in d.columns and d["action_margin_mean"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(d["p"], d["action_margin_mean"], marker="s", ls=":", color="#d62728", alpha=0.7,
                 label="action margin (flat)")
        ax2.set_ylabel("action logit margin", color="#d62728")
        ax2.legend(loc="lower right", fontsize=8)
    ax.axvline(p_star, color="grey", ls="--", alpha=0.5, label=f"p* = {p_star:.2f}")
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.4)
    ax.set_ylim(-0.15, 1.05); ax.set_xlabel("stated accuracy p")
    ax.set_ylabel("credence gap")
    ax.set_title(f"Credence probe ({variant}) — {model_name}\n"
                 "tracks 2p−1 ⇒ represented;  flat ⇒ not represented")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def build_wrapper(cfg: EvalConfig, *, quantize: str | None = None):
    """Construct a ModelWrapper, optionally 4-/8-bit quantized (for the 32B rung on a 46 GB card).

    The quantized path bypasses ``ModelWrapper.__init__`` (which always loads a full-precision model)
    and mirrors its attribute setup for a ``BitsAndBytesConfig`` load — contained here so the shared
    ``model.py`` is untouched (the other instance's training imports it).
    """
    from .model import ModelWrapper
    if not quantize:
        return ModelWrapper(
            cfg.model.model_name, adapter_path=cfg.model.adapter_path,
            dtype=cfg.model.dtype, device_map=cfg.model.device_map,
            use_chat_template=cfg.model.use_chat_template)

    import threading
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    w = ModelWrapper.__new__(ModelWrapper)
    w.model_name = cfg.model.model_name
    w.adapter_path = None
    w.use_chat_template = cfg.model.use_chat_template
    w._lock = threading.Lock()
    w.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if w.tokenizer.pad_token is None:
        w.tokenizer.pad_token = w.tokenizer.eos_token
    w.tokenizer.padding_side = "left"
    qbits = 8 if "8" in str(quantize) else 4
    bnb = BitsAndBytesConfig(
        load_in_4bit=(qbits == 4), load_in_8bit=(qbits == 8),
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    w.model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, quantization_config=bnb, device_map="auto")
    w.model.eval()
    assert hasattr(w, "model") and hasattr(w, "tokenizer")
    return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Credence probe — represented-but-unused diagnostic")
    ap.add_argument("--model", help="HF model id (default: config Qwen2.5-3B-Instruct)")
    ap.add_argument("--adapter", help="PEFT/LoRA adapter dir to load on top of the base")
    ap.add_argument("--p-grid", dest="p_grid", nargs="+", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-n", "--repeats", type=int, default=None)
    ap.add_argument("--variant", nargs="+", default=["outcome"],
                    choices=["outcome", "prediction", "direct", "action"],
                    help="elicitation variant(s); 'action' = forced-choice disposition margin")
    ap.add_argument("--data", default=None, help="dataset path override (e.g. a mechanism variant)")
    ap.add_argument("--quantize", default=None, choices=["4bit", "8bit"],
                    help="load the model quantized (for the 32B rung on a 46 GB card)")
    ap.add_argument("--phrasings", nargs="+", default=["report", "revealed", "observer"])
    ap.add_argument("--action-margin", dest="action_margin", default=None,
                    help="logprob_sweep per-p CSV for the flat-action overlay")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = EvalConfig()
    if args.data:
        cfg = dataclasses.replace(cfg, dataset_path=os.path.abspath(args.data))
    if args.model:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, model_name=args.model))
    if args.adapter:
        cfg = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, adapter_path=os.path.abspath(args.adapter)))
    if args.p_grid:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, p_grid=tuple(args.p_grid)))
    if args.repeats is not None:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, n_repeats=args.repeats))
    if args.limit is not None:
        cfg = dataclasses.replace(cfg, limit=args.limit)

    wrapper = build_wrapper(cfg, quantize=args.quantize)
    df = run_credence_probe(cfg, wrapper, variants=tuple(args.variant),
                            phrasings=tuple(args.phrasings), tag=args.tag,
                            action_margin_csv=args.action_margin)
    out_dir = os.path.abspath(os.path.join(cfg.results_dir, "credence"))
    suffix = f"_{args.tag}" if args.tag else ""
    for variant in args.variant:
        if "variant" in df.columns and (df["variant"] == variant).any():
            out_png = args.out or os.path.join(out_dir, f"credence_gap_by_p{suffix}_{variant}.png")
            plot_credence(df, df.attrs["p_star"], out_png, cfg.model.model_name, variant=variant)
            print(f"plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
