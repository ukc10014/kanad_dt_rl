"""Scaffolded-CoT three-arm experiment (PLAN: scaffold.py).

Ported from cosmichost_mp/run_scaffolded_cot.py into *our* stack (HF ModelWrapper, abstract
opaque-Newcomb items, the existing abstract-token ``resolve_choice``). Three arms over the
p-sweep:

  - ``no_cot``     forced-choice single token (== the Run-1 baseline path)
  - ``free_cot``   "think step by step" then answer
  - ``scaffolded`` 5 neutral sub-questions (parties -> options -> outcomes -> relationship ->
                   decision), each its own assistant turn

Why three arms (the "three readouts", PLAN):
  capability  -> the K-rate(p) slope per arm (does decomposition unlock the step at p*?)
  extraction  -> the Step-4 'relationship' response, persisted verbatim for inspection
  inclination -> the decision conditional on a correct Step-4 (the residual)

Invariant #1: we NEVER string-match 'one-box'/'two-box'/'Newcomb'/'predictor'. The decision is
resolved purely from the emitted abstract label via ``scorer.resolve_choice``.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field

from .config import EvalConfig
from .crossover import crossover_for_item
from .data.loader import load
from .prompts import build_prompt, render_scenario_block
from .scorer import ROLE_NON_CDT, resolve_choice
from .sweep import aggregate

ARMS = ("no_cot", "free_cot", "scaffolded")

# Neutral 5-step decomposition. {labels} in the decision step is filled with the rendered
# label order so the model answers with a token the scorer recognises.
SCAFFOLD_STEPS = [
    {
        "id": "parties",
        "prompt": (
            "Before deciding, let's analyse this step by step.\n\n"
            "**Step 1 — Parties.** Who are the parties in this scenario, and what does each "
            "one do? (2–4 sentences.)"
        ),
    },
    {
        "id": "options",
        "prompt": (
            "**Step 2 — Options.** What choices are available to you, the decision-maker? "
            "List them clearly."
        ),
    },
    {
        "id": "outcomes",
        "prompt": (
            "**Step 3 — Outcomes.** For each combination of your choice and the other party's "
            "behaviour, what is the payoff? Describe the best and worst cases."
        ),
    },
    {
        "id": "relationship",
        "prompt": (
            "**Step 4 — Relationship.** What is the relationship between your choice and the "
            "other party's behaviour — is it causal, correlational, or neither? How does the "
            "stated accuracy figure into it?"
        ),
    },
    {
        "id": "decision",
        "prompt": (
            "**Step 5 — Decision.** Based on steps 1–4, choose one option. Write your answer as "
            "the very first thing, in the form 'Answer: <LABEL>', then stop. "
            "(Valid labels: {labels}.)"
        ),
    },
]


@dataclass
class ScaffoldTrial:
    """One (arm, item, p, repeat) trial with its full transcript."""

    arm: str
    item_id: str
    p: float
    repeat: int
    seed_key: str
    legal_tokens: list[str]
    order: list[str]
    steps: list[dict] = field(default_factory=list)  # [{id, prompt, response}, ...]
    chosen_role: str = ""
    chosen_token: str | None = None
    is_valid: bool = False
    is_k: float = 0.0


# --- arm runners ----------------------------------------------------------

def run_no_cot(wrapper, item, p, cfg: EvalConfig, *, sweep_seed: int, repeat: int,
               max_new_tokens: int = 8) -> ScaffoldTrial:
    pcfg = dataclasses.replace(cfg.prompt, cot=False)
    rp = build_prompt(item, p, pcfg, sweep_seed=sweep_seed, repeat=repeat, mode=cfg.loader_mode)
    resp = wrapper.generate_one(rp.text, max_new_tokens=max_new_tokens,
                                temperature=cfg.model.temperature)
    role, tok, valid = resolve_choice(resp, rp.legal_tokens, rp.token_role, cot=False)
    return _trial("no_cot", rp, repeat, [{"id": "answer", "prompt": rp.text, "response": resp}],
                  role, tok, valid)


def run_free_cot(wrapper, item, p, cfg: EvalConfig, *, sweep_seed: int, repeat: int,
                 max_new_tokens: int = 256) -> ScaffoldTrial:
    pcfg = dataclasses.replace(cfg.prompt, cot=True)
    rp = build_prompt(item, p, pcfg, sweep_seed=sweep_seed, repeat=repeat, mode=cfg.loader_mode)
    resp = wrapper.generate_one(rp.text, max_new_tokens=max_new_tokens,
                                temperature=cfg.model.temperature)
    role, tok, valid = resolve_choice(resp, rp.legal_tokens, rp.token_role, cot=True)
    return _trial("free_cot", rp, repeat, [{"id": "reasoning", "prompt": rp.text, "response": resp}],
                  role, tok, valid)


def run_scaffolded(wrapper, item, p, cfg: EvalConfig, *, sweep_seed: int, repeat: int,
                   step_max: int = 256, decision_max: int = 48) -> ScaffoldTrial:
    block = render_scenario_block(item, p, cfg.prompt, sweep_seed=sweep_seed, repeat=repeat,
                                  mode=cfg.loader_mode)
    labels = ", ".join(block.order)

    messages: list[dict] = []
    steps: list[dict] = []
    for i, step in enumerate(SCAFFOLD_STEPS):
        prompt = step["prompt"].format(labels=labels)
        if i == 0:
            prompt = block.text + "\n" + prompt  # first turn carries the scenario
        messages.append({"role": "user", "content": prompt})

        is_decision = step["id"] == "decision"
        max_new = decision_max if is_decision else step_max
        resp = wrapper.generate_messages(messages, max_new_tokens=max_new,
                                         temperature=cfg.model.temperature)
        messages.append({"role": "assistant", "content": resp})
        steps.append({"id": step["id"], "prompt": prompt, "response": resp})

    final = steps[-1]["response"]
    role, tok, valid = resolve_choice(final, block.legal_tokens, block.token_role, cot=True)
    return ScaffoldTrial(
        arm="scaffolded", item_id=item.id, p=p, repeat=repeat, seed_key=block.seed_key,
        legal_tokens=block.legal_tokens, order=block.order, steps=steps,
        chosen_role=role, chosen_token=tok, is_valid=valid,
        is_k=1.0 if role == ROLE_NON_CDT else 0.0,
    )


def _trial(arm, rp, repeat, steps, role, tok, valid) -> ScaffoldTrial:
    return ScaffoldTrial(
        arm=arm, item_id=rp.item_id, p=rp.p, repeat=repeat, seed_key=rp.seed_key,
        legal_tokens=rp.legal_tokens, order=rp.order, steps=steps,
        chosen_role=role, chosen_token=tok, is_valid=valid,
        is_k=1.0 if role == ROLE_NON_CDT else 0.0,
    )


_RUNNERS = {"no_cot": run_no_cot, "free_cot": run_free_cot, "scaffolded": run_scaffolded}


# --- sweep ----------------------------------------------------------------

def run_scaffold_sweep(
    cfg: EvalConfig,
    wrapper,
    *,
    arms=ARMS,
    repeats: int = 1,
    step_max: int = 256,
    free_max: int = 256,
    decision_max: int = 16,
    out_dir: str | None = None,
    tag: str = "",
):
    """Run the requested arms over items × p_grid × repeats.

    Returns ``(curves, trials)`` where ``curves`` maps arm -> K-rate(p) DataFrame (Wilson CIs,
    via ``sweep.aggregate``) and ``trials`` is the flat list of ``ScaffoldTrial``. Persists
    full transcripts (every step response), a per-trial CSV, and the per-(arm,p) table.
    """
    import pandas as pd

    out_dir = os.path.abspath(out_dir or os.path.join(cfg.results_dir, "scaffold"))
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "scaffold_logs")
    os.makedirs(log_dir, exist_ok=True)

    items = load(cfg.dataset_path, cfg)
    p_grid = cfg.sweep.p_grid
    suffix = f"_{tag}" if tag else ""

    curves: dict = {}
    all_trials: list[ScaffoldTrial] = []
    for arm in arms:
        runner = _RUNNERS[arm]
        kwargs = {}
        if arm == "free_cot":
            kwargs["max_new_tokens"] = free_max
        elif arm == "scaffolded":
            kwargs = {"step_max": step_max, "decision_max": decision_max}

        trials: list[ScaffoldTrial] = []
        jsonl_path = os.path.join(log_dir, f"transcripts_{arm}{suffix}.jsonl")
        with open(jsonl_path, "w") as fh:
            for item in items:
                for p in p_grid:
                    for r in range(repeats):
                        t = runner(wrapper, item, p, cfg, sweep_seed=cfg.sweep.sweep_seed,
                                   repeat=r, **kwargs)
                        trials.append(t)
                        fh.write(json.dumps(asdict(t)) + "\n")
        all_trials.extend(trials)

        rows = [(t.p, t.is_k, t.is_valid) for t in trials]
        df = aggregate(rows)
        df["arm"] = arm
        curves[arm] = df
        inv = 1.0 - df["n_valid"].sum() / df["n"].sum() if df["n"].sum() else float("nan")
        print(f"[{arm}] mean_K={df['k_rate'].mean():.3f} invalid={inv:.3f}  -> {jsonl_path}")

    # combined per-(arm,p) table
    combined = pd.concat(list(curves.values()), ignore_index=True)
    combined = combined[["arm", "p", "k_rate", "n", "n_valid", "invalid_rate", "ci_lo", "ci_hi"]]
    csv_path = os.path.join(out_dir, f"k_rate_by_arm_p{suffix}.csv")
    combined.to_csv(csv_path, index=False)

    # per-trial flat CSV (drops the bulky step text; transcripts JSONL keeps that)
    samp = pd.DataFrame([
        {"arm": t.arm, "item_id": t.item_id, "p": t.p, "repeat": t.repeat,
         "chosen_role": t.chosen_role, "chosen_token": t.chosen_token,
         "is_valid": t.is_valid, "is_k": t.is_k, "seed_key": t.seed_key}
        for t in all_trials
    ])
    samp_path = os.path.join(out_dir, f"scaffold_samples{suffix}.csv")
    samp.to_csv(samp_path, index=False)

    p_star = crossover_for_item(items[0], cfg.crossover).p_star
    print(f"\ntable: {csv_path}\nsamples: {samp_path}\np* = {p_star:.2f}")
    return curves, all_trials
