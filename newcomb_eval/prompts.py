"""Prompt construction & ``p`` injection (PLAN.md §2 prompts.py).

Determinism: all randomisation (which abstract token pool, role->token assignment,
option order) is seeded from ``(item_id, p, sweep_seed, repeat)`` so a given sample is
reproducible and stable across runs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import PromptConfig
from .data.schema import NewcombItem


@dataclass
class RenderedPrompt:
    text: str
    legal_tokens: list[str]          # the two abstract labels the model may emit
    token_role: dict                 # label -> "cdt" | "non_cdt"
    p: float
    item_id: str
    order: list[str] = field(default_factory=list)  # labels in displayed order
    seed_key: str = ""


@dataclass
class ScenarioBlock:
    """The scenario + options block *without* the closing instruction.

    Shared by ``build_prompt`` (which appends the forced-choice/CoT instruction) and the
    scaffold runner (which appends its own step prompts). Carries the same seeded
    ``legal_tokens``/``token_role``/``order`` so a scaffolded completion is scored against
    exactly the labels the model was shown.
    """

    text: str                        # ends with a trailing newline (ready to append to)
    legal_tokens: list[str]
    token_role: dict
    order: list[str]
    p: float
    item_id: str
    seed_key: str = ""


def _rng(item_id: str, p: float, sweep_seed: int, repeat: int) -> random.Random:
    return random.Random(f"{item_id}|{p:.6f}|{sweep_seed}|{repeat}")


def _format_p(p: float, fmt: str) -> str:
    return fmt.format(p=p)


def render_scenario_block(
    item: NewcombItem,
    p: float,
    prompt_cfg: PromptConfig,
    *,
    sweep_seed: int = 0,
    repeat: int = 0,
    mode: str = "text_injection",
) -> ScenarioBlock:
    """Render the scenario + options block (no closing instruction).

    Steps (PLAN.md §2):
      - render the scenario and inject the stated predictor accuracy ``p``;
      - assign a randomised abstract token to each option role and a randomised order.

    The returned ``text`` ends with a trailing newline so a caller can append either the
    forced-choice instruction (``build_prompt``) or a scaffold step prompt.
    """
    rng = _rng(item.id, p, sweep_seed, repeat)
    p_str = _format_p(p, prompt_cfg.accuracy_format)

    # --- inject p ---
    if mode == "text_injection":
        if prompt_cfg.accuracy_marker not in item.scenario:
            raise ValueError(
                f"{item.id}: text-injection mode but marker {prompt_cfg.accuracy_marker!r} absent"
            )
        scenario = item.scenario.replace(prompt_cfg.accuracy_marker, p_str)
    else:  # templated: scenario is structural, append the accuracy sentence
        scenario = item.scenario.rstrip() + " " + prompt_cfg.accuracy_sentence.format(p=p)

    # --- assign abstract tokens to roles, randomise order ---
    pool = list(rng.choice(prompt_cfg.token_pools))
    rng.shuffle(pool)
    tok_non_cdt, tok_cdt = pool[0], pool[1]
    token_role = {tok_non_cdt: "non_cdt", tok_cdt: "cdt"}

    options = [
        (tok_non_cdt, item.non_cdt_text),
        (tok_cdt, item.cdt_text),
    ]
    rng.shuffle(options)  # randomise display order
    order = [tok for tok, _ in options]

    lines = [scenario, ""]
    for tok, text in options:
        lines.append(f"{tok}. {text}")
    lines.append("")  # trailing blank => text ends with "\n"

    return ScenarioBlock(
        text="\n".join(lines),
        legal_tokens=sorted(token_role.keys()),
        token_role=token_role,
        order=order,
        p=p,
        item_id=item.id,
        seed_key=f"{item.id}|{p:.6f}|{sweep_seed}|{repeat}",
    )


def build_prompt(
    item: NewcombItem,
    p: float,
    prompt_cfg: PromptConfig,
    *,
    sweep_seed: int = 0,
    repeat: int = 0,
    mode: str = "text_injection",
) -> RenderedPrompt:
    """Render one (item, p) into a forced-choice (or CoT) prompt + scorer metadata.

    Thin wrapper over ``render_scenario_block`` that appends the fixed closing instruction
    (answer with exactly one legal token). Output is byte-identical to the pre-refactor form.
    """
    block = render_scenario_block(
        item, p, prompt_cfg, sweep_seed=sweep_seed, repeat=repeat, mode=mode
    )
    instruction = prompt_cfg.cot_instruction if prompt_cfg.cot else prompt_cfg.instruction
    # block.text ends with "\n"; add one more for the blank line before the instruction.
    text = block.text + "\n" + instruction + f" (Valid labels: {', '.join(block.order)}.)"

    return RenderedPrompt(
        text=text,
        legal_tokens=block.legal_tokens,
        token_role=block.token_role,
        p=p,
        item_id=item.id,
        order=block.order,
        seed_key=block.seed_key,
    )
