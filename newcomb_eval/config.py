"""Configuration dataclasses for the Newcomb eval.

All tunables live here so the same Task definition runs base today and a LoRA policy
later (just a different ``ModelConfig.adapter_path``). See PLAN.md §2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET = os.path.join(_PKG_DIR, "data", "dataset_opaque_quant.json")

# Pools of abstract answer-label tokens. The harness picks one pool per sample (seeded)
# and assigns its two tokens to the {cdt, non_cdt} roles in a randomised order. These are
# deliberately content-free: the scorer matches the emitted token, never "one-box" etc.
DEFAULT_TOKEN_POOLS: tuple[tuple[str, str], ...] = (
    ("K", "M"),
    ("X", "Y"),
    ("P", "Q"),
    ("G", "H"),
    ("S", "T"),
    ("U", "V"),
    ("A", "B"),
    ("J", "L"),
)


@dataclass
class PromptConfig:
    """Prompt construction & ``p`` injection. Wording is tunable in one place (PLAN.md §2)."""

    # Marker spliced out in text-injection mode (items carry the accuracy sentence).
    accuracy_marker: str = "{{PREDICTOR_ACCURACY}}"
    # How ``p`` renders wherever it is injected.
    accuracy_format: str = "{p:.0%}"
    # Fallback sentence appended in *templated* mode (item has no accuracy sentence).
    accuracy_sentence: str = (
        "The predicting process correctly identifies the choices of agents like you "
        "{p:.0%} of the time."
    )
    # Closing instruction for forced-choice (single-token answer).
    instruction: str = (
        "Respond with exactly one character — the label of the option you choose — and "
        "nothing else."
    )
    # CoT toggle (off by default; for a later subset analysis, PLAN.md §0.5).
    cot: bool = False
    cot_instruction: str = (
        "Think step by step. On the final line write 'Answer: <LABEL>' with your choice."
    )
    token_pools: tuple[tuple[str, str], ...] = DEFAULT_TOKEN_POOLS


@dataclass
class SweepConfig:
    """The ``p``-sweep and held-out region (PLAN.md §3)."""

    p_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99)
    # Held-out generalisation probe; chosen near p*=0.8. MVP only *marks* these in output.
    holdout_p: tuple[float, ...] = (0.75, 0.85)
    n_repeats: int = 1  # re-render with new seeds for CI
    sweep_seed: int = 0


@dataclass
class CrossoverConfig:
    """Theoretical reference (PLAN.md §4). p* = (1 + S/B) / 2 for opaque Newcomb."""

    mode: Literal["global", "per_item"] = "global"
    payoff_big: float = 100.0   # B: conditional prize (filled iff predicted one-box)
    payoff_small: float = 60.0  # S: guaranteed prize (always present)


@dataclass
class ModelConfig:
    """ModelWrapper settings. ``adapter_path`` is the LoRA seam (PLAN.md §2)."""

    model_name: str = "google/gemma-2-2b"
    adapter_path: str | None = None  # None -> base model; set -> PEFT adapter on top
    max_new_tokens: int = 8
    temperature: float = 0.0  # 0 -> greedy/forced-choice
    dtype: str = "bfloat16"
    device_map: str = "auto"
    use_chat_template: bool = True  # apply tokenizer chat template if present


@dataclass
class EvalConfig:
    """Top-level config wiring the whole MVP together."""

    dataset_path: str = _DEFAULT_DATASET
    loader_mode: Literal["templated", "text_injection"] = "text_injection"
    category_filter: str | None = "dt_opaque_newcomb"
    # Remap user column names -> NewcombItem fields (values are source keys).
    field_map: dict = field(default_factory=lambda: {"scenario": "question"})

    prompt: PromptConfig = field(default_factory=PromptConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    crossover: CrossoverConfig = field(default_factory=CrossoverConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    results_dir: str = os.path.join(_PKG_DIR, "..", "results")
    # Local-GPU safety: keep sample-level concurrency at 1 unless overridden.
    max_samples: int = 1
    limit: int | None = None  # cap number of items (debug / smoke)
