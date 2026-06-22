"""Inspect Task wiring: dataset x p-grid -> samples, solver -> scorer (PLAN.md §2 task.py).

Kept parameterised by EvalConfig + a ModelWrapper so the *same* task definition runs the
base model today and a LoRA policy later — just pass a different ModelWrapper. No
base-model-only assumptions here (PLAN.md §6).
"""
from __future__ import annotations

import anyio
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, solver

from .config import EvalConfig
from .data.loader import load
from .prompts import build_prompt
from .scorer import newcomb_scorer


def build_dataset(items, cfg: EvalConfig) -> MemoryDataset:
    """Cross-product every item with every p in the grid (x n_repeats) into Inspect Samples."""
    sweep = cfg.sweep
    holdout = set(sweep.holdout_p)
    samples: list[Sample] = []
    for item in items:
        for p in sweep.p_grid:
            for r in range(sweep.n_repeats):
                rp = build_prompt(
                    item, p, cfg.prompt,
                    sweep_seed=sweep.sweep_seed, repeat=r, mode=cfg.loader_mode,
                )
                non_cdt_label = next(t for t, role in rp.token_role.items() if role == "non_cdt")
                samples.append(
                    Sample(
                        input=rp.text,
                        target=non_cdt_label,  # the "K" answer; Inspect accuracy == raw K-rate
                        id=f"{item.id}|p={p}|r={r}",
                        metadata={
                            "item_id": item.id,
                            "p": p,
                            "repeat": r,
                            "legal_tokens": rp.legal_tokens,
                            "token_role": rp.token_role,
                            "order": rp.order,
                            "is_holdout": p in holdout,
                            "cot": cfg.prompt.cot,
                            "payoff_big": item.payoff_big,
                            "payoff_small": item.payoff_small,
                            "behavior_category": item.behavior_category,
                            "strata": item.meta,
                        },
                    )
                )
    return MemoryDataset(samples)


@solver
def newcomb_solver(model_wrapper, cfg: EvalConfig):
    """Render is already baked into Sample.input; here we just call the wrapper and store output."""
    max_new = cfg.model.max_new_tokens
    temp = cfg.model.temperature
    model_name = cfg.model.model_name

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input_text
        # Run blocking HF generate off the event loop; the wrapper's lock serialises GPU use.
        completion = await anyio.to_thread.run_sync(
            model_wrapper.generate_one, prompt, max_new, temp
        )
        state.output = ModelOutput.from_content(model=model_name, content=completion)
        return state

    return solve


def build_task(cfg: EvalConfig, model_wrapper) -> Task:
    """Programmatic Task builder used by sweep.py (pass a live ModelWrapper)."""
    items = load(cfg.dataset_path, cfg)
    return Task(
        dataset=build_dataset(items, cfg),
        solver=newcomb_solver(model_wrapper, cfg),
        scorer=newcomb_scorer(),
        name="newcomb_p_sweep",
    )


@task
def newcomb_task(cfg: EvalConfig | None = None) -> Task:
    """Inspect CLI entrypoint: builds its own ModelWrapper from cfg.model.

    Usage: ``inspect eval newcomb_eval/task.py`` (uses default EvalConfig). Programmatic
    callers should prefer build_task() to share one ModelWrapper across a sweep.
    """
    from .model import ModelWrapper

    cfg = cfg or EvalConfig()
    wrapper = ModelWrapper(
        cfg.model.model_name,
        adapter_path=cfg.model.adapter_path,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
        use_chat_template=cfg.model.use_chat_template,
    )
    return build_task(cfg, wrapper)
