"""Dataset loader (PLAN.md §2 data/loader.py).

Loads a user JSON file (a list of source objects) into ``list[NewcombItem]`` with a
field-remap table, synthesises stable ids, optionally filters by behaviour category, and
validates every item against the schema contract.

Two modes (PLAN.md §2):
  - templated:      scenario is structural; prompts.py appends the accuracy sentence.
  - text_injection: scenario carries the {{PREDICTOR_ACCURACY}} marker; prompts.py splices p in.
"""
from __future__ import annotations

import json
import os

from .schema import NewcombItem, validate

# Source keys copied straight through when present (same names as the Perez-style schema).
_PASSTHROUGH = (
    "non_cdt_position",
    "answer_matching_behavior",
    "answer_not_matching_behavior",
    "answers",
    "behavior_category",
    "match_endorsed_by",
    "nonmatch_endorsed_by",
    "payoff_big",
    "payoff_small",
)

_CATEGORY_SHORT = {
    "dt_opaque_newcomb": "opaque",
    "dt_transparent_newcomb": "transparent",
    "dt_counterfactual_mugging": "cfmugging",
    "dt_xor_blackmail": "xor",
    "dt_smoking_lesion": "smoking",
    "dt_parfit_hitchhiker": "hitchhiker",
}


def _short(cat: str) -> str:
    return _CATEGORY_SHORT.get(cat, cat or "item")


def _remap(raw: dict, field_map: dict) -> dict:
    """Apply field_map (NewcombItem field -> source key) over a copy of raw."""
    out = dict(raw)
    for dst, src in (field_map or {}).items():
        if src in raw:
            out[dst] = raw[src]
    return out


def load(path: str, cfg) -> list[NewcombItem]:
    """Load and validate items from ``path`` under EvalConfig-like ``cfg``.

    ``cfg`` must expose: field_map, loader_mode, category_filter, crossover.mode,
    prompt.accuracy_marker.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of items, got {type(data).__name__}")

    items: list[NewcombItem] = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}[{i}]: item is not an object")
        mapped = _remap(raw, getattr(cfg, "field_map", {}))

        scenario = mapped.get("scenario")
        if scenario is None:
            raise ValueError(
                f"{path}[{i}]: no 'scenario' after remap (field_map={getattr(cfg,'field_map',{})})"
            )

        cat = mapped.get("behavior_category", "")
        item_id = mapped.get("id") or f"{_short(cat)}_{i:02d}"

        kwargs = {k: mapped[k] for k in _PASSTHROUGH if k in mapped}
        meta = dict(mapped.get("meta", {}))
        # Carry the source 'system' and answers map into meta for downstream analyses.
        if "system" in raw:
            meta.setdefault("system", raw["system"])
        meta.setdefault("source_index", i)

        item = NewcombItem(id=item_id, scenario=scenario, meta=meta, **kwargs)
        validate(
            item,
            mode=cfg.loader_mode,
            crossover_mode=getattr(getattr(cfg, "crossover", None), "mode", "global"),
            accuracy_marker=getattr(getattr(cfg, "prompt", None), "accuracy_marker",
                                    "{{PREDICTOR_ACCURACY}}"),
        )
        items.append(item)

    cat_filter = getattr(cfg, "category_filter", None)
    if cat_filter:
        items = [it for it in items if it.behavior_category == cat_filter]
        if not items:
            raise ValueError(f"{path}: no items match category_filter={cat_filter!r}")

    limit = getattr(cfg, "limit", None)
    if limit is not None:
        items = items[:limit]
    return items


def load_example_item(cfg=None) -> NewcombItem:
    """Load the shipped contract example (data/example_item.json)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "example_item.json")
    with open(path) as f:
        raw = json.load(f)
    raw.pop("_comment", None)
    # Default remap mirrors EvalConfig: scenario <- question.
    field_map = getattr(cfg, "field_map", {"scenario": "question"}) if cfg else {"scenario": "question"}
    mapped = _remap(raw, field_map)
    kwargs = {k: mapped[k] for k in _PASSTHROUGH if k in mapped}
    return NewcombItem(
        id=mapped.get("id", "example_00"),
        scenario=mapped["scenario"],
        meta={"system": raw.get("system", "")},
        **kwargs,
    )
