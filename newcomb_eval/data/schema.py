"""NewcombItem schema + validation (PLAN.md §2 data/schema.py).

An item stores the *structural* problem plus role metadata. Surface tokens and option
order are assigned at render time by prompts.py — so ``non_cdt_position`` here is a
*role pointer* into the source ``answers``, not a fixed surface label the model sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Detects a hard-coded predictor accuracy like "80%", "0.8 of the time", "80 percent".
_ACCURACY_RE = re.compile(
    r"(\d{1,3}\s*%|\d{1,3}\s*percent|0?\.\d+\s+of\s+the\s+time)",
    re.IGNORECASE,
)


class ItemValidationError(ValueError):
    """Raised when an item violates the schema contract."""


@dataclass
class NewcombItem:
    id: str
    scenario: str  # problem body; carries the accuracy marker (text-injection mode)
    # Role pointers into ``answers`` (keys "A"/"B" from the source file).
    non_cdt_position: str            # which source option is the non-CDT (EDT/FDT) choice
    answer_matching_behavior: str    # Perez-schema: the "matching" (non-CDT) role
    answer_not_matching_behavior: str
    answers: dict                    # {"A": "...", "B": "..."} source option texts
    behavior_category: str = ""
    match_endorsed_by: list = field(default_factory=list)
    nonmatch_endorsed_by: list = field(default_factory=list)
    payoff_big: float | None = None    # B: conditional prize
    payoff_small: float | None = None  # S: guaranteed prize
    meta: dict = field(default_factory=dict)

    @property
    def non_cdt_text(self) -> str:
        """Surface text of the non-CDT (one-box / "K") option."""
        return self.answers[self.non_cdt_position]

    @property
    def cdt_text(self) -> str:
        """Surface text of the CDT (two-box) option."""
        return self.answers[self.answer_not_matching_behavior]


def validate(
    item: NewcombItem,
    *,
    mode: str = "text_injection",
    crossover_mode: str = "global",
    accuracy_marker: str = "{{PREDICTOR_ACCURACY}}",
) -> NewcombItem:
    """Validate one item against the contract; raise ItemValidationError on violation.

    Raises on: missing role fields; a scenario that already hard-codes a predictor
    accuracy (templated mode); a missing accuracy marker (text-injection mode); or
    payoffs absent when crossover_mode == "per_item". (PLAN.md §2.)
    """
    if not item.id:
        raise ItemValidationError("item has no id")

    # Role fields must be present and point at real options.
    for fld in ("non_cdt_position", "answer_matching_behavior", "answer_not_matching_behavior"):
        val = getattr(item, fld)
        if not val:
            raise ItemValidationError(f"{item.id}: missing role field '{fld}'")
        if val not in item.answers:
            raise ItemValidationError(
                f"{item.id}: role field '{fld}'={val!r} not a key of answers {list(item.answers)}"
            )
    if item.answer_matching_behavior == item.answer_not_matching_behavior:
        raise ItemValidationError(f"{item.id}: matching and non-matching roles are identical")
    # In this schema the "matching" behaviour is the non-CDT (one-box) choice.
    if item.answer_matching_behavior != item.non_cdt_position:
        raise ItemValidationError(
            f"{item.id}: answer_matching_behavior must equal non_cdt_position for opaque Newcomb"
        )
    if len(item.answers) != 2:
        raise ItemValidationError(f"{item.id}: expected exactly 2 answers, got {len(item.answers)}")

    if mode == "text_injection":
        if accuracy_marker not in item.scenario:
            raise ItemValidationError(
                f"{item.id}: text-injection mode requires the accuracy marker "
                f"{accuracy_marker!r} in the scenario"
            )
    elif mode == "templated":
        # Strip the marker before checking, so a marker isn't mistaken for hard-coded p.
        body = item.scenario.replace(accuracy_marker, "")
        if _ACCURACY_RE.search(body):
            raise ItemValidationError(
                f"{item.id}: templated mode but scenario hard-codes a predictor accuracy; "
                f"remove it so the harness can inject p"
            )
    else:
        raise ItemValidationError(f"{item.id}: unknown loader mode {mode!r}")

    if crossover_mode == "per_item":
        if item.payoff_big is None or item.payoff_small is None:
            raise ItemValidationError(
                f"{item.id}: crossover mode 'per_item' requires payoff_big and payoff_small"
            )

    return item
