"""PLAN.md §7: example_item loads + validates; templated-mode catches a hard-coded
accuracy; text-injection requires the marker."""
import dataclasses

import pytest

from newcomb_eval.config import EvalConfig
from newcomb_eval.data.loader import load, load_example_item
from newcomb_eval.data.schema import ItemValidationError, NewcombItem, validate


def test_example_item_loads_and_validates():
    cfg = EvalConfig()
    item = load_example_item(cfg)
    assert item.behavior_category == "dt_opaque_newcomb"
    assert item.non_cdt_text == "Take only container Q."
    validate(item, mode="text_injection", crossover_mode="per_item")


def test_full_dataset_loads_filtered_to_opaque():
    cfg = EvalConfig()
    items = load(cfg.dataset_path, cfg)
    assert len(items) == 20
    assert all(it.behavior_category == "dt_opaque_newcomb" for it in items)
    assert all("{{PREDICTOR_ACCURACY}}" in it.scenario for it in items)


def test_ids_are_unique_and_stable():
    cfg = EvalConfig()
    ids = [it.id for it in load(cfg.dataset_path, cfg)]
    assert len(ids) == len(set(ids))
    assert ids[0] == "opaque_00"


def _base_item(scenario):
    return NewcombItem(
        id="t0",
        scenario=scenario,
        non_cdt_position="A",
        answer_matching_behavior="A",
        answer_not_matching_behavior="B",
        answers={"A": "Take only Q.", "B": "Take both."},
    )


def test_templated_mode_flags_hardcoded_accuracy():
    item = _base_item("F is correct 80% of the time. Choose.")
    with pytest.raises(ItemValidationError):
        validate(item, mode="templated")


def test_templated_mode_accepts_clean_scenario():
    item = _base_item("F has analysed your choice. Choose.")
    validate(item, mode="templated")  # should not raise


def test_text_injection_requires_marker():
    item = _base_item("F has analysed your choice. Choose.")  # no marker
    with pytest.raises(ItemValidationError):
        validate(item, mode="text_injection")


def test_role_must_point_to_real_option():
    bad = NewcombItem(
        id="t1",
        scenario="... {{PREDICTOR_ACCURACY}} ...",
        non_cdt_position="C",  # not a key of answers
        answer_matching_behavior="C",
        answer_not_matching_behavior="B",
        answers={"A": "x", "B": "y"},
    )
    with pytest.raises(ItemValidationError):
        validate(bad, mode="text_injection")


def test_per_item_crossover_requires_payoffs():
    item = _base_item("... {{PREDICTOR_ACCURACY}} ...")  # no payoffs
    with pytest.raises(ItemValidationError):
        validate(item, mode="text_injection", crossover_mode="per_item")
