"""Computation-transplant diagnostic helpers (pure tests; no model/GPU)."""

from newcomb_eval.config import EvalConfig
from newcomb_eval.data.schema import NewcombItem
from newcomb_eval.transplant import (
    CONDITIONS,
    ROLE_CDT,
    ROLE_NON_CDT,
    build_transplant_prompt,
    ev_summary,
    role_tokens,
    transplant_block,
)


def _item():
    return NewcombItem(
        id="toy",
        scenario=(
            "A process filled two boxes. Box BIG has 100 points if it predicted only BIG. "
            "Box SMALL has 60 points. Accuracy {{PREDICTOR_ACCURACY}}."
        ),
        non_cdt_position="A",
        answer_matching_behavior="A",
        answer_not_matching_behavior="B",
        answers={"A": "Take only BIG.", "B": "Take both BIG and SMALL."},
    )


def test_ev_summary_selects_cdt_below_crossover_and_non_cdt_above():
    lo = ev_summary(0.5, 100, 60)
    tie = ev_summary(0.8, 100, 60)
    hi = ev_summary(0.99, 100, 60)
    assert lo["ev_non_cdt"] == 50
    assert lo["ev_cdt"] == 110
    assert lo["optimal_role"] == ROLE_CDT
    assert tie["ev_non_cdt"] == tie["ev_cdt"] == 80
    assert tie["optimal_role"] == "tie"
    assert hi["ev_non_cdt"] == 99
    assert hi["ev_cdt"] == 61
    assert hi["optimal_role"] == ROLE_NON_CDT


def test_role_tokens_finds_abstract_labels_independent_of_order():
    assert role_tokens({"K": ROLE_NON_CDT, "M": ROLE_CDT}) == ("K", "M")
    assert role_tokens({"X": ROLE_CDT, "Y": ROLE_NON_CDT}) == ("Y", "X")


def test_transplant_blocks_are_ordered_interventions():
    kwargs = dict(
        p=0.5,
        B=100,
        S=60,
        non_cdt_token="K",
        cdt_token="M",
        ev_non_cdt_value=50,
        ev_cdt_value=110,
        optimal_token="M",
    )
    assert transplant_block("none", **kwargs) == ""
    variables = transplant_block("variables", **kwargs)
    assert "Stated accuracy = 0.5" in variables
    assert "conditional_amount = 100" in variables and "guaranteed_amount = 60" in variables
    assert "`K`" in variables and "EV for label" not in variables
    # collision-proof: no bare single-letter payoff variables in the aid
    assert "= p ×" not in variables and "reward B =" not in variables and "reward S =" not in variables
    formulas = transplant_block("formulas", **kwargs)
    assert "EV for label `K` = accuracy × conditional_amount" in formulas
    assert "50 points" not in formulas
    numeric = transplant_block("numeric_evs", **kwargs)
    assert "EV for label `K` = 50 points" in numeric
    assert "accuracy × conditional_amount" not in numeric
    full = transplant_block("full", **kwargs)
    assert ("accuracy × conditional_amount" in full and "50 points" in full
            and "higher expected-value label is `M`" in full)
    assert "equal expected value" in transplant_block("comparison", **{**kwargs, "optimal_token": "tie"})


def test_build_transplant_prompt_preserves_abstract_labels_and_records_optimal_role():
    cfg = EvalConfig()
    cfg.loader_mode = "text_injection"
    tp = build_transplant_prompt(_item(), 0.5, cfg, "full", sweep_seed=0, repeat=0)
    assert set(tp.legal_tokens) == set(tp.token_role)
    assert tp.optimal_role == ROLE_CDT
    assert tp.optimal_token == tp.cdt_token
    assert "Helpful calculation aid:" in tp.text
    assert f"The higher expected-value label is `{tp.cdt_token}`" in tp.text
    assert "Valid labels:" in tp.text


def test_all_conditions_are_supported():
    kwargs = dict(
        p=0.7,
        B=100,
        S=60,
        non_cdt_token="A",
        cdt_token="B",
        ev_non_cdt_value=70,
        ev_cdt_value=90,
        optimal_token="B",
    )
    for condition in CONDITIONS:
        assert isinstance(transplant_block(condition, **kwargs), str)
