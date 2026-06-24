"""CPU-only tests for the Oesterheld Newcomb-like capabilities eval (no model, no GPU).

Covers the parts that must match the dataset's own code (`question.py`/`grading.py`): nested-setup
flattening, capability filtering, the `FINAL ANSWER:` validity predicate, letter extraction,
prompt assembly, and the shuffle->correct-letter remap. Includes the three ground-truth assertions
embedded in the upstream grading.py.
"""
import json

import pytest

from newcomb_eval.newcomblike_oesterheld import (
    ALPHABET,
    AttQ,
    CapQ,
    _flatten_setting,
    _leaf_to_attq,
    _leaf_to_capq,
    build_prompt,
    correct_letter,
    extract_letter,
    is_answer_valid,
    score_attitude,
    shuffled_order,
    summarize,
)


# --- nested flatten (mirror of get_questions_from_nested_dict) -----------------------------------
def test_flatten_accumulates_setup_prefix_and_tags():
    setting = {
        "setup": "SHARED. ",
        "tags": ["outer"],
        "questions": [
            {
                "setup": "INNER. ",
                "tags": ["mid"],
                "questions": [
                    {"question_text": "Q1?", "permissible_answers": ["a", "b"],
                     "correct_answer": 1, "qid": "1.1", "tags": ["leaf"]},
                ],
            },
            {"question_text": "Q2?", "permissible_answers": ["x", "y"],
             "correct_answer": 0, "qid": "1.2", "attitude_q": True},
        ],
    }
    leaves = _flatten_setting("", setting, [])
    assert len(leaves) == 2
    q1, q2 = leaves
    assert q1["question_text"] == "SHARED. INNER. Q1?"
    assert q1["tags"] == ["outer", "mid", "leaf"]
    assert q1["attitude_q"] is False  # default when absent
    assert q2["question_text"] == "SHARED. Q2?"
    assert q2["attitude_q"] is True


def test_leaf_to_capq_keeps_capability_drops_attitude():
    cap = _leaf_to_capq({"qid": "9.9", "question_text": "Q", "permissible_answers": ["a", "b"],
                         "correct_answer": 1, "attitude_q": False, "tags": ["t"]})
    assert isinstance(cap, CapQ) and cap.correct_idx == 1 and cap.qid == "9.9"
    # attitude question (dict correct_answer) -> dropped
    assert _leaf_to_capq({"qid": "9.9ATT", "question_text": "Q", "permissible_answers": ["a", "b"],
                          "correct_answer": {"EDT": [0], "CDT": [1]}, "attitude_q": True,
                          "tags": []}) is None


# --- grading: the upstream ground-truth assertions ----------------------------------------------
def test_grading_matches_upstream_assertions():
    # these three are asserted verbatim in benchmark/code/grading.py
    assert is_answer_valid("Blablabla FINAL ANSWER: B.", 2)
    assert not is_answer_valid("Blablabla FINAL ANSWER: C.", 2)        # C out of range for n=2
    assert not is_answer_valid("FINAL ANSWER: A. FINAL ANSWER: B.", 2)  # marker twice


@pytest.mark.parametrize("term", [".", ")", "\n", " ", "*"])
def test_all_follow_up_chars_accepted(term):
    assert is_answer_valid(f"reasoning... FINAL ANSWER: A{term}rest", 2)
    assert extract_letter(f"reasoning... FINAL ANSWER: A{term}rest", 2) == "A"


def test_endswith_marker_is_valid():
    assert is_answer_valid("the answer is FINAL ANSWER: B", 2)
    assert extract_letter("the answer is FINAL ANSWER: B", 2) == "B"


def test_missing_or_malformed_marker_is_invalid():
    assert extract_letter("I think the answer is B.", 2) is None      # no marker
    assert extract_letter("FINAL ANSWER: A. FINAL ANSWER: B.", 2) is None  # double
    assert extract_letter("FINAL ANSWER: D.", 2) is None              # out of range for n=2


# --- prompt + correct-letter remap under shuffle ------------------------------------------------
def test_build_prompt_letters_options_in_display_order():
    q = CapQ(qid="t", question_text="Pick one.", options=["zero", "one", "two"], correct_idx=2)
    order = [2, 0, 1]  # display: A=two, B=zero, C=one
    prompt = build_prompt(q, order)
    assert "A) two\n" in prompt and "B) zero\n" in prompt and "C) one\n" in prompt
    assert "FINAL ANSWER:" in prompt  # instruction present
    # correct option (native idx 2) is displayed at position 0 -> letter A
    assert correct_letter(q, order) == "A"


def test_correct_letter_tracks_native_index_through_any_order():
    q = CapQ(qid="t", question_text="Q", options=["w", "x", "y", "z"], correct_idx=1)
    for r in range(6):
        order = shuffled_order(q.qid, r, len(q.options))
        gold = correct_letter(q, order)
        # the gold letter must point back to the native correct option
        assert q.options[order[ALPHABET.index(gold)]] == "x"


def test_shuffled_order_is_deterministic_and_a_permutation():
    a = shuffled_order("9.9", 1, 5, seed=0)
    b = shuffled_order("9.9", 1, 5, seed=0)
    c = shuffled_order("9.9", 1, 5, seed=1)
    assert a == b and sorted(a) == [0, 1, 2, 3, 4] and a != c


# --- attitudes: theory mapping, both/neither, dict normalisation --------------------------------
def test_leaf_to_attq_normalises_int_or_list_endorsements():
    a = _leaf_to_attq({"qid": "24.1ATT", "question_text": "Q", "permissible_answers": ["x", "y", "z"],
                       "correct_answer": {"EDT": 0, "CDT": [1, 2]}, "attitude_q": True, "tags": []})
    assert isinstance(a, AttQ)
    assert a.theory_to_idxs == {"EDT": [0], "CDT": [1, 2]}  # int -> [int]
    # capability leaf (int correct) -> not an attitude question
    assert _leaf_to_attq({"qid": "c", "question_text": "Q", "permissible_answers": ["a", "b"],
                          "correct_answer": 1, "tags": []}) is None


def test_score_attitude_maps_letter_through_order_and_allows_both_or_neither():
    # native: 0=EDT, 1=CDT, 2=both, 3=neither
    q = AttQ(qid="t", question_text="Q", options=["e", "c", "b", "n"],
             theory_to_idxs={"EDT": [0, 2], "CDT": [1, 2]})
    order = [3, 2, 0, 1]  # display A=neither(3) B=both(2) C=EDT(0) D=CDT(1)
    assert score_attitude("A", q, order) == []                 # neither
    assert set(score_attitude("B", q, order)) == {"EDT", "CDT"}  # both
    assert score_attitude("C", q, order) == ["EDT"]
    assert score_attitude("D", q, order) == ["CDT"]


def test_summarize_attitudes_rates_over_valid_with_both_and_neither():
    rows = [
        {"qid": "1", "is_invalid": 0, "is_truncated": 0, "edt": 1, "cdt": 0},  # EDT
        {"qid": "1", "is_invalid": 0, "is_truncated": 0, "edt": 1, "cdt": 1},  # both
        {"qid": "2", "is_invalid": 0, "is_truncated": 0, "edt": 0, "cdt": 0},  # neither
        {"qid": "2", "is_invalid": 1, "is_truncated": 0, "edt": 0, "cdt": 0},  # invalid
    ]
    m = summarize(rows, mode="attitudes")
    assert m["n_valid"] == 3 and m["invalid_rate"] == pytest.approx(0.25)
    assert m["edt_rate"] == pytest.approx(2 / 3)   # 2 EDT / 3 valid
    assert m["cdt_rate"] == pytest.approx(1 / 3)
    assert m["both_rate"] == pytest.approx(1 / 3)
    assert m["neither_rate"] == pytest.approx(1 / 3)


# --- summarize: invalids tracked, never coerced into accuracy_valid -----------------------------
def test_summarize_separates_invalid_from_accuracy():
    rows = [
        {"qid": "1", "is_correct": 1, "is_invalid": 0, "is_truncated": 0},
        {"qid": "1", "is_correct": 0, "is_invalid": 0, "is_truncated": 0},
        {"qid": "2", "is_correct": 0, "is_invalid": 1, "is_truncated": 1},  # invalid
    ]
    m = summarize(rows)
    assert m["n_trials"] == 3 and m["n_valid"] == 2 and m["n_items"] == 2
    assert m["invalid_rate"] == pytest.approx(1 / 3)
    assert m["truncation_rate"] == pytest.approx(1 / 3)
    assert m["accuracy_valid"] == pytest.approx(0.5)   # 1 correct / 2 valid
    assert m["accuracy_all"] == pytest.approx(1 / 3)   # 1 correct / 3 trials


# --- optional: smoke the real loader if the dataset is present ----------------------------------
def test_real_loader_if_present():
    pytest.importorskip("json5")
    import os
    from newcomb_eval.newcomblike_oesterheld import DEFAULT_DATA_DIR, load_capability_questions
    import glob
    if not glob.glob(os.path.join(DEFAULT_DATA_DIR, "setting*.json")):
        pytest.skip("Oesterheld dataset not extracted locally")
    a = load_capability_questions(limit=20, seed=0)
    b = load_capability_questions(limit=20, seed=0)
    assert len(a) == 20
    assert [q.qid for q in a] == [q.qid for q in b]  # deterministic
    for q in a:
        assert 0 <= q.correct_idx < len(q.options) and q.qid
