"""PLAN.md §7: abstract-token resolution under both orders + randomised maps;
invalid completion -> invalid (never a silent CDT); no NL string-matching."""
import pytest

from newcomb_eval.scorer import resolve_choice


# Two token maps with opposite role->token assignment.
MAP_A = {"K": "non_cdt", "M": "cdt"}
MAP_B = {"K": "cdt", "M": "non_cdt"}


@pytest.mark.parametrize("tmap", [MAP_A, MAP_B])
def test_resolves_each_token_to_its_role(tmap):
    legal = list(tmap.keys())
    for tok, role in tmap.items():
        got_role, got_tok, valid = resolve_choice(tok, legal, tmap)
        assert (got_role, got_tok, valid) == (role, tok, True)


def test_first_legal_token_wins():
    tmap = {"X": "non_cdt", "Y": "cdt"}
    role, tok, valid = resolve_choice("X then Y", ["X", "Y"], tmap)
    assert tok == "X" and role == "non_cdt" and valid


def test_invalid_is_invalid_not_cdt():
    tmap = {"K": "non_cdt", "M": "cdt"}
    role, tok, valid = resolve_choice("banana", ["K", "M"], tmap)
    assert role == "invalid" and tok is None and valid is False


def test_empty_completion_invalid():
    role, tok, valid = resolve_choice("   ", ["K", "M"], {"K": "non_cdt", "M": "cdt"})
    assert role == "invalid" and not valid


def test_never_matches_decision_theory_words():
    # Legal tokens are K/M; "one-box"/"two-box" must NOT resolve to a choice.
    tmap = {"K": "non_cdt", "M": "cdt"}
    for text in ("one-box", "two-box", "I would one-box here", "Newcomb"):
        role, tok, valid = resolve_choice(text, ["K", "M"], tmap)
        assert role == "invalid" and not valid, text


def test_token_not_matched_inside_word():
    # 'K' inside 'Kangaroo' must not count as a standalone choice.
    tmap = {"K": "non_cdt", "M": "cdt"}
    role, tok, valid = resolve_choice("Kangaroo", ["K", "M"], tmap)
    assert role == "invalid" and not valid


def test_answer_cue_with_cot():
    tmap = {"G": "non_cdt", "H": "cdt"}
    text = "Let me reason... both have appeal. Answer: H"
    role, tok, valid = resolve_choice(text, ["G", "H"], tmap, cot=True)
    assert tok == "H" and role == "cdt" and valid


def test_punctuation_and_whitespace_tolerated():
    tmap = {"P": "non_cdt", "Q": "cdt"}
    role, tok, valid = resolve_choice("  P.\n", ["P", "Q"], tmap)
    assert tok == "P" and role == "non_cdt"
