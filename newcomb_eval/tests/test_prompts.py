"""PLAN.md §7: determinism, seed-sensitivity, and correct p injection."""
from newcomb_eval.config import EvalConfig
from newcomb_eval.data.loader import load
from newcomb_eval.prompts import build_prompt


def _item():
    cfg = EvalConfig()
    return load(cfg.dataset_path, cfg)[0], cfg


def test_same_seed_identical_prompt():
    item, cfg = _item()
    a = build_prompt(item, 0.8, cfg.prompt, sweep_seed=0, repeat=0)
    b = build_prompt(item, 0.8, cfg.prompt, sweep_seed=0, repeat=0)
    assert a.text == b.text
    assert a.token_role == b.token_role
    assert a.order == b.order


def test_different_seed_changes_tokens_or_order():
    item, cfg = _item()
    a = build_prompt(item, 0.8, cfg.prompt, sweep_seed=0, repeat=0)
    # Scan repeats; at least one must differ in token assignment or order.
    differs = any(
        (build_prompt(item, 0.8, cfg.prompt, sweep_seed=0, repeat=r).order != a.order)
        or (build_prompt(item, 0.8, cfg.prompt, sweep_seed=0, repeat=r).token_role != a.token_role)
        for r in range(1, 8)
    )
    assert differs


def test_p_injected_and_marker_removed():
    item, cfg = _item()
    rp = build_prompt(item, 0.8, cfg.prompt, sweep_seed=0, repeat=0)
    assert "{{PREDICTOR_ACCURACY}}" not in rp.text
    assert "80%" in rp.text


def test_p_value_tracks_argument():
    item, cfg = _item()
    assert "50%" in build_prompt(item, 0.5, cfg.prompt).text
    assert "99%" in build_prompt(item, 0.99, cfg.prompt).text


def test_legal_tokens_and_roles_consistent():
    item, cfg = _item()
    rp = build_prompt(item, 0.7, cfg.prompt)
    assert len(rp.legal_tokens) == 2
    assert set(rp.token_role.values()) == {"cdt", "non_cdt"}
    assert set(rp.legal_tokens) == set(rp.token_role.keys())
    # Both tokens must actually appear in the rendered options.
    for tok in rp.legal_tokens:
        assert f"{tok}." in rp.text


def test_no_decision_theory_words_as_labels():
    item, cfg = _item()
    rp = build_prompt(item, 0.8, cfg.prompt)
    for banned in ("one-box", "two-box", "Newcomb"):
        assert banned not in rp.text
