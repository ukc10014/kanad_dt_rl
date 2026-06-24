"""CPU-only tests for the credence probe (no model, no GPU).

Covers: conditional-prize entity extraction on every dataset item; the committed-action prefix
references the seeded label + the dataset option text (module authors no decision-theory phrasing);
2-way credence scoring math; and the aggregation/headline that turns per-sample credences into the
gap-vs-(2p-1) signal that distinguishes "represented" from "not-represented".
"""
import math
import threading

import pytest

from newcomb_eval.config import EvalConfig
from newcomb_eval.credence_probe import (
    ACTIONS,
    aggregate_credence,
    build_assistant_prefix,
    build_credence_base,
    conditional_prize_name,
    credence_cue,
    credence_headline,
    score_credence,
)
from newcomb_eval.data.loader import load
from newcomb_eval.model import ModelWrapper
from newcomb_eval.prompts import render_scenario_block

FORBIDDEN = ("one-box", "two-box", "Newcomb", "predictor")


def _items():
    cfg = EvalConfig()
    return load(cfg.dataset_path, cfg), cfg


# --- entity extraction ---------------------------------------------------------------------------
def test_conditional_prize_name_on_every_item():
    items, _ = _items()
    assert len(items) >= 20
    for it in items:
        name = conditional_prize_name(it.scenario)
        assert name, f"{it.id}: empty entity"
        assert name in it.scenario, f"{it.id}: extracted {name!r} not a substring of scenario"
        # the entity is the conditional (100-point) prize, never the guaranteed (60-point) one
        assert "60 points" not in name and "100" not in name


def test_conditional_prize_name_specific_cases():
    assert conditional_prize_name("X. Container Q holds 100 points if F predicted.") == "Container Q"
    assert conditional_prize_name("Y. The MAJOR prize is worth 100 points if it predicted.") == "The MAJOR prize"
    assert conditional_prize_name("Z. Envelope GOLD contains 100 points if PRIOR judged.") == "Envelope GOLD"
    assert conditional_prize_name("no marker here") is None


# --- prompt assembly -----------------------------------------------------------------------------
class _FmtStub:
    """Wrapper stub exposing only _format (identity) for build_credence_base."""

    def __init__(self):
        self._lock = threading.Lock()

    def _format(self, prompt):
        return "<<USER>>" + prompt + "<<ASSISTANT>>"


def test_build_assistant_prefix_uses_dataset_text_not_authored_phrasing():
    pre = build_assistant_prefix("K", "Take only container Q.", "the prize holds exactly ")
    assert "option K:" in pre and "Take only container Q." in pre
    assert pre.endswith("the prize holds exactly ")


def test_build_credence_base_references_seeded_label_and_option():
    items, cfg = _items()
    it = items[0]
    block = render_scenario_block(it, 0.8, cfg.prompt, mode=cfg.loader_mode)
    onebox_label = next(t for t, r in block.token_role.items() if r == "non_cdt")
    cdt_label = next(t for t, r in block.token_role.items() if r == "cdt")

    base_ma, tf, te = build_credence_base(_FmtStub(), block, it, action="major_alone",
                                          phrasing="report", variant="outcome",
                                          entity=conditional_prize_name(it.scenario))
    # major_alone -> non_cdt label + non_cdt option text; "100"/"0" credence tokens
    assert f"option {onebox_label}:" in base_ma and it.non_cdt_text in base_ma
    assert (tf, te) == ("100", "0")

    base_bo, _, _ = build_credence_base(_FmtStub(), block, it, action="both",
                                        phrasing="report", variant="outcome",
                                        entity=conditional_prize_name(it.scenario))
    assert f"option {cdt_label}:" in base_bo and it.cdt_text in base_bo


def test_credence_tokens_never_forbidden_strings():
    for variant in ("outcome", "prediction"):
        for phrasing in ("report", "revealed", "observer"):
            cue, tf, te = credence_cue(variant, phrasing, onebox_label="K", entity="Box K")
            assert tf in ("100", "True") and te in ("0", "False")
            for tok in (tf, te):
                assert tok not in FORBIDDEN  # scorer compares these, never a DT label


# --- scoring -------------------------------------------------------------------------------------
class _TwoWayStub:
    def __init__(self, out):
        self._out = out

    def answer_2way(self, text, a, b):
        return self._out


def test_score_credence_passes_through_two_way():
    out = score_credence(_TwoWayStub({"p_non_cdt": 0.83, "margin": 1.6, "is_k": 1.0}),
                         "base", "100", "0")
    assert abs(out["credence_full"] - 0.83) < 1e-9
    assert out["is_degenerate"] is False


def test_score_credence_flags_unresolved_token():
    out = score_credence(_TwoWayStub({"p_non_cdt": float("nan"), "margin": float("nan"),
                                      "is_k": float("nan")}), "base", "100", "0")
    assert out["is_degenerate"] is True


def test_real_two_way_math_matches_logprob_renorm():
    """Sanity: answer_2way's 2-way softmax over two single-token logits == manual softmax."""
    import torch
    torch.manual_seed(0)
    # exercised indirectly; here just assert the renorm identity the probe relies on
    logits = torch.tensor([2.0, -1.0])
    p = torch.softmax(logits, dim=-1)[0].item()
    m = max(2.0, -1.0)
    manual = math.exp(2.0 - m) / (math.exp(2.0 - m) + math.exp(-1.0 - m))
    assert abs(p - manual) < 1e-6


# --- aggregation ---------------------------------------------------------------------------------
def _synth_rows(cred_fn, ps=(0.5, 0.8, 0.99), items=("i1", "i2", "i3"), phrasings=("report",)):
    rows = []
    for it in items:
        for p in ps:
            for ph in phrasings:
                for action in ACTIONS:
                    rows.append({"item_id": it, "p": p, "action": action, "phrasing": ph,
                                 "credence_full": cred_fn(p, action)})
    return rows


def test_aggregate_edt_reproduces_2pm1():
    rows = _synth_rows(lambda p, a: p if a == "major_alone" else 1 - p)
    df = aggregate_credence(rows, p_star=0.8)
    for _, r in df.iterrows():
        assert abs(r["gap_mean"] - (2 * r["p"] - 1)) < 1e-9
        assert abs(r["gap_mean"] - r["ref_2pm1"]) < 1e-9
        assert abs(r["sum_check_mean"] - 1.0) < 1e-9      # p + (1-p) = 1
        assert not r["is_degenerate"]
    assert abs(df[df["p"] == 0.5].iloc[0]["gap_mean"]) < 1e-9  # symmetry control


def test_aggregate_cdt_is_flat_zero():
    rows = _synth_rows(lambda p, a: 0.5)
    df = aggregate_credence(rows, p_star=0.8)
    assert (df["gap_mean"].abs() < 1e-9).all()


def test_aggregate_flags_degenerate_saturation():
    rows = _synth_rows(lambda p, a: 0.995)  # pinned high regardless of action/p
    df = aggregate_credence(rows, p_star=0.8)
    assert df["is_degenerate"].all()


def test_aggregate_computes_phrasing_sd():
    # two phrasings disagree for the same (item,p,action) -> nonzero phrasing SD
    rows = []
    for p in (0.5, 0.8, 0.99):
        for a in ACTIONS:
            rows.append({"item_id": "i1", "p": p, "action": a, "phrasing": "report",
                         "credence_full": 0.3})
            rows.append({"item_id": "i1", "p": p, "action": a, "phrasing": "revealed",
                         "credence_full": 0.7})
    df = aggregate_credence(rows, p_star=0.8)
    assert (df["phrasing_sd_mean"] > 0.1).all()


def test_aggregate_joins_action_margin():
    rows = _synth_rows(lambda p, a: p if a == "major_alone" else 1 - p)
    df = aggregate_credence(rows, p_star=0.8, action_margin_by_p={0.5: 3.4, 0.8: 3.3, 0.99: 3.5})
    assert abs(df[df["p"] == 0.5].iloc[0]["action_margin_mean"] - 3.4) < 1e-9


# --- headline ------------------------------------------------------------------------------------
def test_headline_edt_slope_near_one_cdt_near_zero():
    edt = credence_headline(_synth_rows(lambda p, a: p if a == "major_alone" else 1 - p), p_star=0.8)
    assert abs(edt["fit_slope_gap_on_2pm1"] - 1.0) < 1e-6   # gap == 2p-1 == x  => slope 1
    assert abs(edt["gap_at_0.5"]) < 1e-9
    assert edt["rmse_vs_2pm1"] < 1e-6

    cdt = credence_headline(_synth_rows(lambda p, a: 0.5), p_star=0.8)
    assert abs(cdt["fit_slope_gap_on_2pm1"]) < 1e-6
    assert cdt["rmse_vs_2pm1"] > 0.3   # flat 0 is far from the rising 2p-1


def test_headline_reports_action_margin_contrast():
    hl = credence_headline(_synth_rows(lambda p, a: p if a == "major_alone" else 1 - p),
                           p_star=0.8, action_margin_by_p={0.5: 3.4, 0.8: 3.3, 0.99: 3.5})
    assert abs(hl["action_margin_hi_lo"] - (3.5 - 3.4)) < 1e-9


# --- direct (free-form numeric) elicitation parsing ----------------------------------------------
def test_parse_probability_forms():
    from newcomb_eval.credence_probe import parse_probability
    assert parse_probability("85%") == pytest.approx(0.85)
    assert parse_probability("about 90, I think") == pytest.approx(0.90)
    assert parse_probability("0.7") == pytest.approx(0.7)
    assert parse_probability("100") == pytest.approx(1.0)
    assert parse_probability("250") == pytest.approx(1.0)        # clamped
    assert parse_probability("no number here") is None
    assert parse_probability("") is None


# --- ladder coherence primitives -----------------------------------------------------------------
def test_ladder_monotonicity():
    from newcomb_eval.credence_ladder import monotonicity
    assert monotonicity({0.5: 0.0, 0.8: 0.3, 0.99: 0.9}) == pytest.approx(1.0)   # rising
    assert monotonicity({0.5: 0.9, 0.8: 0.3, 0.99: 0.0}) == pytest.approx(0.0)   # falling
    assert monotonicity({0.5: 0.0, 0.8: 0.5, 0.99: 0.4}) == pytest.approx(0.5)   # one of two steps


def test_ladder_series_agreement():
    from newcomb_eval.credence_ladder import series_agreement
    a = {0.5: 0.0, 0.8: 0.6, 0.99: 0.98}
    assert series_agreement(a, a) == pytest.approx(1.0)                  # identical -> r=1
    flipped = {k: -v for k, v in a.items()}
    assert series_agreement(a, flipped) == pytest.approx(-1.0)           # anti -> r=-1
    assert series_agreement(a, {0.5: 0.0, 0.8: 0.0, 0.99: 0.0}) != series_agreement(a, a)  # no var -> nan


# --- instrument fix: baseline-subtracted gap + one-sided saturation -------------------------------
def test_aggregate_adds_baseline_subtracted_gap():
    # gap = 2p-1 + 0.4 pedestal: gap_adj must recover 2p-1 (0 at p=0.5)
    # gap = (p+0.2)-(1-p-0.2) = 2p-0.6, i.e. (2p-1) + 0.4 pedestal
    rows = _synth_rows(lambda p, a: (p + 0.2) if a == "major_alone" else (1 - p - 0.2))
    df = aggregate_credence(rows, p_star=0.8)
    base = df[df["p"] == 0.5].iloc[0]
    assert base["gap_mean"] == pytest.approx(0.4)            # pedestal present in raw gap
    assert base["gap_adj_mean"] == pytest.approx(0.0)        # removed at baseline
    hi = df[df["p"] == 0.99].iloc[0]
    assert hi["gap_adj_mean"] == pytest.approx((2 * 0.99 - 1))   # = raw gap minus the 0.8 pedestal


def test_headline_adjusted_slope_strips_pedestal_and_flags_saturation():
    # forced-token failure mode: major_alone pinned at 1.0 for every p, both flat -> p-blind pedestal
    sat = credence_headline(_synth_rows(lambda p, a: 1.0 if a == "major_alone" else 0.27), p_star=0.8)
    assert sat["onesided_saturation"] is True
    assert abs(sat["fit_slope_gap_adj_on_2pm1"]) < 1e-6          # no p-modulation once de-pedestalled
    assert sat["gap_at_0.5"] > 0.5                              # big bias pedestal

    # clean p-tracking on a pedestal: gap = 2p-1 + 0.3 -> adjusted slope ~1, not saturated
    clean = credence_headline(_synth_rows(lambda p, a: (p + 0.15) if a == "major_alone" else (1 - p - 0.15)),
                              p_star=0.8)
    assert clean["fit_slope_gap_adj_on_2pm1"] == pytest.approx(1.0, abs=1e-6)
    assert clean["onesided_saturation"] is False


# --- mechanism dataset generation -----------------------------------------------------------------
def test_mechanism_generator_swaps_clause_and_keeps_marker(tmp_path):
    import json
    from newcomb_eval.gen_mechanism_dataset import BASE_CLAUSE, MECHANISMS, generate, LEVELS
    paths = generate(out_dir=str(tmp_path))
    assert len(paths) == len(MECHANISMS) == len(LEVELS)
    for level, _ in LEVELS:
        items = json.load(open(tmp_path / f"dataset_mech_{level}.json"))
        assert len(items) == 20
        for it in items:
            assert "{{PREDICTOR_ACCURACY}}" in it["question"]   # marker preserved -> loader-valid
            assert it["mechanism_level"] == level
        if level == "m3":
            assert any("exact copy" in it["question"] for it in items)  # FDT framing present
            assert all(BASE_CLAUSE not in it["question"] for it in items)  # statistical clause gone
