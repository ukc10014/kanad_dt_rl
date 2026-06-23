"""CoT inspector: record assembly (reward / EV-optimal / mismatch flag) + HTML render."""
import dataclasses

from newcomb_eval.config import EvalConfig
from newcomb_eval.cot_inspect import inspect_records, render_html


class FakeWrapper:
    """Always 'reasons' the same and reads a fixed answer distribution (no GPU)."""

    def __init__(self, p_non_cdt: float, margin: float):
        self._p, self._m = p_non_cdt, margin

    def generate_one(self, text, max_new_tokens=128, temperature=0.7):
        return "Reasoning about the boxes...\nso my expected values are..."

    def _format(self, prompt):
        return prompt

    def answer_2way(self, text, non_cdt, cdt):
        return {"p_non_cdt": self._p, "margin": self._m, "is_k": 1.0 if self._p >= 0.5 else 0.0}


def _one_item_cfg():
    return dataclasses.replace(EvalConfig(), limit=1)  # 1 item; p* = 0.8, B=100, S=60


def test_records_reward_and_optimality():
    cfg = _one_item_cfg()
    # always answers non_cdt (one-box) — EV-optimal only above p*=0.8.
    recs = inspect_records(cfg, FakeWrapper(0.9, 2.0), n_samples=1, p_grid=(0.5, 0.99))
    by_p = {r["p"]: r for r in recs}
    assert by_p[0.5]["answer_role"] == "non_cdt"
    assert by_p[0.5]["is_optimal"] is False     # one-box at p<p* is sub-optimal
    assert by_p[0.5]["reward"] == 50.0          # ev_non_cdt(0.5) = 0.5*100
    assert by_p[0.99]["is_optimal"] is True      # one-box at p>p* is optimal
    assert by_p[0.99]["reward"] == 99.0


def test_low_margin_flagged_in_html():
    cfg = _one_item_cfg()
    recs = inspect_records(cfg, FakeWrapper(0.52, 0.1), n_samples=1, p_grid=(0.8,))
    assert abs(recs[0]["margin"]) < 1.0
    html = render_html(recs, "test")
    assert "low-margin" in html
    assert "low" in html.split("<tbody>")[1]   # the row carries the low-margin class
    assert "<pre>" in html                       # CoT body rendered (and escaped)


def test_record_count_items_x_p_x_samples():
    cfg = _one_item_cfg()
    recs = inspect_records(cfg, FakeWrapper(0.9, 3.0), n_samples=2, p_grid=(0.9, 0.99))
    assert len(recs) == 1 * 2 * 2  # 1 item × 2 p × 2 samples
