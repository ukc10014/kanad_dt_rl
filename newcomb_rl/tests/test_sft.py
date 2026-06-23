"""SFT/STaR harvest: correct+confident filtering, balance, yield report (no GPU)."""
import dataclasses
import random

from newcomb_rl.rl_config import RLOOConfig
from newcomb_rl.sft import Example, _balance_across_p, harvest, yield_table


class FakeWrapper:
    """Always 'reasons' the same and reads a fixed (is_k, margin) answer."""

    def __init__(self, is_k: float, margin: float):
        self._is_k, self._m = is_k, margin

    def _format(self, prompt):
        return prompt

    def generate_one(self, text, max_new_tokens=160, temperature=0.8):
        return "Reasoning... so the expected value is..."

    def answer_2way(self, text, non_cdt, cdt):
        return {"p_non_cdt": 0.9, "margin": self._m, "is_k": self._is_k}


def _cfg():
    return dataclasses.replace(RLOOConfig(), eval=dataclasses.replace(
        RLOOConfig().eval, limit=1))  # 1 item; p* = 0.8, train_p = .5..99


def test_harvest_keeps_only_correct_high_p_when_model_always_one_boxes():
    cfg = _cfg()
    # always answers non_cdt (one-box), confident → kept only where optimal==non_cdt (p>0.8)
    ex, report = harvest(cfg, FakeWrapper(1.0, 3.0), samples_per_cell=2, oversample_low=2,
                         balance=False)
    assert all(e.role == "non_cdt" and e.p in (0.9, 0.99) for e in ex)
    assert report[0.9]["kept"] == 2 and report[0.99]["kept"] == 2
    assert report[0.5]["kept"] == 0                       # one-box at p<p* is wrong → rejected
    assert report[0.5]["total"] == 2 * 2                  # low p oversampled (samples × oversample)
    assert report[0.99]["total"] == 2                     # high p not oversampled


def test_harvest_margin_filter_rejects_low_confidence():
    cfg = _cfg()
    ex, report = harvest(cfg, FakeWrapper(1.0, 0.2), samples_per_cell=3, balance=False)  # margin<1
    assert ex == []
    assert all(report[p]["kept"] == 0 for p in report)


def test_balance_across_p_downsamples_to_min_bucket():
    rng = random.Random(0)
    exs = ([Example("i", 0.5, "", "", "M", "cdt", 2.0)] * 1 +
           [Example("i", 0.9, "", "", "K", "non_cdt", 2.0)] * 5)
    out = _balance_across_p(exs, rng)
    counts = {}
    for e in out:
        counts[e.p] = counts.get(e.p, 0) + 1
    assert counts == {0.5: 1, 0.9: 1}                     # both buckets capped at min (=1)


def test_yield_table_renders_rows():
    report = {0.5: {"kept": 0, "total": 8}, 0.99: {"kept": 4, "total": 4}}
    t = yield_table(report)
    assert "0.50" in t and "0.99" in t and "kept/total" in t
