"""Pivot A: answer-logprob renormalisation + sweep aggregation (no GPU)."""
import math
import threading

from newcomb_eval.logprob_sweep import aggregate_logprobs, roles_to_tokens
from newcomb_eval.model import ModelWrapper


class _Stub:
    """Minimal duck-type exercising ModelWrapper.answer_logprobs without loading a model.

    answer_logprobs only touches self._lock, self._format, self._continuation_logprob.
    """

    def __init__(self, logprobs: dict):
        self._lock = threading.Lock()
        self._lp = logprobs

    def _format(self, prompt):
        return prompt

    def _continuation_logprob(self, base, cont):
        return self._lp[cont]


def test_roles_to_tokens():
    assert roles_to_tokens({"K": "non_cdt", "M": "cdt"}) == ("K", "M")
    assert roles_to_tokens({"X": "cdt", "Y": "non_cdt"}) == ("Y", "X")


def test_answer_logprobs_renormalises_two_way():
    # logP(K)=log .75, logP(M)=log .25 -> P(non_cdt) = .75 exactly
    stub = _Stub({"K": math.log(0.75), "M": math.log(0.25)})
    out = ModelWrapper.answer_logprobs(stub, "prompt", "K", "M")
    assert abs(out["p_non_cdt"] - 0.75) < 1e-9
    assert out["is_k"] == 1.0
    assert abs(out["margin"] - (math.log(0.75) - math.log(0.25))) < 1e-9


def test_answer_logprobs_argmax_can_be_cdt():
    stub = _Stub({"K": math.log(0.2), "M": math.log(0.8)})
    out = ModelWrapper.answer_logprobs(stub, "prompt", "K", "M")
    assert abs(out["p_non_cdt"] - 0.2) < 1e-9
    assert out["is_k"] == 0.0
    assert out["margin"] < 0


def test_aggregate_logprobs_means_and_krate():
    rows = [
        {"p": 0.5, "p_non_cdt": 0.6, "margin": 1.0, "is_k": 1.0},
        {"p": 0.5, "p_non_cdt": 0.4, "margin": -1.0, "is_k": 0.0},
        {"p": 0.9, "p_non_cdt": 0.9, "margin": 2.0, "is_k": 1.0},
        {"p": 0.9, "p_non_cdt": 0.7, "margin": 1.0, "is_k": 1.0},
    ]
    df = aggregate_logprobs(rows)
    lo = df[df["p"] == 0.5].iloc[0]
    hi = df[df["p"] == 0.9].iloc[0]
    assert abs(lo["p_non_cdt_mean"] - 0.5) < 1e-9
    assert abs(lo["k_rate_argmax"] - 0.5) < 1e-9
    assert abs(hi["p_non_cdt_mean"] - 0.8) < 1e-9
    assert hi["k_rate_argmax"] == 1.0
    assert lo["n"] == 2 and hi["n"] == 2
    # hi has a higher continuous one-box signal than lo (the graded slope)
    assert hi["p_non_cdt_mean"] > lo["p_non_cdt_mean"]
