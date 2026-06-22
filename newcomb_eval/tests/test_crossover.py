"""PLAN.md §7: p* matches a hand-worked example; EV_nonCDT(p*) == EV_CDT(p*)."""
import math

import pytest

from newcomb_eval.crossover import crossover_p, ev_cdt, ev_non_cdt


def test_pstar_default_payoffs():
    # B=100, S=60  ->  p* = (1 + 60/100)/2 = 0.8  (hand-worked).
    assert crossover_p(100, 60) == pytest.approx(0.8)


def test_pstar_canonical_newcomb_near_half():
    # Canonical big>>small pushes p* to the floor (~0.5). (PLAN.md §3 flag.)
    assert crossover_p(1000, 1) == pytest.approx(0.5005)


def test_ev_equal_at_crossover():
    for B, S in [(100, 60), (50, 30), (200, 120), (1000, 1), (80, 48)]:
        ps = crossover_p(B, S)
        assert ev_non_cdt(ps, B, S) == pytest.approx(ev_cdt(ps, B, S))


def test_ev_ordering_around_crossover():
    B, S = 100, 60
    ps = crossover_p(B, S)
    # Above p*, one-box (non-CDT) wins; below, two-box (CDT) wins.
    assert ev_non_cdt(ps + 0.1, B, S) > ev_cdt(ps + 0.1, B, S)
    assert ev_non_cdt(ps - 0.1, B, S) < ev_cdt(ps - 0.1, B, S)


def test_degenerate_payoffs_raise():
    with pytest.raises(ValueError):
        crossover_p(0, 10)
    with pytest.raises(ValueError):
        # S > B would push p* above 1.
        crossover_p(10, 100)
