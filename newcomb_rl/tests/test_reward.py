"""Reward optima are the experiment's ground truth: evidential -> track-p, causal -> two-box."""
import pytest

from newcomb_eval.crossover import crossover_p, ev_cdt, ev_non_cdt
from newcomb_eval.scorer import ROLE_CDT, ROLE_INVALID, ROLE_NON_CDT
from newcomb_rl.reward import (
    ARM_CAUSAL,
    ARM_EVIDENTIAL,
    compute_reward,
    optimal_role,
    realized_points,
)

B, S = 100.0, 60.0
PSTAR = crossover_p(B, S)  # 0.8


def test_evidential_ev_matches_crossover():
    for p in [0.5, 0.7, 0.8, 0.9, 0.99]:
        assert compute_reward(ROLE_NON_CDT, p, B, S, arm=ARM_EVIDENTIAL) == pytest.approx(ev_non_cdt(p, B, S))
        assert compute_reward(ROLE_CDT, p, B, S, arm=ARM_EVIDENTIAL) == pytest.approx(ev_cdt(p, B, S))


def test_evidential_optimum_tracks_pstar():
    # Below p*: two-box (CDT) is EV-best; above p*: one-box (non-CDT).
    assert optimal_role(ARM_EVIDENTIAL, PSTAR - 0.1, B, S) == ROLE_CDT
    assert optimal_role(ARM_EVIDENTIAL, PSTAR + 0.1, B, S) == ROLE_NON_CDT


def test_causal_optimum_is_always_two_box():
    for p in [0.1, 0.5, 0.8, 0.99]:
        assert optimal_role(ARM_CAUSAL, p, B, S) == ROLE_CDT
        # two-box strictly dominates one-box by S, for any p (causal arm).
        one = compute_reward(ROLE_NON_CDT, p, B, S, arm=ARM_CAUSAL)
        two = compute_reward(ROLE_CDT, p, B, S, arm=ARM_CAUSAL)
        assert two - one == pytest.approx(S)


def test_invalid_is_zero_both_arms():
    for arm in (ARM_EVIDENTIAL, ARM_CAUSAL):
        assert compute_reward(ROLE_INVALID, 0.8, B, S, arm=arm) == 0.0


def test_realized_points_logic():
    # one-box gets B iff full; two-box gets B(iff full)+S; invalid 0.
    assert realized_points(ROLE_NON_CDT, True, B, S) == B
    assert realized_points(ROLE_NON_CDT, False, B, S) == 0.0
    assert realized_points(ROLE_CDT, True, B, S) == B + S
    assert realized_points(ROLE_CDT, False, B, S) == S
    assert realized_points(ROLE_INVALID, True, B, S) == 0.0


def test_realized_evidential_mean_approaches_ev():
    import random
    rng = random.Random(0)
    p = 0.7
    n = 4000
    one = sum(compute_reward(ROLE_NON_CDT, p, B, S, arm=ARM_EVIDENTIAL, mode="realized", rng=rng)
              for _ in range(n)) / n
    assert one == pytest.approx(ev_non_cdt(p, B, S), abs=4.0)  # ~ p*B = 70


def test_causal_realized_two_box_dominates_with_shared_fill():
    # With a shared per-group fill, two-box >= one-box sample-by-sample (by S when comparing same fill).
    for fill in (True, False):
        one = compute_reward(ROLE_NON_CDT, 0.5, B, S, arm=ARM_CAUSAL, mode="realized", causal_fill=fill)
        two = compute_reward(ROLE_CDT, 0.5, B, S, arm=ARM_CAUSAL, mode="realized", causal_fill=fill)
        assert two - one == pytest.approx(S)
