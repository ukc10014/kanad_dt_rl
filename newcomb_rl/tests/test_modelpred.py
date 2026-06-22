"""Pivot C: the model-based-predictor arm uses the evidential reward with a (model-sourced) p."""
from newcomb_eval.scorer import ROLE_CDT, ROLE_INVALID, ROLE_NON_CDT

from newcomb_rl.reward import (
    ARM_EVIDENTIAL,
    ARM_MODELPRED,
    MODE_EV,
    MODE_REALIZED,
    compute_reward,
    evidential_reward_ev,
    optimal_role,
)

B, S = 100.0, 60.0


def test_modelpred_ev_equals_evidential():
    # The reward formula is identical to evidential; only the *source* of p differs (rollout-time).
    for p in (0.5, 0.7, 0.8, 0.9, 0.99):
        for role in (ROLE_NON_CDT, ROLE_CDT):
            assert compute_reward(role, p, B, S, arm=ARM_MODELPRED, mode=MODE_EV) == \
                   evidential_reward_ev(role, p, B, S)


def test_modelpred_invalid_is_zero():
    assert compute_reward(ROLE_INVALID, 0.9, B, S, arm=ARM_MODELPRED, mode=MODE_EV) == 0.0
    assert compute_reward(ROLE_INVALID, 0.9, B, S, arm=ARM_MODELPRED, mode=MODE_REALIZED) == 0.0


def test_modelpred_optimal_role_tracks_p():
    # p* = (1 + S/B)/2 = 0.8 ; one-box iff p > p*
    assert optimal_role(ARM_MODELPRED, 0.9, B, S) == ROLE_NON_CDT
    assert optimal_role(ARM_MODELPRED, 0.6, B, S) == ROLE_CDT
    # and it agrees with the evidential arm at every p (same formula)
    for p in (0.5, 0.7, 0.8, 0.9):
        assert optimal_role(ARM_MODELPRED, p, B, S) == optimal_role(ARM_EVIDENTIAL, p, B, S)
