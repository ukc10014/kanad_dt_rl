"""Lever 1b: EV-balanced paired sampler — equal below/above p*, paired on the same item."""
from newcomb_rl.rl_config import RLOOConfig
from newcomb_rl.sampler import NewcombSampler


def _sampler():
    return NewcombSampler(RLOOConfig())


def test_sample_paired_count_and_pairing():
    s = _sampler()
    pairs = 4
    out = s.sample_paired(pairs)
    assert len(out) == 2 * pairs
    for k in range(pairs):
        lo, hi = out[2 * k], out[2 * k + 1]
        assert lo.item_id == hi.item_id          # paired on the same item
        assert lo.p < lo.p_star <= hi.p_star      # first is below p*, ...
        assert hi.p > hi.p_star                    # ... second is above p*


def test_sample_paired_is_ev_balanced():
    s = _sampler()
    out = s.sample_paired(6)
    below = sum(1 for sp in out if sp.p < sp.p_star)
    above = sum(1 for sp in out if sp.p > sp.p_star)
    assert below == above == 6  # equal pressure each side of the crossover


def test_sample_paired_respects_train_split():
    s = _sampler()
    train = set(s.train_p)
    holdout = set(s.cfg.eval.sweep.holdout_p)
    out = s.sample_paired(8)
    for sp in out:
        assert sp.p in train
        assert sp.p not in holdout


def test_sample_paired_distinct_p_within_pair():
    s = _sampler()
    out = s.sample_paired(10)
    for k in range(10):
        assert out[2 * k].p != out[2 * k + 1].p
