"""Sampler must only ever draw TRAIN p values (never held-out), and be seed-deterministic."""
from newcomb_rl.rl_config import RLOOConfig
from newcomb_rl.sampler import NewcombSampler


def test_only_train_p_drawn_never_holdout():
    cfg = RLOOConfig()
    hold = set(cfg.eval.sweep.holdout_p)
    s = NewcombSampler(cfg)
    draws = s.sample(400)
    ps = {sp.p for sp in draws}
    assert ps.issubset(set(cfg.train_p_grid))
    assert ps.isdisjoint(hold)


def test_payoffs_and_pstar_attached():
    cfg = RLOOConfig()
    s = NewcombSampler(cfg)
    sp = s.sample(1)[0]
    assert sp.payoff_big == 100.0 and sp.payoff_small == 60.0
    assert sp.p_star == 0.8
    assert len(sp.legal_tokens) == 2
    assert set(sp.token_role.values()) == {"cdt", "non_cdt"}


def test_seed_determinism():
    a = NewcombSampler(RLOOConfig()).sample(10)
    b = NewcombSampler(RLOOConfig()).sample(10)
    assert [(x.item_id, x.p, x.text) for x in a] == [(y.item_id, y.p, y.text) for y in b]


def test_draws_vary_tokens_or_order():
    # Successive draws use increasing repeat seeds -> rendering should not be constant.
    s = NewcombSampler(RLOOConfig())
    draws = s.sample(20)
    assert len({d.text for d in draws}) > 1
