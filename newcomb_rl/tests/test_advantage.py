"""RLOO leave-one-out advantage math."""
import pytest
import torch

from newcomb_rl.rloo import loo_advantage


def test_loo_matches_explicit_leave_one_out():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0])
    A = loo_advantage(r)
    K = len(r)
    for i in range(K):
        others = torch.cat([r[:i], r[i + 1:]])
        assert A[i].item() == pytest.approx(r[i].item() - others.mean().item(), abs=1e-5)


def test_loo_zero_when_all_equal():
    r = torch.full((6,), 0.5)
    assert torch.allclose(loo_advantage(r), torch.zeros(6))


def test_loo_centered_sum_zero():
    # leave-one-out advantages are a scaled centering, so they sum to ~0.
    r = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    assert loo_advantage(r).sum().abs().item() < 1e-5


def test_loo_singleton_is_zero():
    assert torch.allclose(loo_advantage(torch.tensor([2.0])), torch.zeros(1))


def test_loo_equals_scaled_mean_centering():
    r = torch.tensor([2.0, 0.0, 0.0, 0.0])
    K = len(r)
    expected = (K / (K - 1.0)) * (r - r.mean())
    assert torch.allclose(loo_advantage(r), expected)
