"""Phase-3 enablers: train-p restriction (per-p ablation) + LoRA target_modules (attn/MLP)."""
import dataclasses

from newcomb_rl.rl_config import RLOOConfig
from newcomb_rl.sampler import NewcombSampler


def test_train_p_grid_default_excludes_holdout():
    cfg = RLOOConfig()  # holdout {0.75, 0.85}
    assert cfg.train_p_grid == (0.5, 0.6, 0.7, 0.8, 0.9, 0.99)


def test_train_p_grid_high_only():
    cfg = dataclasses.replace(RLOOConfig(), train_p_min=0.81)  # p* = 0.8 → keep p>p*
    assert cfg.train_p_grid == (0.9, 0.99)


def test_train_p_grid_low_only():
    cfg = dataclasses.replace(RLOOConfig(), train_p_max=0.79)
    assert cfg.train_p_grid == (0.5, 0.6, 0.7)


def test_sampler_respects_restricted_train_p():
    cfg = dataclasses.replace(RLOOConfig(), train_p_min=0.81)
    s = NewcombSampler(cfg)
    assert set(s.train_p) == {0.9, 0.99}
    assert all(sp.p in (0.9, 0.99) for sp in s.sample(12))


def test_lora_targets_default_and_override():
    assert RLOOConfig().lora_targets == (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    attn = dataclasses.replace(RLOOConfig(), lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"))
    assert "gate_proj" not in attn.lora_targets
