"""Offline end-to-end wiring test: build_task -> Inspect eval -> scorer -> aggregate,
driven by a fake model (no GPU / no real LLM). Validates the whole spine deterministically.
"""
import re

import dataclasses

from newcomb_eval.config import EvalConfig
from newcomb_eval.sweep import run_sweep


class FakeOneBox:
    """A 'model' that always one-boxes: returns the abstract label of the 'only' option."""

    _line = re.compile(r"^(\S+)\.\s+(.*)$")

    def generate_one(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0) -> str:
        for ln in prompt.splitlines():
            m = self._line.match(ln.strip())
            if m and "only" in m.group(2).lower():
                return m.group(1)
        return "?"  # would score invalid


class FakeInvalid:
    def generate_one(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0) -> str:
        return "this is not a label"


def _cfg(tmp_path):
    cfg = EvalConfig()
    cfg.limit = 2
    cfg.results_dir = str(tmp_path)
    cfg.sweep = dataclasses.replace(cfg.sweep, p_grid=(0.5, 0.8, 0.99), holdout_p=(0.8,), n_repeats=1)
    return cfg


def test_one_box_model_gives_full_k_rate(tmp_path):
    cfg = _cfg(tmp_path)
    df = run_sweep(cfg, FakeOneBox())
    assert list(df["p"]) == [0.5, 0.8, 0.99]
    assert (df["k_rate"] == 1.0).all()        # always one-box -> K-rate 1
    assert (df["invalid_rate"] == 0.0).all()  # always a legal token
    assert (df["p_star"] == 0.8).all()
    assert df.loc[df["p"] == 0.8, "is_holdout"].iloc[0]
    # 2 items x 1 repeat per p.
    assert (df["n"] == 2).all()


def test_invalid_model_excluded_from_k_rate(tmp_path):
    cfg = _cfg(tmp_path)
    df = run_sweep(cfg, FakeInvalid())
    assert (df["invalid_rate"] == 1.0).all()
    assert (df["n_valid"] == 0).all()
    # No valid samples -> K-rate is NaN, not silently 0.
    assert df["k_rate"].isna().all()


def test_csv_written(tmp_path):
    cfg = _cfg(tmp_path)
    df = run_sweep(cfg, FakeOneBox())
    import os
    assert os.path.exists(df.attrs["csv_path"])
