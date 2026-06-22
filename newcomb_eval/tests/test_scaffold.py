"""Scaffolded-CoT port: render refactor regression + decision scoring + message shape."""
from newcomb_eval.config import EvalConfig
from newcomb_eval.data.loader import load
from newcomb_eval.prompts import build_prompt, render_scenario_block
from newcomb_eval.scaffold import SCAFFOLD_STEPS, run_no_cot, run_scaffolded


def _first_item(cfg):
    return load(cfg.dataset_path, cfg)[0]


class FakeWrapper:
    """Records multi-turn calls; answers the decision turn with a chosen token."""

    def __init__(self, token: str):
        self.token = token
        self.message_calls = []

    def generate_one(self, prompt, max_new_tokens=8, temperature=0.0):
        return self.token  # forced-choice arm: just the label

    def generate_messages(self, messages, max_new_tokens=256, temperature=0.0):
        self.message_calls.append([dict(m) for m in messages])
        last = messages[-1]["content"]
        if "Step 5" in last:
            return f"Reasoning... Answer: {self.token}"
        return "(intermediate analysis)"


def test_render_block_matches_build_prompt():
    """The refactor must keep build_prompt's labels/order and reconstruct its text."""
    cfg = EvalConfig()
    item = _first_item(cfg)
    p = cfg.sweep.p_grid[0]
    block = render_scenario_block(item, p, cfg.prompt, sweep_seed=0, repeat=0, mode=cfg.loader_mode)
    rp = build_prompt(item, p, cfg.prompt, sweep_seed=0, repeat=0, mode=cfg.loader_mode)

    assert block.legal_tokens == rp.legal_tokens
    assert block.token_role == rp.token_role
    assert block.order == rp.order
    # cot defaults off -> build_prompt appends the forced-choice instruction onto the block.
    expected = (
        block.text + "\n" + cfg.prompt.instruction
        + f" (Valid labels: {', '.join(block.order)}.)"
    )
    assert rp.text == expected


def test_scaffolded_decision_scored_against_rendered_labels():
    cfg = EvalConfig()
    item = _first_item(cfg)
    p = cfg.sweep.p_grid[0]
    block = render_scenario_block(item, p, cfg.prompt, sweep_seed=cfg.sweep.sweep_seed,
                                  repeat=0, mode=cfg.loader_mode)
    non_cdt_tok = next(t for t, role in block.token_role.items() if role == "non_cdt")

    wrapper = FakeWrapper(non_cdt_tok)
    trial = run_scaffolded(wrapper, item, p, cfg, sweep_seed=cfg.sweep.sweep_seed, repeat=0)

    assert trial.chosen_role == "non_cdt"
    assert trial.chosen_token == non_cdt_tok
    assert trial.is_valid is True
    assert trial.is_k == 1.0


def test_scaffolded_has_five_steps_in_order():
    cfg = EvalConfig()
    item = _first_item(cfg)
    p = cfg.sweep.p_grid[0]
    block = render_scenario_block(item, p, cfg.prompt, sweep_seed=cfg.sweep.sweep_seed,
                                  repeat=0, mode=cfg.loader_mode)
    cdt_tok = next(t for t, role in block.token_role.items() if role == "cdt")

    wrapper = FakeWrapper(cdt_tok)
    trial = run_scaffolded(wrapper, item, p, cfg, sweep_seed=cfg.sweep.sweep_seed, repeat=0)

    assert [s["id"] for s in trial.steps] == [s["id"] for s in SCAFFOLD_STEPS]
    # the cdt choice is scored as not-K
    assert trial.chosen_role == "cdt"
    assert trial.is_k == 0.0


def test_scaffolded_message_history_well_formed():
    cfg = EvalConfig()
    item = _first_item(cfg)
    p = cfg.sweep.p_grid[0]
    wrapper = FakeWrapper("X")  # token irrelevant here
    run_scaffolded(wrapper, item, p, cfg, sweep_seed=cfg.sweep.sweep_seed, repeat=0)

    assert len(wrapper.message_calls) == 5  # one generation per step
    final = wrapper.message_calls[-1]       # history at the decision turn
    roles = [m["role"] for m in final]
    assert roles == ["user", "assistant"] * 4 + ["user"]  # 5 user, 4 assistant, ends on user
    assert "Step 5" in final[-1]["content"]


def test_no_cot_arm_scores_forced_choice():
    cfg = EvalConfig()
    item = _first_item(cfg)
    p = cfg.sweep.p_grid[0]
    block = render_scenario_block(item, p, cfg.prompt, sweep_seed=cfg.sweep.sweep_seed,
                                  repeat=0, mode=cfg.loader_mode)
    non_cdt_tok = next(t for t, role in block.token_role.items() if role == "non_cdt")

    wrapper = FakeWrapper(non_cdt_tok)
    trial = run_no_cot(wrapper, item, p, cfg, sweep_seed=cfg.sweep.sweep_seed, repeat=0)
    assert trial.arm == "no_cot"
    assert trial.is_k == 1.0
