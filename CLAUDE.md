# CLAUDE.md — kanad_dt_rl

Working guidance for Claude Code in this repo. **`PLAN.md` is the source of truth** for the
full spec; this file is the short operational contract. When they conflict, `PLAN.md` wins —
but flag the conflict rather than silently diverging.

## What this project is
An **Inspect (`inspect_ai`) eval** measuring whether a small open model tracks
the **stated predictor accuracy `p`** in abstract-token Newcomb problems. Base policy is
**Qwen2.5-3B-Instruct** (default in `config.py`); Gemma-2-2b-it is an alternative to check
(see the model-choice note in PLAN.md §2). Headline output:
a **K-rate (non-CDT selection rate) vs `p`** curve per model, with the theoretical crossover
`p*` overlaid. Day-one scope is **measurement only** — no weight updates, LoRA, or RLOO.

## Non-negotiable invariants (violating these breaks the experiment)
1. **Abstract tokens only.** Answer labels are randomised abstract tokens (`K`/`M`, `α`/`β`,
   random letter pairs) with randomised option order. **Never** string-match on "one-box",
   "two-box", "Newcomb", or "predictor" anywhere — especially in `scorer.py`. The scorer
   resolves the emitted token against per-sample `token_role`.
2. **`p` is injected via the prompt**, never baked into the dataset. One item × N `p`-values
   = N samples. The harness owns the sweep; the dataset supplies the structural item.
3. **Inspect framework** (`Task`/`solver`/`scorer`/`Dataset`) so the later RL eval reuses the
   same task + scorer.
4. **Never preclude LoRA/RLOO.** `ModelWrapper` must take `adapter_path: str | None = None`
   and must not wrap the model in anything blocking gradient flow / PEFT attachment.
   `task.py`, `scorer.py`, `crossover.py` must stay free of base-model-only assumptions.
5. **Forced-choice by default**, single answer token. CoT is a toggle, off by default.
6. **Determinism.** All randomisation (token map, option order) is seeded from
   `(item_id, p, sweep_seed)`. Same triple → identical prompt.
7. **Invalid answers are tracked, never coerced.** An unparseable completion → `invalid`,
   never a silent CDT. Report invalid-rate separately.

## File layout
See `PLAN.md §1`. Core modules under `newcomb_eval/`: `config.py`, `data/{schema,loader}.py`,
`prompts.py`, `model.py`, `task.py`, `scorer.py`, `crossover.py`, `sweep.py`, `plot.py`,
`run_mvp.py`, `tests/`.

## Definition of done (day one)
The smoke-test checklist in `PLAN.md §7` must pass:
- `tests/` for scorer, prompts, crossover, loader all green.
- `run_mvp.py` runs a tiny 2-item × 3-`p` grid on base Gemma-small → table + plot +
  invalid-rate, no errors.
- Grep-clean: no "one-box"/"two-box"/"Newcomb" string-matching in `scorer.py`.

## Conventions
- Python, dataclasses for config (`EvalConfig`, `PromptConfig`, `SweepConfig`).
- Tunable prompt wording lives in `PromptConfig` (one place), not scattered in code.
- `prompts.build_prompt(...)` returns a `RenderedPrompt(text, legal_tokens, token_role)`.
- Results tables are tidy: one row per `(p, stratum)` → `p, k_rate, n, invalid_rate, ci`.
- **Persist raw generations/transcripts**, not just aggregated summaries — needed for later
  audit of capability-vs-disposition.

## Open inputs from the user (do not invent silently — ask)
1. Dataset file shape → field-remap + `loader.mode` (templated vs text-injection).
2. Payoff source → `crossover.mode` (`per_item` vs `global`); sets whether `p*` is per-item.
3. Approved predictor-accuracy sentence wording in `PromptConfig`.

## Deferred (do NOT build day one, but keep the seam): see `PLAN.md §8`
The decision-theory **fingerprint** eval (four problem types → 4-tuple signature classifying
CDT/EDT/FDT/incoherent) is the *next* eval and the post-RL probe — not today. It reuses
`task.py`/`scorer.py`/`ModelWrapper` unchanged; adds `signature.py` + a comprehension gate.

## Backup / git
- Push from the **tmux window, not a VS Code/Cursor terminal** (the editor injects ambient
  GitHub creds that bypass the scoped token). Plain `git push` here prompts for the PAT.
- Per-repo identity is `ukc10014 <…noreply>`; do not set `--global`.
- Exit ritual: commit → push → verify on GitHub → revoke the fine-grained token.
