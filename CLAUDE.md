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

## Sanity gates — catch confounds before trusting any number
A metric is only as good as its denominator. Before *interpreting or reporting* a headline number
(K-rate, slope, reward), check the cheap confounds below — and if one is elevated, **quarantine the
dependent result: re-run/fix, don't footnote-and-proceed.** (This bit us: Run 3/4's CoT "slope
+0.50" was largely a 38–48%-invalid artifact — flagged as a caveat but reported as a finding
anyway; robust re-scoring flattened it.)
- **Invalid / non-parse rate is not neutral noise.** Invalids are scored into a class (here
  not-K = 0), so a high, `p`-uneven invalid rate **biases the slope**, it doesn't just widen the CI.
  Heuristic: invalid > ~5% on a cell ⇒ that cell is suspect; > ~20% ⇒ do **not** interpret the
  slope until the format/parser is fixed.
- **Truncation.** `gen_len == max_new_tokens` ⇒ the answer may be cut off (the scaffold decision
  step truncated at 16 tok → 100% invalid). Want `gen_len` well under the cap, or extract the
  answer structurally (forced `"Answer:"` continuation), not by hoping the label appears inline.
- **Format mismatch.** The model may emit the option *text* ("take only Q"), not the abstract
  *label* ("Q") — the actual root cause of the CoT invalids. Don't rely on the label appearing in
  free text.
- **Degenerate output.** Empty / prompt-echo / repetition / a constant token regardless of input
  (base model without a chat template → 100% invalid). A flat metric can be a *broken measurement*,
  not a real null — rule that out first.
- **n and noise.** Slopes/CIs from n ≲ 10 per cell are noise (we saw a slope flip +0.5→−0.2 at
  n=10). State n; don't over-read a single-run slope sign.
- **Read the raw transcripts, don't just persist them.** A glance at a few generations catches all
  of the above in seconds — persisting (below) is necessary but not sufficient.

## GPU runs (serial vs. batch)
Unattended/batch GPU work — sweeps, ablations, self-contained baselines/probes whose results
*aggregate* and need no human between runs — lives in **`OVERNIGHT.md`**, a living queue with
smoke-before-batch + memory-safe-default rules. Most work at this stage is **serial** (each result
informs the next move); only fire-and-forget aggregating runs go overnight. Before stepping away,
grab candidates from `OVERNIGHT.md`'s "overnight-friendly" list; keep serial experiments attended.

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
- **⚠ On this box (migrated 2026-06-26) the VS Code env LEAKS INTO tmux** — the tmux server was
  started under VS Code Server, so `GIT_ASKPASS` + `VSCODE_GIT_ASKPASS_*` + `VSCODE_GIT_IPC_HANDLE`
  are present even in the tmux window. `git push` then authenticates *silently* via VS Code's
  GitHub credential (over the IPC socket) — **not** the scoped PAT, and **no prompt**. So the
  "tmux not editor" rule above is defeated here, and revoking the fine-grained token does NOT
  de-authorize such a push (a different, broader VS Code credential was used).
  - **To force the scoped-token path (PAT prompt):**
    ```bash
    env -u GIT_ASKPASS -u VSCODE_GIT_ASKPASS_NODE -u VSCODE_GIT_ASKPASS_MAIN \
        -u VSCODE_GIT_IPC_HANDLE GIT_TERMINAL_PROMPT=1 git push
    ```
    (or `unset` those four vars in the session first, or start a clean `env -i tmux` from a plain
    SSH login). Check `echo $GIT_ASKPASS` before pushing — if non-empty, you're on the VS Code path.
- Per-repo identity is `ukc10014 <…noreply>`; do not set `--global`.
- Exit ritual: commit → push → verify on GitHub → revoke the fine-grained token. **Verify which
  credential actually pushed** (`$GIT_ASKPASS` empty ⇒ scoped PAT; set ⇒ VS Code creds) so the
  revoke step is meaningful.
