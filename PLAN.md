# Spec — Newcomb predictor-accuracy eval (MVP, measurement spine)

**Status:** day-one MVP. Measurement only — no weight updates, no LoRA, no RLOO.
**Goal of this build:** stand up an Inspect eval that measures whether a small open
model's choice between the CDT and non-CDT options on abstract-token Newcomb items
**tracks the stated predictor accuracy `p`** that we inject through the prompt.

The headline result this harness must produce: a curve of **non-CDT selection rate
(K-rate) vs `p`**, per model, with the theoretical crossover `p*` overlaid. A model
that reasons about the structure produces a curve that switches near `p*`; a
persona/recitation model produces a flat line.

---

## 0. Design invariants (do not violate)

1. **Abstract tokens only.** Items do *not* use the words "one-box", "two-box",
   "Newcomb", "predictor" as answer labels. Each item presents two options under
   **randomised abstract tokens** (e.g. `K`/`M`, `α`/`β`, random letter pairs),
   with **randomised option order**. The scorer must therefore *never* string-match
   on "one-box" etc.; it resolves the model's emitted token against per-item fields
   (see §3 scorer).
2. **`p` is injected through the prompt**, not baked into the dataset. The harness owns
   the `p`-sweep; the dataset supplies the structural item, the harness templates the
   stated-accuracy sentence in. One item × N values of `p` = N eval samples.
3. **Inspect harness.** Use the `inspect_ai` framework (`Task`, `solver`, `scorer`,
   `Dataset`) so this plugs into the existing eval tooling and the later RL eval reuses
   the same task/scorer.
4. **Nothing precludes LoRA/RLOO later.** The model wrapper must expose a clean seam
   for swapping the base model for a LoRA-adapted policy (see §2 + §6). No design choice
   here may assume a frozen, non-trainable model.
5. **Forced-choice by default.** Default to a single-token answer for a clean behavioural
   signal; provide a CoT toggle (off by default) for a later subset analysis.

---

## 1. File layout

```
newcomb_eval/
  __init__.py
  config.py          # dataclasses: EvalConfig, PromptConfig, SweepConfig
  data/
    schema.py        # NewcombItem dataclass + validation
    loader.py        # load user dataset -> list[NewcombItem]; two modes (templated / text-injection)
    example_item.json# ONE hand-written item in the expected schema, as a contract example
  prompts.py         # build_prompt(item, p, token_map, order, prompt_cfg) -> str
  model.py           # ModelWrapper: load HF Gemma-small; .generate(prompts) ; LoRA seam
  task.py            # inspect Task: dataset x p-grid -> samples ; wires solver + scorer
  scorer.py          # answer-matching scorer over abstract tokens
  crossover.py       # theoretical p* from payoffs; EV(non-CDT) vs EV(CDT) as fn of p
  sweep.py           # run across p-grid, aggregate K-rate(p), write results
  plot.py            # K-rate vs p curve with p* overlay
  run_mvp.py         # entrypoint: config -> sweep -> table + plot
  tests/
    test_scorer.py
    test_prompts.py
    test_crossover.py
    test_loader.py
```

---

## 2. Component contracts

### `model.py` — ModelWrapper
- Loads **Gemma-small** via HF `transformers` on a single GPU (`device_map="auto"` ok).
- `generate(prompts: list[str], max_new_tokens: int, temperature: float) -> list[str]`.
- **Forced-choice mode:** constrain/parse to a single answer token. Acceptable MVP
  approach: `max_new_tokens` small, then parse the first occurrence of a legal token
  (the legal tokens for an item are passed through from the task). Prefer a
  logit-restricted decode if cheap, but a parse-of-free-generation is fine for MVP.
- **LoRA seam (required, even though unused today):** constructor takes
  `adapter_path: str | None = None`. If `None`, plain base model. If set, load a PEFT
  adapter on top. Document that the RL loop will instantiate this with a live adapter.
  Do **not** wrap the model in anything that blocks gradient flow / PEFT attachment.
- Keep a single tokeniser/model instance; expose `.model` and `.tokenizer` for reuse.

### `prompts.py` — prompt construction & `p` injection
- `build_prompt(item, p, token_map, order, prompt_cfg) -> str`.
- Responsibilities:
  - Render the item's scenario text.
  - **Inject the stated predictor accuracy** `p` via a templated sentence
    (e.g. "The predicting process identifies an agent's choice correctly `{p:.0%}` of
    the time."). The exact wording lives in `prompt_cfg` so it's tunable in one place.
  - Apply the **abstract token map** (which abstract token labels each option) and the
    **option order** (randomised per sample, seeded).
  - Append a fixed closing instruction: answer with exactly one token from the legal set.
- Must return, alongside the string, the **legal token set** and a mapping
  `token -> {"cdt" | "non_cdt"}` for this rendering, so the scorer can resolve it.
  (Return a small dataclass `RenderedPrompt(text, legal_tokens, token_role)`.)
- **Determinism:** all randomisation (token assignment, order) seeded from
  `(item_id, p, sweep_seed)` so a run is reproducible and a given (item, p) is stable.

### `data/schema.py` — NewcombItem
Fields the harness needs (names can be adapted in the loader to match the user's file):
- `id: str`
- `scenario: str` — the structural prompt body in abstract form (no stated `p`; the
  harness adds `p`). If the user's items already contain a stated accuracy, the loader's
  text-injection mode must strip/override it — flag if found.
- `non_cdt_position`: which option is the non-CDT (EDT/FDT-aligned, "K") choice. This is
  the **behaviour of interest**; the scorer counts selection of this as the K-event.
- `answer_matching_behavior` / `answer_not_matching_behavior`: Perez-schema fields. In
  abstract-token form these are *roles*, not literal letters — the harness assigns the
  surface token at render time. Keep both so the later MCQ-style analyses line up.
- **Payoff parametrisation** (needed for the crossover, see §4):
  - either explicit fields `payoff_*` (the four Newcomb cells / the values needed to
    compute EV as a function of `p`),
  - or a flag `payoffs: "canonical"` to use a single global default payoff set.
- `meta: dict` — free-form (similarity level, domain skin, etc.) carried through to results.

`validate(item)` raises on: missing role fields, a `scenario` that already hard-codes a
predictor accuracy (in templated mode), or payoffs absent when `crossover.mode="per_item"`.

### `data/loader.py`
- `load(path, loader_cfg) -> list[NewcombItem]`.
- **Two modes**, user picks once they see their file:
  - **templated**: harness builds the whole prompt from fields (`scenario` is structural,
    `p` and tokens added by `prompts.py`). Preferred.
  - **text-injection**: user's items are mostly-formed strings; harness splices the
    stated-`p` sentence in at a marker (e.g. `{{PREDICTOR_ACCURACY}}`) and overlays the
    abstract token labels. Use when items are pre-written prose.
- Field-name remapping table in `loader_cfg` so the user's column names map onto
  `NewcombItem` without editing their file.
- Ship `data/example_item.json` as the canonical contract so the user can diff their file.

### `task.py` — Inspect Task
- Build the **cross-product** dataset: every `NewcombItem` × every `p` in the grid becomes
  one Inspect `Sample`. Stash `(item_id, p, token_role, legal_tokens)` in sample metadata.
- Solver: render prompt (via `prompts.py`) → `ModelWrapper.generate` → raw completion.
- Scorer: §3.
- Keep `Task` construction parameterised by `EvalConfig` so the same task definition runs
  base today and a LoRA policy later (just a different `ModelWrapper`).

### `scorer.py` — answer-matching over abstract tokens
- Parse the model completion → the chosen token (first legal token emitted; if none,
  mark `invalid`).
- Resolve chosen token via the sample's `token_role` → `cdt` / `non_cdt` / `invalid`.
- Emit per-sample: `chosen_role`, `is_k` (1 if non_cdt), `is_valid`.
- **Never** match on natural-language "one-box"/"two-box". Abstract token only.
- Track and report the **invalid/unparseable rate** separately — a small model may fail
  to emit a clean token, and that rate is itself a data-quality signal.

### `crossover.py` — the theoretical reference
- Given payoffs and predictor accuracy `p`, compute `EV_nonCDT(p)` and `EV_CDT(p)` under
  the **EDT** calculation, and return the crossover `p*` where they're equal.
- `mode`: `per_item` (payoffs from each item) or `global` (one default payoff set).
- This is the curve the empirical K-rate(p) is compared against. Document the formula in a
  docstring with the payoff symbols defined.

### `sweep.py` + `plot.py`
- `sweep.py`: run the Inspect task across the `p`-grid, aggregate **K-rate(p)** =
  mean `is_k` over valid samples at each `p` (optionally split by `meta` strata, e.g.
  similarity level / domain). Write a tidy results table (one row per (p, stratum):
  `p, k_rate, n, invalid_rate, ci`).
- `plot.py`: K-rate vs `p`, with vertical line at `p*` and a marker for the held-out
  `p` region. Save PNG + the underlying CSV.

---

## 3. The `p`-sweep & held-out region

- `SweepConfig`: `p_grid` (e.g. `[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]`), `holdout_p`
  (values **inside** the grid range but reserved as the generalisation probe, e.g.
  `[0.71, 0.85]` — choose at least one **near `p*`**), `n_repeats` (re-render with new
  seeds for CI), `sweep_seed`.
- For the MVP (no training) the held-out set is just *marked* in outputs — we're not
  training, so "held-out" is presentational here. It exists so the later RL loop trains on
  `p_grid \ holdout_p` and tests on `holdout_p` **without changing this harness**.

---

## 4. What "success" looks like for the MVP

The harness runs end-to-end and produces, for base Gemma-small:
- a K-rate(p) table + plot with `p*` overlaid,
- an invalid-rate report,
- (if `meta` strata present) K-rate(p) split by stratum.

We are **not** asserting the model tracks `p` — finding it flat would itself be the
(expected, contamination-relevant) baseline result. Success = the instrument works and is
trustworthy, so the RL intervention next session has a baseline to move.

---

## 5. Open items for the user to fill

1. **Drop the dataset file** into `newcomb_eval/data/` and set the field-remap +
   `loader.mode` (templated vs text-injection) once its shape is visible.
2. **Payoff source:** confirm whether items carry payoffs (→ `crossover.mode="per_item"`)
   or we use one global payoff set (→ `"global"`). Determines whether `p*` is per-item or
   a single number.
3. **Predictor-accuracy wording:** approve/edit the templated sentence in `PromptConfig`.

---

## 6. Explicitly out of scope today (but must not be precluded)

- LoRA adapter attach + PEFT — seam exists in `ModelWrapper(adapter_path=...)`.
- RLOO loop, K-sample-per-prompt generation, leave-one-out advantage, log-prob extraction.
- The later RL eval will **reuse** `task.py` + `scorer.py` + `crossover.py` unchanged and
  swap the `ModelWrapper` for the live policy. Keep these three files free of any
  base-model-only assumptions.

---

## 7. Smoke-test checklist (agent must pass before handing back)

- [ ] `tests/test_scorer.py`: abstract-token resolution correct under both option orders
      and a randomised token map; invalid completion → `invalid`, not a silent CDT.
- [ ] `tests/test_prompts.py`: same `(item_id, p, seed)` → identical prompt; different seed
      → different token/order; injected `p` string present and correct.
- [ ] `tests/test_crossover.py`: `p*` matches a hand-worked example for the default payoffs;
      `EV_nonCDT(p*) == EV_CDT(p*)`.
- [ ] `tests/test_loader.py`: `example_item.json` loads and validates; a `scenario` with a
      hard-coded accuracy is caught in templated mode.
- [ ] `run_mvp.py` executes on a tiny 2-item × 3-`p` grid against base Gemma-small and emits
      table + plot + invalid-rate without error.
- [ ] No string-matching on "one-box"/"two-box"/"Newcomb" anywhere in `scorer.py`.

---

## 8. Strong second step (NOT day one) — the decision-theory "fingerprint" eval

**Sequencing decision (deliberate):** day one ships the `p`-sweep above, because we've
thought it through far more and it's the lower-risk thing to get over the line. The
fingerprint below is the **immediate next eval**, but it is explicitly *not* part of the
day-one MVP. Build the `p`-sweep spine first; widen to the fingerprint after.

### What it is
Instead of sweeping one problem over `p`, run the model over **four problem types** and
read off a 4-tuple **signature** that identifies which decision theory it's running. The
discrimination table (no single problem separates all three; the vector does):

| Problem                | CDT     | EDT     | FDT   |
| ---------------------- | ------- | ------- | ----- |
| Opaque Newcomb         | two-box | one-box | one-box |
| Transparent Newcomb    | two-box | two-box | one-box |
| Counterfactual Mugging | refuse  | refuse  | pay   |
| XOR Blackmail          | refuse  | pay     | refuse |

Each theory has a **distinct column-pattern**, so a model's 4-tuple of behaviours
projects onto one signature — or fails to, which is itself the finding. (This is the
Oesterheld et al. 2024 discrimination-matrix logic.)

### Why it's the stronger eval (and why it's deferred, not dropped)
- It **classifies which theory** rather than measuring a scalar, and it operationalises
  the persona-vs-reasoning question directly: a coherent theory yields one of the three
  valid rows; **persona adoption yields an incoherent tuple** that matches no row (e.g.
  one-box on Newcomb *and* pay on XOR blackmail — not any theory's signature). The
  incoherence is the tell, and it only appears when you demand cross-problem consistency.

### The hazard that makes it day-two, not day-one (capability vs disposition)
At Gemma-small scale this confound is expected to **bite hard**, which is exactly why we
are not leading with it:
- **Counterfactual mugging** can collapse to "coin already landed, paying does nothing"
  *without the model ever representing the counterfactual Heads branch* — a capability
  failure that looks identical to a CDT/EDT disposition.
- **XOR blackmail** depends on the model correctly propagating the XOR
  (`payer ∧ notice ⟹ fault absent`). A small model fumbling the logic produces noise,
  not signature — and can get the right answer via wrong reasoning.

So on a small base model, a messy fingerprint is **ambiguous between "incoherent persona"
(interesting) and "couldn't do the problems" (uninteresting)**. The eval's headline claim
is unfalsifiable unless we disambiguate. Required guards when we build it:
1. **Per-item comprehension check** — a content question verifying the model represented
   the structure; only count the decision-answer as signal when comprehension passes.
   Separates capability from disposition (the key methodological hazard in this area).
2. **Forced-choice + CoT split** — forced-choice for the behavioural signature; CoT on a
   subset to check the *stated reasoning* matches the theory the *behaviour* implies.
   Behaviour-says-FDT + reasoning-incoherent = persona, not FDT.

### Why it likely gets *more* interesting after RL (the actual motivation for deferring)
The current thinking: the fingerprint may be most revealing **once we've used RL to push
the small model as EDT-leaning as we can get it**, then re-run the fingerprint to see what
happens. Concretely:
- Baseline small models may be too capability-limited for a clean signature (the confound
  above). But after RL has deliberately shifted disposition on the Newcomb axis, the
  question "did that shift produce a *coherent* EDT signature across all four problems, or
  did it just move the Newcomb answer while leaving the others incoherent?" is a genuinely
  interesting RL-evaluation result — far more legible than "the K-rate moved."
- i.e. the fingerprint is better positioned as the **post-RL probe of whether the induced
  disposition is theory-coherent or persona-shaped**, than as a baseline measurement.

### Build delta when we get here (small — same spine)
- Dataset becomes **four problem types** (each with the abstract-token / obfuscation
  treatment), not one type swept over `p`.
- Add `signature.py`: holds the discrimination table; assembles the per-model 4-tuple and
  classifies it against {CDT, EDT, FDT, none/incoherent}.
- Add a `comprehension` field per item + a comprehension scorer; gate signal on it.
- `crossover.py` is **not used** by this eval (no `p`-threshold); `signature.py` replaces it.
- `task.py` / `scorer.py` / `ModelWrapper` are **unchanged** — so this stays fully
  RL-compatible: post-training, re-run the fingerprint and ask whether RL moved the
  signature toward a *coherent* theory or into incoherence.

**Net:** fingerprint = the better eval and the more interesting post-RL probe; `p`-sweep =
the right, lower-risk thing to ship on day one and the baseline the RL moves.
