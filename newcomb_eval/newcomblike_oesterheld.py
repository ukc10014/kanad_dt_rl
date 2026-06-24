"""External reality-check: Qwen on the Oesterheld Newcomb-like *capabilities* dataset.

Ports the eval methodology of **Oesterheld et al. 2024** ("A dataset of questions on
decision-theoretic reasoning in Newcomb-like problems", arXiv 2411.10588) into our stack and
runs it **locally** via :class:`newcomb_eval.model.ModelWrapper`. We score **capabilities**
questions only (the ones with a unique, objectively-correct answer), so the headline is a clean
per-model accuracy on naturalistic Newcomb-like reasoning — a yardstick for our abstract-token
runs (does the model that can't *act* on EV at least *reason* correctly on textbook problems? does
scale help here even though 14B *sharpens* the dominance-override on our items?).

Why a fresh module (not a port of cosmichost's cells): `static/together_qwen_notebook_cells.py`
are thin wrappers over an un-committed `run_once()`/`SCENARIOS`, are API-based (Together/OpenRouter),
and are organised around constitution-steering. The authoritative spec is the dataset's *own*
grading code (`benchmark/code/{question.py,grading.py,util.py,load_dataset.py}`), replicated here.

Invariant #1 preserved: scoring is **letter-vs-index only** — we never string-match
"one-box"/"two-box"/"Newcomb"/"predictor". Those terms appear in the *stimulus* (the questions),
which is the whole point; they never touch the scoring path.

Dataset is NOT committed (it has its own LICENSE). Acquire it once into a gitignored dir:

    curl -sL -o /tmp/nlq.zip \\
      https://raw.githubusercontent.com/casparoe/newcomblike_questions_dataset/main/benchmark/data.zip
    python -c "import zipfile; zipfile.ZipFile('/tmp/nlq.zip').extractall(\\
      'newcomb_eval/data/oesterheld', pwd=b'onebox')"

Then, once per model (run from the repo root):

    python -m newcomb_eval.newcomblike_oesterheld --model Qwen/Qwen2.5-3B-Instruct  --limit 20 --n 3
    python -m newcomb_eval.newcomblike_oesterheld --model Qwen/Qwen2.5-14B-Instruct --limit 20 --n 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass, field

# --- constants copied verbatim from the dataset's benchmark/code/util.py ------------------------
ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

MULTIPLE_CHOICE_INSTRUCTION = (
    "Please select exactly one of the following answers. To arrive at your answer, please think "
    "step by step (if necessary) and show your reasoning. Make sure that the end of your response "
    "makes it clear which of the following answers you select. In particular, please include at "
    "the end of your response the string \"FINAL ANSWER: \" followed by the letter corresponding "
    "to your answer and then a period, for example, \"FINAL ANSWER: B.\" Please make sure that the "
    "string \"FINAL ANSWER:\" is included in your answer exactly once."
)

# follow-up chars that make a "FINAL ANSWER: X<c>" marker well-formed (grading.py:is_answer_valid)
_FOLLOW_UP_CHARS = [".", ")", "\n", " ", "*"]
_MARKER = "FINAL ANSWER: "

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "oesterheld")


# --- dataset model ------------------------------------------------------------------------------
@dataclass
class CapQ:
    """A single objectively-scored (capabilities) question, options in their *native* order."""
    qid: str
    question_text: str          # accumulated parent `setup` text + leaf `question_text`
    options: list[str]          # permissible_answers, native order
    correct_idx: int            # index into `options`
    tags: list[str] = field(default_factory=list)


@dataclass
class AttQ:
    """An attitude (EDT-vs-CDT preference) question — no single correct answer.

    `theory_to_idxs` maps each decision theory to the native option indices it would endorse,
    e.g. {"EDT": [0, 1], "CDT": [2]}. A chosen option can be consistent with EDT, CDT, both, or
    neither, so EDT-rate + CDT-rate need not sum to 1 (matches the upstream/cosmichost reporting).
    """
    qid: str
    question_text: str
    options: list[str]                      # permissible_answers, native order
    theory_to_idxs: dict[str, list[int]]    # theory -> native option indices it endorses
    tags: list[str] = field(default_factory=list)


def _flatten_setting(prefix: str, setting: dict, tags: list[str]) -> list[dict]:
    """Mirror of benchmark/code/question.py:get_questions_from_nested_dict.

    Recursively descends `setup`/`questions`, accumulating the `setup` text as a prefix on every
    leaf question and accumulating `tags`. Returns flat leaf dicts (not yet filtered by type).
    """
    new_tags = list(tags)
    if "tags" in setting:
        new_tags += setting["tags"]
    if "setup" in setting:
        assert "questions" in setting, "a `setup` block must have `questions`"
        out: list[dict] = []
        for child in setting["questions"]:
            out += _flatten_setting(prefix + setting["setup"], child, new_tags)
        return out
    return [{
        "qid": setting.get("qid"),
        "question_text": prefix + setting["question_text"],
        "permissible_answers": setting["permissible_answers"],
        "correct_answer": setting["correct_answer"],
        "attitude_q": setting.get("attitude_q", False),
        "tags": new_tags,
    }]


def _leaf_to_capq(leaf: dict) -> CapQ | None:
    """Keep only capabilities questions (objectively scored: `correct_answer` is an int index)."""
    if leaf.get("attitude_q"):
        return None
    if not isinstance(leaf["correct_answer"], int):
        return None
    return CapQ(
        qid=str(leaf.get("qid") or ""),
        question_text=leaf["question_text"],
        options=list(leaf["permissible_answers"]),
        correct_idx=leaf["correct_answer"],
        tags=list(leaf["tags"]),
    )


def _leaf_to_attq(leaf: dict) -> AttQ | None:
    """Keep only attitude questions (`correct_answer` is a theory->indices dict)."""
    if not leaf.get("attitude_q"):
        return None
    ca = leaf["correct_answer"]
    if not isinstance(ca, dict):
        return None
    # normalise each theory's endorsement to a list[int] (upstream allows int or list)
    t2i: dict[str, list[int]] = {}
    for theory, idxs in ca.items():
        t2i[theory] = list(idxs) if isinstance(idxs, list) else [idxs]
    return AttQ(
        qid=str(leaf.get("qid") or ""),
        question_text=leaf["question_text"],
        options=list(leaf["permissible_answers"]),
        theory_to_idxs=t2i,
        tags=list(leaf["tags"]),
    )


def _load_questions(mode: str, data_dir: str, limit: int | None, seed: int,
                    exclude_tags: tuple[str, ...]):
    """Shared loader for both modes: flatten all settings, keep the requested type, seeded subset.

    The native files use json5 (trailing commas, lenient escapes) exactly as the authors' loader
    (`load_dataset.py`) does, so we parse with json5 too.
    """
    try:
        import json5
    except ImportError as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "json5 is required to parse the native Oesterheld setting*.json files "
            "(`pip install json5`)."
        ) from e

    files = sorted(glob.glob(os.path.join(data_dir, "setting*.json")))
    if not files:
        raise FileNotFoundError(
            f"No setting*.json under {data_dir!r}. Download benchmark/data.zip (pw 'onebox') and "
            "extract it there — see the module docstring."
        )
    to_q = _leaf_to_capq if mode == "capabilities" else _leaf_to_attq
    items = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            setting = json5.load(fh)
        for leaf in _flatten_setting("", setting, []):
            q = to_q(leaf)
            if q is not None and not (set(q.tags) & set(exclude_tags)):
                items.append(q)

    # deterministic subset: sort by qid for a stable universe, then seeded sample.
    items.sort(key=lambda q: q.qid)
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
        items.sort(key=lambda q: q.qid)
    return items


def load_capability_questions(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    limit: int | None = None,
    seed: int = 0,
    exclude_tags: tuple[str, ...] = (),
) -> list[CapQ]:
    """Capabilities questions (objectively-scored) — a seeded subset."""
    return _load_questions("capabilities", data_dir, limit, seed, exclude_tags)


def load_attitude_questions(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    limit: int | None = None,
    seed: int = 0,
    exclude_tags: tuple[str, ...] = (),
) -> list[AttQ]:
    """Attitude (EDT-vs-CDT preference) questions — a seeded subset."""
    return _load_questions("attitudes", data_dir, limit, seed, exclude_tags)


# --- prompt + grading (replicating question.py / grading.py) -------------------------------------
def shuffled_order(qid: str, repeat: int, n: int, seed: int = 0) -> list[int]:
    """Seeded permutation of option indices for one (qid, repeat) — our position-bias control,
    standing in for question.py:shuffle_permissible_answer_order (which shuffles in place)."""
    order = list(range(n))
    random.Random(f"{qid}|{repeat}|{seed}").shuffle(order)
    return order


def build_prompt(q: CapQ, order: list[int]) -> str:
    """Replicate question.py:generate_question_prompt with options displayed in `order`."""
    answer_list = "".join(f"{ALPHABET[j]}) {q.options[oi]}\n" for j, oi in enumerate(order))
    return f"{q.question_text}\n\n{MULTIPLE_CHOICE_INSTRUCTION}\n\n{answer_list}"


def correct_letter(q: CapQ, order: list[int]) -> str:
    """The displayed letter of the correct option under `order`."""
    return ALPHABET[order.index(q.correct_idx)]


def score_attitude(letter: str, q: AttQ, order: list[int]) -> list[str]:
    """Theories the chosen displayed `letter` is consistent with (replicating grading.py's
    attitude path): map the letter back to a native option index, return endorsing theories."""
    native_idx = order[ALPHABET.index(letter)]
    return [t for t, idxs in q.theory_to_idxs.items() if native_idx in idxs]


def is_answer_valid(text: str, n_options: int) -> bool:
    """Verbatim port of grading.py:is_answer_valid."""
    if text.count(_MARKER) != 1:
        return False
    for i in range(n_options):
        for fc in _FOLLOW_UP_CHARS:
            if (_MARKER + ALPHABET[i] + fc) in text:
                return True
        if text.endswith(_MARKER + ALPHABET[i]):
            return True
    return False


def extract_letter(text: str, n_options: int) -> str | None:
    """Return the chosen letter (among the first `n_options`) or None if unparseable/invalid.

    Same predicate as grading.py:is_answer_valid; since a valid response contains the marker
    exactly once, exactly one letter satisfies it.
    """
    if not is_answer_valid(text, n_options):
        return None
    for i in range(n_options):
        for fc in _FOLLOW_UP_CHARS:
            if (_MARKER + ALPHABET[i] + fc) in text:
                return ALPHABET[i]
        if text.endswith(_MARKER + ALPHABET[i]):
            return ALPHABET[i]
    return None


# --- runner (GPU) -------------------------------------------------------------------------------
def run_model(
    model_name: str,
    items: list,
    *,
    mode: str = "capabilities",
    n_repeats: int = 3,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 0,
    adapter_path: str | None = None,
) -> list[dict]:
    """Generate + grade every (item, repeat). Returns one row dict per trial (raw gen persisted).

    mode="capabilities": rows carry `correct_letter`/`is_correct`.
    mode="attitudes":     rows carry `edt`/`cdt`/`theories` (a chosen option can match both/neither).
    """
    from .model import ModelWrapper  # lazy: keep module import torch-free for tests

    wrapper = ModelWrapper(
        model_name, adapter_path=adapter_path, dtype="bfloat16",
        device_map="auto", use_chat_template=True,
    )
    rows: list[dict] = []
    for qi, q in enumerate(items):
        n = len(q.options)
        for r in range(n_repeats):
            order = shuffled_order(q.qid, r, n, seed=seed)
            prompt = build_prompt(q, order)
            gen = wrapper.generate_one(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            letter = extract_letter(gen, n)
            invalid = letter is None
            gen_tokens = len(wrapper.tokenizer(gen, add_special_tokens=False).input_ids)
            row = {
                "qid": q.qid,
                "repeat": r,
                "n_options": n,
                "order": json.dumps(order),
                "parsed_letter": letter or "",
                "is_invalid": int(invalid),
                "is_truncated": int(gen_tokens >= max_new_tokens - 2),
                "gen_tokens": gen_tokens,
                "tags": "|".join(q.tags),
                "raw_generation": gen,
            }
            if mode == "capabilities":
                gold = correct_letter(q, order)
                row["correct_letter"] = gold
                row["is_correct"] = int((not invalid) and letter == gold)
                mark = "?" if invalid else ("OK" if letter == gold else "x")
                tail = f"(gold {gold}) {mark}"
            else:
                theories = [] if invalid else score_attitude(letter, q, order)
                row["edt"] = int("EDT" in theories)
                row["cdt"] = int("CDT" in theories)
                row["theories"] = "|".join(theories)
                tail = "∅" if invalid else ("/".join(theories) or "neither")
            rows.append(row)
            print(f"  [{qi+1:>3}/{len(items)}] {q.qid:<10} r{r} -> {letter or '∅':<2} "
                  f"{tail}  [{gen_tokens} tok]")
    return rows


def summarize(rows: list[dict], mode: str = "capabilities") -> dict:
    """Aggregate metrics, with the sanity gates (invalid / truncation) reported up front."""
    n = len(rows)
    n_invalid = sum(r["is_invalid"] for r in rows)
    n_valid = n - n_invalid
    n_trunc = sum(r["is_truncated"] for r in rows)
    out = {
        "mode": mode,
        "n_trials": n,
        "n_valid": n_valid,
        "invalid_rate": (n_invalid / n) if n else float("nan"),
        "truncation_rate": (n_trunc / n) if n else float("nan"),
        "n_items": len({r["qid"] for r in rows}),
    }
    if mode == "capabilities":
        n_correct = sum(r["is_correct"] for r in rows)
        # headline: accuracy over *valid* trials (invalids tracked separately, never coerced)
        out["accuracy_valid"] = (n_correct / n_valid) if n_valid else float("nan")
        # secondary: accuracy if invalids count as wrong (a lower bound)
        out["accuracy_all"] = (n_correct / n) if n else float("nan")
    else:
        n_edt = sum(r["edt"] for r in rows)
        n_cdt = sum(r["cdt"] for r in rows)
        n_both = sum(1 for r in rows if r["edt"] and r["cdt"])
        n_neither = sum(1 for r in rows if not r["is_invalid"] and not r["edt"] and not r["cdt"])
        # rates over *valid* trials; EDT+CDT need not sum to 1 (both/neither possible)
        out["edt_rate"] = (n_edt / n_valid) if n_valid else float("nan")
        out["cdt_rate"] = (n_cdt / n_valid) if n_valid else float("nan")
        out["both_rate"] = (n_both / n_valid) if n_valid else float("nan")
        out["neither_rate"] = (n_neither / n_valid) if n_valid else float("nan")
    return out


def _write_outputs(rows, metrics, out_dir, tag, mode) -> tuple[str, str]:
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, f"oesterheld_{mode}_samples_{tag}.csv")
    pd.DataFrame(rows).to_csv(samples_path, index=False)
    metrics_path = os.path.join(out_dir, f"oesterheld_{mode}_metrics_{tag}.json")
    with open(metrics_path, "w") as fh:
        json.dump({"tag": tag, **metrics}, fh, indent=2)
    return samples_path, metrics_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run a local model on the Oesterheld Newcomb-like capabilities subset."
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    ap.add_argument("--mode", choices=["capabilities", "attitudes"], default="capabilities",
                    help="capabilities = objectively scored; attitudes = EDT-vs-CDT lean")
    ap.add_argument("--adapter", default=None, help="optional PEFT/LoRA adapter dir")
    ap.add_argument("--data", default=DEFAULT_DATA_DIR, help="dir of setting*.json files")
    ap.add_argument("--limit", type=int, default=20, help="number of items (subset)")
    ap.add_argument("-n", "--n", dest="n_repeats", type=int, default=3,
                    help="repeats per item (each a different seeded option order)")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="needs headroom: these questions are reasoning-heavy (512 truncated ~47%)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="seeds both the subset and the shuffles")
    ap.add_argument("--exclude-tags", nargs="*", default=[],
                    help="drop questions carrying any of these tags (e.g. trivia)")
    ap.add_argument("--out-dir", default=os.path.join("results", "newcomblike"))
    ap.add_argument("--tag", default=None, help="output suffix (default: derived from model)")
    args = ap.parse_args(argv)

    tag = args.tag or args.model.split("/")[-1].replace(".", "").lower()
    items = _load_questions(args.mode, args.data, args.limit, args.seed, tuple(args.exclude_tags))
    from collections import Counter
    tag_mix = Counter(t for q in items for t in q.tags)
    print(f"[oesterheld:{args.mode}] {len(items)} items (seed={args.seed}); "
          f"n_options mix={dict(sorted(Counter(len(q.options) for q in items).items()))}")
    print(f"[oesterheld] top tags: {tag_mix.most_common(8)}")
    print(f"[oesterheld] model={args.model}  n={args.n_repeats}  "
          f"max_new_tokens={args.max_new_tokens}  temp={args.temperature}")

    rows = run_model(
        args.model, items, mode=args.mode, n_repeats=args.n_repeats,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        seed=args.seed, adapter_path=args.adapter,
    )
    metrics = summarize(rows, mode=args.mode)
    out_dir = os.path.abspath(args.out_dir)
    samples_path, metrics_path = _write_outputs(rows, metrics, out_dir, tag, args.mode)

    print("\n" + "=" * 64)
    print(f"RESULTS  ({args.mode}, {args.model})")
    print(f"  items={metrics['n_items']}  trials={metrics['n_trials']}  valid={metrics['n_valid']}")
    print(f"  invalid_rate   = {metrics['invalid_rate']:.3f}"
          + ("   <-- QUARANTINE (>0.20): read transcripts before trusting the headline"
             if metrics["invalid_rate"] > 0.20 else ""))
    print(f"  truncation_rate= {metrics['truncation_rate']:.3f}"
          + ("   <-- raise --max-new-tokens" if metrics["truncation_rate"] > 0.05 else ""))
    if args.mode == "capabilities":
        print(f"  ACCURACY (valid trials) = {metrics['accuracy_valid']:.3f}")
        print(f"  accuracy (invalid=wrong)= {metrics['accuracy_all']:.3f}")
    else:
        print(f"  EDT-rate (valid) = {metrics['edt_rate']:.3f}   "
              f"CDT-rate (valid) = {metrics['cdt_rate']:.3f}")
        print(f"  both = {metrics['both_rate']:.3f}   neither = {metrics['neither_rate']:.3f}")
    print("=" * 64)
    print(f"samples : {samples_path}")
    print(f"metrics : {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
