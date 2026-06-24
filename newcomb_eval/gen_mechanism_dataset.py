"""Generate predictor-mechanism-credibility variants of the opaque-Newcomb dataset (CPU, no model).

Every base item shares the exact predictor clause
``"<subj> correctly identifies the choices of agents like you {{PREDICTOR_ACCURACY}} of the time"``
— a **reference-class statistic**, the least binding (most dominance-friendly) framing. We swap ONLY
that clause for a credibility-graded mechanism, holding payoff structure, abstract labels, and the
``{{PREDICTOR_ACCURACY}}`` marker fixed, so the mechanism story is the only thing that varies:

  m0     statistical (≈ current) — a population base rate; a causal reasoner can treat it as
         non-binding on its individual choice.
  m0pad  length-matched placebo — m0 + neutral verbiage, NO added mechanism credibility. Controls
         the "longer story" confound: if m0pad ≈ m0 while m1..m3 move, it's credibility not length.
  m1     individualized predictive model of *your* reasoning.
  m2     scan of your decision process, run forward.
  m3     exact copy of your decision procedure (FDT-strong): the forecast IS your procedure's output,
         so the correlation has a credible common cause — dominance reasoning itself can favour 1-box.

Writes ``newcomb_eval/data/dataset_mech_<level>.json`` (gitignored under results? no — these are
inputs; keep them in data/). Idempotent. CLI: ``python -m newcomb_eval.gen_mechanism_dataset``.
"""
from __future__ import annotations

import json
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_BASE = os.path.join(_DATA_DIR, "dataset_opaque_quant.json")
_ACC = "{{PREDICTOR_ACCURACY}}"
BASE_CLAUSE = f"correctly identifies the choices of agents like you {_ACC} of the time"

# Each value replaces BASE_CLAUSE (the subject "<X> " precedes it, so start with a verb phrase).
MECHANISMS = {
    "m0": BASE_CLAUSE,
    "m0pad": (f"correctly identifies the choices of agents like you {_ACC} of the time; this rate "
              "has been carefully recorded and independently audited over a long series of prior trials"),
    "m1": (f"has built a detailed predictive model of your particular way of reasoning and, by "
           f"simulating that model, anticipates your specific choice {_ACC} of the time"),
    "m2": (f"has scanned your full decision-making process in detail and run that very process "
           f"forward to completion, so its forecast matches your choice {_ACC} of the time"),
    "m3": (f"works by executing an exact copy of your own decision procedure on this very problem, "
           f"so its forecast is identical to whatever your procedure outputs and matches your "
           f"choice {_ACC} of the time"),
}

# ordering + display labels for the mechanism aggregator's x-axis (credibility increasing)
LEVELS = [("m0", "statistical"), ("m0pad", "stat+pad"), ("m1", "indiv-model"),
          ("m2", "process-scan"), ("m3", "exact-copy")]


def generate(base_path: str = _BASE, out_dir: str = _DATA_DIR) -> list[str]:
    items = json.load(open(base_path))
    missing = [it.get("question", "")[:40] for it in items if BASE_CLAUSE not in it["question"]]
    if missing:
        raise ValueError(f"{len(missing)} item(s) lack the base clause; cannot swap mechanism: {missing}")
    written = []
    for level, clause in MECHANISMS.items():
        variant = []
        for it in items:
            v = dict(it)
            v["question"] = it["question"].replace(BASE_CLAUSE, clause)
            v["mechanism_level"] = level
            assert _ACC in v["question"], f"{it.get('question_id', '?')}: lost accuracy marker"
            variant.append(v)
        path = os.path.join(out_dir, f"dataset_mech_{level}.json")
        with open(path, "w") as fh:
            json.dump(variant, fh, indent=2)
        written.append(path)
    return written


def main(argv=None) -> int:
    paths = generate()
    for p in paths:
        n = len(json.load(open(p)))
        print(f"wrote {p}  ({n} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
