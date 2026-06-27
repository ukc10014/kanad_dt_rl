"""Build dataset_payoff_template.json from dataset_mech_m3.json by templatising the payoffs.

The m3 items bake the payoffs into the question text ("60 points", "100 points"). To run the
payoff ablation (vary S so p* moves) we need those numbers injectable like {{PREDICTOR_ACCURACY}}.
This replaces the literal payoff numbers with {{PAYOFF_SMALL}} / {{PAYOFF_BIG}} markers; the empty
box ("0 points") is left as-is. The driver (payoff_ablation.py) substitutes concrete B/S per config.

All 20 m3 items share the same payoff phrasing (S=60, B=100, empty=0) and the binding exact-copy
predictor framing — the substitution is a safe two-pattern regex.
"""
from __future__ import annotations
import json, os, re

SRC = os.path.join(os.path.dirname(__file__), "data", "dataset_mech_m3.json")
OUT = os.path.join(os.path.dirname(__file__), "data", "dataset_payoff_template.json")


def templatise(q: str) -> str:
    q = re.sub(r"\b100 point", "{{PAYOFF_BIG}} point", q)
    q = re.sub(r"\b60 point", "{{PAYOFF_SMALL}} point", q)
    return q


def main() -> int:
    src = json.load(open(SRC))
    out = []
    for it in src:
        it2 = dict(it)
        it2["question"] = templatise(it["question"])
        # sanity: both markers present, no stray bare payoff numbers left before "point"
        assert "{{PAYOFF_BIG}}" in it2["question"] and "{{PAYOFF_SMALL}}" in it2["question"], it.get("id")
        assert "{{PREDICTOR_ACCURACY}}" in it2["question"]
        out.append(it2)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}  ({len(out)} items templatised: payoffs -> markers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
