"""Answer-matching scorer over abstract tokens (PLAN.md §3).

Hard rule: we NEVER string-match on natural-language "one-box"/"two-box"/"Newcomb".
The chosen option is resolved purely from the abstract label the model emits, against the
per-sample ``token_role`` mapping. An unparseable completion is ``invalid`` — never a
silent CDT.
"""
from __future__ import annotations

import re

from inspect_ai.scorer import Score, Target, mean, scorer, stderr

# Result roles
ROLE_NON_CDT = "non_cdt"
ROLE_CDT = "cdt"
ROLE_INVALID = "invalid"

_ANSWER_CUE = re.compile(r"answer\s*[:\-]?\s*", re.IGNORECASE)


def _find_token(text: str, legal_tokens: list[str], *, ignore_case: bool) -> str | None:
    """Return the first legal token that appears standalone in ``text``, else None."""
    flags = re.IGNORECASE if ignore_case else 0
    # Longest-first so a multi-char token isn't pre-empted by a substring.
    ordered = sorted(legal_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in ordered)
    # Standalone: not flanked by other alphanumerics (so 'A' won't match inside 'Apple').
    rx = re.compile(rf"(?<![A-Za-z0-9])({pattern})(?![A-Za-z0-9])", flags)
    m = rx.search(text)
    if not m:
        return None
    hit = m.group(1)
    if ignore_case:
        # Map back to the canonical-cased legal token.
        for t in legal_tokens:
            if t.lower() == hit.lower():
                return t
        return None
    return hit


def resolve_choice(
    completion: str,
    legal_tokens: list[str],
    token_role: dict,
    *,
    cot: bool = False,
) -> tuple[str, str | None, bool]:
    """Resolve a model completion to (chosen_role, chosen_token, is_valid).

    Forced-choice: take the first legal token emitted. With CoT, prefer the text after the
    last "Answer:" cue. Matching is tried case-sensitively first (the prompt shows uppercase
    labels), then case-insensitively as a fallback. No legal token found -> invalid.
    """
    text = (completion or "").strip()
    if not text:
        return ROLE_INVALID, None, False

    search_space = text
    if cot:
        cues = list(_ANSWER_CUE.finditer(text))
        if cues:
            search_space = text[cues[-1].end():]

    tok = _find_token(search_space, legal_tokens, ignore_case=False)
    if tok is None:
        tok = _find_token(search_space, legal_tokens, ignore_case=True)
    if tok is None and cot:
        # Fall back to scanning the whole completion if the post-cue slice had nothing.
        tok = _find_token(text, legal_tokens, ignore_case=False) or _find_token(
            text, legal_tokens, ignore_case=True
        )

    if tok is None:
        return ROLE_INVALID, None, False
    return token_role.get(tok, ROLE_INVALID), tok, True


@scorer(metrics=[mean(), stderr()])
def newcomb_scorer():
    """Inspect scorer. value = is_k (1 if non-CDT chosen, else 0).

    Invalid completions score 0 but carry ``is_valid=False`` in metadata so sweep.py can
    compute K-rate over *valid* samples only and report the invalid rate separately.
    """

    async def score(state, target: Target) -> Score:
        md = state.metadata or {}
        legal = md.get("legal_tokens", [])
        roles = md.get("token_role", {})
        completion = state.output.completion if state.output else ""
        role, tok, is_valid = resolve_choice(
            completion, legal, roles, cot=bool(md.get("cot", False))
        )
        is_k = 1.0 if role == ROLE_NON_CDT else 0.0
        return Score(
            value=is_k,
            answer=tok or "",
            explanation=f"chosen_role={role} is_valid={is_valid} raw={completion!r}",
            metadata={
                "chosen_role": role,
                "chosen_token": tok,
                "is_k": is_k,
                "is_valid": is_valid,
                "p": md.get("p"),
                "item_id": md.get("item_id"),
                "is_holdout": md.get("is_holdout", False),
                "strata": md.get("strata", {}),
            },
        )

    return score
