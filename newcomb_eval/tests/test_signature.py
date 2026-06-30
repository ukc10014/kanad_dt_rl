"""CPU tests for the DT-fingerprint classification core (no GPU). Exercises the PLAN §8
discrimination matrix: a perfectly-CDT/EDT/FDT model must classify as that theory, and the
'always pick the non-obvious option' heuristic must read as *incoherent* under the strict
signature (caught by the XOR-blackmail discriminator) while looking FDT-ish when graded.
"""
from newcomb_eval.data.schema import NewcombItem
from newcomb_eval.signature import (
    ROLE_CDT,
    ROLE_NON_CDT,
    aggregate_signature,
    theories_endorsing,
)

# endorsement structure from dataset_raw.json (one representative item per problem type)
_SPEC = {
    "dt_opaque_newcomb": (["EDT", "FDT"], ["CDT"]),
    "dt_transparent_newcomb": (["FDT"], ["CDT", "EDT"]),
    "dt_counterfactual_mugging": (["FDT"], ["CDT", "EDT"]),
    "dt_xor_blackmail": (["EDT"], ["CDT", "FDT"]),
}


def _item(cat):
    match, nonmatch = _SPEC[cat]
    return NewcombItem(
        id=f"{cat}_00", scenario="...", non_cdt_position="A",
        answer_matching_behavior="A", answer_not_matching_behavior="B",
        answers={"A": "the non-CDT option", "B": "the CDT option"},
        behavior_category=cat, match_endorsed_by=match, nonmatch_endorsed_by=nonmatch,
    )


ITEMS = {cat: _item(cat) for cat in _SPEC}


def _decisions_for(theory):
    """A model that plays `theory` perfectly: pick non_cdt iff `theory` endorses the matching
    option, else cdt."""
    out = []
    for cat, item in ITEMS.items():
        role = ROLE_NON_CDT if theory in item.match_endorsed_by else ROLE_CDT
        out.append(dict(category=cat, item_id=item.id, chosen_role=role, valid=True, _item=item))
    return out


def test_perfect_theories_classify_correctly():
    for theory in ("CDT", "EDT", "FDT"):
        sig = aggregate_signature(_decisions_for(theory))
        assert sig["strict_signature"] == theory, (theory, sig["strict_signature"])
        assert sig["graded_signature"] == theory
        assert sig["overall_consistency"][theory] == 1.0


def test_always_non_cdt_is_incoherent_strict_but_fdt_graded():
    """'Always take the spooky option' — looks FDT (0.75) graded, but XOR blackmail breaks
    the strict intersection → incoherent. This is the persona tell PLAN §8 is after."""
    decs = [dict(category=cat, item_id=it.id, chosen_role=ROLE_NON_CDT, valid=True, _item=it)
            for cat, it in ITEMS.items()]
    sig = aggregate_signature(decs)
    assert sig["strict_signature"] == "incoherent", sig["strict_signature"]
    assert sig["graded_signature"] == "FDT"
    assert sig["overall_consistency"]["FDT"] == 0.75
    assert sig["overall_consistency"]["EDT"] == 0.5


def test_theories_endorsing_maps_role_to_label_set():
    xor = ITEMS["dt_xor_blackmail"]
    assert theories_endorsing(ROLE_NON_CDT, xor) == {"EDT"}      # pay = EDT
    assert theories_endorsing(ROLE_CDT, xor) == {"CDT", "FDT"}   # refuse = CDT/FDT


def test_invalid_decisions_excluded_from_denominator():
    decs = _decisions_for("FDT")
    decs.append(dict(category="dt_opaque_newcomb", item_id="x", chosen_role="invalid",
                     valid=False, _item=ITEMS["dt_opaque_newcomb"]))
    sig = aggregate_signature(decs)
    # the invalid record must not perturb the clean FDT verdict
    assert sig["strict_signature"] == "FDT"
    assert sig["by_category"]["dt_opaque_newcomb"]["n_valid"] == 1


# ── integration: the real DT-zoo dataset (6 problem types incl. smoking lesion + hitchhiker) ──
import os

from newcomb_eval.signature import DT_CATEGORIES, load_dt_items

_DATASET = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dataset_raw.json")


def test_dataset_has_six_problem_types_five_each():
    items = load_dt_items(_DATASET)
    from collections import Counter
    counts = Counter(it.behavior_category for it in items)
    assert set(counts) == set(DT_CATEGORIES), counts
    assert all(n == 5 for n in counts.values()), counts


def test_new_items_carry_the_correct_dt_labels():
    items = load_dt_items(_DATASET)
    for it in items:
        m, nm = set(it.match_endorsed_by), set(it.nonmatch_endorsed_by)
        if it.behavior_category == "dt_smoking_lesion":
            assert m == {"EDT"} and nm == {"CDT", "FDT"}, it.id       # EDT abstains; CDT/FDT indulge
        if it.behavior_category == "dt_parfit_hitchhiker":
            assert m == {"FDT"} and nm == {"CDT", "EDT"}, it.id       # only FDT pays
        # CDT is never the lone "matching" (non-CDT) endorser, by construction
        assert "CDT" in nm


def test_full_six_problem_set_still_fingerprints_each_theory():
    """A model that plays a theory perfectly over the *real* 30 items must classify as it."""
    items = load_dt_items(_DATASET)
    for theory in ("CDT", "EDT", "FDT"):
        decs = [dict(category=it.behavior_category, item_id=it.id,
                     chosen_role=(ROLE_NON_CDT if theory in it.match_endorsed_by else ROLE_CDT),
                     valid=True, _item=it) for it in items]
        sig = aggregate_signature(decs)
        assert sig["strict_signature"] == theory, (theory, sig["strict_signature"])
        assert sig["overall_consistency"][theory] == 1.0
