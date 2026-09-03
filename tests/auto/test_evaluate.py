"""Tests for stickler.auto: zero-config evaluation of vanilla pydantic models.

Covers the pipeline that turns a plain ``pydantic.BaseModel`` into a scored
stickler evaluation: type-driven comparator inference, name-token refinement,
the tricky pydantic types that break the JSON-schema path (Optional, enum,
datetime, nested models, lists), weight hints, and provenance.
"""

import datetime
import decimal
import enum
from typing import List, Optional

import pytest
from pydantic import BaseModel

import stickler
from stickler.auto.inference import infer_field_config, unwrap_optional


class Status(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class Line(BaseModel):
    sku: str
    qty: int
    price: float


class Invoice(BaseModel):
    invoice_id: str
    amount: float
    when: datetime.date
    status: Status
    customer_name: str
    note: Optional[str] = None
    lines: List[Line] = []


def _invoice(**overrides):
    base = dict(
        invoice_id="A1",
        amount=100.00,
        when=datetime.date(2020, 1, 1),
        status=Status.OPEN,
        customer_name="Acme Corporation",
        note="paid in full",
        lines=[Line(sku="S1", qty=2, price=9.99), Line(sku="S2", qty=1, price=4.50)],
    )
    base.update(overrides)
    return Invoice(**base)


# --- end-to-end -------------------------------------------------------------


def test_identical_scores_perfect():
    gt = _invoice()
    result = stickler.evaluate(gt, _invoice())
    assert result.overall_score == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in result.field_scores.values())


def test_tolerant_float_within_tolerance_matches():
    # 100.00 vs 100.001 is within the inferred relative tolerance.
    gt = _invoice(amount=100.00)
    pred = _invoice(amount=100.001)
    result = stickler.evaluate(gt, pred)
    assert result.field_scores["amount"] == pytest.approx(1.0)


def test_date_matches_exactly_not_levenshtein():
    gt = _invoice(when=datetime.date(2020, 1, 1))
    pred = _invoice(when=datetime.date(2020, 1, 1))
    assert stickler.evaluate(gt, pred).field_scores["when"] == pytest.approx(1.0)
    # The discriminating case: "2020-01-01" vs "2020-01-02" is 1 edit apart
    # (Levenshtein ~0.9) but a different date; DateComparator must score 0.
    off_by_a_day = _invoice(when=datetime.date(2020, 1, 2))
    result = stickler.evaluate(gt, off_by_a_day)
    assert result.field_scores["when"] == pytest.approx(0.0)
    # The post-clip score alone cannot tell the two comparators apart:
    # Levenshtein's raw 0.9 is under the 0.95 threshold and clips to 0.0 too.
    # Assert the comparator and the raw similarity, which do diverge.
    assert result.explain()["when"]["comparator"] == "DateComparator"
    assert result.explain()["when"]["raw_similarity"] == pytest.approx(0.0)


def test_enum_uses_exact():
    gt = _invoice(status=Status.OPEN)
    pred = _invoice(status=Status.CLOSED)
    assert stickler.evaluate(gt, pred).field_scores["status"] == pytest.approx(0.0)


def test_optional_field_none_does_not_crash():
    gt = _invoice(note=None)
    pred = _invoice(note=None)
    result = stickler.evaluate(gt, pred)
    assert "note" in result.field_scores


def test_reordered_list_matches_via_hungarian():
    gt = _invoice()
    reordered = _invoice(
        lines=[Line(sku="S2", qty=1, price=4.50), Line(sku="S1", qty=2, price=9.99)]
    )
    assert stickler.evaluate(gt, reordered).field_scores["lines"] == pytest.approx(1.0)


def test_auto_path_handles_models_regardless_of_json_schema_path():
    # The premise: evaluate() works on a model with Optional/enum/date fields
    # via live annotations, without depending on the JSON-schema round-trip.
    # (We do NOT assert the schema path crashes: that is an implementation
    # detail that may improve independently, e.g. via #127-style fixes.)
    result = stickler.evaluate(_invoice(), _invoice())
    assert result.overall_score == pytest.approx(1.0)


# --- inference --------------------------------------------------------------


def test_unwrap_optional():
    assert unwrap_optional(Optional[str]) == (str, True)
    assert unwrap_optional(str) == (str, False)


@pytest.mark.parametrize(
    "field_name,annotation,expected",
    [
        ("active", bool, "ExactComparator"),
        ("count", int, "NumericComparator"),
        ("ratio", float, "NumericComparator"),
        ("label", str, "LevenshteinComparator"),
        ("invoice_id", str, "ExactComparator"),  # name-token overrides type
        ("total_amount", float, "NumericComparator"),
        ("customer_name", str, "LevenshteinComparator"),
    ],
)
def test_infer_comparator(field_name, annotation, expected):
    from pydantic.fields import FieldInfo

    spec = infer_field_config(field_name, FieldInfo(annotation=annotation))
    assert spec.comparator_name == expected


# The rows above use field names that carry name tokens (``count`` hits the
# quantity rule, ``total_amount`` the money rule), so a name rule can satisfy
# them even if the type branch returns something else entirely. These pin the
# type dispatch table itself, with token-free names and a provenance assertion
# proving the TYPE signal is what fired.
@pytest.mark.parametrize(
    "annotation,expected,provenance_prefix",
    [
        (bool, "ExactComparator", "type:bool"),
        (int, "NumericComparator", "type:int"),
        (float, "NumericComparator", "type:float"),
        (decimal.Decimal, "NumericComparator", "type:Decimal"),
        (datetime.date, "DateComparator", "type:date"),
        (datetime.datetime, "DateComparator", "type:datetime"),
        (str, "LevenshteinComparator", "type:str"),
        (Status, "ExactComparator", "type:"),
        (dict, "ANLSStarComparator", "type:dict"),
    ],
)
def test_type_dispatch_table(annotation, expected, provenance_prefix):
    from pydantic.fields import FieldInfo

    # "val" carries no name token, so nothing can mask the type branch.
    spec = infer_field_config("val", FieldInfo(annotation=annotation))
    assert spec.comparator_name == expected
    assert spec.provenance[0].startswith(provenance_prefix)


def test_weights_uniform_by_default():
    spec = stickler.eval_for(Invoice).explain()
    assert all(v["weight"] == 1.0 for v in spec.values())


def test_weight_hints_bumps_id_and_amount():
    # Contract: weight_hints raises importance-signalling fields above the
    # uniform 1.0 baseline and above a plain text field, without pinning the
    # exact bump magnitudes (which are tunable).
    hinted = stickler.eval_for(Invoice, weight_hints=True).explain()
    uniform = stickler.eval_for(Invoice).explain()
    assert all(v["weight"] == 1.0 for v in uniform.values())
    assert hinted["invoice_id"]["weight"] > 1.0
    assert hinted["amount"]["weight"] > 1.0
    assert hinted["invoice_id"]["weight"] >= hinted["customer_name"]["weight"]


def test_never_auto_selects_semantic_or_llm():
    spec = stickler.eval_for(Invoice).explain()
    chosen = {v["comparator"] for v in spec.values()}
    assert not (chosen & {"SemanticComparator", "BERTComparator", "LLMComparator"})


# --- batch & guards ---------------------------------------------------------


def test_eval_for_is_cached():
    spec_a = stickler.eval_for(Invoice)
    spec_b = stickler.eval_for(Invoice)
    assert spec_a.eval_model is spec_b.eval_model


def test_shadow_cache_is_keyed_on_options():
    """Cache MISSES matter as much as hits: options must be in the key.

    test_eval_for_is_cached pins that a repeat call reuses the shadow class.
    Nothing pinned the converse, so dropping an option from the cache key made
    the second call silently reuse the class compiled for the first one, i.e.
    the same inputs scored differently depending on call order within a
    process. That is invisible in any single-option test run.
    """
    assert (
        stickler.eval_for(Invoice, match_threshold=0.5).eval_model
        is not stickler.eval_for(Invoice, match_threshold=0.9).eval_model
    )
    assert (
        stickler.eval_for(Invoice, weight_hints=False).eval_model
        is not stickler.eval_for(Invoice, weight_hints=True).eval_model
    )


@pytest.mark.parametrize("order", [(0.5, 0.9), (0.9, 0.5)])
def test_match_threshold_is_honored_regardless_of_call_order(order):
    """match_threshold must drive the result, in either evaluation order."""
    gt = _invoice(lines=[Line(sku="S1", qty=2, price=9.99)])
    # One line item, similarity ~0.66: matched at 0.5, not matched at 0.9.
    pred = _invoice(lines=[Line(sku="S1", qty=2, price=100.00)])
    scores = {
        t: stickler.evaluate(gt, pred, match_threshold=t).f1 for t in order
    }
    assert scores[0.5] > scores[0.9]


def test_eval_spec_reuse_matches_evaluate():
    gt, pred = _invoice(), _invoice(amount=100.001)
    spec = stickler.eval_for(Invoice)
    assert spec.evaluate(gt, pred).overall_score == pytest.approx(
        stickler.evaluate(gt, pred).overall_score
    )


def test_mismatched_classes_raise():
    class Other(BaseModel):
        z: str

    with pytest.raises(TypeError):
        stickler.evaluate(_invoice(), Other(z="a"))


def test_explain_has_provenance():
    spec = stickler.eval_for(Invoice).explain()
    assert spec["amount"]["why"]  # non-empty provenance trail
    assert spec["invoice_id"]["source"] == "name-token"
