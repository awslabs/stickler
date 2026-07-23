"""Tests for stickler.auto: zero-config evaluation of vanilla pydantic models.

Covers the pipeline that turns a plain ``pydantic.BaseModel`` into a scored
stickler evaluation: type-driven comparator inference, name-token refinement,
the tricky pydantic types that break the JSON-schema path (Optional, enum,
datetime, nested models, lists), weight hints, and provenance.
"""

import datetime
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


def test_json_schema_path_still_crashes_but_auto_does_not():
    # Guards the core premise: the naive path fails on Optional, ours doesn't.
    with pytest.raises(ValueError):
        stickler.StructuredModel.from_json_schema(Invoice.model_json_schema())
    # auto path is fine
    stickler.evaluate(_invoice(), _invoice())


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


def test_weights_uniform_by_default():
    spec = stickler.eval_for(Invoice).explain()
    assert all(v["weight"] == 1.0 for v in spec.values())


def test_weight_hints_bumps_id_and_amount():
    spec = stickler.eval_for(Invoice, weight_hints=True).explain()
    assert spec["invoice_id"]["weight"] == 3.0
    assert spec["amount"]["weight"] == 2.5


def test_never_auto_selects_semantic_or_llm():
    spec = stickler.eval_for(Invoice).explain()
    chosen = {v["comparator"] for v in spec.values()}
    assert not (chosen & {"SemanticComparator", "BERTComparator", "LLMComparator"})


# --- batch & guards ---------------------------------------------------------


def test_eval_for_is_cached():
    spec_a = stickler.eval_for(Invoice)
    spec_b = stickler.eval_for(Invoice)
    assert spec_a.eval_model is spec_b.eval_model


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
