"""
Regression test for first-field None auto-population bug.

Validates that ComparableField defaults to None correctly for all fields,
regardless of position. Previously the first field would auto-populate
with a FieldInfo object instead of remaining None.

See: https://github.com/awslabs/stickler/issues/17
"""

from typing import Any, Optional

from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class TwoFieldModel(StructuredModel):
    code: Optional[str] = ComparableField(weight=1.0)
    description: Optional[str] = ComparableField(weight=1.0)


def test_first_field_none_stays_none():
    """First field set to None should remain None, not become FieldInfo."""
    data = TwoFieldModel(**{"code": None})
    assert data.code is None
    assert data.description is None


def test_second_field_none_stays_none():
    """Second field set to None should remain None."""
    data = TwoFieldModel(**{"code": "C", "description": None})
    assert data.code == "C"
    assert data.description is None


def test_both_fields_none():
    """Both fields explicitly None."""
    data = TwoFieldModel(**{"code": None, "description": None})
    assert data.code is None
    assert data.description is None


def test_omitted_fields_default_to_none():
    """Fields not provided at all should default to None."""
    data = TwoFieldModel(**{})
    assert data.code is None
    assert data.description is None


def test_first_field_none_classification():
    """When first field is None in pred but has value in GT, should be FN not FD.

    This is the core classification error from the issue: if the first field
    incorrectly becomes FieldInfo, the comparison sees two non-null values
    and classifies as FD instead of FN.
    """
    gt = TwoFieldModel(code="ABC", description="test")
    pred = TwoFieldModel(**{"code": None, "description": "test"})

    result = gt.compare_with(pred, include_confusion_matrix=True)
    code_cm = result["confusion_matrix"]["fields"]["code"]["overall"]

    assert code_cm["fn"] == 1, "Missing prediction should be FN"
    assert code_cm["fd"] == 0, "Should not be FD"
    assert code_cm["tp"] == 0
