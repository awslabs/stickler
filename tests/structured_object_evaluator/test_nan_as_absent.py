"""A float NaN is absent, the same as ``None`` (#367).

NaN is how pandas and numpy spell a missing number. Read as a value it scored
two identical NaNs 0.0, because NaN equals nothing under IEEE 754. Read as
absent it takes ``None``'s place in every confusion-matrix cell.
"""

from decimal import Decimal
from typing import List, Optional

import numpy as np
import pytest

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.field_helper import FieldHelper
from stickler.structured_object_evaluator.models.null_helper import NullHelper
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

NAN = float("nan")


class Reading(StructuredModel):
    value: Optional[float] = ComparableField(
        comparator=NumericComparator(relative_tolerance=0.001), default=None
    )


class Item(StructuredModel):
    sku: str = ComparableField(comparator=LevenshteinComparator(), default="a")
    value: Optional[float] = ComparableField(
        comparator=NumericComparator(), default=None
    )


class Order(StructuredModel):
    items: Optional[List[Item]] = ComparableField(default=None)


def _counts(gt, pred):
    result = Reading(value=gt).compare_with(
        Reading(value=pred), include_confusion_matrix=True
    )
    overall = result["confusion_matrix"]["fields"]["value"]["overall"]
    counts = {k: overall[k] for k in ("tp", "tn", "fa", "fd", "fn") if overall[k]}
    return result["field_scores"]["value"], counts


@pytest.mark.parametrize(
    "gt, pred, score, counts",
    [
        (NAN, NAN, 1.0, {"tn": 1}),
        (NAN, None, 1.0, {"tn": 1}),
        (None, NAN, 1.0, {"tn": 1}),
        (NAN, 5.0, 0.0, {"fa": 1}),
        (5.0, NAN, 0.0, {"fn": 1}),
    ],
)
def test_nan_classifies_like_none(gt, pred, score, counts):
    assert _counts(gt, pred) == (score, counts)


def test_numpy_nan_is_absent():
    assert _counts(np.float64("nan"), np.float64("nan")) == (1.0, {"tn": 1})


def test_comparator_reads_nan_as_missing():
    cmp = NumericComparator(relative_tolerance=0.001)
    assert cmp.compare(NAN, NAN) == 1.0
    assert cmp.compare(NAN, None) == 1.0
    assert cmp.compare(NAN, 1.0) == 0.0
    assert cmp.compare(1.0, NAN) == 0.0


def test_decimal_nan_is_still_a_non_match():
    # Only a float NaN is absent; a Decimal NaN still reaches the guard.
    assert NumericComparator().compare(Decimal("NaN"), Decimal("NaN")) == 0.0


def test_nan_in_a_list_element_field_is_absent():
    gt = Order(items=[Item(sku="a", value=NAN)])
    pred = Order(items=[Item(sku="a", value=NAN)])
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert result["field_scores"]["items"] == 1.0
    leaf = result["confusion_matrix"]["fields"]["items"]["fields"]["value"]
    assert leaf["overall"]["tn"] == 1
    assert leaf["overall"]["fd"] == 0


def test_null_predicates_agree_on_nan():
    assert NullHelper.is_nan(NAN)
    assert not NullHelper.is_nan(0.0)
    assert not NullHelper.is_nan("nan")
    assert NullHelper.is_effectively_null_for_primitives(NAN)
    assert FieldHelper.is_null_value(NAN)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "gt, pred, score, count, reason",
    [
        (NAN, NAN, 1.0, "tn", "exact match"),
        (NAN, None, 1.0, "tn", "exact match"),
        (None, NAN, 1.0, "tn", "exact match"),
        (NAN, 5.0, 0.0, "fa", "false alarm (unmatched prediction)"),
        (5.0, NAN, 0.0, "fn", "false negative (unmatched ground truth)"),
        (0.0, 0.0, 1.0, "tp", "exact match"),
        (0.0, 5.0, 0.0, "fd", "below threshold (0.000 < 0.5)"),
    ],
)
def test_field_reports_agree_with_missing_value_counts(
    nested, gt, pred, score, count, reason
):
    class ReportItem(Item):
        # The matching sku must keep the pair accepted when value is missing.
        match_threshold = 0.4

    class ReportOrder(StructuredModel):
        items: List[ReportItem] = ComparableField()

    if nested:
        expected = ReportOrder(items=[ReportItem(value=gt)])
        actual = ReportOrder(items=[ReportItem(value=pred)])
        path = "items[0].value"
    else:
        expected, actual = Reading(value=gt), Reading(value=pred)
        path = "value"

    result = expected.compare_with(
        actual, include_confusion_matrix=True, document_field_comparisons=True
    )
    fields = result["confusion_matrix"]["fields"]
    if nested:
        assert fields["items"]["overall"]["tp"] == 1
        fields = fields["items"]["fields"]
    assert fields["value"]["overall"][count] == 1
    row = next(r for r in result["field_comparisons"] if r["expected_key"] == path)
    assert row["actual_key"] == path
    assert row["match"] is (count in ("tp", "tn"))
    assert row["score"] == score
    assert row["reason"] == reason
