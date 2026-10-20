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
