"""The deprecated ``confusion_matrix_calculator`` module still imports and classifies."""

import importlib
import sys
from typing import Optional

import pytest

from stickler import ComparableField, ExactComparator, StructuredModel

MODULE = "stickler.structured_object_evaluator.models.confusion_matrix_calculator"


class Invoice(StructuredModel):
    vendor: Optional[str] = ComparableField(comparator=ExactComparator())


def _import_fresh():
    sys.modules.pop(MODULE, None)
    with pytest.warns(DeprecationWarning, match="removed in 1.1.0"):
        return importlib.import_module(MODULE)


def test_importing_it_warns():
    _import_fresh()


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        ("Acme", "Acme", {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}),
        ("Acme", "Zzz", {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0}),
        ("Acme", None, {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 1}),
        (None, "Acme", {"tp": 0, "fa": 1, "fd": 0, "fp": 1, "tn": 0, "fn": 0}),
        (None, None, {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0}),
    ],
)
def test_it_classifies_a_field_as_in_1_0_1(gt, pred, expected):
    module = _import_fresh()
    result = module.ConfusionMatrixCalculator.classify_field_for_confusion_matrix(
        Invoice(vendor=gt), "vendor", pred
    )
    assert result == expected
