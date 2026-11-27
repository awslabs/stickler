"""Tests for the examples in docs/docs/Guides/Best-Practices/README.md.

Keep the models and calibration pairs below identical to the guide so the
documentation can't drift from real behaviour.
"""

from typing import List

import pytest

from stickler import (
    ComparableField,
    ExactComparator,
    FuzzyComparator,
    LevenshteinComparator,
    NumericComparator,
    StructuredModel,
)

THRESHOLD = 0.75


@pytest.mark.parametrize(
    "gt,pred,expected_match",
    [
        ("Acme Corporation", "Acme Corporaton", True),   # 0.94
        ("Acme Corporation", "Beta Industries", False),  # 0.13
        ("Jon Smith", "John Smith", True),               # 0.90
        ("123 Main St", "456 Oak Ave", False),           # 0.18
    ],
)
def test_calibration(gt, pred, expected_match):
    score = LevenshteinComparator().compare(gt, pred)
    assert (score >= THRESHOLD) == expected_match


class LineItem(StructuredModel):
    description: str = ComparableField(comparator=FuzzyComparator(), weight=1.0)
    quantity: int = ComparableField(comparator=ExactComparator(), weight=1.2)
    unit_price: float = ComparableField(
        comparator=NumericComparator(absolute_tolerance=0.01), weight=1.5
    )


class Invoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=ExactComparator(), weight=3.0)
    total_amount: float = ComparableField(
        comparator=NumericComparator(absolute_tolerance=0.01), weight=2.5
    )
    vendor_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )
    line_items: List[LineItem] = ComparableField(weight=1.0)
    internal_notes: str = ComparableField(comparator=FuzzyComparator(), weight=0.2)


def _invoice_data(invoice_id: str) -> dict:
    return {
        "invoice_id": invoice_id,
        "total_amount": 1250.00,
        "vendor_name": "Acme Corporation",
        "line_items": [
            {"description": "Widget A", "quantity": 10, "unit_price": 50.00},
        ],
        "internal_notes": "Processed by system A",
    }


def test_weights_example_scores_mismatched_invoice_id_as_zero():
    ground_truth = Invoice(**_invoice_data("INV-1"))
    prediction = Invoice(**_invoice_data("INV-2"))

    result = ground_truth.compare_with(prediction)

    assert result["field_scores"]["invoice_id"] == 0.0