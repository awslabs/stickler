"""
Test JSON handling capabilities in StructuredModel.

This test suite validates:
1. Creating models from JSON using from_json
2. Handling missing fields in prediction vs ground truth
3. Handling extra fields in prediction vs ground truth
"""

import pytest
from typing import Dict, Any, List, Optional
from pydantic import Field

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.utils.compare_json import compare_json
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator


class InvoiceModel(StructuredModel):
    """Model for invoice data extraction."""

    invoice_number: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.9,
        weight=2.0,  # Must match for the entire invoice to be considered a match
    )
    invoice_date: Optional[str] = Field(
        default=None,
        json_schema_extra=lambda schema: schema.update(
            {
                "x-comparison": {
                    "comparator_type": "LevenshteinComparator",
                    "threshold": 0.8,
                    "weight": 1.0,
                }
            }
        ),
    )
    total_amount: float = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.95, weight=2.0
    )
    vendor_name: Optional[str] = Field(
        default=None,
        json_schema_extra=lambda schema: schema.update(
            {
                "x-comparison": {
                    "comparator_type": "LevenshteinComparator",
                    "threshold": 0.7,
                    "weight": 1.0,
                }
            }
        ),
    )


def test_from_json_creation():
    """Test creating a StructuredModel from JSON data."""

    # JSON data representing an invoice
    invoice_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-01-15",
        "total_amount": 1250.50,
        "vendor_name": "ACME Corp",
    }

    # Create model from JSON
    invoice = InvoiceModel.from_json(invoice_json)

    # Verify fields were correctly parsed
    assert invoice.invoice_number == "INV-2023-001"
    assert invoice.invoice_date == "2023-01-15"
    assert invoice.total_amount == 1250.50
    assert invoice.vendor_name == "ACME Corp"


def test_missing_fields_handling():
    """Test handling of missing fields in prediction vs ground truth."""

    # Ground truth with all fields
    gt_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-01-15",
        "total_amount": 1250.50,
        "vendor_name": "ACME Corp",
    }

    # Prediction missing some fields
    pred_json = {
        "invoice_number": "INV-2023-001",
        "total_amount": 1250.50,
        # Missing invoice_date and vendor_name
    }

    # Convert JSONs to models and compare
    gt = InvoiceModel.from_json(gt_json)
    pred = InvoiceModel.from_json(pred_json)

    comparison = gt.compare_with(pred)

    # Invoice number and total amount should match
    assert comparison["field_scores"]["invoice_number"] == 1.0
    assert comparison["field_scores"]["total_amount"] == 1.0

    # Missing fields should have default values in the model
    assert pred.invoice_date is None
    assert pred.vendor_name is None

    # But all_fields_matched should be False due to missing fields
    assert comparison["all_fields_matched"] == False

    # Overall score should be calculated based on fields present in both models
    # with weights taken into account
    expected_score = (2.0 * 1.0 + 2.0 * 1.0) / (
        2.0 + 1.0 + 2.0 + 1.0
    )  # Weighted average
    assert abs(comparison["overall_score"] - expected_score) < 0.01


def test_extra_fields_handling():
    """Test handling of extra fields in prediction vs ground truth."""

    # Ground truth with standard fields
    gt_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-01-15",
        "total_amount": 1250.50,
        "vendor_name": "ACME Corp",
    }

    # Prediction with extra fields not in the model
    pred_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-01-15",
        "total_amount": 1250.50,
        "vendor_name": "ACME Corp",
        "payment_terms": "Net 30",  # Extra field
        "currency": "USD",  # Extra field
        "notes": "Please pay on time",  # Extra field
    }

    # Convert JSONs to models
    gt = InvoiceModel.from_json(gt_json)
    pred = InvoiceModel.from_json(pred_json)

    # Extra fields should be stored in the extra_fields attribute
    assert "payment_terms" in pred.extra_fields
    assert "currency" in pred.extra_fields
    assert "notes" in pred.extra_fields
    assert pred.extra_fields["payment_terms"] == "Net 30"
    assert pred.extra_fields["currency"] == "USD"
    assert pred.extra_fields["notes"] == "Please pay on time"

    # Compare models
    comparison = gt.compare_with(pred)

    # All defined fields should match perfectly
    assert comparison["field_scores"]["invoice_number"] == 1.0
    assert comparison["field_scores"]["invoice_date"] == 1.0
    assert comparison["field_scores"]["total_amount"] == 1.0
    assert comparison["field_scores"]["vendor_name"] == 1.0

    # Overall score should be perfect
    assert comparison["overall_score"] == 1.0
    assert comparison["all_fields_matched"] == True


def test_compare_json_utility():
    """Test the compare_json utility function for direct JSON comparison."""

    # Ground truth JSON
    gt_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-01-15",
        "total_amount": 1250.50,
        "vendor_name": "ACME Corp",
    }

    # Prediction JSON with slight differences
    pred_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023/01/15",  # Different date format
        "total_amount": 1250.00,  # Slightly different amount
        "vendor_name": "ACME Corporation",  # Different vendor name
        "payment_terms": "Net 30",  # Extra field
    }

    # Use compare_json utility
    comparison = compare_json(gt_json, pred_json, InvoiceModel)

    # Check results
    assert comparison["field_scores"]["invoice_number"] == 1.0  # Perfect match
    assert comparison["field_scores"]["invoice_date"] < 1.0  # Similar but not perfect
    assert comparison["field_scores"]["total_amount"] < 1.0  # Similar but not perfect
    assert comparison["field_scores"]["vendor_name"] < 1.0  # Similar but not perfect

    # Overall score should be calculated properly
    assert 0.0 < comparison["overall_score"] < 1.0


def test_error_handling_in_compare_json():
    """Test error handling in compare_json when JSON is invalid."""

    # Valid ground truth JSON
    gt_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-01-15",
        "total_amount": 1250.50,
        "vendor_name": "ACME Corp",
    }

    # Invalid prediction JSON (total_amount should be numeric)
    pred_json = {
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-01-15",
        "total_amount": "invalid value",  # Not a number
        "vendor_name": "ACME Corp",
    }

    # Use compare_json utility
    comparison = compare_json(gt_json, pred_json, InvoiceModel)

    # Check that error is handled gracefully
    assert "error" in comparison
    assert comparison["overall_score"] == 0.0
