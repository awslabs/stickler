#!/usr/bin/env python3
# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""
Pytest verification tests for BulkStructuredModelEvaluator functionality.
Converted from cline/simple_verification.py
"""

import pytest
from typing import List, Optional

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)


# Test models
class Contact(StructuredModel):
    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Transaction(StructuredModel):
    date: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    description: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    amount: float = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class BankStatement(StructuredModel):
    accountNumber: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    contact: Contact = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    transactions: List[Transaction] = ComparableField(aggregate=False, weight=1.0)


@pytest.fixture
def sample_data():
    """Fixture providing sample bank statement data."""
    return {
        "accountNumber": "1234567890",
        "contact": {"phone": "555-123-4567", "email": "test@example.com"},
        "transactions": [
            {"date": "2023-01-01", "description": "ATM Withdrawal", "amount": -100.0},
            {"date": "2023-01-02", "description": "Direct Deposit", "amount": 2000.0},
        ],
    }


@pytest.fixture
def evaluator():
    """Fixture providing a BulkStructuredModelEvaluator instance."""
    return BulkStructuredModelEvaluator(BankStatement, verbose=True)


class TestBulkEvaluatorVerification:
    """Verification tests for BulkStructuredModelEvaluator."""

    def test_evaluator_creation(self, evaluator):
        """Test that evaluator can be created successfully."""
        assert evaluator is not None
        assert evaluator._processed_count == 0

    def test_model_creation(self, sample_data):
        """Test that models can be created from sample data."""
        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        assert gt_model is not None
        assert pred_model is not None
        assert gt_model.accountNumber == sample_data["accountNumber"]
        assert len(gt_model.transactions) == 2

    def test_perfect_match_update_and_compute(self, evaluator, sample_data):
        """Test update and compute with perfect match data."""
        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        # Test update
        evaluator.update(gt_model, pred_model, "test_doc_1")
        assert evaluator._processed_count == 1

        # Test compute
        result = evaluator.compute()
        assert result is not None

        # Verify results
        accuracy = result.metrics.get("cm_accuracy")
        assert accuracy is not None
        assert isinstance(accuracy, (int, float))
        assert accuracy == 1.0  # Perfect match should be 100% accurate

        assert len(result.field_metrics) > 0

    def test_nested_field_paths(self, evaluator, sample_data):
        """Test that all expected nested field paths are found."""
        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        evaluator.update(gt_model, pred_model, "test_doc")
        result = evaluator.compute()

        # Check nested field paths
        field_paths = set(result.field_metrics.keys())

        # Verify expected paths exist
        expected_paths = {
            "accountNumber",
            "contact",
            "contact.phone",
            "contact.email",
            "transactions",
            "transactions.date",
            "transactions.description",
            "transactions.amount",
        }

        missing_paths = expected_paths - field_paths
        assert len(missing_paths) == 0, f"Missing expected paths: {missing_paths}"

        # Verify specific nested paths
        assert "contact.phone" in field_paths
        assert "contact.email" in field_paths
        assert any("transactions." in path for path in field_paths)

    def test_partial_match_processing(self, evaluator, sample_data):
        """Test processing with partial match data."""
        gt_model = BankStatement(**sample_data)

        # Create different data for partial match
        different_data = sample_data.copy()
        different_data["accountNumber"] = "DIFFERENT_ACCOUNT"
        different_data["contact"] = sample_data["contact"].copy()
        different_data["contact"]["email"] = "different@example.com"

        pred_model = BankStatement(**different_data)

        # Process both documents
        evaluator.update(gt_model, gt_model, "perfect_match")  # Perfect match
        evaluator.update(gt_model, pred_model, "partial_match")  # Partial match

        result = evaluator.compute()

        assert evaluator._processed_count == 2

        # Accuracy should be between 0 and 1 (not perfect due to partial match)
        accuracy = result.metrics.get("cm_accuracy")
        assert 0 <= accuracy < 1.0

    def test_multiple_document_processing(self, evaluator, sample_data):
        """Test processing multiple documents."""
        gt_model = BankStatement(**sample_data)

        # Process multiple documents with different accuracy levels
        evaluator.update(gt_model, gt_model, "doc1")  # Perfect match

        different_data = sample_data.copy()
        different_data["accountNumber"] = "DIFFERENT"
        pred_model = BankStatement(**different_data)
        evaluator.update(gt_model, pred_model, "doc2")  # Partial match

        result = evaluator.compute()

        assert evaluator._processed_count == 2
        assert len(result.field_metrics) > 0

        # Should have metrics for all expected fields
        field_paths = set(result.field_metrics.keys())
        assert "accountNumber" in field_paths
        assert "contact" in field_paths

    def test_result_consistency(self, sample_data):
        """Test that results are consistent across multiple runs."""
        evaluator1 = BulkStructuredModelEvaluator(BankStatement)
        evaluator2 = BulkStructuredModelEvaluator(BankStatement)

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        # Process same data in both evaluators
        evaluator1.update(gt_model, pred_model, "doc1")
        evaluator2.update(gt_model, pred_model, "doc1")

        result1 = evaluator1.compute()
        result2 = evaluator2.compute()

        # Results should be identical
        assert result1.metrics["cm_accuracy"] == result2.metrics["cm_accuracy"]
        assert len(result1.field_metrics) == len(result2.field_metrics)

    def test_field_metrics_structure(self, evaluator, sample_data):
        """Test that field metrics have the expected structure."""
        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        evaluator.update(gt_model, pred_model, "test_doc")
        result = evaluator.compute()

        # Check that each field metric has expected keys
        for field_name, metrics in result.field_metrics.items():
            assert "cm_accuracy" in metrics
            assert "cm_precision" in metrics
            assert "cm_recall" in metrics
            assert "cm_f1" in metrics

            # Check that metric values are numeric
            for metric_name, value in metrics.items():
                if metric_name.startswith("cm_"):
                    assert isinstance(value, (int, float))
                    assert 0 <= value <= 1
