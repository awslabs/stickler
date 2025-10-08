#!/usr/bin/env python3
# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""
Comprehensive parity testing for BulkStructuredModelEvaluator.

This test suite validates that BulkStructuredModelEvaluator produces identical results
to individual StructuredModel.compare_with() calls, ensuring the bulk evaluator is
a faithful aggregation mechanism.
"""

import pytest
import json
from typing import List, Optional, Dict, Any, Tuple
from pydantic import Field

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.exact import ExactComparator
from stickler.comparators.fuzzy import FuzzyComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.utils.process_evaluation import ProcessEvaluation


# Test Model Definitions - Progressive Complexity


class SimpleModel(StructuredModel):
    """Simple flat model for basic testing."""

    field1: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    field2: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    optional_field: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Contact(StructuredModel):
    """Contact model for nested object testing."""

    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    name: str = ComparableField(comparator=FuzzyComparator(), threshold=0.8, weight=1.0)


class Transaction(StructuredModel):
    """Transaction model for list field testing."""

    date: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    description: str = ComparableField(
        comparator=FuzzyComparator(), threshold=0.7, weight=1.0
    )
    amount: float = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class NestedModel(StructuredModel):
    """Model with nested objects for testing."""

    simple_field: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    nested_contact: Contact = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    count: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class ListModel(StructuredModel):
    """Model with list fields for testing."""

    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    transactions: List[Transaction] = ComparableField(aggregate=False, weight=1.0)
    tags: List[str] = ComparableField(aggregate=False, weight=1.0)


class DeepNestedModel(StructuredModel):
    """Complex model with deep nesting and lists."""

    account_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    primary_contact: Contact = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    nested_data: NestedModel = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    transaction_list: List[Transaction] = ComparableField(aggregate=False, weight=1.0)
    metadata: Dict[str, str] = Field(default_factory=dict)


# Validation Helper Functions


def assert_metrics_equal(
    direct_result: Dict[str, Any], bulk_result: ProcessEvaluation, test_name: str = ""
):
    """
    Compare direct compare_with result to bulk evaluator result.

    Args:
        direct_result: Result from model.compare_with()
        bulk_result: Result from BulkStructuredModelEvaluator.compute()
        test_name: Name of test for debugging
    """
    direct_cm = direct_result["confusion_matrix"]

    # Compare overall metrics
    overall_direct = direct_cm["overall"]
    overall_bulk = bulk_result.metrics

    for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
        assert overall_direct[metric] == overall_bulk[metric], (
            f"{test_name}: Overall {metric} mismatch - direct: {overall_direct[metric]}, bulk: {overall_bulk[metric]}"
        )

    # Compare derived metrics
    if "derived" in direct_cm["overall"]:
        derived_direct = direct_cm["overall"]["derived"]
        for derived_metric in ["cm_precision", "cm_recall", "cm_f1", "cm_accuracy"]:
            if derived_metric in derived_direct and derived_metric in overall_bulk:
                assert (
                    abs(derived_direct[derived_metric] - overall_bulk[derived_metric])
                    < 0.0001
                ), (
                    f"{test_name}: Overall derived {derived_metric} mismatch - direct: {derived_direct[derived_metric]}, bulk: {overall_bulk[derived_metric]}"
                )

    # Compare field-level metrics
    fields_direct = direct_cm["fields"]
    fields_bulk = bulk_result.field_metrics

    # Check that all direct fields exist in bulk result
    assert_field_metrics_equal(fields_direct, fields_bulk, test_name, "")


def assert_field_metrics_equal(
    direct_fields: Dict[str, Any],
    bulk_fields: Dict[str, Any],
    test_name: str,
    path_prefix: str,
):
    """
    Recursively compare field-level metrics.

    Args:
        direct_fields: Field metrics from direct compare_with()
        bulk_fields: Field metrics from bulk evaluator
        test_name: Test name for debugging
        path_prefix: Current field path prefix
    """
    for field_name, field_data in direct_fields.items():
        current_path = f"{path_prefix}.{field_name}" if path_prefix else field_name

        if isinstance(field_data, dict):
            # Handle different field structures
            if "overall" in field_data:
                # Hierarchical structure with overall + fields
                overall_metrics = field_data["overall"]

                assert current_path in bulk_fields, (
                    f"{test_name}: Missing field path '{current_path}' in bulk result. Available: {list(bulk_fields.keys())}"
                )

                bulk_field_metrics = bulk_fields[current_path]

                # Compare basic metrics
                for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                    if metric in overall_metrics:
                        assert overall_metrics[metric] == bulk_field_metrics[metric], (
                            f"{test_name}: Field {current_path} {metric} mismatch - direct: {overall_metrics[metric]}, bulk: {bulk_field_metrics[metric]}"
                        )

                # Compare derived metrics if present
                if "derived" in overall_metrics:
                    derived_metrics = overall_metrics["derived"]
                    for derived_metric in [
                        "cm_precision",
                        "cm_recall",
                        "cm_f1",
                        "cm_accuracy",
                    ]:
                        if (
                            derived_metric in derived_metrics
                            and derived_metric in bulk_field_metrics
                        ):
                            assert (
                                abs(
                                    derived_metrics[derived_metric]
                                    - bulk_field_metrics[derived_metric]
                                )
                                < 0.0001
                            ), (
                                f"{test_name}: Field {current_path} derived {derived_metric} mismatch"
                            )

                # Recursively check nested fields
                if "fields" in field_data:
                    assert_field_metrics_equal(
                        field_data["fields"], bulk_fields, test_name, current_path
                    )

            elif "fields" in field_data:
                # Field with nested structure but no overall
                if current_path in bulk_fields:
                    # Compare direct metrics for this field if present
                    bulk_field_metrics = bulk_fields[current_path]
                    for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                        if metric in field_data:
                            assert field_data[metric] == bulk_field_metrics[metric], (
                                f"{test_name}: Field {current_path} {metric} mismatch"
                            )

                # Process nested fields
                assert_field_metrics_equal(
                    field_data["fields"], bulk_fields, test_name, current_path
                )

            elif "nested_fields" in field_data:
                # List field structure
                if current_path in bulk_fields:
                    bulk_field_metrics = bulk_fields[current_path]
                    for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                        if metric in field_data:
                            assert field_data[metric] == bulk_field_metrics[metric], (
                                f"{test_name}: List field {current_path} {metric} mismatch"
                            )

                # Check nested fields from list items
                nested_fields = field_data["nested_fields"]
                for nested_field_name, nested_metrics in nested_fields.items():
                    nested_path = f"{current_path}.{nested_field_name}"

                    if nested_path in bulk_fields:
                        bulk_nested_metrics = bulk_fields[nested_path]
                        for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                            if metric in nested_metrics:
                                assert (
                                    nested_metrics[metric]
                                    == bulk_nested_metrics[metric]
                                ), (
                                    f"{test_name}: Nested field {nested_path} {metric} mismatch"
                                )

            else:
                # Simple field with direct metrics
                if current_path in bulk_fields:
                    bulk_field_metrics = bulk_fields[current_path]
                    for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                        if metric in field_data:
                            assert field_data[metric] == bulk_field_metrics[metric], (
                                f"{test_name}: Simple field {current_path} {metric} mismatch"
                            )


def aggregate_compare_with_results(
    document_pairs: List[Tuple[StructuredModel, StructuredModel, str]],
) -> Dict[str, Any]:
    """
    Manually aggregate multiple compare_with results for ground truth.

    Args:
        document_pairs: List of (gt_model, pred_model, doc_id) tuples

    Returns:
        Manually aggregated confusion matrix results
    """
    # Initialize aggregators
    overall_aggregate = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "fd": 0, "fa": 0}
    field_aggregate = {}

    for gt_model, pred_model, doc_id in document_pairs:
        # Get individual comparison result
        individual_result = gt_model.compare_with(
            pred_model, include_confusion_matrix=True
        )
        cm = individual_result["confusion_matrix"]

        # Aggregate overall metrics
        for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
            overall_aggregate[metric] += cm["overall"][metric]

        # Aggregate field metrics
        aggregate_field_metrics(cm["fields"], field_aggregate, "")

    return {"overall": overall_aggregate, "fields": field_aggregate}


def aggregate_field_metrics(
    fields_dict: Dict[str, Any], aggregated_dict: Dict[str, Any], prefix: str
):
    """
    Recursively aggregate field metrics - mirrors the logic in BulkStructuredModelEvaluator.
    """
    for field_name, field_data in fields_dict.items():
        current_path = f"{prefix}.{field_name}" if prefix else field_name

        if not isinstance(field_data, dict):
            continue

        # Initialize field in aggregate if not present
        if current_path not in aggregated_dict:
            aggregated_dict[current_path] = {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "fd": 0,
                "fa": 0,
            }

        # Handle different field structures - matches BulkStructuredModelEvaluator logic
        if "overall" in field_data:
            # Hierarchical field with overall metrics
            for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                if metric in field_data["overall"]:
                    aggregated_dict[current_path][metric] += field_data["overall"][
                        metric
                    ]

            # Process nested fields
            if "fields" in field_data:
                aggregate_field_metrics(
                    field_data["fields"], aggregated_dict, current_path
                )

        elif "nested_fields" in field_data:
            # List field with nested_fields
            for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                if metric in field_data:
                    aggregated_dict[current_path][metric] += field_data[metric]

            # Process nested fields from list items
            for nested_field_name, nested_metrics in field_data[
                "nested_fields"
            ].items():
                nested_path = f"{current_path}.{nested_field_name}"
                if nested_path not in aggregated_dict:
                    aggregated_dict[nested_path] = {
                        "tp": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": 0,
                        "fd": 0,
                        "fa": 0,
                    }

                for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                    if metric in nested_metrics:
                        aggregated_dict[nested_path][metric] += nested_metrics[metric]

            # Also process fields if present
            if "fields" in field_data:
                aggregate_field_metrics(
                    field_data["fields"], aggregated_dict, current_path
                )

        elif "fields" in field_data:
            # Field with nested structure
            for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                if metric in field_data:
                    aggregated_dict[current_path][metric] += field_data[metric]

            # Process nested fields
            aggregate_field_metrics(field_data["fields"], aggregated_dict, current_path)

        else:
            # Simple field with direct metrics
            for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                if metric in field_data:
                    aggregated_dict[current_path][metric] += field_data[metric]


def assert_aggregation_equal(
    expected_aggregate: Dict[str, Any],
    actual_result: ProcessEvaluation,
    test_name: str = "",
):
    """
    Compare manually aggregated results to bulk evaluator results.
    """
    # Compare overall metrics
    expected_overall = expected_aggregate["overall"]
    actual_overall = actual_result.metrics

    for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
        assert expected_overall[metric] == actual_overall[metric], (
            f"{test_name}: Aggregated overall {metric} mismatch - expected: {expected_overall[metric]}, actual: {actual_overall[metric]}"
        )

    # Compare field metrics
    expected_fields = expected_aggregate["fields"]
    actual_fields = actual_result.field_metrics

    for field_path, expected_metrics in expected_fields.items():
        assert field_path in actual_fields, (
            f"{test_name}: Missing field path '{field_path}' in actual results"
        )

        actual_metrics = actual_fields[field_path]
        for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
            assert expected_metrics[metric] == actual_metrics[metric], (
                f"{test_name}: Aggregated field {field_path} {metric} mismatch - expected: {expected_metrics[metric]}, actual: {actual_metrics[metric]}"
            )


# Test Classes


class TestSingleDocumentParity:
    """Test that single document processing matches direct compare_with() calls."""

    def test_simple_model_perfect_match(self):
        """Test perfect match with simple flat model."""
        data = {"field1": "test", "field2": 42, "optional_field": "optional"}
        gt_model = SimpleModel(**data)
        pred_model = SimpleModel(**data)

        # Direct comparison
        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        # Bulk evaluator
        bulk_evaluator = BulkStructuredModelEvaluator(SimpleModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "simple_perfect_match")

    def test_simple_model_complete_mismatch(self):
        """Test complete mismatch with simple flat model."""
        gt_data = {"field1": "test1", "field2": 42, "optional_field": "opt1"}
        pred_data = {"field1": "test2", "field2": 43, "optional_field": "opt2"}

        gt_model = SimpleModel(**gt_data)
        pred_model = SimpleModel(**pred_data)

        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        bulk_evaluator = BulkStructuredModelEvaluator(SimpleModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "simple_complete_mismatch")

    def test_simple_model_partial_match(self):
        """Test partial match with simple flat model."""
        gt_data = {"field1": "test", "field2": 42, "optional_field": "opt1"}
        pred_data = {
            "field1": "test",
            "field2": 43,
            "optional_field": "opt1",
        }  # field2 differs

        gt_model = SimpleModel(**gt_data)
        pred_model = SimpleModel(**pred_data)

        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        bulk_evaluator = BulkStructuredModelEvaluator(SimpleModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "simple_partial_match")

    def test_nested_model_perfect_match(self):
        """Test perfect match with nested object model."""
        contact_data = {
            "phone": "555-123-4567",
            "email": "test@example.com",
            "name": "John Doe",
        }
        data = {"simple_field": "test", "nested_contact": contact_data, "count": 5}

        gt_model = NestedModel(**data)
        pred_model = NestedModel(**data)

        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        bulk_evaluator = BulkStructuredModelEvaluator(NestedModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "nested_perfect_match")

    def test_nested_model_nested_field_mismatch(self):
        """Test nested field mismatch in nested object model."""
        gt_contact = {
            "phone": "555-123-4567",
            "email": "test@example.com",
            "name": "John Doe",
        }
        pred_contact = {
            "phone": "555-123-4567",
            "email": "different@example.com",
            "name": "John Doe",
        }  # email differs

        gt_data = {"simple_field": "test", "nested_contact": gt_contact, "count": 5}
        pred_data = {"simple_field": "test", "nested_contact": pred_contact, "count": 5}

        gt_model = NestedModel(**gt_data)
        pred_model = NestedModel(**pred_data)

        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        bulk_evaluator = BulkStructuredModelEvaluator(NestedModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "nested_field_mismatch")

    def test_list_model_perfect_match(self):
        """Test perfect match with list fields."""
        transactions = [
            {"date": "2023-01-01", "description": "Purchase", "amount": 100.0},
            {"date": "2023-01-02", "description": "Refund", "amount": -50.0},
        ]

        data = {
            "name": "Test Account",
            "transactions": transactions,
            "tags": ["personal", "savings"],
        }

        gt_model = ListModel(**data)
        pred_model = ListModel(**data)

        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        bulk_evaluator = BulkStructuredModelEvaluator(ListModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "list_perfect_match")

    def test_list_model_list_item_mismatch(self):
        """Test mismatch in list item fields."""
        gt_transactions = [
            {"date": "2023-01-01", "description": "Purchase", "amount": 100.0},
            {"date": "2023-01-02", "description": "Refund", "amount": -50.0},
        ]

        pred_transactions = [
            {"date": "2023-01-01", "description": "Purchase", "amount": 100.0},
            {
                "date": "2023-01-02",
                "description": "Different Description",
                "amount": -50.0,
            },  # description differs
        ]

        gt_data = {
            "name": "Test Account",
            "transactions": gt_transactions,
            "tags": ["personal"],
        }
        pred_data = {
            "name": "Test Account",
            "transactions": pred_transactions,
            "tags": ["personal"],
        }

        gt_model = ListModel(**gt_data)
        pred_model = ListModel(**pred_data)

        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        bulk_evaluator = BulkStructuredModelEvaluator(ListModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "list_item_mismatch")

    def test_deep_nested_model_complex_match(self):
        """Test complex deeply nested model with mixed matches."""
        contact_data = {
            "phone": "555-123-4567",
            "email": "test@example.com",
            "name": "John Doe",
        }
        nested_contact_data = {
            "phone": "555-987-6543",
            "email": "nested@example.com",
            "name": "Jane Smith",
        }
        nested_data = {
            "simple_field": "nested",
            "nested_contact": nested_contact_data,
            "count": 3,
        }
        transactions = [{"date": "2023-01-01", "description": "Test", "amount": 200.0}]

        data = {
            "account_id": "ACC123",
            "primary_contact": contact_data,
            "nested_data": nested_data,
            "transaction_list": transactions,
            "metadata": {"key": "value"},
        }

        gt_model = DeepNestedModel(**data)
        pred_model = DeepNestedModel(**data)

        direct_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        bulk_evaluator = BulkStructuredModelEvaluator(DeepNestedModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        assert_metrics_equal(direct_result, bulk_result, "deep_nested_complex_match")


class TestMultiDocumentAggregation:
    """Test that bulk evaluator correctly aggregates multiple compare_with() results."""

    def test_two_identical_documents_doubling(self):
        """Test that processing same document pair twice doubles all metrics."""
        data = {"field1": "test", "field2": 42, "optional_field": "optional"}
        gt_model = SimpleModel(**data)
        pred_model = SimpleModel(**data)

        # Get single document result for reference
        single_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        # Process the same document twice with bulk evaluator
        bulk_evaluator = BulkStructuredModelEvaluator(SimpleModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_evaluator.update(gt_model, pred_model, "doc2")
        bulk_result = bulk_evaluator.compute()

        # All metrics should be exactly doubled
        single_cm = single_result["confusion_matrix"]["overall"]
        bulk_overall = bulk_result.metrics

        for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
            expected = single_cm[metric] * 2
            actual = bulk_overall[metric]
            assert expected == actual, (
                f"Doubling failed for {metric}: expected {expected}, got {actual}"
            )

        # Verify field metrics are also doubled
        single_fields = single_result["confusion_matrix"]["fields"]
        bulk_fields = bulk_result.field_metrics

        # Check a few key fields are doubled
        for field_path in bulk_fields:
            if field_path in ["field1", "field2", "optional_field"]:
                for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                    if (
                        field_path in single_fields
                        and metric in single_fields[field_path]
                    ):
                        expected = single_fields[field_path][metric] * 2
                        actual = bulk_fields[field_path][metric]
                        assert expected == actual, (
                            f"Field {field_path} {metric} doubling failed: expected {expected}, got {actual}"
                        )

    def test_mixed_results_aggregation(self):
        """Test aggregation of documents with different match patterns."""
        # Document 1: Perfect match
        perfect_data = {"field1": "test", "field2": 42, "optional_field": "opt"}
        gt1 = SimpleModel(**perfect_data)
        pred1 = SimpleModel(**perfect_data)

        # Document 2: Complete mismatch
        gt2_data = {"field1": "test1", "field2": 42, "optional_field": "opt1"}
        pred2_data = {"field1": "test2", "field2": 43, "optional_field": "opt2"}
        gt2 = SimpleModel(**gt2_data)
        pred2 = SimpleModel(**pred2_data)

        # Document 3: Partial match (only field1 matches)
        gt3_data = {"field1": "same", "field2": 100, "optional_field": "opt3"}
        pred3_data = {"field1": "same", "field2": 200, "optional_field": "opt4"}
        gt3 = SimpleModel(**gt3_data)
        pred3 = SimpleModel(**pred3_data)

        # Manual aggregation using individual compare_with calls
        document_pairs = [
            (gt1, pred1, "doc1"),
            (gt2, pred2, "doc2"),
            (gt3, pred3, "doc3"),
        ]
        expected_aggregate = aggregate_compare_with_results(document_pairs)

        # Bulk evaluator processing
        bulk_evaluator = BulkStructuredModelEvaluator(SimpleModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt1, pred1, "doc1")
        bulk_evaluator.update(gt2, pred2, "doc2")
        bulk_evaluator.update(gt3, pred3, "doc3")
        actual_result = bulk_evaluator.compute()

        # Compare aggregated results
        assert_aggregation_equal(
            expected_aggregate, actual_result, "mixed_results_aggregation"
        )

    def test_nested_model_aggregation(self):
        """Test aggregation with nested object models."""
        # Document 1: Perfect match
        contact1 = {
            "phone": "555-123-4567",
            "email": "test@example.com",
            "name": "John Doe",
        }
        data1 = {"simple_field": "test", "nested_contact": contact1, "count": 5}
        gt1 = NestedModel(**data1)
        pred1 = NestedModel(**data1)

        # Document 2: Nested field mismatch (email differs)
        contact2_gt = {
            "phone": "555-987-6543",
            "email": "test2@example.com",
            "name": "Jane Smith",
        }
        contact2_pred = {
            "phone": "555-987-6543",
            "email": "different@example.com",
            "name": "Jane Smith",
        }
        gt2_data = {"simple_field": "test2", "nested_contact": contact2_gt, "count": 10}
        pred2_data = {
            "simple_field": "test2",
            "nested_contact": contact2_pred,
            "count": 10,
        }
        gt2 = NestedModel(**gt2_data)
        pred2 = NestedModel(**pred2_data)

        # Manual aggregation
        document_pairs = [(gt1, pred1, "doc1"), (gt2, pred2, "doc2")]
        expected_aggregate = aggregate_compare_with_results(document_pairs)

        # Bulk evaluator processing
        bulk_evaluator = BulkStructuredModelEvaluator(NestedModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt1, pred1, "doc1")
        bulk_evaluator.update(gt2, pred2, "doc2")
        actual_result = bulk_evaluator.compute()

        # Compare results
        assert_aggregation_equal(
            expected_aggregate, actual_result, "nested_model_aggregation"
        )

    def test_list_model_aggregation(self):
        """Test aggregation with list field models."""
        # Document 1: Perfect match
        transactions1 = [
            {"date": "2023-01-01", "description": "Purchase", "amount": 100.0},
            {"date": "2023-01-02", "description": "Refund", "amount": -50.0},
        ]
        data1 = {
            "name": "Account1",
            "transactions": transactions1,
            "tags": ["personal"],
        }
        gt1 = ListModel(**data1)
        pred1 = ListModel(**data1)

        # Document 2: List item mismatch (description differs in second transaction)
        transactions2_gt = [
            {"date": "2023-02-01", "description": "Deposit", "amount": 200.0},
            {"date": "2023-02-02", "description": "Withdrawal", "amount": -75.0},
        ]
        transactions2_pred = [
            {"date": "2023-02-01", "description": "Deposit", "amount": 200.0},
            {
                "date": "2023-02-02",
                "description": "Different Description",
                "amount": -75.0,
            },
        ]
        gt2_data = {
            "name": "Account2",
            "transactions": transactions2_gt,
            "tags": ["business"],
        }
        pred2_data = {
            "name": "Account2",
            "transactions": transactions2_pred,
            "tags": ["business"],
        }
        gt2 = ListModel(**gt2_data)
        pred2 = ListModel(**pred2_data)

        # Manual aggregation
        document_pairs = [(gt1, pred1, "doc1"), (gt2, pred2, "doc2")]
        expected_aggregate = aggregate_compare_with_results(document_pairs)

        # Bulk evaluator processing
        bulk_evaluator = BulkStructuredModelEvaluator(ListModel)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt1, pred1, "doc1")
        bulk_evaluator.update(gt2, pred2, "doc2")
        actual_result = bulk_evaluator.compute()

        # Compare results
        assert_aggregation_equal(
            expected_aggregate, actual_result, "list_model_aggregation"
        )

    def test_batch_processing_vs_individual_updates(self):
        """Test that batch processing produces same results as individual updates."""
        # Create test data
        documents = []
        for i in range(5):
            data = {"field1": f"test{i}", "field2": i * 10, "optional_field": f"opt{i}"}
            gt_model = SimpleModel(**data)
            # Add slight variation to predictions
            pred_data = {
                "field1": f"test{i}",
                "field2": i * 10 + (1 if i % 2 == 0 else 0),
                "optional_field": f"opt{i}",
            }
            pred_model = SimpleModel(**pred_data)
            documents.append((gt_model, pred_model, f"doc{i}"))

        # Individual updates
        individual_evaluator = BulkStructuredModelEvaluator(SimpleModel)
        individual_evaluator.reset()
        for gt_model, pred_model, doc_id in documents:
            individual_evaluator.update(gt_model, pred_model, doc_id)
        individual_result = individual_evaluator.compute()

        # Batch processing
        batch_evaluator = BulkStructuredModelEvaluator(SimpleModel)
        batch_evaluator.reset()
        batch_evaluator.update_batch(documents)
        batch_result = batch_evaluator.compute()

        # Results should be identical
        assert individual_result.metrics == batch_result.metrics, (
            "Individual vs batch processing overall metrics mismatch"
        )
        assert individual_result.field_metrics == batch_result.field_metrics, (
            "Individual vs batch processing field metrics mismatch"
        )

    def test_large_dataset_aggregation(self):
        """Test aggregation with larger number of documents."""
        # Create 50 documents with varying match patterns
        documents = []
        for i in range(50):
            base_data = {
                "field1": f"test{i % 5}",
                "field2": i,
                "optional_field": f"opt{i % 3}",
            }
            gt_model = SimpleModel(**base_data)

            # Create prediction with controlled mismatch pattern
            if i % 4 == 0:  # 25% complete mismatch
                pred_data = {
                    "field1": f"different{i}",
                    "field2": i + 1000,
                    "optional_field": f"different{i}",
                }
            elif i % 4 == 1:  # 25% partial match (only field1 matches)
                pred_data = {
                    "field1": f"test{i % 5}",
                    "field2": i + 100,
                    "optional_field": f"different{i}",
                }
            else:  # 50% perfect match
                pred_data = base_data.copy()

            pred_model = SimpleModel(**pred_data)
            documents.append((gt_model, pred_model, f"doc{i}"))

        # Manual aggregation (ground truth)
        expected_aggregate = aggregate_compare_with_results(documents)

        # Bulk evaluator processing
        bulk_evaluator = BulkStructuredModelEvaluator(SimpleModel, verbose=False)
        bulk_evaluator.reset()
        for gt_model, pred_model, doc_id in documents:
            bulk_evaluator.update(gt_model, pred_model, doc_id)
        actual_result = bulk_evaluator.compute()

        # Verify aggregation is correct
        assert_aggregation_equal(
            expected_aggregate, actual_result, "large_dataset_aggregation"
        )

        # Verify processed count is correct
        assert bulk_evaluator._processed_count == 50, (
            f"Expected 50 processed documents, got {bulk_evaluator._processed_count}"
        )


if __name__ == "__main__":
    # Run specific test for debugging
    test_class = TestSingleDocumentParity()
    test_class.test_simple_model_perfect_match()
    print("Single document parity tests passed!")

    # Run multi-document aggregation tests
    test_class2 = TestMultiDocumentAggregation()
    test_class2.test_two_identical_documents_doubling()
    test_class2.test_mixed_results_aggregation()
    print("Multi-document aggregation tests passed!")
