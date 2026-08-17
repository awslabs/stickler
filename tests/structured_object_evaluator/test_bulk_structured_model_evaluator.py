#!/usr/bin/env python3

"""
Comprehensive tests for BulkStructuredModelEvaluator.

This test suite validates the stateful bulk evaluation functionality,
memory efficiency, error handling, state management, and distributed processing
capabilities of the new BulkStructuredModelEvaluator.
"""

import json
import math
from typing import List, Optional

import pandas as pd
import pytest

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.utils.process_evaluation import ProcessEvaluation


# Test Models
class Contact(StructuredModel):
    """Contact model for testing nested object evaluation."""

    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Transaction(StructuredModel):
    """Transaction model for testing list processing."""

    date: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    description: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    amount: float = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class BankStatement(StructuredModel):
    """Bank statement model with nested objects and lists."""

    accountNumber: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    contact: Contact = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    transactions: List[Transaction] = ComparableField(weight=1.0)


class TestBasicFunctionality:
    """Test basic functionality of the stateful evaluator."""

    @pytest.fixture
    def sample_data(self):
        """Sample bank statement data for testing."""
        return {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567", "email": "test@example.com"},
            "transactions": [
                {
                    "date": "2023-01-01",
                    "description": "ATM Withdrawal",
                    "amount": -100.0,
                },
                {
                    "date": "2023-01-02",
                    "description": "Direct Deposit",
                    "amount": 2000.0,
                },
            ],
        }

    def test_reset_clears_state(self, sample_data):
        """Test that reset() properly clears all accumulated state."""
        evaluator = BulkStructuredModelEvaluator(BankStatement)

        # Process some data
        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)
        evaluator.update(gt_model, pred_model, "doc1")

        # Verify state is accumulated
        assert evaluator._processed_count == 1
        assert len(evaluator._confusion_matrix["overall"]) > 0

        # Reset and verify state is cleared
        evaluator.reset()
        assert evaluator._processed_count == 0
        assert sum(evaluator._confusion_matrix["overall"].values()) == 0
        assert len(evaluator._errors) == 0

    def test_update_single_document(self, sample_data):
        """Test update() method with single document pair."""
        evaluator = BulkStructuredModelEvaluator(BankStatement)
        evaluator.reset()

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)  # Perfect match

        evaluator.update(gt_model, pred_model, "doc1")

        # Verify processing completed
        assert evaluator._processed_count == 1

        # Verify overall metrics accumulated
        overall_metrics = evaluator._confusion_matrix["overall"]
        assert overall_metrics["tp"] > 0  # Should have true positives for perfect match

    def test_update_batch_multiple_documents(self, sample_data):
        """Test update_batch() method with multiple documents."""
        evaluator = BulkStructuredModelEvaluator(BankStatement)
        evaluator.reset()

        # Create batch data
        gt_model1 = BankStatement(**sample_data)
        pred_model1 = BankStatement(**sample_data)

        # Different data for second document
        different_data = sample_data.copy()
        different_data["accountNumber"] = "DIFFERENT"
        gt_model2 = BankStatement(**sample_data)
        pred_model2 = BankStatement(**different_data)

        batch_data = [
            (gt_model1, pred_model1, "doc1"),
            (gt_model2, pred_model2, "doc2"),
        ]

        evaluator.update_batch(batch_data)

        # Verify both documents processed
        assert evaluator._processed_count == 2

    def test_compute_returns_correct_metrics(self, sample_data):
        """Test that compute() returns ProcessEvaluation with correct structure."""
        evaluator = BulkStructuredModelEvaluator(BankStatement)
        evaluator.reset()

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)
        evaluator.update(gt_model, pred_model, "doc1")

        result = evaluator.compute()

        # Verify return type and structure
        assert isinstance(result, ProcessEvaluation)
        assert "tp" in result.metrics
        assert "cm_accuracy" in result.metrics
        assert "accountNumber" in result.field_metrics
        assert "contact" in result.field_metrics


class TestStatefulBehavior:
    """Test stateful accumulation behavior."""

    @pytest.fixture
    def evaluator(self):
        return BulkStructuredModelEvaluator(BankStatement)

    @pytest.fixture
    def perfect_match_data(self):
        data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567", "email": "test@example.com"},
            "transactions": [
                {"date": "2023-01-01", "description": "Test", "amount": 100.0}
            ],
        }
        return BankStatement(**data), BankStatement(**data)

    def test_accumulation_across_updates(self, evaluator, perfect_match_data):
        """Test that metrics accumulate correctly across multiple update() calls."""
        evaluator.reset()
        gt_model, pred_model = perfect_match_data

        # Process first document
        evaluator.update(gt_model, pred_model, "doc1")
        first_tp = evaluator._confusion_matrix["overall"]["tp"]

        # Process second document
        evaluator.update(gt_model, pred_model, "doc2")
        second_tp = evaluator._confusion_matrix["overall"]["tp"]

        # Verify accumulation
        assert second_tp == first_tp * 2  # Should double with second identical document
        assert evaluator._processed_count == 2

    def test_current_metrics_vs_final_compute(self, evaluator, perfect_match_data):
        """Test that get_current_metrics() and compute() return equivalent results."""
        evaluator.reset()
        gt_model, pred_model = perfect_match_data

        evaluator.update(gt_model, pred_model, "doc1")

        current_metrics = evaluator.get_current_metrics()
        final_metrics = evaluator.compute()

        # Should return identical metrics
        assert current_metrics.metrics == final_metrics.metrics
        assert current_metrics.field_metrics == final_metrics.field_metrics

    def test_multiple_reset_cycles(self, evaluator, perfect_match_data):
        """Test that evaluator can be reset and reused multiple times."""
        gt_model, pred_model = perfect_match_data

        for cycle in range(3):
            evaluator.reset()
            evaluator.update(gt_model, pred_model, f"doc_{cycle}")

            result = evaluator.compute()
            assert evaluator._processed_count == 1
            assert result.metrics["cm_accuracy"] == 1.0  # Perfect match each time


class TestMemoryEfficiency:
    """Test memory efficiency and scalability characteristics."""

    def test_large_dataset_processing(self):
        """Test processing larger number of documents without memory issues."""
        evaluator = BulkStructuredModelEvaluator(BankStatement, verbose=False)
        evaluator.reset()

        # Create simple test data
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [
                {"date": "2023-01-01", "description": "Test", "amount": 100.0}
            ],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        # Process many documents
        num_docs = 5000
        for i in range(num_docs):
            evaluator.update(gt_model, pred_model, f"doc_{i}")

        result = evaluator.compute()
        assert evaluator._processed_count == num_docs
        assert result.metrics["cm_accuracy"] == 1.0  # All perfect matches

    def test_memory_usage_stays_bounded(self):
        """Test that memory usage doesn't grow linearly with document count."""
        evaluator = BulkStructuredModelEvaluator(BankStatement, verbose=False)

        # Simple data to minimize memory footprint variations
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],  # Empty to minimize complexity
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        evaluator.reset()

        # Process documents and verify state remains bounded
        for i in range(1000):
            evaluator.update(gt_model, pred_model, f"doc_{i}")

        # Memory usage should be bounded by the confusion matrix size, not document count
        # The confusion matrix should have fixed number of fields regardless of document count
        field_count = len(evaluator._confusion_matrix["fields"])
        assert (
            field_count < 20
        )  # Should be small number of fields, not growing with doc count


class TestErrorHandling:
    """Test error handling and recovery behavior."""

    @pytest.fixture
    def evaluator_no_elide(self):
        return BulkStructuredModelEvaluator(BankStatement, elide_errors=False)

    @pytest.fixture
    def evaluator_elide(self):
        return BulkStructuredModelEvaluator(BankStatement, elide_errors=True)

    def test_error_accumulation_mode(self, evaluator_no_elide):
        """Test that errors are accumulated when elide_errors=False."""
        evaluator = evaluator_no_elide
        evaluator.reset()

        # Create valid data
        valid_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }
        valid_gt = BankStatement(**valid_data)
        valid_pred = BankStatement(**valid_data)

        # Process valid document
        evaluator.update(valid_gt, valid_pred, "valid_doc")

        # Process invalid data by passing None (should cause error)
        try:
            evaluator.update(None, valid_pred, "invalid_doc")
        except Exception:
            pass  # Expected to fail

        # Should have error recorded but continue processing
        assert len(evaluator._errors) > 0
        assert evaluator._errors[0]["doc_id"] == "invalid_doc"

    def test_error_elision_mode(self, evaluator_elide):
        """Test that errors are skipped when elide_errors=True."""
        evaluator = evaluator_elide
        evaluator.reset()

        # Similar test but should skip errors silently
        valid_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }
        valid_gt = BankStatement(**valid_data)
        valid_pred = BankStatement(**valid_data)

        evaluator.update(valid_gt, valid_pred, "valid_doc")

        try:
            evaluator.update(None, valid_pred, "invalid_doc")
        except Exception:
            pass

        # Should not have errors recorded when eliding
        assert len(evaluator._errors) == 0

    def test_partial_failure_recovery(self, evaluator_no_elide):
        """Test recovery after partial failures in batch processing."""
        evaluator = evaluator_no_elide
        evaluator.reset()

        valid_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        # Create mixed batch with valid and invalid data
        valid_gt = BankStatement(**valid_data)
        valid_pred = BankStatement(**valid_data)

        batch_data = [
            (valid_gt, valid_pred, "doc1"),
            (valid_gt, valid_pred, "doc2"),  # This should work
        ]

        evaluator.update_batch(batch_data)

        # Should have processed valid documents despite any errors
        assert evaluator._processed_count >= 2


class TestAdvancedFeatures:
    """Test advanced features like state management and distributed processing."""

    @pytest.fixture
    def sample_evaluator_with_data(self):
        evaluator = BulkStructuredModelEvaluator(BankStatement)
        evaluator.reset()

        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [
                {"date": "2023-01-01", "description": "Test", "amount": 100.0}
            ],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)
        evaluator.update(gt_model, pred_model, "doc1")

        return evaluator

    def test_state_serialization_deserialization(self, sample_evaluator_with_data):
        """Test that state can be serialized and deserialized correctly."""
        evaluator1 = sample_evaluator_with_data

        # Get state from first evaluator
        state = evaluator1.get_state()

        # Create new evaluator and load state
        evaluator2 = BulkStructuredModelEvaluator(BankStatement)
        evaluator2.load_state(state)

        # Both should produce identical results
        result1 = evaluator1.compute()
        result2 = evaluator2.compute()

        assert result1.metrics == result2.metrics
        assert evaluator1._processed_count == evaluator2._processed_count

    def test_state_merging_distributed(self):
        """Test merging states from multiple evaluator instances."""
        # Create two evaluators processing different data
        evaluator1 = BulkStructuredModelEvaluator(BankStatement)
        evaluator2 = BulkStructuredModelEvaluator(BankStatement)

        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        # Process different documents in each evaluator
        evaluator1.reset()
        evaluator1.update(gt_model, pred_model, "doc1")

        evaluator2.reset()
        evaluator2.update(gt_model, pred_model, "doc2")

        # Merge second evaluator's state into first
        state2 = evaluator2.get_state()
        evaluator1.merge_state(state2)

        # Should have processed both documents
        assert evaluator1._processed_count == 2

    def test_checkpointing_resume(self):
        """Test checkpointing and resuming evaluation."""
        evaluator = BulkStructuredModelEvaluator(BankStatement)
        evaluator.reset()

        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        # Process some documents
        evaluator.update(gt_model, pred_model, "doc1")
        evaluator.update(gt_model, pred_model, "doc2")

        # Save checkpoint
        checkpoint = evaluator.get_state()

        # Continue processing
        evaluator.update(gt_model, pred_model, "doc3")

        # Create new evaluator and resume from checkpoint
        resumed_evaluator = BulkStructuredModelEvaluator(BankStatement)
        resumed_evaluator.load_state(checkpoint)
        resumed_evaluator.update(gt_model, pred_model, "doc3")

        # Both should have same final result
        result1 = evaluator.compute()
        result2 = resumed_evaluator.compute()

        assert result1.metrics == result2.metrics


class TestCompatibility:
    """Test compatibility with existing systems and data formats."""

    def test_matches_single_evaluator_results(self):
        """Test that bulk evaluator produces same results as compare_with for single document."""
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [
                {"date": "2023-01-01", "description": "Test", "amount": 100.0}
            ],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        # Single evaluator result
        single_result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        # Bulk evaluator result
        bulk_evaluator = BulkStructuredModelEvaluator(BankStatement)
        bulk_evaluator.reset()
        bulk_evaluator.update(gt_model, pred_model, "doc1")
        bulk_result = bulk_evaluator.compute()

        # Results should match for single document
        assert (
            bulk_result.metrics["cm_accuracy"]
            == single_result["confusion_matrix"]["overall"]["derived"]["cm_accuracy"]
        )

    def test_nested_field_aggregation(self):
        """Test that nested fields are properly aggregated with correct paths."""
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567", "email": "test@example.com"},
            "transactions": [
                {"date": "2023-01-01", "description": "Test1", "amount": 100.0},
                {"date": "2023-01-02", "description": "Test2", "amount": 200.0},
            ],
        }

        evaluator = BulkStructuredModelEvaluator(BankStatement)
        evaluator.reset()

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)
        evaluator.update(gt_model, pred_model, "doc1")

        result = evaluator.compute()

        # Check that nested fields are accessible with proper paths
        field_paths = set(result.field_metrics.keys())

        # Should have nested contact fields
        expected_contact_fields = {"contact.phone", "contact.email"}
        assert expected_contact_fields.issubset(field_paths), (
            f"Missing contact fields in {field_paths}"
        )

        # Should have nested transaction fields
        expected_transaction_fields = {
            "transactions.date",
            "transactions.description",
            "transactions.amount",
        }
        assert expected_transaction_fields.issubset(field_paths), (
            f"Missing transaction fields in {field_paths}"
        )

    def test_legacy_dataframe_wrapper(self):
        """Test legacy DataFrame compatibility wrapper."""
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        # Create DataFrame in expected format
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "expected": json.dumps(sample_data),
                    "predicted": json.dumps(sample_data),
                },
                {
                    "doc_id": "doc2",
                    "expected": json.dumps(sample_data),
                    "predicted": json.dumps(sample_data),
                },
            ]
        )

        evaluator = BulkStructuredModelEvaluator(BankStatement)
        result = evaluator.evaluate_dataframe(df)

        assert isinstance(result, ProcessEvaluation)
        assert result.metrics["cm_accuracy"] == 1.0  # Perfect matches
        assert evaluator._processed_count == 2


class TestPerformance:
    """Test performance characteristics and scalability."""

    def test_streaming_vs_batch_equivalence(self):
        """Test that streaming and batch processing produce equivalent results."""
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        # Streaming processing
        stream_evaluator = BulkStructuredModelEvaluator(BankStatement)
        stream_evaluator.reset()
        for i in range(10):
            stream_evaluator.update(gt_model, pred_model, f"doc_{i}")
        stream_result = stream_evaluator.compute()

        # Batch processing
        batch_evaluator = BulkStructuredModelEvaluator(BankStatement)
        batch_evaluator.reset()
        batch_data = [(gt_model, pred_model, f"doc_{i}") for i in range(10)]
        batch_evaluator.update_batch(batch_data)
        batch_result = batch_evaluator.compute()

        # Results should be identical
        assert stream_result.metrics == batch_result.metrics

    def test_incremental_vs_bulk_processing(self):
        """Test that incremental processing produces same results as bulk processing."""
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        # Create test data with some variations
        data_variations = []
        for i in range(5):
            data = sample_data.copy()
            data["accountNumber"] = f"123456789{i}"
            data_variations.append(data)

        # Incremental processing
        incremental_evaluator = BulkStructuredModelEvaluator(BankStatement)
        incremental_evaluator.reset()

        for i, data in enumerate(data_variations):
            gt_model = BankStatement(**sample_data)  # Always same ground truth
            pred_model = BankStatement(**data)  # Varying predictions
            incremental_evaluator.update(gt_model, pred_model, f"doc_{i}")

        incremental_result = incremental_evaluator.compute()

        # "Bulk" processing using batch method
        bulk_evaluator = BulkStructuredModelEvaluator(BankStatement)
        bulk_evaluator.reset()

        batch_data = []
        for i, data in enumerate(data_variations):
            gt_model = BankStatement(**sample_data)
            pred_model = BankStatement(**data)
            batch_data.append((gt_model, pred_model, f"doc_{i}"))

        bulk_evaluator.update_batch(batch_data)
        bulk_result = bulk_evaluator.compute()

        # Results should be identical
        assert incremental_result.metrics == bulk_result.metrics

    def test_scalability_characteristics(self):
        """Test that evaluator maintains performance characteristics at scale."""
        import time

        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        evaluator = BulkStructuredModelEvaluator(BankStatement, verbose=False)

        # Test small scale
        evaluator.reset()
        start_time = time.time()
        for i in range(100):
            evaluator.update(gt_model, pred_model, f"doc_{i}")
        small_scale_time = time.time() - start_time

        # Test larger scale
        evaluator.reset()
        start_time = time.time()
        for i in range(1000):
            evaluator.update(gt_model, pred_model, f"doc_{i}")
        large_scale_time = time.time() - start_time

        # Should scale approximately linearly (within reasonable bounds)
        # Allow for some overhead but shouldn't be more than 15x slower for 10x data
        time_ratio = large_scale_time / small_scale_time
        assert time_ratio < 15, f"Scaling poorly: {time_ratio}x time for 10x documents"


class TestUpdateFromComparisonResult:
    """Test update_from_comparison_result() and aggregate_from_comparisons()."""

    @pytest.fixture
    def sample_comparison_results(self):
        """Generate pre-computed comparison results using compare_with()."""
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567", "email": "test@example.com"},
            "transactions": [
                {"date": "2023-01-01", "description": "Test", "amount": 100.0}
            ],
        }
        different_data = {
            "accountNumber": "DIFFERENT",
            "contact": {"phone": "555-999-9999", "email": "test@example.com"},
            "transactions": [
                {"date": "2023-01-01", "description": "Test", "amount": 100.0}
            ],
        }

        gt_model = BankStatement(**sample_data)
        perfect_pred = BankStatement(**sample_data)
        partial_pred = BankStatement(**different_data)

        perfect_result = gt_model.compare_with(
            perfect_pred, include_confusion_matrix=True
        )
        partial_result = gt_model.compare_with(
            partial_pred, include_confusion_matrix=True
        )

        return [perfect_result, partial_result]

    def test_aggregate_from_comparisons(self, sample_comparison_results):
        """Test the standalone aggregate_from_comparisons function."""
        from stickler import aggregate_from_comparisons

        result = aggregate_from_comparisons(sample_comparison_results)

        assert isinstance(result, ProcessEvaluation)
        assert result.document_count == 2
        assert "tp" in result.metrics
        assert "cm_precision" in result.metrics
        assert "cm_recall" in result.metrics
        assert "cm_f1" in result.metrics
        assert len(result.field_metrics) > 0

    def test_aggregate_from_comparisons_empty_list(self):
        """Test aggregate_from_comparisons with empty input."""
        from stickler import aggregate_from_comparisons

        result = aggregate_from_comparisons([])

        assert isinstance(result, ProcessEvaluation)
        assert result.document_count == 0

    def test_aggregate_from_comparisons_matches_bulk_evaluator(
        self, sample_comparison_results
    ):
        """Verify standalone function matches using BulkStructuredModelEvaluator directly."""
        from stickler import aggregate_from_comparisons

        # Standalone function
        standalone_result = aggregate_from_comparisons(sample_comparison_results)

        # Manual evaluator
        evaluator = BulkStructuredModelEvaluator()
        for r in sample_comparison_results:
            evaluator.update_from_comparison_result(r)
        evaluator_result = evaluator.compute()

        assert standalone_result.metrics == evaluator_result.metrics
        assert standalone_result.field_metrics == evaluator_result.field_metrics

    def test_update_from_comparison_result_missing_confusion_matrix(self):
        """Caller-misuse precondition re-raises rather than being silently
        folded into the per-doc error counter; otherwise a malformed
        comparison_result would just bump fn and look like a normal miss."""
        evaluator = BulkStructuredModelEvaluator(elide_errors=False)
        with pytest.raises(ValueError, match="confusion_matrix"):
            evaluator.update_from_comparison_result(
                {"overall_score": 0.5}, "bad_doc"
            )
        assert evaluator._errors == []
        assert evaluator._confusion_matrix["overall"]["fn"] == 0

    def test_update_from_comparison_result_accumulates(self):
        """Test that multiple calls accumulate correctly."""
        sample_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        gt_model = BankStatement(**sample_data)
        pred_model = BankStatement(**sample_data)

        comparison_result = gt_model.compare_with(
            pred_model, include_confusion_matrix=True
        )

        evaluator = BulkStructuredModelEvaluator()
        evaluator.update_from_comparison_result(comparison_result, "doc1")
        first_tp = evaluator._confusion_matrix["overall"]["tp"]

        evaluator.update_from_comparison_result(comparison_result, "doc2")
        second_tp = evaluator._confusion_matrix["overall"]["tp"]

        assert second_tp == first_tp * 2
        assert evaluator._processed_count == 2


# Fixture model lives at module scope so Pydantic can resolve forward refs
# inside the test class below.


class _WeightedInvoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=LevenshteinComparator(), weight=10.0)
    note: str = ComparableField(comparator=LevenshteinComparator(), weight=0.1)
    total: float = ComparableField(comparator=NumericComparator(), weight=5.0)


class TestWeightedOverallScore:
    """Weight-aware aggregate score in bulk evaluation."""

    def _make_invoice_pair(self, gt_kwargs, pred_kwargs):
        return (
            _WeightedInvoice(**gt_kwargs),
            _WeightedInvoice(**pred_kwargs),
        )

    def test_single_doc_weighted_overall_matches_compare_with(self):
        gt, pred = self._make_invoice_pair(
            {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
            {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
        )

        expected = gt.compare_with(pred, include_confusion_matrix=True)["overall_score"]

        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        evaluator.update(gt, pred)
        result = evaluator.compute()

        assert result.metrics["weighted_overall_score"] == pytest.approx(expected)

    def test_multi_doc_weighted_overall_is_arithmetic_mean(self):
        pairs = [
            self._make_invoice_pair(
                {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
                {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
            ),
            self._make_invoice_pair(
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
            ),
            self._make_invoice_pair(
                {"invoice_id": "INV-3", "note": "hi", "total": 30.0},
                {"invoice_id": "INV-3", "note": "bye", "total": 30.0},
            ),
        ]

        per_doc = [
            gt.compare_with(pred, include_confusion_matrix=True)["overall_score"]
            for gt, pred in pairs
        ]
        expected = sum(per_doc) / len(per_doc)

        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        for gt, pred in pairs:
            evaluator.update(gt, pred)
        result = evaluator.compute()

        assert result.metrics["weighted_overall_score"] == pytest.approx(expected)

    def test_non_uniform_weights_diverge_from_cm_f1(self):
        """Weighted score reflects per-field weights; cm_f1 treats fields equally.

        With invoice_id (w=10) near-matching, note (w=0.1) matching, and total
        (w=5) matching, cm_f1 sees "3 true positives" uniformly while the
        weighted score is dominated by the heavy invoice_id similarity.
        """
        gt = _WeightedInvoice(invoice_id="INV-1", note="short", total=100.0)
        pred = _WeightedInvoice(invoice_id="INV-9", note="short", total=100.0)

        per_doc = gt.compare_with(pred, include_confusion_matrix=True)

        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        evaluator.update(gt, pred)
        result = evaluator.compute()

        assert result.metrics["weighted_overall_score"] == pytest.approx(
            per_doc["overall_score"]
        )
        # Pin the exact value so a math regression (e.g., divisor swap)
        # can't pass by just happening to stay different from cm_f1.
        assert result.metrics["weighted_overall_score"] == pytest.approx(
            0.8675, abs=1e-3
        )
        assert result.metrics["cm_f1"] == pytest.approx(1.0)

    def test_state_roundtrip_preserves_weighted_score(self):
        pairs = [
            self._make_invoice_pair(
                {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
                {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
            ),
            self._make_invoice_pair(
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
            ),
            self._make_invoice_pair(
                {"invoice_id": "INV-3", "note": "hi", "total": 30.0},
                {"invoice_id": "INV-3", "note": "bye", "total": 30.0},
            ),
        ]

        first = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        for gt, pred in pairs[:2]:
            first.update(gt, pred)
        state = first.get_state()

        resumed = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        resumed.load_state(state)
        resumed.update(*pairs[2])
        resumed_result = resumed.compute()

        direct = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        for gt, pred in pairs:
            direct.update(gt, pred)
        direct_result = direct.compute()

        assert resumed_result.metrics["weighted_overall_score"] == pytest.approx(
            direct_result.metrics["weighted_overall_score"]
        )

    def test_load_state_tolerates_old_state_without_score_keys(self):
        """Old state dicts without score keys must load, score defaults to 0.0."""
        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        # Simulate an old state dict with no score_* keys.
        old_state = {
            "confusion_matrix": {"overall": {}, "fields": {}},
            "errors": [],
            "processed_count": 0,
            "start_time": 0.0,
            "target_schema": "_WeightedInvoice",
            "elide_errors": False,
        }

        evaluator.load_state(old_state)
        result = evaluator.compute()

        assert result.metrics["weighted_overall_score"] == 0.0
        assert evaluator._overall_score_count == 0

    def test_per_field_mean_score_at_nested_paths(self):
        """Nested paths like contact.phone should emit their own mean_score."""
        matching = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567", "email": "test@example.com"},
            "transactions": [{"date": "2023-01-01", "description": "x", "amount": 1.0}],
        }
        wrong_phone = dict(matching)
        wrong_phone["contact"] = {
            "phone": "555-999-9999",
            "email": "test@example.com",
        }

        evaluator = BulkStructuredModelEvaluator(target_schema=BankStatement)
        evaluator.update(BankStatement(**matching), BankStatement(**matching))
        evaluator.update(BankStatement(**matching), BankStatement(**wrong_phone))
        result = evaluator.compute()

        # contact.phone: 1.0 (match) + 0.0 (mismatch) = mean 0.5.
        assert "contact.phone" in result.field_metrics
        assert result.field_metrics["contact.phone"]["mean_score"] == pytest.approx(0.5)

        # accountNumber matched twice: mean 1.0.
        assert result.field_metrics["accountNumber"]["mean_score"] == pytest.approx(1.0)

    def test_update_from_comparison_result_tolerates_missing_score_keys(self):
        """Minimal dict without overall_score / fields must not raise."""
        evaluator = BulkStructuredModelEvaluator()
        evaluator.update_from_comparison_result(
            {"confusion_matrix": {"overall": {"tp": 1}, "fields": {}}}, "doc1"
        )

        assert evaluator._processed_count == 1
        assert evaluator._overall_score_count == 0
        result = evaluator.compute()
        assert result.metrics["weighted_overall_score"] == 0.0

    def test_zero_docs_weighted_overall_is_zero(self):
        """Empty evaluator returns 0.0 (not NaN)."""
        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        result = evaluator.compute()

        score = result.metrics["weighted_overall_score"]
        assert score == 0.0
        assert not math.isnan(score)

    def test_merge_state_sums_weighted_score_accumulators(self):
        a_pair = self._make_invoice_pair(
            {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
            {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
        )
        b_pair = self._make_invoice_pair(
            {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
            {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
        )

        a = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        a.update(*a_pair)
        b = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        b.update(*b_pair)

        a.merge_state(b.get_state())
        merged_result = a.compute()

        direct = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        direct.update(*a_pair)
        direct.update(*b_pair)
        direct_result = direct.compute()

        assert merged_result.metrics["weighted_overall_score"] == pytest.approx(
            direct_result.metrics["weighted_overall_score"]
        )

    def test_merge_state_tolerates_old_peer_without_score_keys(self):
        """Merging an old state dict (no score keys) must not raise."""
        pair = self._make_invoice_pair(
            {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
            {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
        )

        a = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        a.update(*pair)

        old_peer_state = {
            "confusion_matrix": {"overall": {}, "fields": {}},
            "errors": [],
            "processed_count": 0,
            "start_time": 0.0,
            "target_schema": "_WeightedInvoice",
            "elide_errors": False,
        }

        a.merge_state(old_peer_state)
        result = a.compute()

        assert result.metrics["weighted_overall_score"] > 0.0
        assert a._overall_score_count == 1

    def test_error_doc_excluded_from_weighted_score_mean(self):
        """Docs that raise during compare_with are excluded from denominator.

        The CM still absorbs the error (fn bump) so error-rate shows up in
        cm_* metrics, but weighted_overall_score only counts successful docs.
        """
        good_pair = self._make_invoice_pair(
            {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
            {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
        )

        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        evaluator.update(*good_pair, doc_id="good")

        # Induce a comparison error via a malformed comparison_result dict.
        # confusion_matrix=None passes the precondition but blows up inside
        # _accumulate_confusion_matrix (None has no `in` membership).
        evaluator.update_from_comparison_result(
            {"confusion_matrix": None}, doc_id="bad"
        )

        good_score = good_pair[0].compare_with(
            good_pair[1], include_confusion_matrix=True
        )["overall_score"]

        result = evaluator.compute()
        assert evaluator._overall_score_count == 1
        assert result.metrics["weighted_overall_score"] == pytest.approx(good_score)
        # CM absorbs the error via fn bump; weighted score does not.
        assert result.metrics["fn"] >= 1
        assert len(result.errors) == 1

    def test_list_of_structured_model_mean_score_at_list_path(self):
        """List[StructuredModel] fields emit mean_score at the list node."""
        from typing import List as _List

        class _Line(StructuredModel):
            name: str = ComparableField(comparator=LevenshteinComparator(), weight=2.0)
            qty: int = ComparableField(comparator=NumericComparator(), weight=1.0)

        class _Order(StructuredModel):
            order_id: str = ComparableField(
                comparator=LevenshteinComparator(), weight=5.0
            )
            items: _List[_Line] = ComparableField(weight=3.0)

        gt = _Order(
            order_id="O-1",
            items=[_Line(name="apple", qty=1), _Line(name="banana", qty=2)],
        )
        pred = _Order(
            order_id="O-1",
            items=[_Line(name="apple", qty=1), _Line(name="banana", qty=3)],
        )

        per_doc = gt.compare_with(pred, include_confusion_matrix=True)
        expected_items_score = per_doc["confusion_matrix"]["fields"]["items"][
            "threshold_applied_score"
        ]

        evaluator = BulkStructuredModelEvaluator(target_schema=_Order)
        evaluator.update(gt, pred)
        result = evaluator.compute()

        assert "items" in result.field_metrics
        assert result.field_metrics["items"]["mean_score"] == pytest.approx(
            expected_items_score
        )
        assert result.field_metrics["order_id"]["mean_score"] == pytest.approx(1.0)

    def test_build_process_evaluation_does_not_mutate_score_accumulators(self):
        """Reading mean_score for cm-only paths must not create defaultdict entries."""
        evaluator = BulkStructuredModelEvaluator()
        # Seed a cm-only path via update_from_comparison_result (no score data).
        evaluator.update_from_comparison_result(
            {"confusion_matrix": {"overall": {"tp": 1}, "fields": {"foo": {"tp": 1}}}},
            doc_id="doc1",
        )

        evaluator.compute()

        # Sums/counts must still be empty — compute() should not have materialized
        # a zero-sum entry just by reading it.
        assert dict(evaluator._field_score_sums) == {}
        assert dict(evaluator._field_score_counts) == {}

    def test_cm_only_path_omits_mean_score_key(self):
        """Paths with CM counts but no score data must not emit mean_score=0.0."""
        evaluator = BulkStructuredModelEvaluator()
        evaluator.update_from_comparison_result(
            {"confusion_matrix": {"overall": {"tp": 1}, "fields": {"foo": {"tp": 1}}}},
            doc_id="doc1",
        )

        result = evaluator.compute()

        assert "foo" in result.field_metrics
        # The distinguishing contract: no mean_score key means "not scored",
        # which is distinct from "scored and got 0.0".
        assert "mean_score" not in result.field_metrics["foo"]

    def test_score_only_path_surfaces_in_field_metrics(self):
        """Paths with score data but no CM counts must still appear in output."""
        evaluator = BulkStructuredModelEvaluator()
        evaluator.update_from_comparison_result(
            {
                "confusion_matrix": {
                    "overall": {"tp": 1},
                    "fields": {"bar": {"threshold_applied_score": 0.8}},
                }
            },
            doc_id="doc1",
        )

        result = evaluator.compute()

        assert "bar" in result.field_metrics
        assert result.field_metrics["bar"]["mean_score"] == pytest.approx(0.8)
        # CM counts default to absent — derived metrics still compute cleanly.
        assert result.field_metrics["bar"].get("tp", 0) == 0

    def test_list_of_model_leaf_omits_mean_score_key(self):
        """Leaves inside List[StructuredModel] get CM counts but no mean_score.

        compare_with() only emits threshold_applied_score at the list parent,
        so nested leaves never accumulate a score. The key is omitted rather
        than reported as a misleading 0.0.
        """
        from typing import List as _List

        class _Line(StructuredModel):
            name: str = ComparableField(comparator=LevenshteinComparator(), weight=2.0)
            qty: int = ComparableField(comparator=NumericComparator(), weight=1.0)

        class _Order(StructuredModel):
            order_id: str = ComparableField(
                comparator=LevenshteinComparator(), weight=5.0
            )
            items: _List[_Line] = ComparableField(weight=3.0)

        gt = _Order(order_id="O-1", items=[_Line(name="apple", qty=1)])
        pred = _Order(order_id="O-1", items=[_Line(name="apple", qty=1)])

        evaluator = BulkStructuredModelEvaluator(target_schema=_Order)
        evaluator.update(gt, pred)
        evaluator.update(gt, pred)
        result = evaluator.compute()

        # Nested leaves get CM counts bubbled up...
        assert "items.name" in result.field_metrics
        assert result.field_metrics["items.name"].get("tp", 0) > 0
        # ...but no mean_score because compare_with emits it only at the parent.
        assert "mean_score" not in result.field_metrics["items.name"]
        assert "mean_score" not in result.field_metrics["items.qty"]
        # List parent and sibling leaves do surface mean_score.
        assert "mean_score" in result.field_metrics["items"]
        assert "mean_score" in result.field_metrics["order_id"]

    def test_invalid_overall_score_skipped(self):
        """Non-numeric, non-finite, and bool overall_score are silently dropped.

        bool is a subclass of int in Python, so ``_is_valid_score`` rejects
        it explicitly to avoid silently counting ``True`` as ``1.0``.
        """
        for bad in (float("nan"), float("inf"), float("-inf"), None, True, "0.5"):
            evaluator = BulkStructuredModelEvaluator()
            evaluator.update_from_comparison_result(
                {
                    "confusion_matrix": {"overall": {"tp": 1}, "fields": {}},
                    "overall_score": bad,
                },
                doc_id="doc1",
            )
            evaluator.update_from_comparison_result(
                {
                    "confusion_matrix": {"overall": {"tp": 1}, "fields": {}},
                    "overall_score": 1.0,
                },
                doc_id="doc2",
            )

            result = evaluator.compute()
            # Only the valid doc counted in the denominator.
            assert evaluator._overall_score_count == 1, (
                f"expected non-finite {bad!r} to be skipped"
            )
            assert result.metrics["weighted_overall_score"] == pytest.approx(1.0)
            # Both docs counted toward _processed_count / CM.
            assert result.document_count == 2

    def test_reset_clears_weighted_score_accumulators(self):
        """reset() must zero weighted score and per-field score accumulators."""
        gt, pred = self._make_invoice_pair(
            {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
            {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
        )
        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        evaluator.update(gt, pred)
        assert evaluator._overall_score_count == 1
        assert len(evaluator._field_score_sums) > 0

        evaluator.reset()

        assert evaluator._overall_score_sum == 0.0
        assert evaluator._overall_score_count == 0
        assert dict(evaluator._field_score_sums) == {}
        assert dict(evaluator._field_score_counts) == {}

        result = evaluator.compute()
        assert result.metrics["weighted_overall_score"] == 0.0
        assert result.field_metrics == {}

    def test_get_current_metrics_reflects_partial_weighted_score(self):
        """Mid-stream polling via get_current_metrics() sees partial aggregate."""
        pairs = [
            self._make_invoice_pair(
                {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
                {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
            ),
            self._make_invoice_pair(
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
            ),
        ]

        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        evaluator.update(*pairs[0])

        mid = evaluator.get_current_metrics()
        first_doc_score = pairs[0][0].compare_with(
            pairs[0][1], include_confusion_matrix=True
        )["overall_score"]
        assert mid.metrics["weighted_overall_score"] == pytest.approx(first_doc_score)
        # State must not have been cleared by the polling call.
        assert evaluator._overall_score_count == 1

        evaluator.update(*pairs[1])
        final = evaluator.compute()
        expected_mean = (
            first_doc_score
            + pairs[1][0].compare_with(pairs[1][1], include_confusion_matrix=True)[
                "overall_score"
            ]
        ) / 2
        assert final.metrics["weighted_overall_score"] == pytest.approx(expected_mean)

    def test_update_batch_accumulates_weighted_score(self):
        """update_batch() must feed the weighted score path like update() does."""
        pairs = [
            self._make_invoice_pair(
                {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
                {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
            ),
            self._make_invoice_pair(
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
                {"invoice_id": "INV-2", "note": "hi", "total": 20.0},
            ),
        ]
        batch = [(gt, pred, f"doc_{i}") for i, (gt, pred) in enumerate(pairs)]

        evaluator = BulkStructuredModelEvaluator(target_schema=_WeightedInvoice)
        evaluator.update_batch(batch)
        result = evaluator.compute()

        expected = sum(
            gt.compare_with(pred, include_confusion_matrix=True)["overall_score"]
            for gt, pred in pairs
        ) / len(pairs)
        assert result.metrics["weighted_overall_score"] == pytest.approx(expected)
        assert evaluator._overall_score_count == len(pairs)

    def test_aggregate_from_comparisons_exposes_weighted_score(self):
        """The module-level helper must surface weighted_overall_score."""
        from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (  # noqa: E501
            aggregate_from_comparisons,
        )

        gt, pred = self._make_invoice_pair(
            {"invoice_id": "INV-1", "note": "hi", "total": 10.0},
            {"invoice_id": "INV-9", "note": "hi", "total": 10.0},
        )
        comparison = gt.compare_with(pred, include_confusion_matrix=True)

        result = aggregate_from_comparisons([comparison])

        assert "weighted_overall_score" in result.metrics
        assert result.metrics["weighted_overall_score"] == pytest.approx(
            comparison["overall_score"]
        )

    def test_per_path_mean_score_averages_only_observing_docs(self):
        """mean_score denominator at each path = docs that actually scored that path.

        Doc A records scores at paths {foo, bar}; doc B only at {foo}. foo's
        mean must average across 2 docs, bar's across 1.
        """
        evaluator = BulkStructuredModelEvaluator()
        evaluator.update_from_comparison_result(
            {
                "confusion_matrix": {
                    "overall": {"tp": 2},
                    "fields": {
                        "foo": {"tp": 1, "threshold_applied_score": 0.8},
                        "bar": {"tp": 1, "threshold_applied_score": 0.4},
                    },
                }
            },
            doc_id="a",
        )
        evaluator.update_from_comparison_result(
            {
                "confusion_matrix": {
                    "overall": {"tp": 1},
                    "fields": {"foo": {"tp": 1, "threshold_applied_score": 0.2}},
                }
            },
            doc_id="b",
        )

        result = evaluator.compute()

        assert result.field_metrics["foo"]["mean_score"] == pytest.approx(0.5)
        assert result.field_metrics["bar"]["mean_score"] == pytest.approx(0.4)
        assert evaluator._field_score_counts["foo"] == 2
        assert evaluator._field_score_counts["bar"] == 1

    def test_merge_state_across_disjoint_worker_field_sets(self):
        """Workers seeing different field subsets must merge into correct union.

        Models the distributed case: one worker only scored path X, another
        only path Y. After merge, both paths appear with their own per-path
        denominators.
        """
        worker_a = BulkStructuredModelEvaluator()
        worker_a.update_from_comparison_result(
            {
                "confusion_matrix": {
                    "overall": {"tp": 1},
                    "fields": {"only_a": {"tp": 1, "threshold_applied_score": 0.7}},
                }
            },
            doc_id="a",
        )

        worker_b = BulkStructuredModelEvaluator()
        worker_b.update_from_comparison_result(
            {
                "confusion_matrix": {
                    "overall": {"tp": 1},
                    "fields": {"only_b": {"tp": 1, "threshold_applied_score": 0.3}},
                }
            },
            doc_id="b",
        )

        worker_a.merge_state(worker_b.get_state())
        result = worker_a.compute()

        assert "only_a" in result.field_metrics
        assert result.field_metrics["only_a"]["mean_score"] == pytest.approx(0.7)
        assert "only_b" in result.field_metrics
        assert result.field_metrics["only_b"]["mean_score"] == pytest.approx(0.3)
        # Overall aggregate spans both docs.
        assert worker_a._overall_score_count == 0  # no top-level overall_score set
        assert result.document_count == 2


class TestAccumulatorErrorVisibility:
    """A custom accumulator that raises on every doc must surface its
    failure count on ``compute().accumulator_errors`` so the user can
    spot silently-broken metrics. The outer confusion matrix should be
    unaffected because each accumulator runs in its own try/except."""

    class _AlwaysFailingAccumulator:
        """Minimal PostComparisonAccumulator that raises on every accumulate."""

        @property
        def name(self) -> str:
            return "failing_one"

        def reset(self) -> None:
            return None

        def accumulate(self, comparison_result, prediction_raw) -> None:
            raise RuntimeError("intentional accumulator failure")

        def compute(self):
            return None

        def get_state(self):
            return {}

        def load_state(self, state) -> None:
            return None

        def merge_state(self, other_state) -> None:
            return None

    def _baseline_comparison_result(self):
        return {
            "confusion_matrix": {
                "overall": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
                "fields": {},
            }
        }

    def test_accumulator_errors_counted_per_name(self):
        evaluator = BulkStructuredModelEvaluator(
            accumulators=[self._AlwaysFailingAccumulator()],
        )

        for i in range(3):
            evaluator.update_from_comparison_result(
                self._baseline_comparison_result(), doc_id=f"doc_{i}"
            )

        result = evaluator.compute()

        assert result.accumulator_errors == {"failing_one": 3}
        # Confusion matrix is untouched — the outer try/except only fires
        # when accumulation itself fails, not when an accumulator raises.
        assert result.metrics["tp"] == 3
        assert result.document_count == 3

    def test_accumulator_errors_none_when_all_succeed(self):
        """No failures → accumulator_errors stays None to preserve the
        ``additive optional`` contract on ProcessEvaluation."""
        evaluator = BulkStructuredModelEvaluator(
            accumulators=[],
        )
        evaluator.update_from_comparison_result(
            self._baseline_comparison_result(), doc_id="doc_0"
        )
        result = evaluator.compute()
        assert result.accumulator_errors is None

    def test_accumulator_errors_round_trip_through_state(self):
        """Failures must survive get_state / load_state for checkpoint
        recovery and merge across distributed workers."""
        worker_a = BulkStructuredModelEvaluator(
            accumulators=[self._AlwaysFailingAccumulator()],
        )
        worker_a.update_from_comparison_result(
            self._baseline_comparison_result(), doc_id="a"
        )

        worker_b = BulkStructuredModelEvaluator(
            accumulators=[self._AlwaysFailingAccumulator()],
        )
        worker_b.update_from_comparison_result(
            self._baseline_comparison_result(), doc_id="b"
        )
        worker_b.update_from_comparison_result(
            self._baseline_comparison_result(), doc_id="c"
        )

        worker_a.merge_state(worker_b.get_state())
        merged = worker_a.compute()

        assert merged.accumulator_errors == {"failing_one": 3}


class TestJsonlPersistentHandle:
    """Persistent JSONL handle: one open() per evaluator instead of one per
    document. Verifies shape (line count, append on reopen) — not perf."""

    def _gt_pred(self):
        sample = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }
        return BankStatement(**sample), BankStatement(**sample)

    def test_jsonl_writer_writes_one_line_per_doc(self, tmp_path):
        path = tmp_path / "results.jsonl"
        evaluator = BulkStructuredModelEvaluator(
            target_schema=BankStatement,
            individual_results_jsonl=str(path),
        )
        gt, pred = self._gt_pred()
        for i in range(100):
            evaluator.update(gt, pred, doc_id=f"doc_{i}")
        evaluator.compute()  # close() runs at the end of compute()

        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 100
        # Each line must be a complete JSON record (flush-per-line keeps
        # the file valid even on mid-run termination).
        for line in lines:
            record = json.loads(line)
            assert "doc_id" in record
            assert "comparison_result" in record

    def test_jsonl_writer_appends_after_close_and_reopen(self, tmp_path):
        path = tmp_path / "results.jsonl"

        first = BulkStructuredModelEvaluator(
            target_schema=BankStatement,
            individual_results_jsonl=str(path),
        )
        gt, pred = self._gt_pred()
        first.update(gt, pred, doc_id="first_doc")
        first.close()

        second = BulkStructuredModelEvaluator(
            target_schema=BankStatement,
            individual_results_jsonl=str(path),
        )
        second.update(gt, pred, doc_id="second_doc")
        second.close()

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["doc_id"] == "first_doc"
        assert json.loads(lines[1])["doc_id"] == "second_doc"

    def test_close_is_idempotent(self, tmp_path):
        path = tmp_path / "results.jsonl"
        evaluator = BulkStructuredModelEvaluator(
            target_schema=BankStatement,
            individual_results_jsonl=str(path),
        )
        gt, pred = self._gt_pred()
        evaluator.update(gt, pred, doc_id="doc_0")
        evaluator.close()
        # A second close() on a never-opened handle must be a no-op.
        evaluator.close()


class TestLegacyStateMigration:
    """The migration helper at ``_migrate_legacy_acc_states`` must lift
    pre-accumulator (top-level) confidence keys into the new
    ``accumulators`` shape on both load and merge paths."""

    def _legacy_state(self, schema_name: str) -> dict:
        # Old top-level confidence shape — no ``accumulators`` key.
        return {
            "confusion_matrix": {
                "overall": {"tp": 2, "fp": 1, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
                "fields": {
                    "name": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
                    "price": {"tp": 1, "fp": 1, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
                },
            },
            "errors": [],
            "processed_count": 2,
            "start_time": 0.0,
            "target_schema": schema_name,
            "elide_errors": False,
            # Legacy confidence keys live at the top level.
            "keyed_confidence_pairs": {
                "name": [
                    {"is_match": True, "confidence": 0.9, "similarity": 1.0},
                    {"is_match": False, "confidence": 0.2, "similarity": 0.5},
                ],
                "price": [
                    {"is_match": True, "confidence": 0.85, "similarity": 1.0},
                ],
            },
            "confidence_fields_with": 3,
            "confidence_fields_total": 4,
        }

    def test_load_state_migrates_legacy_confidence_keys(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=BankStatement)
        legacy = self._legacy_state(schema_name="BankStatement")

        evaluator.load_state(legacy)
        result = evaluator.compute()

        # Confidence metrics surface despite the legacy top-level shape.
        assert result.confidence_metrics is not None
        cov = result.confidence_metrics["coverage"]
        assert cov["fields_with_confidence"] == 3
        assert cov["fields_total"] == 4
        # AUROC needs both classes — we have one match and one non-match.
        assert result.confidence_metrics["overall"]["auroc"]["value"] is not None

    def test_merge_state_migrates_legacy_confidence_keys(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=BankStatement)
        legacy = self._legacy_state(schema_name="BankStatement")

        evaluator.merge_state(legacy)
        result = evaluator.compute()

        assert result.confidence_metrics is not None
        cov = result.confidence_metrics["coverage"]
        assert cov["fields_with_confidence"] == 3
        assert cov["fields_total"] == 4
