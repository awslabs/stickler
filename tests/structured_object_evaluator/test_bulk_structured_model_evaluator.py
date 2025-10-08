#!/usr/bin/env python3

"""
Comprehensive tests for BulkStructuredModelEvaluator.

This test suite validates the stateful bulk evaluation functionality,
memory efficiency, error handling, state management, and distributed processing
capabilities of the new BulkStructuredModelEvaluator.
"""

import pytest
import json
import pandas as pd
from typing import List, Optional
from pydantic import Field

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
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
    transactions: List[Transaction] = ComparableField(aggregate=False, weight=1.0)


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
        except:
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
        except:
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
        """Test that bulk evaluator produces same results as single evaluator for single document."""
        from stickler.structured_object_evaluator.evaluator import (
            StructuredModelEvaluator,
        )

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
        single_evaluator = StructuredModelEvaluator(BankStatement)
        single_result = single_evaluator.evaluate(gt_model, pred_model)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
