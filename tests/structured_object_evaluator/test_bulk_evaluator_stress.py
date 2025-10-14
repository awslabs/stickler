#!/usr/bin/env python3

"""
Pytest stress tests for BulkStructuredModelEvaluator robustness.
Converted from cline/stress_test_bulk_evaluator.py
"""

import pytest
import time
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
def simple_sample_data():
    """Fixture providing simple sample data for performance tests."""
    return {
        "accountNumber": "1234567890",
        "contact": {"phone": "555-123-4567"},
        "transactions": [
            {"date": "2023-01-01", "description": "Test", "amount": 100.0}
        ],
    }


@pytest.fixture
def rich_sample_data():
    """Fixture providing rich sample data for comprehensive tests."""
    return {
        "accountNumber": "1234567890",
        "contact": {"phone": "555-123-4567", "email": "test@example.com"},
        "transactions": [
            {"date": "2023-01-01", "description": "ATM Withdrawal", "amount": -100.0},
            {"date": "2023-01-02", "description": "Direct Deposit", "amount": 2000.0},
        ],
    }


class TestBulkEvaluatorStress:
    """Stress tests for BulkStructuredModelEvaluator robustness."""

    def test_basic_functionality(self, rich_sample_data):
        """Test basic functionality - does it actually work?"""
        evaluator = BulkStructuredModelEvaluator(BankStatement)

        gt_model = BankStatement(**rich_sample_data)
        pred_model = BankStatement(**rich_sample_data)

        # Test update
        evaluator.update(gt_model, pred_model, "test_doc")
        assert evaluator._processed_count == 1

        # Test compute
        result = evaluator.compute()
        assert result is not None
        assert "cm_accuracy" in result.metrics
        assert len(result.field_metrics) > 0

        # Verify nested paths exist
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
        actual_paths = set(result.field_metrics.keys())
        missing_paths = expected_paths - actual_paths

        assert len(missing_paths) == 0, f"Missing field paths: {missing_paths}"

    @pytest.mark.slow
    def test_large_dataset_processing(self, simple_sample_data):
        """Test large dataset processing - memory and performance."""
        evaluator = BulkStructuredModelEvaluator(BankStatement, verbose=False)

        gt_model = BankStatement(**simple_sample_data)
        pred_model = BankStatement(**simple_sample_data)

        # Process moderate number of documents (reduced for CI/CD)
        num_docs = 1000

        start_time = time.time()
        for i in range(num_docs):
            evaluator.update(gt_model, pred_model, f"doc_{i}")

        processing_time = time.time() - start_time

        result = evaluator.compute()

        # Verify processing completed successfully
        assert evaluator._processed_count == num_docs
        assert result.metrics["cm_accuracy"] == 1.0  # Perfect matches
        assert len(result.field_metrics) > 0

        # Performance should be reasonable (at least 100 docs/sec)
        docs_per_sec = num_docs / processing_time
        assert docs_per_sec > 100, f"Performance too slow: {docs_per_sec:.0f} docs/sec"

    def test_error_handling(self, rich_sample_data):
        """Test error handling - can it handle bad input?"""
        evaluator = BulkStructuredModelEvaluator(BankStatement, elide_errors=False)

        gt_model = BankStatement(**rich_sample_data)
        pred_model = BankStatement(**rich_sample_data)

        # Process valid document
        evaluator.update(gt_model, pred_model, "valid_doc")
        assert evaluator._processed_count == 1

        # Try to break it with None - should handle gracefully or raise
        try:
            evaluator.update(None, pred_model, "invalid_doc1")
            # If it doesn't raise, check that an error was recorded
            result = evaluator.compute()
            if not result.errors:
                pytest.fail(
                    "Expected either an exception or error to be recorded for None input"
                )
        except (TypeError, AttributeError, ValueError):
            # This is acceptable behavior - raising an exception
            pass

        # Try to break it with wrong type - should handle gracefully or raise
        try:
            evaluator.update("not_a_model", pred_model, "invalid_doc2")
            result = evaluator.compute()
            if not result.errors:
                pytest.fail(
                    "Expected either an exception or error to be recorded for invalid input"
                )
        except (TypeError, AttributeError, ValueError):
            # This is acceptable behavior - raising an exception
            pass

        # Try to break it with mismatched models - should handle gracefully or raise
        try:
            evaluator.update(gt_model, "not_a_model", "invalid_doc3")
            result = evaluator.compute()
            if not result.errors:
                pytest.fail(
                    "Expected either an exception or error to be recorded for invalid input"
                )
        except (TypeError, AttributeError, ValueError):
            # This is acceptable behavior - raising an exception
            pass

        # Verify valid processing still works
        result = evaluator.compute()
        assert evaluator._processed_count >= 1  # At least the valid document
        assert "cm_accuracy" in result.metrics

    def test_state_management(self, rich_sample_data):
        """Test state management - serialization, loading, merging."""
        evaluator1 = BulkStructuredModelEvaluator(BankStatement)

        gt_model = BankStatement(**rich_sample_data)
        pred_model = BankStatement(**rich_sample_data)

        # Process some documents
        evaluator1.update(gt_model, pred_model, "doc1")
        evaluator1.update(gt_model, pred_model, "doc2")
        assert evaluator1._processed_count == 2

        # Test get_state
        state = evaluator1.get_state()
        assert state is not None
        assert "processed_count" in state
        assert state["processed_count"] == 2

        # Test load_state
        evaluator2 = BulkStructuredModelEvaluator(BankStatement)
        evaluator2.load_state(state)
        assert evaluator2._processed_count == 2

        # Test state merging
        evaluator3 = BulkStructuredModelEvaluator(BankStatement)
        evaluator3.update(gt_model, pred_model, "doc3")
        assert evaluator3._processed_count == 1

        state3 = evaluator3.get_state()
        evaluator1.merge_state(state3)
        assert evaluator1._processed_count == 3

        # Test that results are consistent
        result1 = evaluator1.compute()
        result2 = evaluator2.compute()

        # Results should be proportionally consistent
        assert result1.metrics["cm_accuracy"] == result2.metrics["cm_accuracy"] == 1.0
        assert len(result1.field_metrics) == len(result2.field_metrics)

    def test_batch_vs_streaming_equivalence(self, rich_sample_data):
        """Test batch vs streaming equivalence."""
        # Create different data variants
        data_variants = []
        for i in range(5):
            data = rich_sample_data.copy()
            data["accountNumber"] = f"123456789{i}"
            data_variants.append(data)

        # Streaming processing
        stream_evaluator = BulkStructuredModelEvaluator(BankStatement)
        stream_evaluator.reset()

        for i, data in enumerate(data_variants):
            gt_model = BankStatement(**rich_sample_data)
            pred_model = BankStatement(**data)
            stream_evaluator.update(gt_model, pred_model, f"stream_doc_{i}")

        stream_result = stream_evaluator.compute()

        # Batch processing
        batch_evaluator = BulkStructuredModelEvaluator(BankStatement)
        batch_evaluator.reset()

        batch_data = []
        for i, data in enumerate(data_variants):
            gt_model = BankStatement(**rich_sample_data)
            pred_model = BankStatement(**data)
            batch_data.append((gt_model, pred_model, f"batch_doc_{i}"))

        batch_evaluator.update_batch(batch_data)
        batch_result = batch_evaluator.compute()

        # Compare results - should be equivalent
        assert stream_evaluator._processed_count == batch_evaluator._processed_count
        assert (
            stream_result.metrics["cm_accuracy"] == batch_result.metrics["cm_accuracy"]
        )
        assert (
            stream_result.metrics["cm_precision"]
            == batch_result.metrics["cm_precision"]
        )
        assert stream_result.metrics["cm_recall"] == batch_result.metrics["cm_recall"]
        assert stream_result.metrics["cm_f1"] == batch_result.metrics["cm_f1"]
        assert len(stream_result.field_metrics) == len(batch_result.field_metrics)

    def test_edge_cases(self):
        """Test edge cases - empty data, None values, etc."""
        evaluator = BulkStructuredModelEvaluator(BankStatement)

        # Test empty transactions
        empty_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567"},
            "transactions": [],
        }

        gt_model = BankStatement(**empty_data)
        pred_model = BankStatement(**empty_data)
        evaluator.update(gt_model, pred_model, "empty_doc")

        # Test None email (optional field)
        none_email_data = {
            "accountNumber": "1234567890",
            "contact": {"phone": "555-123-4567", "email": None},
            "transactions": [],
        }

        gt_model2 = BankStatement(**none_email_data)
        pred_model2 = BankStatement(**none_email_data)
        evaluator.update(gt_model2, pred_model2, "none_email_doc")

        result = evaluator.compute()

        # Should handle edge cases gracefully
        assert evaluator._processed_count == 2
        assert result.metrics["cm_accuracy"] == 1.0  # Perfect matches
        assert len(result.field_metrics) > 0

    def test_memory_efficiency(self, simple_sample_data):
        """Test that memory usage doesn't grow excessively with document count."""
        import gc
        import psutil
        import os

        evaluator = BulkStructuredModelEvaluator(BankStatement, verbose=False)
        gt_model = BankStatement(**simple_sample_data)
        pred_model = BankStatement(**simple_sample_data)

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process documents
        num_docs = 500
        for i in range(num_docs):
            evaluator.update(gt_model, pred_model, f"doc_{i}")

            # Force garbage collection periodically
            if i % 100 == 0:
                gc.collect()

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        result = evaluator.compute()

        # Verify processing completed
        assert evaluator._processed_count == num_docs
        assert result.metrics["cm_accuracy"] == 1.0

        # Memory growth should be reasonable (less than 100MB for 500 docs)
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"

    def test_concurrent_safety_basics(self, simple_sample_data):
        """Test basic thread safety considerations."""
        import threading
        import time

        evaluator = BulkStructuredModelEvaluator(BankStatement, verbose=False)
        gt_model = BankStatement(**simple_sample_data)
        pred_model = BankStatement(**simple_sample_data)

        results = []
        exceptions = []

        def worker(worker_id):
            try:
                for i in range(10):
                    evaluator.update(
                        gt_model, pred_model, f"worker_{worker_id}_doc_{i}"
                    )
                    time.sleep(
                        0.001
                    )  # Small delay to increase chance of race conditions
                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                exceptions.append(e)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should complete without exceptions
        assert len(exceptions) == 0, f"Concurrency exceptions: {exceptions}"
        assert len(results) == 3
        assert evaluator._processed_count == 30

        result = evaluator.compute()
        assert result.metrics["cm_accuracy"] == 1.0
