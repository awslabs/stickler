

"""
Test Optional[List[StructuredModel]] None vs populated edge cases.
These tests validate that None vs populated scenarios generate proper FA/FN metrics.
"""

import pytest
from typing import List, Optional
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


class Transaction(StructuredModel):
    amount: float = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    description: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class Document(StructuredModel):
    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    transactions: Optional[List[Transaction]] = ComparableField(weight=1.0)
    total_amount: Optional[float] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )


class TestNoneHandlingEdgeCases:
    """Test suite for Optional[List[StructuredModel]] None vs populated edge cases."""

    def test_none_vs_populated_list_false_alarm(self):
        """Test GT=None vs Prediction=populated list (should generate FA)."""
        evaluator = StructuredModelEvaluator(threshold=0.7)

        # GT has None, Prediction has transactions (should be FA)
        gt = Document(id="doc1", transactions=None, total_amount=100.0)
        pred_transactions = [
            Transaction(amount=100.0, description="Payment 1"),
            Transaction(amount=50.0, description="Payment 2"),
        ]
        pred = Document(id="doc1", transactions=pred_transactions, total_amount=100.0)

        result = evaluator.evaluate(gt, pred)
        cm = result["confusion_matrix"]

        # Extract transactions metrics
        transactions_field = cm["fields"]["transactions"]
        if "overall" in transactions_field:
            transactions_metrics = transactions_field["overall"]
        else:
            transactions_metrics = transactions_field

        # Should have FA >= 1 (at least 1 false alarm for each transaction)
        assert transactions_metrics.get("fa", 0) >= 1, (
            f"Expected FA >= 1, got FA = {transactions_metrics.get('fa', 0)}"
        )

        # Should have FP >= 1 (false positive includes false alarms)
        assert transactions_metrics.get("fp", 0) >= 1, (
            f"Expected FP >= 1, got FP = {transactions_metrics.get('fp', 0)}"
        )

        # Should not have TP (no true positives when GT is None)
        assert transactions_metrics.get("tp", 0) == 0, (
            f"Expected TP = 0, got TP = {transactions_metrics.get('tp', 0)}"
        )

    def test_populated_list_vs_none_false_negative(self):
        """Test GT=populated list vs Prediction=None (should generate FN)."""
        evaluator = StructuredModelEvaluator(threshold=0.7)

        # GT has transactions, Prediction has None (should be FN)
        gt_transactions = [
            Transaction(amount=100.0, description="Payment 1"),
            Transaction(amount=50.0, description="Payment 2"),
        ]
        gt = Document(id="doc2", transactions=gt_transactions, total_amount=100.0)
        pred = Document(id="doc2", transactions=None, total_amount=100.0)

        result = evaluator.evaluate(gt, pred)
        cm = result["confusion_matrix"]

        transactions_field = cm["fields"]["transactions"]
        if "overall" in transactions_field:
            transactions_metrics = transactions_field["overall"]
        else:
            transactions_metrics = transactions_field

        # Should have FN >= 1 (at least 1 false negative for missing transactions)
        assert transactions_metrics.get("fn", 0) >= 1, (
            f"Expected FN >= 1, got FN = {transactions_metrics.get('fn', 0)}"
        )

        # Should not have TP (no true positives when prediction is None)
        assert transactions_metrics.get("tp", 0) == 0, (
            f"Expected TP = 0, got TP = {transactions_metrics.get('tp', 0)}"
        )

        # Should not have FA (no false alarms when prediction is None)
        assert transactions_metrics.get("fa", 0) == 0, (
            f"Expected FA = 0, got FA = {transactions_metrics.get('fa', 0)}"
        )

    def test_none_vs_none_true_negative(self):
        """Test GT=None vs Prediction=None (should be TN)."""
        evaluator = StructuredModelEvaluator(threshold=0.7)

        # Both documents have None transactions
        gt = Document(id="doc3", transactions=None, total_amount=100.0)
        pred = Document(id="doc3", transactions=None, total_amount=100.0)

        result = evaluator.evaluate(gt, pred)
        cm = result["confusion_matrix"]

        transactions_field = cm["fields"]["transactions"]
        if "overall" in transactions_field:
            transactions_metrics = transactions_field["overall"]
        else:
            transactions_metrics = transactions_field

        # Should have TN = 1 (true negative for both being None)
        assert transactions_metrics.get("tn", 0) == 1, (
            f"Expected TN = 1, got TN = {transactions_metrics.get('tn', 0)}"
        )

        # Should not have any other metrics
        assert transactions_metrics.get("tp", 0) == 0, "Expected TP = 0"
        assert transactions_metrics.get("fp", 0) == 0, "Expected FP = 0"
        assert transactions_metrics.get("fn", 0) == 0, "Expected FN = 0"
        assert transactions_metrics.get("fa", 0) == 0, "Expected FA = 0"
        assert transactions_metrics.get("fd", 0) == 0, "Expected FD = 0"

    def test_empty_list_vs_none_equivalent(self):
        """Test empty list vs None - they are treated as equivalent states."""
        evaluator = StructuredModelEvaluator(threshold=0.7)

        # GT has empty list, Prediction has None
        gt = Document(id="doc4", transactions=[], total_amount=0.0)
        pred = Document(id="doc4", transactions=None, total_amount=0.0)

        result = evaluator.evaluate(gt, pred)
        cm = result["confusion_matrix"]

        # Empty list and None are treated as equivalent states in this implementation
        # GT=[] and Pred=None results in TN (both represent "no data")
        transactions_field = cm["fields"]["transactions"]
        if "overall" in transactions_field:
            transactions_metrics = transactions_field["overall"]
        else:
            transactions_metrics = transactions_field

        # Should have TN = 1 (empty list and None are equivalent - both mean "no data")
        assert transactions_metrics.get("tn", 0) == 1, (
            f"Expected TN = 1, got TN = {transactions_metrics.get('tn', 0)}"
        )
        assert transactions_metrics.get("fn", 0) == 0, (
            f"Expected FN = 0, got FN = {transactions_metrics.get('fn', 0)}"
        )

        # Should not have other metrics
        assert transactions_metrics.get("tp", 0) == 0, "Expected TP = 0"
        assert transactions_metrics.get("fp", 0) == 0, "Expected FP = 0"
        assert transactions_metrics.get("fa", 0) == 0, "Expected FA = 0"
        assert transactions_metrics.get("fd", 0) == 0, "Expected FD = 0"

    def test_direct_model_comparison(self):
        """Test None handling directly through model comparison."""
        # Test None vs populated directly
        gt = Document(id="direct_test", transactions=None, total_amount=100.0)
        pred_transactions = [Transaction(amount=100.0, description="Payment")]
        pred = Document(
            id="direct_test", transactions=pred_transactions, total_amount=100.0
        )

        # Use compare_with method directly
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]

        transactions_field = cm["fields"]["transactions"]

        # Raw compare_with result structure includes metrics under "overall" key
        # Different from evaluator.evaluate() which flattens the structure
        if "overall" in transactions_field:
            transactions_metrics = transactions_field["overall"]
        else:
            transactions_metrics = transactions_field

        # Should properly handle None vs populated scenario
        assert transactions_metrics.get("fa", 0) >= 1, (
            "Should have false alarms for None vs populated"
        )
        assert transactions_metrics.get("tp", 0) == 0, "Should not have true positives"
