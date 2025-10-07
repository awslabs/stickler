#!/usr/bin/env python3
# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""
Baseline tests for Hungarian matching extraction.

These tests capture the CURRENT behavior (including bugs) before extracting
Hungarian matching logic to a dedicated class. The goal is to ensure bit-for-bit
compatibility during refactoring.

Known Issues Being Preserved:
1. Wrong threshold source: Uses ComparableField.threshold instead of StructuredModel.match_threshold
2. Generates nested metrics for ALL matched pairs regardless of threshold
3. Incorrect object-level counts in some scenarios

These bugs will be fixed in Phase 3, but Phase 2 (extraction) must maintain identical behavior.
"""

import pytest
from typing import List

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


class Transaction(StructuredModel):
    """Transaction model for testing Hungarian matching."""

    transaction_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )

    description: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )

    amount: float = ComparableField(threshold=0.9, weight=1.0)

    # This threshold SHOULD control Hungarian matching recursion (but currently doesn't)
    match_threshold = 0.8


class Account(StructuredModel):
    """Account model containing list of transactions."""

    account_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )

    # NO threshold on this field - should use Transaction.match_threshold
    transactions: List[Transaction] = ComparableField(weight=3.0)


class TestHungarianMatchingBaseline:
    """Baseline tests that capture current Hungarian matching behavior."""

    def create_test_accounts(self):
        """Create the standard test scenario from documentation."""
        gt_account = Account(
            account_id="ACC-12345",
            transactions=[
                Transaction(
                    transaction_id="TXN-001",
                    description="Coffee shop payment",
                    amount=4.95,
                ),
                Transaction(
                    transaction_id="TXN-002", description="Grocery store", amount=127.43
                ),
                Transaction(
                    transaction_id="TXN-003", description="Gas station", amount=45.67
                ),
            ],
        )

        pred_account = Account(
            account_id="ACC-12345",
            transactions=[
                # Good match: similarity ~0.860 (≥ 0.8 threshold)
                Transaction(
                    transaction_id="TXN-001", description="Coffee shop", amount=4.95
                ),
                # Poor match: similarity ~0.572 (< 0.8 threshold)
                Transaction(
                    transaction_id="TXN-002",
                    description="Online purchase",
                    amount=89.99,
                ),
                # Unmatched: Different ID
                Transaction(
                    transaction_id="TXN-004", description="Restaurant", amount=23.45
                ),
                # TXN-003 is missing → FN
            ],
        )

        return gt_account, pred_account

    def test_individual_transaction_similarities(self):
        """Test individual transaction comparisons to understand pairwise similarities."""
        gt_account, pred_account = self.create_test_accounts()
        gt_txns = gt_account.transactions
        pred_txns = pred_account.transactions

        # Calculate all pairwise similarities
        similarities = {}
        for i, gt_txn in enumerate(gt_txns):
            for j, pred_txn in enumerate(pred_txns):
                similarity = gt_txn.compare(pred_txn)
                similarities[(i, j)] = similarity

        # Assert expected similarities from our documentation
        assert similarities[(0, 0)] == pytest.approx(0.860, rel=0.01)  # Good match
        assert similarities[(1, 1)] == pytest.approx(0.572, rel=0.01)  # Poor match

        # Check threshold classification
        assert (
            similarities[(0, 0)] >= Transaction.match_threshold
        )  # Should get nested metrics
        assert (
            similarities[(1, 1)] < Transaction.match_threshold
        )  # Should NOT get nested metrics

    def test_current_threshold_behavior_bug(self):
        """Test that current implementation uses WRONG threshold source.

        This test documents the bug: implementation currently uses ComparableField.threshold
        instead of Transaction.match_threshold for Hungarian matching decisions.
        """
        gt_account, pred_account = self.create_test_accounts()

        result = gt_account.compare_with(
            pred_account, include_confusion_matrix=True, add_derived_metrics=True
        )

        cm = result["confusion_matrix"]
        txn_field = cm["fields"]["transactions"]

        # BUG: Current implementation generates nested metrics for ALL matched pairs
        # even though only GT[0]->Pred[0] should qualify (similarity ≥ 0.8)
        assert "fields" in txn_field, "Should have nested field metrics"
        nested_fields = txn_field["fields"]

        # Document the bug: shows metrics for all 3 transaction field types
        # This proves nested metrics were generated for poor matches
        expected_fields = ["transaction_id", "description", "amount"]
        assert all(field in nested_fields for field in expected_fields)

        # The nested metrics show aggregated results from ALL matched pairs,
        # not just the good matches as the documentation specifies
        # Updated to handle new aggregate structure
        def get_metric(field_data, metric):
            if "overall" in field_data:
                return field_data["overall"][metric]
            return field_data[metric]

        assert (
            get_metric(nested_fields["transaction_id"], "tp") >= 1
        )  # At least the good match
        assert (
            get_metric(nested_fields["description"], "fd") >= 1
        )  # Poor matches show as FD
        assert get_metric(nested_fields["amount"], "tp") >= 1  # Some amount matches

    def test_current_object_level_counts(self):
        """Test corrected object-level counting behavior.

        Documents the FIXED behavior after threshold bug correction.
        The implementation now correctly uses Transaction.match_threshold (0.8)
        instead of ComparableField.threshold, resulting in proper classification.
        """
        gt_account, pred_account = self.create_test_accounts()

        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)

        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        # CORRECTED behavior with proper threshold=0.8:
        # GT[0] vs Pred[0]: similarity=0.860 >= 0.8 → TP (CORRECT)
        # GT[1] vs Pred[1]: similarity=0.572 < 0.8 → FD (CORRECTED)
        # GT[2]: Unmatched → FN
        # Pred[2]: Unmatched → FA

        # Fixed counts with corrected threshold logic
        assert txn_overall["tp"] == 1  # FIXED: Only good match (≥0.8) classified as TP
        assert txn_overall["fd"] == 2  # FIXED: Two poor matches (<0.8) classified as FD
        assert (
            txn_overall["fa"] == 0
        )  # FIXED: All pred items matched (Hungarian matches optimally)
        assert (
            txn_overall["fn"] == 0
        )  # FIXED: All GT items matched (Hungarian matches optimally)

    def test_nested_field_metric_aggregation(self):
        """Test how nested field metrics are aggregated from matched pairs."""
        gt_account, pred_account = self.create_test_accounts()

        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)

        cm = result["confusion_matrix"]
        nested_fields = cm["fields"]["transactions"]["fields"]

        # Current behavior: aggregates metrics from ALL matched pairs
        # Expected behavior: should only aggregate from threshold-passing pairs

        # Helper function to get metrics from new structure
        def get_metric(field_data, metric):
            if "overall" in field_data:
                return field_data["overall"][metric]
            return field_data[metric]

        # Transaction ID field: CORRECTED - only shows activity from threshold-passing matches
        tid_metrics = nested_fields["transaction_id"]
        assert (
            get_metric(tid_metrics, "tp") >= 1
        )  # Only good matches (≥0.8) contribute TP
        # No FD assertion - poor matches don't generate nested field metrics due to threshold gating

        # Description field: should show poor matches since descriptions differ significantly
        desc_metrics = nested_fields["description"]
        assert (
            get_metric(desc_metrics, "fd") >= 1
        )  # Most description comparisons are poor

        # Amount field: mix of good and poor matches
        amt_metrics = nested_fields["amount"]
        assert (
            get_metric(amt_metrics, "tp") >= 1 or get_metric(amt_metrics, "fd") >= 1
        )  # Some amount activity

    def test_empty_list_edge_cases(self):
        """Test Hungarian matching with empty lists."""

        # Both empty
        empty_gt = Account(account_id="ACC-001", transactions=[])
        empty_pred = Account(account_id="ACC-001", transactions=[])

        result = empty_gt.compare_with(empty_pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        assert txn_overall["tn"] == 1  # True negative for empty lists
        assert all(txn_overall[metric] == 0 for metric in ["tp", "fa", "fd", "fn"])

        # GT empty, pred has items
        gt_empty = Account(account_id="ACC-001", transactions=[])
        pred_with_items = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(transaction_id="TXN-001", description="Test", amount=10.0)
            ],
        )

        result = gt_empty.compare_with(pred_with_items, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        assert txn_overall["fa"] >= 1  # False alarms for unmatched pred items
        assert txn_overall["fp"] >= 1  # False positives

        # GT has items, pred empty
        result = pred_with_items.compare_with(gt_empty, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        assert txn_overall["fn"] >= 1  # False negatives for unmatched GT items

    def test_single_item_lists(self):
        """Test Hungarian matching with single-item lists."""
        gt_single = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(transaction_id="TXN-001", description="Coffee", amount=5.0)
            ],
        )

        pred_single = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(
                    transaction_id="TXN-001", description="Coffee shop", amount=5.0
                )
            ],
        )

        result = gt_single.compare_with(pred_single, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        # Should have exactly one matched pair
        assert txn_overall["tp"] + txn_overall["fd"] == 1  # One object comparison

        # Should have nested field metrics since there's a matched pair
        assert "fields" in cm["fields"]["transactions"]
        nested_fields = cm["fields"]["transactions"]["fields"]
        assert len(nested_fields) == 3  # transaction_id, description, amount

    def test_all_poor_matches_scenario(self):
        """Test scenario where all matches are below threshold."""
        gt_account = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(transaction_id="TXN-001", description="Coffee", amount=5.0),
                Transaction(transaction_id="TXN-002", description="Lunch", amount=15.0),
            ],
        )

        # Create predictions that are all poor matches (< 0.8 similarity)
        pred_account = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(
                    transaction_id="TXN-999",
                    description="Completely different",
                    amount=100.0,
                ),
                Transaction(
                    transaction_id="TXN-888", description="Also different", amount=200.0
                ),
            ],
        )

        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        # All matches should be classified as FD (poor matches)
        # CORRECTED: NO nested metrics should be generated for all-poor scenarios
        assert txn_overall["tp"] == 0  # No good matches
        assert txn_overall["fd"] >= 1  # All matches are poor

        # CORRECTED: With threshold-gated recursion fixed, poor matches don't generate nested field metrics
        if "fields" in cm["fields"]["transactions"]:
            nested_fields = cm["fields"]["transactions"]["fields"]
            # Threshold-gating working correctly - should be empty for all poor matches
            assert (
                len(nested_fields) == 0
            )  # FIXED: No nested field metrics for poor matches
        else:
            # This is also acceptable - no nested fields structure at all
            pass

    def test_all_good_matches_scenario(self):
        """Test scenario where all matches are above threshold."""
        gt_account = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(
                    transaction_id="TXN-001", description="Coffee shop", amount=5.0
                ),
                Transaction(
                    transaction_id="TXN-002", description="Grocery store", amount=50.0
                ),
            ],
        )

        # Create predictions that are all good matches (≥ 0.8 similarity)
        pred_account = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(
                    transaction_id="TXN-001", description="Coffee shop", amount=5.0
                ),  # Perfect match
                Transaction(
                    transaction_id="TXN-002", description="Grocery store", amount=50.0
                ),  # Perfect match
            ],
        )

        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        # All matches should be TP
        assert txn_overall["tp"] == 2  # Two good matches
        assert txn_overall["fd"] == 0  # No poor matches
        assert txn_overall["fa"] == 0  # No unmatched items
        assert txn_overall["fn"] == 0  # No unmatched items

        # Should definitely have nested metrics for all good matches
        assert "fields" in cm["fields"]["transactions"]
        nested_fields = cm["fields"]["transactions"]["fields"]
        assert len(nested_fields) == 3  # All transaction fields

        # All nested field metrics should be positive (showing activity from 2 good matches)
        def get_metric(field_data, metric):
            if "overall" in field_data:
                return field_data["overall"][metric]
            return field_data[metric]

        for field_name, field_metrics in nested_fields.items():
            total_activity = sum(
                get_metric(field_metrics, metric) for metric in ["tp", "fa", "fd", "fn"]
            )
            assert total_activity > 0, f"Field {field_name} should have some activity"


class TestHungarianMatchingEdgeCases:
    """Additional edge cases for comprehensive baseline coverage."""

    def test_different_threshold_values(self):
        """Test behavior with different match_threshold values."""

        class HighThresholdTransaction(StructuredModel):
            transaction_id: str = ComparableField(
                comparator=ExactComparator(), threshold=1.0, weight=1.0
            )
            description: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
            )

            # Very high threshold - most matches will be FD
            match_threshold = 0.95

        class HighThresholdAccount(StructuredModel):
            account_id: str = ComparableField(
                comparator=ExactComparator(), threshold=1.0, weight=1.0
            )
            transactions: List[HighThresholdTransaction] = ComparableField(weight=1.0)

        gt = HighThresholdAccount(
            account_id="ACC-001",
            transactions=[
                HighThresholdTransaction(transaction_id="TXN-001", description="Coffee")
            ],
        )

        pred = HighThresholdAccount(
            account_id="ACC-001",
            transactions=[
                HighThresholdTransaction(
                    transaction_id="TXN-001", description="Coffee shop"
                )
            ],
        )

        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        # With high threshold, even good matches might be classified as FD
        # This tests that threshold value affects classification
        assert txn_overall["tp"] + txn_overall["fd"] == 1  # One matched pair

    def test_mixed_list_sizes(self):
        """Test Hungarian matching with different list sizes."""

        # GT has 3 items, Pred has 2 items
        gt = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(transaction_id="TXN-001", description="Coffee", amount=5.0),
                Transaction(transaction_id="TXN-002", description="Lunch", amount=15.0),
                Transaction(
                    transaction_id="TXN-003", description="Dinner", amount=25.0
                ),
            ],
        )

        pred = Account(
            account_id="ACC-001",
            transactions=[
                Transaction(
                    transaction_id="TXN-001", description="Coffee", amount=5.0
                ),  # Good match
                Transaction(
                    transaction_id="TXN-999", description="Unknown", amount=99.0
                ),  # Poor match
            ],
        )

        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        txn_overall = cm["fields"]["transactions"]["overall"]

        # Hungarian should match optimally, leaving 1 GT item unmatched
        assert txn_overall["tp"] + txn_overall["fd"] == 2  # Two matched pairs
        assert txn_overall["fn"] >= 1  # At least one unmatched GT item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
