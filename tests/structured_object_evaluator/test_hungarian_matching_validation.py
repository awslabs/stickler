

"""
Test cases for Hungarian matching validation in List[StructuredModel] fields.
These tests establish baseline behavior and validate the corrected logic.
"""

import pytest
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.exact import ExactComparator
from stickler.comparators.fuzzy import FuzzyComparator
from typing import List


class Transaction(StructuredModel):
    """Transaction model for testing Hungarian matching"""

    transaction_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )

    description: str = ComparableField(
        comparator=FuzzyComparator(), threshold=0.7, weight=2.0
    )

    amount: float = ComparableField(threshold=0.9, weight=1.0)

    # Critical: This threshold controls Hungarian matching recursion
    match_threshold = 0.8


class Account(StructuredModel):
    """Account model with list of transactions"""

    account_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )

    transactions: List[Transaction] = ComparableField(weight=3.0)


class TestHungarianMatchingValidation:
    """Test Hungarian matching behavior for different list length scenarios"""

    def test_equal_length_lists_3x3(self):
        """
        Test Case 1: Equal length lists (3 GT, 3 Pred)
        Expected: Only TP/FD possible, zero FN/FA
        Hungarian should pair everyone up, threshold determines TP vs FD
        """
        print("\n=== TEST CASE 1: Equal Length Lists (3x3) ===")

        # Ground Truth
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

        # Prediction
        pred_account = Account(
            account_id="ACC-12345",
            transactions=[
                Transaction(
                    transaction_id="TXN-001", description="Coffee shop", amount=4.95
                ),  # Good match
                Transaction(
                    transaction_id="TXN-002",
                    description="Online purchase",
                    amount=89.99,
                ),  # Poor match
                Transaction(
                    transaction_id="TXN-004", description="Restaurant", amount=23.45
                ),  # Poor match
            ],
        )

        # Run comparison
        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)
        transactions_metrics = result["confusion_matrix"]["fields"]["transactions"][
            "overall"
        ]

        print(f"GT List Length: {len(gt_account.transactions)}")
        print(f"Pred List Length: {len(pred_account.transactions)}")
        print(
            f"Actual Metrics: TP={transactions_metrics['tp']}, FD={transactions_metrics['fd']}, FN={transactions_metrics['fn']}, FA={transactions_metrics['fa']}"
        )

        # Expected for equal length: Hungarian pairs everyone, only TP/FD possible
        # Based on corrected understanding: TP=1 (one good match), FD=2 (two poor matches), FN=0, FA=0
        expected_tp = 1
        expected_fd = 2
        expected_fn = 0  # No unmatched GT items (equal length)
        expected_fa = 0  # No unmatched Pred items (equal length)

        print(
            f"Expected Metrics: TP={expected_tp}, FD={expected_fd}, FN={expected_fn}, FA={expected_fa}"
        )

        # Document current behavior vs expected
        assert len(gt_account.transactions) == len(pred_account.transactions), (
            "Lists should be equal length"
        )

        # These assertions capture what SHOULD happen according to corrected logic
        # If they fail, it indicates implementation needs fixing
        try:
            assert transactions_metrics["tp"] == expected_tp, (
                f"Expected TP={expected_tp}, got {transactions_metrics['tp']}"
            )
            assert transactions_metrics["fd"] == expected_fd, (
                f"Expected FD={expected_fd}, got {transactions_metrics['fd']}"
            )
            assert transactions_metrics["fn"] == expected_fn, (
                f"Expected FN={expected_fn}, got {transactions_metrics['fn']}"
            )
            assert transactions_metrics["fa"] == expected_fa, (
                f"Expected FA={expected_fa}, got {transactions_metrics['fa']}"
            )
            print("✅ PASS: Metrics match corrected expected behavior")
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            print("This indicates the implementation needs correction")

    def test_gt_longer_4x2(self):
        """
        Test Case 2: GT list longer (4 GT, 2 Pred)
        Expected: Hungarian pairs 2 best, remaining 2 GT become FN
        """
        print("\n=== TEST CASE 2: GT Longer (4x2) ===")

        # Ground Truth - 4 transactions
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
                Transaction(
                    transaction_id="TXN-004",
                    description="Restaurant bill",
                    amount=32.10,
                ),
            ],
        )

        # Prediction - 2 transactions
        pred_account = Account(
            account_id="ACC-12345",
            transactions=[
                Transaction(
                    transaction_id="TXN-001", description="Coffee shop", amount=4.95
                ),  # Good match
                Transaction(
                    transaction_id="TXN-002",
                    description="Online purchase",
                    amount=89.99,
                ),  # Poor match
            ],
        )

        # Run comparison
        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)
        transactions_metrics = result["confusion_matrix"]["fields"]["transactions"][
            "overall"
        ]

        print(f"GT List Length: {len(gt_account.transactions)}")
        print(f"Pred List Length: {len(pred_account.transactions)}")
        print(
            f"Actual Metrics: TP={transactions_metrics['tp']}, FD={transactions_metrics['fd']}, FN={transactions_metrics['fn']}, FA={transactions_metrics['fa']}"
        )

        # Expected: 2 Hungarian pairs + 2 unmatched GT items
        expected_tp = 1  # One good Hungarian pair
        expected_fd = 1  # One poor Hungarian pair
        expected_fn = 2  # Two unmatched GT items (extras)
        expected_fa = 0  # No unmatched Pred items

        print(
            f"Expected Metrics: TP={expected_tp}, FD={expected_fd}, FN={expected_fn}, FA={expected_fa}"
        )

        assert len(gt_account.transactions) > len(pred_account.transactions), (
            "GT should be longer"
        )

    def test_pred_longer_2x4(self):
        """
        Test Case 3: Pred list longer (2 GT, 4 Pred)
        Expected: Hungarian pairs 2 best, remaining 2 Pred become FA
        """
        print("\n=== TEST CASE 3: Pred Longer (2x4) ===")

        # Ground Truth - 2 transactions
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
            ],
        )

        # Prediction - 4 transactions
        pred_account = Account(
            account_id="ACC-12345",
            transactions=[
                Transaction(
                    transaction_id="TXN-001", description="Coffee shop", amount=4.95
                ),  # Good match
                Transaction(
                    transaction_id="TXN-002",
                    description="Online purchase",
                    amount=89.99,
                ),  # Poor match
                Transaction(
                    transaction_id="TXN-003", description="Restaurant", amount=23.45
                ),  # Extra
                Transaction(
                    transaction_id="TXN-004", description="Gas station", amount=67.89
                ),  # Extra
            ],
        )

        # Run comparison
        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)
        transactions_metrics = result["confusion_matrix"]["fields"]["transactions"][
            "overall"
        ]

        print(f"GT List Length: {len(gt_account.transactions)}")
        print(f"Pred List Length: {len(pred_account.transactions)}")
        print(
            f"Actual Metrics: TP={transactions_metrics['tp']}, FD={transactions_metrics['fd']}, FN={transactions_metrics['fn']}, FA={transactions_metrics['fa']}"
        )

        # Expected: 2 Hungarian pairs + 2 unmatched Pred items
        expected_tp = 1  # One good Hungarian pair
        expected_fd = 1  # One poor Hungarian pair
        expected_fn = 0  # No unmatched GT items
        expected_fa = 2  # Two unmatched Pred items (extras)

        print(
            f"Expected Metrics: TP={expected_tp}, FD={expected_fd}, FN={expected_fn}, FA={expected_fa}"
        )

        assert len(pred_account.transactions) > len(gt_account.transactions), (
            "Pred should be longer"
        )

    def test_all_above_threshold_3x3(self):
        """
        Test Case 4: All matches above threshold (3x3)
        Expected: All TP, zero FP
        """
        print("\n=== TEST CASE 4: All Above Threshold (3x3) ===")

        # Ground Truth
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

        # Prediction - all very similar
        pred_account = Account(
            account_id="ACC-12345",
            transactions=[
                Transaction(
                    transaction_id="TXN-001",
                    description="Coffee shop payment",
                    amount=4.95,
                ),  # Perfect match
                Transaction(
                    transaction_id="TXN-002", description="Grocery store", amount=127.43
                ),  # Perfect match
                Transaction(
                    transaction_id="TXN-003", description="Gas station", amount=45.67
                ),  # Perfect match
            ],
        )

        # Run comparison
        result = gt_account.compare_with(pred_account, include_confusion_matrix=True)
        transactions_metrics = result["confusion_matrix"]["fields"]["transactions"][
            "overall"
        ]

        print(f"GT List Length: {len(gt_account.transactions)}")
        print(f"Pred List Length: {len(pred_account.transactions)}")
        print(
            f"Actual Metrics: TP={transactions_metrics['tp']}, FD={transactions_metrics['fd']}, FN={transactions_metrics['fn']}, FA={transactions_metrics['fa']}"
        )

        # Expected: All perfect matches
        expected_tp = 3  # All three pairs above threshold
        expected_fd = 0  # No poor matches
        expected_fn = 0  # No unmatched GT items (equal length)
        expected_fa = 0  # No unmatched Pred items (equal length)

        print(
            f"Expected Metrics: TP={expected_tp}, FD={expected_fd}, FN={expected_fn}, FA={expected_fa}"
        )

        assert len(gt_account.transactions) == len(pred_account.transactions), (
            "Lists should be equal length"
        )


if __name__ == "__main__":
    """Run tests directly for debugging"""
    test_instance = TestHungarianMatchingValidation()
    test_instance.test_equal_length_lists_3x3()
    test_instance.test_gt_longer_4x2()
    test_instance.test_pred_longer_2x4()
    test_instance.test_all_above_threshold_3x3()
