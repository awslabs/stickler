#!/usr/bin/env python3
"""
Diagnostic tests to isolate the core comparison logic bug.

BUG: Batch comparison returns 94.8% when individual scores average to 66%.
This is a CRITICAL issue affecting the core functionality of the library.
"""

import pytest
from typing import List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.structured_list_comparator import (
    StructuredListComparator,
)


class Invoice(StructuredModel):
    """Test invoice model - same as Quick Start example."""

    invoice_number: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )

    vendor: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )

    total: float = ComparableField(threshold=0.95, weight=2.0)


class InvoiceBatch(StructuredModel):
    """Test batch model - same as Quick Start example."""

    batch_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )

    invoices: List[Invoice] = ComparableField(weight=3.0)


class TestCoreBugDiagnostic:
    """Diagnostic tests to isolate the 94.8% vs 66% bug."""

    def setup_method(self):
        """Set up test data that reproduces the bug."""
        # Ground truth invoices
        self.gt_invoices = [
            Invoice(
                invoice_number="INV-001", vendor="Company A", total=1000.00
            ),  # Will match perfectly: 1.0
            Invoice(
                invoice_number="INV-002", vendor="Company B", total=2000.00
            ),  # Close match: ~0.8
            Invoice(
                invoice_number="INV-003", vendor="Company C", total=1500.00
            ),  # Complete mismatch: ~0.18
        ]

        # Predicted invoices
        self.pred_invoices = [
            Invoice(
                invoice_number="INV-001", vendor="Company A", total=1000.00
            ),  # Perfect match
            Invoice(
                invoice_number="INV-002", vendor="Company B Ltd", total=2000.00
            ),  # Close match
            Invoice(
                invoice_number="INV-004", vendor="Company D", total=1800.00
            ),  # Complete mismatch
        ]

        # Full batch objects
        self.gt_batch = InvoiceBatch(batch_id="BATCH-001", invoices=self.gt_invoices)
        self.pred_batch = InvoiceBatch(
            batch_id="BATCH-001", invoices=self.pred_invoices
        )

    def test_individual_invoice_scores_baseline(self):
        """Verify individual invoice comparisons work as expected."""
        print("\n=== INDIVIDUAL INVOICE SCORES ===")

        individual_scores = []
        for i, (gt, pred) in enumerate(zip(self.gt_invoices, self.pred_invoices)):
            result = gt.compare_with(pred)
            score = result["overall_score"]
            individual_scores.append(score)
            print(f"Invoice {i + 1}: {score:.3f}")

        # Verify reasonable individual scores
        assert individual_scores[0] >= 0.95, (
            f"Perfect match should be ~1.0, got {individual_scores[0]}"
        )
        assert 0.6 <= individual_scores[1] <= 0.9, (
            f"Close match should be ~0.8, got {individual_scores[1]}"
        )
        assert individual_scores[2] <= 0.3, (
            f"Complete mismatch should be ≤0.3, got {individual_scores[2]}"
        )

        # Calculate expected average
        expected_average = sum(individual_scores) / len(individual_scores)
        print(f"Expected batch average: {expected_average:.3f}")

        # Store for use by other test methods
        self.individual_scores = individual_scores
        self.expected_average = expected_average

    def test_batch_comparison_inflated_score(self):
        """Reproduce the bug: batch comparison giving 94.8% instead of ~66%."""
        print("\n=== BATCH COMPARISON BUG ===")

        # Get individual scores first
        self.test_individual_invoice_scores_baseline()
        individual_scores, expected_average = (
            self.individual_scores,
            self.expected_average,
        )

        # Now test batch comparison
        batch_result = self.gt_batch.compare_with(self.pred_batch)
        actual_batch_score = batch_result["overall_score"]

        print(f"Individual scores: {individual_scores}")
        print(f"Expected batch score: {expected_average:.3f}")
        print(f"Actual batch score: {actual_batch_score:.3f}")
        print(f"Inflation factor: {actual_batch_score / expected_average:.2f}x")

        # THE BUG: This should fail because batch score is way too high
        assert actual_batch_score <= expected_average + 0.15, (
            f"Batch score {actual_batch_score:.3f} is way higher than expected {expected_average:.3f}"
        )

    def test_list_comparator_isolation(self):
        """Test StructuredListComparator directly to isolate the bug."""
        print("\n=== DIRECT LIST COMPARATOR TEST ===")

        try:
            comparator = StructuredListComparator()
            list_result = comparator.compare_lists(self.gt_invoices, self.pred_invoices)

            print(f"Direct list comparison score: {list_result.overall_score:.3f}")

            # Check if this is where the inflation happens
            if hasattr(list_result, "overall_score"):
                assert list_result.overall_score <= 0.8, (
                    f"List comparison score {list_result.overall_score:.3f} seems inflated"
                )

        except Exception as e:
            print(f"List comparator test failed: {e}")
            # This tells us something about the API

    def test_field_weighting_impact(self):
        """Test if the weight=3.0 on invoices field is causing the inflation."""
        print("\n=== FIELD WEIGHTING ANALYSIS ===")

        batch_result = self.gt_batch.compare_with(self.pred_batch)

        print(f"Field scores: {batch_result['field_scores']}")
        print("Field weights:")
        print("  batch_id: weight=1.0")
        print("  invoices: weight=3.0")
        print()

        # Calculate weighted average manually
        batch_id_score = batch_result["field_scores"]["batch_id"]
        invoices_score = batch_result["field_scores"]["invoices"]

        manual_weighted_avg = (batch_id_score * 1.0 + invoices_score * 3.0) / (
            1.0 + 3.0
        )
        actual_overall = batch_result["overall_score"]

        print(f"batch_id score: {batch_id_score:.3f} (weight=1.0)")
        print(f"invoices score: {invoices_score:.3f} (weight=3.0)")
        print(f"Manual weighted avg: {manual_weighted_avg:.3f}")
        print(f"Actual overall: {actual_overall:.3f}")
        print(f"Difference: {abs(manual_weighted_avg - actual_overall):.6f}")

        # Check if weighting math is consistent
        assert abs(manual_weighted_avg - actual_overall) < 0.001, (
            f"Weighted average math seems inconsistent"
        )

    def test_hungarian_matching_validation(self):
        """Verify Hungarian algorithm is matching correctly."""
        print("\n=== HUNGARIAN MATCHING VALIDATION ===")

        # We expect optimal matching to be:
        # gt[0] -> pred[0] (perfect match)
        # gt[1] -> pred[1] (close match)
        # gt[2] -> pred[2] (poor match but best available)

        from stickler.algorithms.hungarian import HungarianMatcher

        # Create similarity matrix manually
        similarity_matrix = []
        for gt_inv in self.gt_invoices:
            row = []
            for pred_inv in self.pred_invoices:
                score = gt_inv.compare_with(pred_inv)["overall_score"]
                row.append(score)
            similarity_matrix.append(row)

        print("Similarity matrix:")
        for i, row in enumerate(similarity_matrix):
            print(f"  GT[{i}]: {[f'{score:.3f}' for score in row]}")

        # Use HungarianMatcher to get assignments
        matcher = HungarianMatcher()
        assignments, _ = matcher.match(self.gt_invoices, self.pred_invoices)
        print(f"Hungarian assignments: {assignments}")

        # Verify expected matching
        expected_assignments = [(0, 0), (1, 1), (2, 2)]
        assert assignments == expected_assignments, (
            f"Hungarian matching unexpected: got {assignments}, expected {expected_assignments}"
        )

    def test_simplified_equal_weights(self):
        """Test with equal weights to see if that fixes the inflation."""
        print("\n=== SIMPLIFIED EQUAL WEIGHTS TEST ===")

        class SimpleInvoice(StructuredModel):
            """Invoice with equal weights to isolate weighting issues."""

            invoice_number: str = ComparableField(
                comparator=LevenshteinComparator(),
                threshold=0.9,
                weight=1.0,  # Equal weight
            )

            vendor: str = ComparableField(
                comparator=LevenshteinComparator(),
                threshold=0.7,
                weight=1.0,  # Equal weight
            )

            total: float = ComparableField(
                threshold=0.95,
                weight=1.0,  # Equal weight
            )

        class SimpleBatch(StructuredModel):
            """Batch with equal weights."""

            batch_id: str = ComparableField(
                comparator=LevenshteinComparator(),
                threshold=0.9,
                weight=1.0,  # Equal weight
            )

            invoices: List[SimpleInvoice] = ComparableField(
                weight=1.0  # Equal weight - NOT 3.0!
            )

        # Create simplified test data
        simple_gt_invoices = [
            SimpleInvoice(invoice_number="INV-001", vendor="Company A", total=1000.00),
            SimpleInvoice(invoice_number="INV-002", vendor="Company B", total=2000.00),
            SimpleInvoice(invoice_number="INV-003", vendor="Company C", total=1500.00),
        ]

        simple_pred_invoices = [
            SimpleInvoice(invoice_number="INV-001", vendor="Company A", total=1000.00),
            SimpleInvoice(
                invoice_number="INV-002", vendor="Company B Ltd", total=2000.00
            ),
            SimpleInvoice(invoice_number="INV-004", vendor="Company D", total=1800.00),
        ]

        simple_gt_batch = SimpleBatch(batch_id="BATCH-001", invoices=simple_gt_invoices)
        simple_pred_batch = SimpleBatch(
            batch_id="BATCH-001", invoices=simple_pred_invoices
        )

        # Test with equal weights
        result = simple_gt_batch.compare_with(simple_pred_batch)
        equal_weight_score = result["overall_score"]

        print(f"Equal weights batch score: {equal_weight_score:.3f}")
        print(
            f"Original weights batch score: {self.gt_batch.compare_with(self.pred_batch)['overall_score']:.3f}"
        )

        # Calculate expected score: (batch_id_score + invoices_score) / 2
        # batch_id matches perfectly = 1.0, invoices ≈ 0.659
        expected_equal_weight_score = (1.0 + 0.659) / 2  # ≈ 0.829

        # Allow some tolerance for the calculation
        assert abs(equal_weight_score - expected_equal_weight_score) <= 0.05, (
            f"Equal weights score {equal_weight_score:.3f} differs too much from expected {expected_equal_weight_score:.3f}"
        )


if __name__ == "__main__":
    # Run the diagnostic tests
    test_instance = TestCoreBugDiagnostic()
    test_instance.setup_method()

    print("🔬 DIAGNOSTIC TESTS FOR CORE COMPARISON BUG")
    print("=" * 60)

    try:
        test_instance.test_individual_invoice_scores_baseline()
        print("✅ Individual scores: REASONABLE")
    except Exception as e:
        print(f"❌ Individual scores: FAILED - {e}")

    try:
        test_instance.test_batch_comparison_inflated_score()
        print("✅ Batch comparison: REASONABLE")
    except Exception as e:
        print(f"❌ Batch comparison: INFLATED - {e}")

    try:
        test_instance.test_list_comparator_isolation()
        print("✅ List comparator: OK")
    except Exception as e:
        print(f"❌ List comparator: ISSUE - {e}")

    try:
        test_instance.test_field_weighting_impact()
        print("✅ Field weighting: CONSISTENT")
    except Exception as e:
        print(f"❌ Field weighting: INCONSISTENT - {e}")

    try:
        test_instance.test_hungarian_matching_validation()
        print("✅ Hungarian matching: CORRECT")
    except Exception as e:
        print(f"❌ Hungarian matching: INCORRECT - {e}")

    try:
        test_instance.test_simplified_equal_weights()
        print("✅ Equal weights: REASONABLE")
    except Exception as e:
        print(f"❌ Equal weights: STILL INFLATED - {e}")
