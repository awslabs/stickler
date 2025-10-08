"""
Unit tests for Quick Start examples to ensure realistic and intuitive scoring.

These tests validate that the examples shown to new users produce reasonable,
explainable results that match intuitive expectations.
"""

import pytest
from typing import List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


class Invoice(StructuredModel):
    """Invoice model matching the Quick Start notebook example."""

    invoice_number: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.9,  # Strict matching for invoice numbers
        weight=2.0,  # High importance
    )

    vendor: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,  # Allow some variation in vendor names
        weight=1.0,
    )

    total: float = ComparableField(
        threshold=0.95,  # Very strict for monetary amounts
        weight=2.0,  # High importance
    )


class InvoiceBatch(StructuredModel):
    """Batch model matching the Quick Start notebook example."""

    batch_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )

    invoices: List[Invoice] = ComparableField(
        weight=3.0  # Most important field
    )


class TestQuickStartExamples:
    """Test cases for Quick Start examples to ensure realistic scoring."""

    def test_individual_invoice_comparison_reasonable_scores(self):
        """Test individual invoice comparison produces reasonable scores."""

        # Test case 1: Perfect match should be 1.0
        gt1 = Invoice(invoice_number="INV-001", vendor="Company A", total=1000.00)
        pred1 = Invoice(invoice_number="INV-001", vendor="Company A", total=1000.00)

        result1 = gt1.compare_with(pred1)
        assert result1["overall_score"] == 1.0, (
            f"Perfect match should be 1.0, got {result1['overall_score']}"
        )

        # Test case 2: Slight vendor variation should be high but not perfect
        gt2 = Invoice(invoice_number="INV-002", vendor="Company B", total=2000.00)
        pred2 = Invoice(invoice_number="INV-002", vendor="Company B Ltd", total=2000.00)

        result2 = gt2.compare_with(pred2)
        # Should be high (vendor passes 0.7 threshold) but not perfect due to vendor difference
        assert 0.7 <= result2["overall_score"] <= 0.95, (
            f"Slight vendor variation should score 0.7-0.95, got {result2['overall_score']}"
        )

        # Test case 3: Complete mismatch should be very low
        gt3 = Invoice(invoice_number="INV-003", vendor="Company C", total=1500.00)
        pred3 = Invoice(invoice_number="INV-004", vendor="Company D", total=1800.00)

        result3 = gt3.compare_with(pred3)
        # All fields are different - should be very low score
        assert result3["overall_score"] <= 0.3, (
            f"Complete mismatch should score ≤0.3, got {result3['overall_score']}"
        )

    def test_batch_comparison_realistic_overall_score(self):
        """Test that batch comparison with mixed results produces realistic overall scores."""

        # Ground truth: 3 invoices
        gt_batch = InvoiceBatch(
            batch_id="BATCH-2024-001",
            invoices=[
                Invoice(invoice_number="INV-001", vendor="Company A", total=1000.00),
                Invoice(invoice_number="INV-002", vendor="Company B", total=2000.00),
                Invoice(invoice_number="INV-003", vendor="Company C", total=1500.00),
            ],
        )

        # Prediction: 1 perfect, 1 close, 1 completely wrong
        pred_batch = InvoiceBatch(
            batch_id="BATCH-2024-001",
            invoices=[
                Invoice(
                    invoice_number="INV-001", vendor="Company A", total=1000.00
                ),  # Perfect match
                Invoice(
                    invoice_number="INV-002", vendor="Company B Ltd", total=2000.00
                ),  # Close match
                Invoice(
                    invoice_number="INV-004", vendor="Company D", total=1800.00
                ),  # Complete mismatch
            ],
        )

        result = gt_batch.compare_with(pred_batch)

        # REASONING: With 1/3 of data completely wrong, overall score should be much lower than 0.9
        # Expected logic:
        # - Invoice 1: ~1.0 (perfect)
        # - Invoice 2: ~0.8 (good but not perfect due to "Company B" vs "Company B Ltd")
        # - Invoice 3: ~0.1 (terrible - everything is different)
        # - Average: (1.0 + 0.8 + 0.1) / 3 = 0.63 ≈ 60-70%

        assert result["overall_score"] <= 0.8, (
            f"Batch with 1/3 completely wrong data should score ≤0.8, got {result['overall_score']:.3f}"
        )

        # Even more conservative - it should definitely be under 85%
        assert result["overall_score"] <= 0.85, (
            f"Batch with significant errors should score ≤0.85, got {result['overall_score']:.3f}"
        )

        print(f"Batch overall score: {result['overall_score']:.3f}")
        print(f"Field scores: {result['field_scores']}")

    def test_evaluator_results_consistent_with_compare_with(self):
        """Test that StructuredModelEvaluator produces consistent results with compare_with."""

        gt_batch = InvoiceBatch(
            batch_id="BATCH-2024-001",
            invoices=[
                Invoice(invoice_number="INV-001", vendor="Company A", total=1000.00),
                Invoice(invoice_number="INV-002", vendor="Company B", total=2000.00),
                Invoice(invoice_number="INV-003", vendor="Company C", total=1500.00),
            ],
        )

        pred_batch = InvoiceBatch(
            batch_id="BATCH-2024-001",
            invoices=[
                Invoice(invoice_number="INV-001", vendor="Company A", total=1000.00),
                Invoice(
                    invoice_number="INV-002", vendor="Company B Ltd", total=2000.00
                ),
                Invoice(invoice_number="INV-004", vendor="Company D", total=1800.00),
            ],
        )

        # Compare using both methods
        compare_result = gt_batch.compare_with(pred_batch)

        evaluator = StructuredModelEvaluator()
        eval_result = evaluator.evaluate(gt_batch, pred_batch)

        # ANLS scores should be reasonably close
        anls_diff = abs(
            compare_result["overall_score"] - eval_result["overall"]["anls_score"]
        )
        assert anls_diff <= 0.1, (
            f"compare_with and evaluator ANLS should be close. "
            f"compare_with: {compare_result['overall_score']:.3f}, "
            f"evaluator: {eval_result['overall']['anls_score']:.3f}, "
            f"diff: {anls_diff:.3f}"
        )

    def test_hungarian_algorithm_behavior(self):
        """Test Hungarian algorithm matching behavior with mixed quality matches."""

        # Case where optimal matching should prefer good matches over poor ones
        gt_invoices = [
            Invoice(invoice_number="INV-001", vendor="Company A", total=1000.00),
            Invoice(invoice_number="INV-002", vendor="Company B", total=2000.00),
        ]

        pred_invoices = [
            Invoice(
                invoice_number="INV-999", vendor="Wrong Company", total=9999.00
            ),  # Poor match
            Invoice(
                invoice_number="INV-001", vendor="Company A", total=1000.00
            ),  # Perfect match
        ]

        gt_batch = InvoiceBatch(batch_id="TEST", invoices=gt_invoices)
        pred_batch = InvoiceBatch(batch_id="TEST", invoices=pred_invoices)

        result = gt_batch.compare_with(pred_batch)

        # Hungarian should match INV-001 correctly and leave INV-002 unmatched
        # This should result in a moderate score, not high
        assert result["overall_score"] <= 0.7, (
            f"Mixed quality matches should produce moderate scores, got {result['overall_score']:.3f}"
        )

    def test_field_weight_impact(self):
        """Test that field weights actually impact the overall score appropriately."""

        # Test invoice with high-weight field errors vs low-weight field errors
        gt = Invoice(invoice_number="INV-001", vendor="Company A", total=1000.00)

        # Case 1: High-weight field (invoice_number, weight=2.0) is wrong
        pred_high_weight_error = Invoice(
            invoice_number="INV-999", vendor="Company A", total=1000.00
        )
        result1 = gt.compare_with(pred_high_weight_error)

        # Case 2: Low-weight field (vendor, weight=1.0) is wrong
        pred_low_weight_error = Invoice(
            invoice_number="INV-001", vendor="Wrong Company", total=1000.00
        )
        result2 = gt.compare_with(pred_low_weight_error)

        # High-weight field error should impact score more than low-weight field error
        assert result1["overall_score"] < result2["overall_score"], (
            f"High-weight field error should impact score more. "
            f"High-weight error: {result1['overall_score']:.3f}, "
            f"Low-weight error: {result2['overall_score']:.3f}"
        )


if __name__ == "__main__":
    # Run the tests and show results
    test_instance = TestQuickStartExamples()

    print("🧪 Testing Quick Start Example Scoring...")

    try:
        test_instance.test_individual_invoice_comparison_reasonable_scores()
        print("✅ Individual invoice scoring: PASSED")
    except AssertionError as e:
        print(f"❌ Individual invoice scoring: FAILED - {e}")

    try:
        test_instance.test_batch_comparison_realistic_overall_score()
        print("✅ Batch comparison scoring: PASSED")
    except AssertionError as e:
        print(f"❌ Batch comparison scoring: FAILED - {e}")

    try:
        test_instance.test_evaluator_results_consistent_with_compare_with()
        print("✅ Evaluator consistency: PASSED")
    except AssertionError as e:
        print(f"❌ Evaluator consistency: FAILED - {e}")

    try:
        test_instance.test_hungarian_algorithm_behavior()
        print("✅ Hungarian algorithm behavior: PASSED")
    except AssertionError as e:
        print(f"❌ Hungarian algorithm behavior: FAILED - {e}")

    try:
        test_instance.test_field_weight_impact()
        print("✅ Field weight impact: PASSED")
    except AssertionError as e:
        print(f"❌ Field weight impact: FAILED - {e}")
