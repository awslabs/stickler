

"""Test to verify the single traversal optimization works correctly."""

import time
from typing import List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator


class TestSingleTraversalOptimization:
    """Test suite for single traversal optimization."""

    def test_single_traversal_verification(self):
        """Test that single traversal produces consistent field_scores and confusion_matrix."""

        class SimpleProduct(StructuredModel):
            product_id: str = ComparableField(
                default="",
                comparator=LevenshteinComparator(),
                threshold=0.8,
                weight=2.0,
            )
            name: str = ComparableField(
                default="",
                comparator=LevenshteinComparator(),
                threshold=0.7,
                weight=1.0,
            )
            price: float = ComparableField(
                default=0.0,
                comparator=LevenshteinComparator(),
                threshold=0.9,
                weight=1.0,
            )

        # Create test instances with partial matches
        gt_product = SimpleProduct(product_id="PROD-123", name="Widget A", price=29.99)
        pred_product = SimpleProduct(
            product_id="PROD-123", name="Widget B", price=29.99
        )

        # Test single traversal approach
        result = gt_product.compare_with(
            pred_product, include_confusion_matrix=True, add_derived_metrics=True
        )

        # Verify we have both field_scores and confusion_matrix
        assert "field_scores" in result
        assert "confusion_matrix" in result
        assert "overall_score" in result
        assert "all_fields_matched" in result

        # Extract scores from both places
        field_scores = result["field_scores"]
        confusion_matrix = result["confusion_matrix"]
        overall_score = result["overall_score"]
        all_fields_matched = result["all_fields_matched"]

        # Verify field scores are consistent with confusion matrix scoring data
        for field_name, field_score in field_scores.items():
            cm_field = confusion_matrix["fields"][field_name]
            assert "threshold_applied_score" in cm_field
            cm_score = cm_field["threshold_applied_score"]

            # Scores should be identical from single traversal
            assert abs(field_score - cm_score) < 1e-10, (
                f"Field {field_name}: field_score={field_score}, cm_score={cm_score}"
            )

        # Verify overall score consistency
        cm_overall_score = confusion_matrix["overall"]["similarity_score"]
        assert abs(overall_score - cm_overall_score) < 1e-10, (
            f"Overall score mismatch: {overall_score} vs {cm_overall_score}"
        )

        # Verify all_fields_matched consistency
        cm_all_fields_matched = confusion_matrix["overall"]["all_fields_matched"]
        assert all_fields_matched == cm_all_fields_matched, (
            f"All fields matched mismatch: {all_fields_matched} vs {cm_all_fields_matched}"
        )

        # Verify we have the expected scores for this test case
        # product_id: perfect match (1.0) >= 0.8 threshold -> 1.0
        # name: partial match (0.8888...) >= 0.7 threshold -> 0.8888...
        # price: perfect match (1.0) >= 0.9 threshold -> 1.0
        assert field_scores["product_id"] == 1.0
        assert field_scores["name"] > 0.7  # Above threshold
        assert field_scores["price"] == 1.0

        # Verify derived metrics exist in confusion matrix
        assert "derived" in confusion_matrix["overall"]
        assert "cm_precision" in confusion_matrix["overall"]["derived"]
        assert "cm_recall" in confusion_matrix["overall"]["derived"]
        assert "cm_f1" in confusion_matrix["overall"]["derived"]

    def test_performance_comparison_demo(self):
        """Demonstrate that we're getting everything in a single pass."""

        class ComplexProduct(StructuredModel):
            match_threshold = 0.6  # Moved from List[ComplexProduct] field

            product_id: str = ComparableField(
                default="",
                comparator=LevenshteinComparator(),
                threshold=0.8,
                weight=2.0,
            )
            name: str = ComparableField(
                default="",
                comparator=LevenshteinComparator(),
                threshold=0.7,
                weight=1.0,
            )
            price: float = ComparableField(
                default=0.0,
                comparator=LevenshteinComparator(),
                threshold=0.9,
                weight=1.0,
            )

        class Order(StructuredModel):
            order_id: str = ComparableField(
                default="",
                comparator=LevenshteinComparator(),
                threshold=0.8,
                weight=2.0,
            )
            customer_name: str = ComparableField(
                default="",
                comparator=LevenshteinComparator(),
                threshold=0.7,
                weight=1.0,
            )
            products: List[ComplexProduct] = ComparableField(default=[], weight=3.0)

        # Create complex nested structures
        gt_order = Order(
            order_id="ORD-123",
            customer_name="John Doe",
            products=[
                ComplexProduct(product_id="PROD-A", name="Widget", price=29.99),
                ComplexProduct(product_id="PROD-B", name="Gadget", price=19.99),
            ],
        )

        pred_order = Order(
            order_id="ORD-123",
            customer_name="John Doe",
            products=[
                ComplexProduct(product_id="PROD-A", name="Widget", price=29.99),
                ComplexProduct(
                    product_id="PROD-C", name="Thingamajig", price=39.99
                ),  # Different product
            ],
        )

        # Single traversal gets everything at once
        start_time = time.time()

        result = gt_order.compare_with(
            pred_order, include_confusion_matrix=True, add_derived_metrics=True
        )

        end_time = time.time()
        duration = end_time - start_time

        # Verify we got comprehensive results from single traversal
        assert "field_scores" in result
        assert "confusion_matrix" in result
        assert "overall_score" in result

        # Verify hierarchical structure captured correctly
        cm = result["confusion_matrix"]
        assert "overall" in cm
        assert "fields" in cm
        assert "products" in cm["fields"]
        assert "overall" in cm["fields"]["products"]  # Hierarchical structure
        assert "fields" in cm["fields"]["products"]  # Nested fields captured

        # Performance should be reasonable (less than 1 second for this small example)
        assert duration < 1.0, f"Single traversal took too long: {duration:.3f}s"

        # Should have some similarity score
        assert result["overall_score"] > 0.0, "Should have some similarity"
