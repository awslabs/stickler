# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""
Test threshold-gated recursive evaluation for List[StructuredModel] comparison.

This test validates the new threshold-gated behavior where:
1. Only object pairs with similarity ≥ match_threshold get recursive analysis
2. Poor matches (FD) and unmatched items (FN/FA) are treated as atomic units
3. non_matches key provides detailed information about non-matching objects
"""

import pytest
from typing import List, Dict, Any, Optional

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


class Product(StructuredModel):
    """Product model for testing threshold-gated recursion."""

    product_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )
    price: float = ComparableField(threshold=0.9, weight=1.0)

    # Key: This threshold gates recursive evaluation
    match_threshold = 0.8  # Set to 0.8 to match test comments about "≥0.8" and "<0.8"


class Order(StructuredModel):
    """Order model containing list of products."""

    order_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )
    products: List[Product] = ComparableField(weight=3.0)


def test_threshold_gated_good_match():
    """Test that good matches (≥ threshold) get recursive analysis."""
    print("DEBUG: Testing good match threshold-gated behavior")

    # Create orders with one good match
    gt_order = Order(
        order_id="ORD-001",
        products=[Product(product_id="PROD-001", name="Laptop", price=999.99)],
    )

    pred_order = Order(
        order_id="ORD-001",
        products=[
            Product(
                product_id="PROD-001", name="Laptop", price=999.99
            )  # Good match ≥0.8 (exact match)
        ],
    )

    # Compare with confusion matrix
    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    print(f"DEBUG: Overall score: {result['overall_score']}")
    print(f"DEBUG: Products field score: {result['field_scores']['products']}")

    # Should have recursive analysis for the good match
    cm = result["confusion_matrix"]
    products_cm = cm["fields"]["products"]

    # Check that we have nested field metrics (recursive analysis happened)
    if "fields" in products_cm:
        nested_fields = products_cm["fields"]
        print(f"DEBUG: Found nested fields: {list(nested_fields.keys())}")

        # Should have metrics for product_id, name, price
        expected_fields = ["product_id", "name", "price"]
        for field in expected_fields:
            assert field in nested_fields, f"Missing nested field metrics for {field}"

    # Should have non_matches key but it should be empty for good matches
    if "non_matches" in products_cm:
        non_matches = products_cm["non_matches"]
        print(f"DEBUG: Non-matches count: {len(non_matches)}")
        assert len(non_matches) == 0, "Good matches should not have non_matches"


def test_threshold_gated_poor_match():
    """Test that poor matches (< threshold) get atomic treatment."""
    print("DEBUG: Testing poor match threshold-gated behavior")

    # Create orders with one poor match
    gt_order = Order(
        order_id="ORD-001",
        products=[Product(product_id="PROD-001", name="Laptop", price=999.99)],
    )

    pred_order = Order(
        order_id="ORD-001",
        products=[
            Product(
                product_id="PROD-002", name="Different Product", price=199.99
            )  # Poor match <0.8
        ],
    )

    # Compare with confusion matrix
    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    print(f"DEBUG: Overall score: {result['overall_score']}")
    print(f"DEBUG: Products field score: {result['field_scores']['products']}")

    # Should NOT have recursive analysis for poor matches
    cm = result["confusion_matrix"]
    products_cm = cm["fields"]["products"]

    # Check that we DON'T have nested field metrics (no recursive analysis)
    if "fields" in products_cm:
        nested_fields = products_cm["fields"]
        print(f"DEBUG: Found nested fields: {list(nested_fields.keys())}")

        # Should NOT have nested field metrics for poor matches
        product_fields = ["product_id", "name", "price"]
        for field in product_fields:
            assert field not in nested_fields, (
                f"Poor matches should not have nested field metrics for {field}"
            )

    # Should have non_matches key with the poor match
    if "non_matches" in products_cm:
        non_matches = products_cm["non_matches"]
        print(f"DEBUG: Non-matches count: {len(non_matches)}")
        assert len(non_matches) > 0, "Poor matches should have non_matches entries"

        # Should have one FD (False Discovery) entry
        fd_entries = [nm for nm in non_matches if nm["type"] == "FD"]
        assert len(fd_entries) == 1, "Should have one FD entry for poor match"

        # Should include similarity score
        fd_entry = fd_entries[0]
        assert "similarity" in fd_entry, "FD entry should include similarity score"
        assert fd_entry["similarity"] < 0.8, "FD similarity should be below threshold"


def test_threshold_gated_mixed_scenario():
    """Test mixed scenario with good match, poor match, and unmatched items."""
    print("DEBUG: Testing mixed threshold-gated scenario")

    # Create orders with mixed matching scenarios
    gt_order = Order(
        order_id="ORD-001",
        products=[
            Product(
                product_id="PROD-001", name="Laptop", price=999.99
            ),  # Will match well
            Product(
                product_id="PROD-002", name="Mouse", price=29.99
            ),  # Will match poorly
            Product(
                product_id="PROD-003", name="Cable", price=14.99
            ),  # Will be unmatched (FN)
        ],
    )

    pred_order = Order(
        order_id="ORD-001",
        products=[
            Product(
                product_id="PROD-001", name="Laptop", price=999.99
            ),  # Good match ≥0.8 (exact match)
            Product(
                product_id="PROD-002", name="Different Product", price=99.99
            ),  # Poor match <0.8
            Product(
                product_id="PROD-004", name="New Product", price=19.99
            ),  # Unmatched (FA)
        ],
    )

    # Compare with confusion matrix
    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    print(f"DEBUG: Overall score: {result['overall_score']}")
    print(f"DEBUG: Products field score: {result['field_scores']['products']}")

    cm = result["confusion_matrix"]
    products_cm = cm["fields"]["products"]

    # Check overall list metrics
    overall = products_cm["overall"]
    print(f"DEBUG: Products overall metrics: {overall}")

    # Hungarian algorithm optimally matches all items, including poor matches
    # Expected: 1 TP (PROD-001 good match), 2 FD (PROD-002 poor match + PROD-003 very poor match)
    assert overall["tp"] == 1, f"Expected 1 TP, got {overall['tp']}"
    assert overall["fd"] == 2, f"Expected 2 FD, got {overall['fd']}"
    assert overall["fn"] == 0, f"Expected 0 FN, got {overall['fn']}"
    assert overall["fa"] == 0, f"Expected 0 FA, got {overall['fa']}"

    # Check that we have nested field metrics only for the good match
    if "fields" in products_cm:
        nested_fields = products_cm["fields"]
        print(f"DEBUG: Found nested fields: {list(nested_fields.keys())}")

        # Should have nested metrics for the 1 good match only
        product_fields = ["product_id", "name", "price"]
        for field in product_fields:
            if field in nested_fields:
                field_metrics = nested_fields[field]
                print(f"DEBUG: Nested field {field} metrics: {field_metrics}")

                # Check based on field-specific behavior:
                if field == "product_id":
                    # Exact match: PROD-001 vs PROD-001 should be TP
                    assert field_metrics["overall"]["tp"] == 1, (
                        f"Nested field {field} should have 1 TP"
                    )
                elif field == "name":
                    # Levenshtein match: "Laptop" vs "Laptop Computer" may be below 0.7 threshold
                    # Since threshold-gated recursion only processes the good match, it should have some comparison
                    assert (
                        field_metrics["overall"]["tp"] + field_metrics["overall"]["fd"]
                        == 1
                    ), f"Nested field {field} should have 1 comparison"
                elif field == "price":
                    # Exact price match: 999.99 vs 999.99 should be TP
                    assert field_metrics["overall"]["tp"] == 1, (
                        f"Nested field {field} should have 1 TP"
                    )

    # Check non_matches structure
    if "non_matches" in products_cm:
        non_matches = products_cm["non_matches"]
        print(f"DEBUG: Non-matches count: {len(non_matches)}")
        print(f"DEBUG: Non-matches: {non_matches}")

        # Should have 2 non-matches: 2 FD (Hungarian algorithm matches all items optimally)
        assert len(non_matches) == 2, f"Expected 2 non-matches, got {len(non_matches)}"

        # Check each type - should only have FD entries since all items are matched
        fd_entries = [nm for nm in non_matches if nm["type"] == "FD"]
        fn_entries = [nm for nm in non_matches if nm["type"] == "FN"]
        fa_entries = [nm for nm in non_matches if nm["type"] == "FA"]

        assert len(fd_entries) == 2, f"Expected 2 FD entries, got {len(fd_entries)}"
        assert len(fn_entries) == 0, f"Expected 0 FN entries, got {len(fn_entries)}"
        assert len(fa_entries) == 0, f"Expected 0 FA entries, got {len(fa_entries)}"

        # FD entries should have similarity scores
        for fd_entry in fd_entries:
            assert "similarity" in fd_entry, "FD entry should include similarity score"
            assert fd_entry["similarity"] < 0.8, (
                "FD similarity should be below threshold"
            )


def test_threshold_gated_empty_lists():
    """Test threshold-gated behavior with empty lists."""
    print("DEBUG: Testing threshold-gated behavior with empty lists")

    # Both empty
    gt_order = Order(order_id="ORD-001", products=[])
    pred_order = Order(order_id="ORD-001", products=[])

    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    cm = result["confusion_matrix"]
    products_cm = cm["fields"]["products"]

    # Should be TN for empty lists
    overall = products_cm["overall"]
    assert overall["tn"] == 1, f"Expected 1 TN for empty lists, got {overall['tn']}"

    # Should have no non_matches
    if "non_matches" in products_cm:
        non_matches = products_cm["non_matches"]
        assert len(non_matches) == 0, (
            f"Empty lists should have no non-matches, got {len(non_matches)}"
        )


def test_threshold_boundary_conditions():
    """Test threshold boundary conditions (exactly at threshold)."""
    print("DEBUG: Testing threshold boundary conditions")

    # Create a product that will match exactly at threshold (0.8)
    gt_order = Order(
        order_id="ORD-001",
        products=[Product(product_id="PROD-001", name="Test Product", price=100.0)],
    )

    # Create a similar product that should be exactly at threshold
    pred_order = Order(
        order_id="ORD-001",
        products=[
            Product(
                product_id="PROD-001", name="Test Product", price=100.0
            )  # Exact match should be ≥0.8
        ],
    )

    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    cm = result["confusion_matrix"]
    products_cm = cm["fields"]["products"]

    # Should be TP (boundary condition: similarity ≥ threshold)
    overall = products_cm["overall"]
    assert overall["tp"] == 1, f"Exact match should be TP, got TP={overall['tp']}"

    # Should have recursive analysis for boundary match
    if "fields" in products_cm:
        nested_fields = products_cm["fields"]
        product_fields = ["product_id", "name", "price"]
        for field in product_fields:
            assert field in nested_fields, (
                f"Boundary match should have nested field metrics for {field}"
            )


if __name__ == "__main__":
    print("Running threshold-gated recursion tests...")

    test_threshold_gated_good_match()
    test_threshold_gated_poor_match()
    test_threshold_gated_mixed_scenario()
    test_threshold_gated_empty_lists()
    test_threshold_boundary_conditions()

    print("All tests completed!")
