# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""
Test case for notebook regression where Attributes.name TP count should be 2.

This test reproduces the exact scenario from testing_nested_object_results.ipynb
where the Attributes.name field should have TP=2 but is currently broken.
"""

import pytest
from typing import List, Optional

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


class Attribute(StructuredModel):
    """Attribute model from notebook example."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.99, weight=1.0
    )


class Product(StructuredModel):
    """Product model from notebook example."""

    product_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )

    price: float = ComparableField(threshold=0.9, weight=1.0)

    # Override the default match threshold for Products
    match_threshold = 0.75
    attributes: Optional[List[Attribute]] = ComparableField()


class Order(StructuredModel):
    """Order model from notebook example."""

    order_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )

    customer_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    products: List[Product] = ComparableField(weight=3.0)


def test_notebook_attributes_name_tp_count():
    """Test that reproduces the notebook scenario where Attributes.name TP should be 2.

    CRITICAL REGRESSION: This test should pass but currently fails.
    The Attributes.name field should have TP=2 because:
    - GT has 3 products, each with 1 attribute with name="one two"
    - Pred has 2 products, each with 1 attribute with name="one two"
    - Hungarian matching matches 2 products (PROD-001 and PROD-003)
    - Each matched product has matching attributes with name="one two"
    - Therefore: Attributes.name should have TP=2
    """

    # Create GT order (3 products, each with 1 attribute)
    gt_order = Order(
        order_id="ORD-12345",
        customer_name="Jane Smith",
        products=[
            Product(
                product_id="PROD-001",
                name="Laptop Computer",
                price=999.9,
                attributes=[Attribute(name="one two")],
            ),
            Product(
                product_id="PROD-002",
                name="Wireless Mouse",
                price=29.99,
                attributes=[Attribute(name="one two")],
            ),
            Product(
                product_id="PROD-003",
                name="HDMI Cable",
                price=14.99,
                attributes=[Attribute(name="one two")],
            ),
        ],
    )

    # Create prediction order (missing_item case: 2 products, each with 1 attribute)
    pred_order = Order(
        order_id="ORD-12345",
        customer_name="Jane Smith",
        products=[
            Product(
                product_id="PROD-001",
                name="Laptop",
                price=999.99,
                attributes=[Attribute(name="one two")],
            ),
            Product(
                product_id="PROD-003",
                name="HDMI Cable",
                price=14.99,
                attributes=[Attribute(name="one two")],
            ),
            # PROD-002 is missing
        ],
    )

    # Compare with confusion matrix
    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    # Navigate to the attributes.name field in the confusion matrix
    # The path should be: confusion_matrix -> fields -> products.attributes.name
    cm = result["confusion_matrix"]
    products_fields = cm["fields"]["products"]["fields"]

    # Check if attributes field exists and has nested name field
    if "attributes" in products_fields:
        attributes_field = products_fields["attributes"]

        # Check if attributes has nested fields structure
        if "fields" in attributes_field and "name" in attributes_field["fields"]:
            name_metrics = attributes_field["fields"]["name"]

            # Helper function to get metrics from new aggregate structure
            def get_metric(field_data, metric):
                if "overall" in field_data:
                    return field_data["overall"][metric]
                return field_data.get(metric, 0)

            # CRITICAL: This should be 2 but is currently broken
            actual_tp = get_metric(name_metrics, "tp")

            print(f"DEBUG: Attributes.name TP count: {actual_tp}")
            print(f"DEBUG: Full name metrics: {name_metrics}")

            assert actual_tp == 2, (
                f"Expected Attributes.name TP=2, got {actual_tp}. "
                f"This indicates a regression in nested list attribute counting. "
                f"Full metrics: {name_metrics}"
            )
        else:
            pytest.fail(
                "attributes field does not have expected nested fields.name structure"
            )
    else:
        pytest.fail("products.attributes field not found in confusion matrix")


def test_debug_nested_structure():
    """Debug test to understand the current structure."""

    # Simplified case for debugging
    gt_order = Order(
        order_id="ORD-12345",
        customer_name="Jane Smith",
        products=[
            Product(
                product_id="PROD-001",
                name="Laptop",
                price=999.99,
                attributes=[Attribute(name="test")],
            )
        ],
    )

    pred_order = Order(
        order_id="ORD-12345",
        customer_name="Jane Smith",
        products=[
            Product(
                product_id="PROD-001",
                name="Laptop",
                price=999.99,
                attributes=[Attribute(name="test")],
            )
        ],
    )

    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    print("\n=== DEBUG: STRUCTURE ANALYSIS ===")
    print(f"CM keys: {list(result['confusion_matrix'].keys())}")
    print(f"CM fields keys: {list(result['confusion_matrix']['fields'].keys())}")

    if "products" in result["confusion_matrix"]["fields"]:
        products_field = result["confusion_matrix"]["fields"]["products"]
        print(f"Products field keys: {list(products_field.keys())}")

        if "fields" in products_field:
            products_nested = products_field["fields"]
            print(f"Products nested field keys: {list(products_nested.keys())}")

            if "attributes" in products_nested:
                attributes_field = products_nested["attributes"]
                print(f"Attributes field structure: {attributes_field}")

                if "fields" in attributes_field:
                    print(
                        f"Attributes nested fields: {list(attributes_field['fields'].keys())}"
                    )
                    if "name" in attributes_field["fields"]:
                        print(
                            f"Attributes.name metrics: {attributes_field['fields']['name']}"
                        )


if __name__ == "__main__":
    # Run the debug test first to understand the structure
    test_debug_nested_structure()
    print("\n" + "=" * 50)

    # Then run the actual test that should fail
    try:
        test_notebook_attributes_name_tp_count()
        print("✓ Test passed - regression is fixed!")
    except AssertionError as e:
        print(f"✗ Test failed - regression confirmed: {e}")
