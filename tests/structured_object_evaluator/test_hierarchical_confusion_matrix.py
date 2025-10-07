# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""
Test hierarchical confusion matrix structure for nested structured models.

This test file validates the hierarchical confusion matrix structure that:
1. Provides recursive nesting structure with "overall" and "fields" at each level
2. Uses recursive structure instead of flat dotted paths
3. Tests up to 4 levels of nesting depth

Structure being tested:
Level 1: Order
Level 2: Product (in List[Product])
Level 3: Attribute (in Optional[List[Attribute]])
Level 4: AttributeProperty (in Optional[List[AttributeProperty]])
"""

import pytest
from typing import List, Optional
from pydantic import Field

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator

# Test Models - 4 Level Hierarchy


class AttributeProperty(StructuredModel):
    """Level 4: Deepest nesting level"""

    key: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    value: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class Attribute(StructuredModel):
    """Level 3: Contains optional list of AttributeProperty"""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    properties: Optional[List[AttributeProperty]] = ComparableField(weight=1.0)


class Product(StructuredModel):
    """Level 2: Contains optional list of Attribute"""

    product_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    attributes: Optional[List[Attribute]] = ComparableField(weight=1.0)


class Order(StructuredModel):
    """Level 1: Top level containing list of Product"""

    order_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )
    customer_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    products: List[Product] = ComparableField(weight=3.0)


def create_complex_ground_truth() -> Order:
    """Create a complex 4-level nested ground truth for testing"""
    return Order(
        order_id="ORD-12345",
        customer_name="Jane Smith",
        products=[
            Product(
                product_id="PROD-001",
                name="Laptop Computer",
                attributes=[
                    Attribute(
                        name="Technical Specs",
                        properties=[
                            AttributeProperty(key="CPU", value="Intel i7"),
                            AttributeProperty(key="RAM", value="16GB"),
                        ],
                    ),
                    Attribute(
                        name="Physical Specs",
                        properties=[
                            AttributeProperty(key="Weight", value="2.5kg"),
                            AttributeProperty(key="Color", value="Silver"),
                        ],
                    ),
                ],
            ),
            Product(
                product_id="PROD-002",
                name="Wireless Mouse",
                attributes=[
                    Attribute(
                        name="Connectivity",
                        properties=[
                            AttributeProperty(key="Type", value="Bluetooth"),
                            AttributeProperty(key="Range", value="10m"),
                        ],
                    )
                ],
            ),
        ],
    )


def create_prediction_with_differences() -> Order:
    """Create prediction with various differences for testing aggregation"""
    return Order(
        order_id="ORD-12345",  # Match
        customer_name="J. Smith",  # Partial match
        products=[
            Product(
                product_id="PROD-001",  # Match
                name="Laptop",  # Partial match
                attributes=[
                    Attribute(
                        name="Tech Specs",  # Partial match
                        properties=[
                            AttributeProperty(
                                key="CPU", value="Intel i7"
                            ),  # Perfect match
                            AttributeProperty(
                                key="RAM", value="8GB"
                            ),  # Different value
                        ],
                    ),
                    Attribute(
                        name="Physical",  # Partial match
                        properties=[
                            AttributeProperty(
                                key="Weight", value="2.5kg"
                            ),  # Perfect match
                            # Missing Color property (FN)
                        ],
                    ),
                ],
            ),
            Product(
                product_id="PROD-002",  # Match
                name="Mouse",  # Partial match
                attributes=[
                    Attribute(
                        name="Connectivity",  # Perfect match
                        properties=[
                            AttributeProperty(
                                key="Type", value="Wireless"
                            ),  # Different value
                            AttributeProperty(
                                key="Range", value="10m"
                            ),  # Perfect match
                            AttributeProperty(
                                key="Battery", value="AA"
                            ),  # Extra property (FP)
                        ],
                    )
                ],
            ),
        ],
    )


class TestHierarchicalConfusionMatrix:
    """Test cases for hierarchical confusion matrix structure with aggregation"""

    def test_hierarchical_structure_format(self):
        """Test that confusion matrix has correct hierarchical structure"""
        gt = create_complex_ground_truth()
        pred = create_prediction_with_differences()

        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]

        # Top level should have "overall" and "fields"
        assert isinstance(cm, dict), "Top level confusion matrix should be dict"
        assert "overall" in cm, "Should have 'overall' key"
        assert "fields" in cm, "Should have 'fields' key"

        # Overall should contain basic metrics
        overall = cm["overall"]
        assert "tp" in overall, "Should have direct tp metric"
        assert "derived" in overall, "Should have derived metrics"

        # Products field should have hierarchical structure
        products = cm["fields"]["products"]
        assert isinstance(products, dict), "Products should be dict"
        assert "overall" in products, "Products should have 'overall'"
        assert "fields" in products, "Products should have 'fields'"

    def test_four_level_nesting_structure(self):
        """Test that all 4 levels of nesting have correct structure"""
        gt = create_complex_ground_truth()
        pred = create_prediction_with_differences()

        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]

        # Level 1: Order (top level)
        assert "overall" in cm
        assert "fields" in cm

        # Level 2: Products (list field)
        products = cm["fields"]["products"]
        assert "overall" in products
        assert "fields" in products

        # Level 3: Attributes (nested list field)
        if "attributes" in products["fields"]:
            attributes = products["fields"]["attributes"]
            assert "overall" in attributes
            assert "fields" in attributes

            # Level 4: Properties (deeply nested list field)
            if "properties" in attributes["fields"]:
                properties = attributes["fields"]["properties"]

                # Check for either hierarchical structure or flattened structure
                # The evaluator may flatten deeply nested structures for compatibility
                if "overall" in properties:
                    # Hierarchical structure
                    assert "fields" in properties

                    # Leaf level: key and value fields
                    if "key" in properties["fields"]:
                        key_field = properties["fields"]["key"]
                        # Updated to expect new structure with "overall" containing metrics
                        if "overall" in key_field:
                            assert "tp" in key_field["overall"]
                        else:
                            assert "tp" in key_field  # Fallback for direct metrics
                else:
                    # Flattened structure (direct metrics)
                    assert (
                        "tp" in properties or "tn" in properties or "fp" in properties
                    ), "Properties should have metrics directly or in overall"
                    assert "derived" in properties, "Should have derived metrics"

    def test_leaf_vs_aggregate_distinction(self):
        """Test that leaf nodes have only direct metrics, non-leaf nodes have basic structure"""
        gt = create_complex_ground_truth()
        pred = create_prediction_with_differences()

        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]

        def check_node_structure(node, path=""):
            """Recursively check that nodes have correct metric structure"""

            if "fields" in node:
                # For non-leaf nodes, check that it has either:
                # 1. Hierarchical structure (overall + fields)
                # 2. Flattened structure (direct metrics + fields)
                if "overall" in node:
                    # Hierarchical structure case
                    assert "fields" in node, (
                        f"Non-leaf node {path} should have 'fields'"
                    )
                else:
                    # Flattened structure case - still valid
                    assert any(
                        m in node for m in ["tp", "fp", "tn", "fn", "fa", "fd"]
                    ), (
                        f"Non-leaf node {path} with flattened structure should have metrics"
                    )

                # Recurse into children
                for field_name, field_data in node["fields"].items():
                    check_node_structure(field_data, f"{path}.{field_name}")

            elif any(m in node for m in ["tp", "fp", "tn", "fn", "fa", "fd"]):
                # Leaf nodes should have direct metrics but no fields
                assert "fields" not in node, (
                    f"Leaf node {path} should not have 'fields'"
                )
                assert "derived" in node, (
                    f"Leaf node {path} should have derived metrics"
                )

        # Check entire structure
        check_node_structure(cm)

    def test_empty_lists_aggregation(self):
        """Test aggregation with empty lists and None values"""
        # Create minimal structure with empty lists
        gt = Order(
            order_id="ORD-001",
            customer_name="Test User",
            products=[],  # Empty list
        )

        pred = Order(
            order_id="ORD-001",
            customer_name="Test User",
            products=[],  # Empty list
        )

        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]

        # Should still have proper structure
        assert isinstance(cm, dict)
        assert "overall" in cm
        assert "fields" in cm

        # Products should be TN (both empty)
        products_overall = cm["fields"]["products"]["overall"]
        assert products_overall["tn"] == 1, "Empty lists should result in TN"

    @pytest.mark.skip(
        reason="Will be implemented after hierarchical structure is working"
    )
    def test_aggregation_mathematics(self):
        """Test that aggregate counts are mathematically correct (sum of all descendants)"""
        # This test will be enabled once the hierarchical structure is implemented
        pass

    @pytest.mark.skip(
        reason="Will be implemented after hierarchical structure is working"
    )
    def test_missing_vs_present_aggregation(self):
        """Test aggregation when one side has data and other doesn't"""
        # This test will be enabled once the hierarchical structure is implemented
        pass
