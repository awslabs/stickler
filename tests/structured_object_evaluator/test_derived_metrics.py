# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""
Test the derived metrics post-processing functionality.
"""

import pytest
import json
from typing import List, Optional
from pydantic import Field
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


# Define hierarchical test models
class Attribute(StructuredModel):
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.99, weight=1.0
    )


class Product(StructuredModel):
    product_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )

    price: float = ComparableField(threshold=0.9, weight=1.0)

    match_threshold = 0.6  # Updated from List[Product] field
    attributes: Optional[List[Attribute]] = ComparableField()


class Order(StructuredModel):
    order_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )

    customer_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    products: List[Product] = ComparableField(weight=3.0)


class TestDerivedMetrics:
    """Test suite for derived metrics post-processing functionality."""

    def test_derived_metrics_simple(self):
        """Test derived metrics on a simple case."""

        class SimpleModel(StructuredModel):
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
            )
            value: int = ComparableField(threshold=0.9, weight=1.0)

        gt = SimpleModel(name="test", value=100)
        pred = SimpleModel(name="test", value=100)

        # Get clean recursive result
        clean_result = gt.compare_recursive(pred)

        # Add derived metrics
        result_with_derived = gt._add_derived_metrics_to_result(clean_result)

        # Validate structure
        assert "overall" in result_with_derived
        assert "derived" in result_with_derived["overall"]
        assert "fields" in result_with_derived

        # Access field metrics which should be in "overall" sub-key
        name_field = result_with_derived["fields"]["name"]
        value_field = result_with_derived["fields"]["value"]

        # Check if the field has "overall" key (hierarchical structure)
        if "overall" in name_field:
            assert "derived" in name_field["overall"]
        else:
            assert "derived" in name_field

        if "overall" in value_field:
            assert "derived" in value_field["overall"]
        else:
            assert "derived" in value_field

        # Check derived metrics content
        overall_derived = result_with_derived["overall"]["derived"]
        assert "cm_precision" in overall_derived
        assert "cm_recall" in overall_derived
        assert "cm_f1" in overall_derived
        assert "cm_accuracy" in overall_derived

    def test_derived_metrics_hierarchical(self):
        """Test derived metrics on hierarchical structures."""

        gt_order = Order(
            order_id="ORD-001",
            customer_name="Test Customer",
            products=[
                Product(
                    product_id="PROD-001",
                    name="Test Product",
                    price=100.0,
                    attributes=[Attribute(name="attr1"), Attribute(name="attr2")],
                )
            ],
        )

        pred_order = Order(
            order_id="ORD-001",
            customer_name="Test Customer",
            products=[
                Product(
                    product_id="PROD-001",
                    name="Test Product",
                    price=100.0,
                    attributes=[
                        Attribute(name="attr1"),
                        Attribute(name="attr3"),  # Different attribute
                    ],
                )
            ],
        )

        # Get clean recursive result
        clean_result = gt_order.compare_recursive(pred_order)

        # Add derived metrics
        result_with_derived = gt_order._add_derived_metrics_to_result(clean_result)

        # Validate overall structure
        assert "overall" in result_with_derived
        assert "derived" in result_with_derived["overall"]

        # Validate fields structure
        assert "fields" in result_with_derived
        assert "products" in result_with_derived["fields"]

        # Check products field has hierarchical structure with derived metrics
        products_field = result_with_derived["fields"]["products"]
        assert "overall" in products_field
        assert "derived" in products_field["overall"]
        assert "fields" in products_field

        # Check attributes field has hierarchical structure with derived metrics
        attributes_field = products_field["fields"]["attributes"]
        assert "overall" in attributes_field
        assert "derived" in attributes_field["overall"]
        assert "fields" in attributes_field
        assert "name" in attributes_field["fields"]

        # Updated to expect new structure with "overall" containing derived metrics
        name_field = attributes_field["fields"]["name"]
        if "overall" in name_field:
            assert "derived" in name_field["overall"]
        else:
            assert "derived" in name_field  # Fallback for direct metrics

    def test_derived_metrics_validation(self):
        """Test that derived metrics are calculated correctly."""

        class SimpleModel(StructuredModel):
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
            )

        # Create case with known metrics: 1 TP, 1 FN
        gt = SimpleModel(name="test")
        pred = SimpleModel(name="different")

        # Get clean recursive result
        clean_result = gt.compare_recursive(pred)
        result_with_derived = gt._add_derived_metrics_to_result(clean_result)

        # Check the name field should have FD (false discovery) due to mismatch
        name_field = result_with_derived["fields"]["name"]

        # The metrics may be in the "overall" sub-key for hierarchical structure
        if "overall" in name_field:
            metrics = name_field["overall"]
        else:
            metrics = name_field

        assert metrics["tp"] == 0
        assert metrics["fd"] == 1
        assert metrics["fp"] == 1

        # Check derived metrics
        if "overall" in name_field and "derived" in name_field["overall"]:
            name_derived = name_field["overall"]["derived"]
        else:
            name_derived = name_field["derived"]
        assert name_derived["cm_precision"] == 0.0  # TP/(TP+FP) = 0/(0+1) = 0
        assert name_derived["cm_recall"] == 0.0  # TP/(TP+FN) = 0/(0+1) = 0
        assert name_derived["cm_f1"] == 0.0  # 2*P*R/(P+R) = 0 when P=R=0

    def test_derived_metrics_integration(self):
        """Test that derived metrics work with compare_with method."""

        class SimpleModel(StructuredModel):
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
            )
            value: int = ComparableField(threshold=0.9, weight=1.0)

        gt = SimpleModel(name="test", value=100)
        pred = SimpleModel(name="test", value=200)  # Value doesn't match

        # Test with derived metrics enabled
        result = gt.compare_with(
            pred, include_confusion_matrix=True, add_derived_metrics=True
        )

        # Should have confusion matrix with derived metrics
        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]

        # Overall should have derived metrics
        assert "overall" in cm
        assert "derived" in cm["overall"]

        # Fields should have derived metrics
        assert "fields" in cm
        assert "name" in cm["fields"]
        assert "value" in cm["fields"]

        # Check if metrics are in "overall" key or directly in the field
        name_field = cm["fields"]["name"]
        value_field = cm["fields"]["value"]

        if "overall" in name_field:
            assert "derived" in name_field["overall"]
        else:
            assert "derived" in name_field

        if "overall" in value_field:
            assert "derived" in value_field["overall"]
        else:
            assert "derived" in value_field

        # Validate specific metrics
        name_field = cm["fields"]["name"]
        value_field = cm["fields"]["value"]

        if "overall" in name_field:
            name_metrics = name_field["overall"]
            name_derived = name_field["overall"]["derived"]
        else:
            name_metrics = name_field
            name_derived = name_field["derived"]

        if "overall" in value_field:
            value_metrics = value_field["overall"]
            value_derived = value_field["overall"]["derived"]
        else:
            value_metrics = value_field
            value_derived = value_field["derived"]

        # Name should match (TP=1)
        assert name_metrics["tp"] == 1
        assert name_derived["cm_precision"] == 1.0
        assert name_derived["cm_recall"] == 1.0

        # Value should not match (FD=1)
        assert value_metrics["fd"] == 1
        assert value_derived["cm_precision"] == 0.0
        assert value_derived["cm_recall"] == 0.0
