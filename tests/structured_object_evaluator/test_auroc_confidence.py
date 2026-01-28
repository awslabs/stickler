"""
Test cases for AUROC confidence metric calculation.
"""

from typing import List, Optional

import pytest

from stickler.comparators import LevenshteinComparator, NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class Product(StructuredModel):
    product_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )
    price: float = ComparableField(comparator=NumericComparator())
    sku: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.95)
    description: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7
    )


class Address(StructuredModel):
    street: str = ComparableField(comparator=LevenshteinComparator())
    city: str = ComparableField(comparator=LevenshteinComparator())
    zip_code: str = ComparableField(comparator=LevenshteinComparator())


class Customer(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator())
    email: str = ComparableField(comparator=LevenshteinComparator())
    address: Address = ComparableField(comparator=LevenshteinComparator())
    orders: List[Product] = ComparableField()


def test_perfect_calibration_auroc():
    """Test AUROC with well-calibrated confidence scores."""
    gt = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    pred = Product.from_json(
        {
            "product_name": {
                "value": "Widget Pro",
                "confidence": 0.95,
            },  # High conf, correct
            "price": {"value": 29.99, "confidence": 0.90},  # High conf, correct
            "sku": {"value": "XYZ789", "confidence": 0.30},  # Low conf, incorrect
            "description": {
                "value": "Good widget",
                "confidence": 0.25,
            },  # Low conf, incorrect
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert auroc > 0.8, (
        f"Expected high AUROC for well-calibrated confidence, got {auroc}"
    )


def test_poor_calibration_auroc():
    """Test AUROC with poorly-calibrated confidence scores."""
    gt = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    pred = Product.from_json(
        {
            "product_name": {
                "value": "Wrong Name",
                "confidence": 0.95,
            },  # High conf, incorrect
            "price": {"value": 50.00, "confidence": 0.90},  # High conf, incorrect
            "sku": {"value": "ABC123", "confidence": 0.20},  # Low conf, correct
            "description": {
                "value": "Perfect description",
                "confidence": 0.15,
            },  # Low conf, correct
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert auroc < 0.5, (
        f"Expected low AUROC for poorly-calibrated confidence, got {auroc}"
    )


def test_no_confidence_data():
    """Test AUROC when no confidence data is available."""
    gt = Product(product_name="Widget Pro", price=29.99, sku="ABC123")
    pred = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert auroc == 0.5, (
        f"Expected default AUROC of 0.5 when no confidence data, got {auroc}"
    )


def test_all_matches_auroc():
    """Test AUROC when all fields match (edge case)."""
    gt = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    pred = Product.from_json(
        {
            "product_name": {"value": "Widget Pro", "confidence": 0.95},
            "price": {"value": 29.99, "confidence": 0.80},
            "sku": {"value": "ABC123", "confidence": 0.70},
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert auroc == 0.5, f"Expected AUROC of 0.5 when all matches, got {auroc}"


def test_all_non_matches_auroc():
    """Test AUROC when no fields match (edge case)."""
    gt = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    pred = Product.from_json(
        {
            "product_name": {"value": "Completely Different", "confidence": 0.95},
            "price": {"value": 999.99, "confidence": 0.80},
            "sku": {"value": "WRONG", "confidence": 0.70},
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert auroc == 0.5, f"Expected AUROC of 0.5 when no matches, got {auroc}"


def test_nested_structure_auroc():
    """Test AUROC with nested StructuredModel fields."""
    gt = Customer(
        name="John Doe",
        email="john@example.com",
        address=Address(street="123 Main St", city="New York", zip_code="10001"),
        orders=[],
    )

    pred = Customer.from_json(
        {
            "name": {"value": "John Doe", "confidence": 0.95},  # High conf, correct
            "email": {
                "value": "john@example.com",
                "confidence": 0.85,
            },  # High conf, correct
            "address": {
                "street": {
                    "value": "123 Main St",
                    "confidence": 0.90,
                },  # High conf, correct
                "city": {
                    "value": "Boston",
                    "confidence": 0.70,
                },  # Medium conf, incorrect
                "zip_code": {
                    "value": "10001",
                    "confidence": 0.95,
                },  # High conf, correct
            },
            "orders": [],
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    # This is actually well-calibrated: high confidence for correct, medium confidence for incorrect
    assert auroc >= 0.8, (
        f"Expected high AUROC for well-calibrated nested confidence, got {auroc}"
    )


def test_mixed_confidence_scenarios():
    """Test AUROC with realistic mixed confidence scenarios."""
    gt = Product(
        product_name="Widget Pro", price=29.99, sku="ABC123", description="Great widget"
    )

    pred = Product.from_json(
        {
            "product_name": {
                "value": "Widget Pro",
                "confidence": 0.95,
            },  # High conf, correct
            "price": {"value": 30.50, "confidence": 0.60},  # Medium conf, close match
            "sku": {
                "value": "ABC124",
                "confidence": 0.85,
            },  # High conf, close but wrong
            "description": {
                "value": "Excellent widget",
                "confidence": 0.40,
            },  # Low conf, similar
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert 0.0 <= auroc <= 1.0, f"AUROC should be between 0 and 1, got {auroc}"


def test_auroc_requires_field_comparisons():
    """Test that AUROC calculation requires document_field_comparisons=True."""
    gt = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    pred = Product.from_json(
        {
            "product_name": {"value": "Widget Pro", "confidence": 0.95},
            "price": {"value": 29.99, "confidence": 0.80},
        }
    )

    with pytest.raises(ValueError, match="No field comparisons found"):
        gt.compare_with(
            pred,
            add_confidence_metrics=True,
            document_field_comparisons=False,  # This should cause an error
        )


def test_list_confidence_auroc():
    """Test AUROC with list items containing confidence scores."""
    gt = Customer(
        name="John Doe",
        email="john@example.com",
        address=Address(street="123 Main St", city="New York", zip_code="10001"),
        orders=[
            Product(
                product_name="Laptop Pro",
                price=1299.99,
                sku="LAP001",
                description="Professional laptop",
            ),
            Product(
                product_name="Mouse Wireless",
                price=49.99,
                sku="MOU001",
                description="Wireless mouse",
            ),
        ],
    )

    pred = Customer.from_json(
        {
            "name": {"value": "John Doe", "confidence": 0.95},
            "email": {"value": "john@example.com", "confidence": 0.85},
            "address": {
                "street": {"value": "123 Main St", "confidence": 0.90},
                "city": {"value": "New York", "confidence": 0.98},
                "zip_code": {"value": "10001", "confidence": 0.95},
            },
            "orders": [
                {
                    "product_name": {
                        "value": "Laptop Pro",
                        "confidence": 0.89,
                    },  # High conf, correct
                    "price": {
                        "value": 1299.99,
                        "confidence": 0.76,
                    },  # Medium conf, correct
                    "sku": {
                        "value": "LAP001",
                        "confidence": 0.92,
                    },  # High conf, correct
                    "description": {
                        "value": "Professional laptop",
                        "confidence": 0.91,
                    },  # High conf, correct
                },
                {
                    "product_name": {
                        "value": "Wrong Mouse",
                        "confidence": 0.85,
                    },  # High conf, incorrect
                    "price": {"value": 49.99, "confidence": 0.83},  # High conf, correct
                    "sku": {
                        "value": "MOU001",
                        "confidence": 0.94,
                    },  # High conf, correct
                    "description": {
                        "value": "Wireless mouse",
                        "confidence": 0.25,
                    },  # Low conf, correct
                },
            ],
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert 0.0 <= auroc <= 1.0, f"AUROC should be between 0 and 1, got {auroc}"

    # Should have confidence data for list items
    confidences = pred.get_all_confidences()
    assert "orders[0].product_name" in confidences
    assert "orders[1].sku" in confidences


def test_mixed_list_confidence_calibration():
    """Test AUROC with mixed calibration in list items."""
    gt = Customer(
        name="Jane Smith",
        email="jane@example.com",
        address=Address(street="456 Oak Ave", city="Boston", zip_code="02101"),
        orders=[
            Product(product_name="Tablet", price=599.99, sku="TAB001"),
            Product(product_name="Keyboard", price=129.99, sku="KEY001"),
        ],
    )

    # Well-calibrated: high confidence for correct, low confidence for incorrect
    pred = Customer.from_json(
        {
            "name": {"value": "Jane Smith", "confidence": 0.95},  # High conf, correct
            "email": {
                "value": "jane@example.com",
                "confidence": 0.90,
            },  # High conf, correct
            "address": {
                "street": {
                    "value": "456 Oak Ave",
                    "confidence": 0.85,
                },  # High conf, correct
                "city": {"value": "Boston", "confidence": 0.92},  # High conf, correct
                "zip_code": {
                    "value": "02101",
                    "confidence": 0.88,
                },  # High conf, correct
            },
            "orders": [
                {
                    "product_name": {
                        "value": "Tablet",
                        "confidence": 0.93,
                    },  # High conf, correct
                    "price": {
                        "value": 599.99,
                        "confidence": 0.87,
                    },  # High conf, correct
                    "sku": {
                        "value": "WRONG_TAB",
                        "confidence": 0.25,
                    },  # Low conf, incorrect
                },
                {
                    "product_name": {
                        "value": "Wrong Device",
                        "confidence": 0.30,
                    },  # Low conf, incorrect
                    "price": {
                        "value": 999.99,
                        "confidence": 0.20,
                    },  # Low conf, incorrect
                    "sku": {
                        "value": "KEY001",
                        "confidence": 0.95,
                    },  # High conf, correct
                },
            ],
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    assert auroc > 0.7, (
        f"Expected high AUROC for well-calibrated list confidence, got {auroc}"
    )


def test_empty_list_confidence():
    """Test AUROC with empty lists but mixed field calibration."""
    gt = Customer(
        name="Empty Customer",
        email="empty@example.com",
        address=Address(street="Empty St", city="Empty City", zip_code="00000"),
        orders=[],
    )

    pred = Customer.from_json(
        {
            "name": {
                "value": "Empty Customer",
                "confidence": 0.95,
            },  # High conf, correct
            "email": {
                "value": "wrong@different.com",
                "confidence": 0.25,
            },  # Low conf, incorrect
            "address": {
                "street": {
                    "value": "Empty St",
                    "confidence": 0.90,
                },  # High conf, correct
                "city": {
                    "value": "Empty City",
                    "confidence": 0.88,
                },  # High conf, correct
                "zip_code": {
                    "value": "99999",
                    "confidence": 0.20,
                },  # Low conf, incorrect
            },
            "orders": [],
        }
    )

    result = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )

    assert "auroc_confidence_metric" in result
    auroc = result["auroc_confidence_metric"]
    # Well-calibrated: high confidence for correct, low confidence for incorrect
    assert auroc > 0.8, (
        f"Expected high AUROC for well-calibrated confidence with empty lists, got {auroc}"
    )


def test_auroc_integration():
    """Test full integration of AUROC feature."""
    gt = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    pred = Product.from_json(
        {
            "product_name": {"value": "Widget Pro", "confidence": 0.95},
            "price": {"value": 29.99, "confidence": 0.80},
            "sku": {"value": "WRONG", "confidence": 0.30},
        }
    )

    # Test without AUROC
    result_without = gt.compare_with(pred, document_field_comparisons=True)
    assert "auroc_confidence_metric" not in result_without

    # Test with AUROC
    result_with = gt.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )
    assert "auroc_confidence_metric" in result_with
    assert isinstance(result_with["auroc_confidence_metric"], float)
    assert 0.0 <= result_with["auroc_confidence_metric"] <= 1.0
