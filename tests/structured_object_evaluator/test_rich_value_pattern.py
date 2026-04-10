"""
Tests for the Rich Value Pattern.

A rich value is a dict with a "value" key plus optional metadata keys
(confidence, bbox, etc.). The RichValueHelper unwraps these during
from_json(), extracting the value for the model field and storing
metadata separately.

These tests verify that:
- Rich values with only "value" (no metadata) are unwrapped correctly
- Rich values with confidence are unwrapped and confidence is stored
- Rich values with non-confidence metadata are unwrapped (future bbox etc.)
- Plain values still work unchanged
- Mixed rich/plain values work in the same model
- Nested and list structures handle rich values correctly
- Confidence is optional and its absence doesn't break anything
"""

from typing import List

from stickler.comparators import (
    ExactComparator,
    LevenshteinComparator,
    NumericComparator,
)
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.rich_value_helper import (
    RichValueHelper,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# ── Test models ──

class Product(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    price: float = ComparableField(comparator=NumericComparator(), threshold=0.5)
    sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0)


class Address(StructuredModel):
    street: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)
    city: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)


class Customer(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    address: Address = ComparableField()
    orders: List[Product] = ComparableField()


# ── Detection tests ──

class TestRichValueDetection:
    def test_value_plus_confidence_is_rich(self):
        assert RichValueHelper._is_rich_value({"value": "Widget", "confidence": 0.9})

    def test_value_only_is_rich(self):
        """A dict with just 'value' is treated as a rich value."""
        assert RichValueHelper._is_rich_value({"value": "Widget"})

    def test_value_plus_bbox_is_rich(self):
        """Future metadata types are detected as rich values."""
        assert RichValueHelper._is_rich_value({"value": "Widget", "bbox": [0.1, 0.2, 0.3, 0.4]})

    def test_value_plus_multiple_metadata_is_rich(self):
        assert RichValueHelper._is_rich_value({
            "value": "Widget", "confidence": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4]
        })

    def test_no_value_key_is_not_rich(self):
        assert not RichValueHelper._is_rich_value({"name": "Widget", "confidence": 0.9})

    def test_plain_string_is_not_rich(self):
        assert not RichValueHelper._is_rich_value("Widget")

    def test_plain_number_is_not_rich(self):
        assert not RichValueHelper._is_rich_value(42)

    def test_empty_dict_is_not_rich(self):
        assert not RichValueHelper._is_rich_value({})


# ── Unwrapping tests ──

class TestRichValueUnwrapping:
    def test_value_with_confidence_unwraps(self):
        data = {"name": {"value": "Widget", "confidence": 0.9}, "price": 29.99}
        unwrapped, confidences = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget", "price": 29.99}
        assert confidences == {"name": 0.9}

    def test_value_only_unwraps_no_confidence(self):
        """Rich value with just 'value' unwraps but produces no confidence entry."""
        data = {"name": {"value": "Widget"}, "price": 29.99}
        unwrapped, confidences = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget", "price": 29.99}
        assert confidences == {}

    def test_value_with_bbox_only_unwraps_no_confidence(self):
        """Rich value with bbox but no confidence produces no confidence entry."""
        data = {"name": {"value": "Widget", "bbox": [0.1, 0.2, 0.3, 0.4]}}
        unwrapped, confidences = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget"}
        assert confidences == {}

    def test_value_with_confidence_and_bbox_extracts_confidence(self):
        """When both confidence and bbox are present, confidence is extracted."""
        data = {"name": {"value": "Widget", "confidence": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4]}}
        unwrapped, confidences = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget"}
        assert confidences == {"name": 0.9}

    def test_plain_values_pass_through(self):
        data = {"name": "Widget", "price": 29.99}
        unwrapped, confidences = RichValueHelper.process_rich_values(data)
        assert unwrapped == data
        assert confidences == {}

    def test_nested_rich_values(self):
        data = {
            "name": {"value": "Jane", "confidence": 0.95},
            "address": {
                "street": {"value": "123 Main", "confidence": 0.85},
                "city": "Boston",
            },
        }
        unwrapped, confidences = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Jane", "address": {"street": "123 Main", "city": "Boston"}}
        assert confidences == {"name": 0.95, "address.street": 0.85}

    def test_list_rich_values(self):
        data = {
            "items": [
                {"value": "Widget", "confidence": 0.9},
                {"value": "Gadget"},
                "PlainItem",
            ]
        }
        unwrapped, confidences = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"items": ["Widget", "Gadget", "PlainItem"]}
        assert confidences == {"items[0]": 0.9}


# ── from_json integration tests ──

class TestFromJsonRichValues:
    def test_confidence_rich_values(self):
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "ABC123", "confidence": 0.7},
        })
        assert pred.name == "Widget"
        assert pred.price == 29.99
        assert pred.get_field_confidence("name") == 0.9
        assert pred.get_field_confidence("price") == 0.8

    def test_value_only_rich_values(self):
        """Rich values without confidence unwrap correctly, no confidence stored."""
        pred = Product.from_json({
            "name": {"value": "Widget"},
            "price": {"value": 29.99},
            "sku": "ABC123",
        })
        assert pred.name == "Widget"
        assert pred.price == 29.99
        assert pred.sku == "ABC123"
        assert pred.get_field_confidence("name") is None
        assert pred.get_field_confidence("price") is None
        assert pred.get_all_confidences() == {}

    def test_bbox_only_rich_values(self):
        """Rich values with bbox but no confidence work correctly."""
        pred = Product.from_json({
            "name": {"value": "Widget", "bbox": [0.1, 0.2, 0.3, 0.4]},
            "price": 29.99,
            "sku": "ABC123",
        })
        assert pred.name == "Widget"
        assert pred.get_field_confidence("name") is None

    def test_mixed_rich_and_plain(self):
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": 29.99,
            "sku": {"value": "ABC123"},
        })
        assert pred.name == "Widget"
        assert pred.price == 29.99
        assert pred.sku == "ABC123"
        assert pred.get_field_confidence("name") == 0.9
        assert pred.get_field_confidence("price") is None
        assert pred.get_field_confidence("sku") is None

    def test_nested_model_rich_values_without_confidence(self):
        pred = Customer.from_json({
            "name": {"value": "Jane"},
            "address": {
                "street": {"value": "123 Main"},
                "city": {"value": "Boston", "confidence": 0.85},
            },
            "orders": [],
        })
        assert pred.name == "Jane"
        assert pred.address.street == "123 Main"
        assert pred.address.city == "Boston"
        assert pred.get_field_confidence("name") is None
        assert pred.get_field_confidence("address.city") == 0.85

    def test_list_items_with_mixed_rich_values(self):
        pred = Customer.from_json({
            "name": "Jane",
            "address": {"street": "123 Main", "city": "Boston"},
            "orders": [
                {
                    "name": {"value": "Widget", "confidence": 0.9},
                    "price": {"value": 29.99},
                    "sku": "ABC",
                },
                {
                    "name": {"value": "Gadget", "bbox": [0.1, 0.2, 0.3, 0.4]},
                    "price": 49.99,
                    "sku": {"value": "DEF", "confidence": 0.7},
                },
            ],
        })
        assert pred.orders[0].name == "Widget"
        assert pred.orders[1].name == "Gadget"
        assert pred.get_field_confidence("orders[0].name") == 0.9
        assert pred.get_field_confidence("orders[0].price") is None
        assert pred.get_field_confidence("orders[1].name") is None
        assert pred.get_field_confidence("orders[1].sku") == 0.7


# ── Comparison still works with rich values ──

class TestComparisonWithRichValues:
    def test_compare_with_value_only_rich_values(self):
        """Comparison works when predictions use value-only rich values."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"value": "Widget"},
            "price": {"value": 29.99},
            "sku": {"value": "ABC123"},
        })
        result = gt.compare_with(pred)
        assert result["overall_score"] > 0.9

    def test_confidence_metrics_with_partial_confidence(self):
        """Confidence metrics work when only some fields have confidence."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 99.99},  # value-only, no confidence
            "sku": {"value": "ABC123", "confidence": 0.8},
        })
        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )
        cm = result["confidence_metrics"]
        # Only name and sku should appear in fields (price has no confidence)
        assert "name" in cm["fields"]
        assert "sku" in cm["fields"]
        assert "price" not in cm["fields"]
        assert cm["coverage"]["fields_with_confidence"] == 2
        assert cm["coverage"]["fields_total"] == 3

    def test_confidence_metrics_with_zero_confidence_fields(self):
        """When no fields have confidence, metrics still work gracefully."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"value": "Widget"},
            "price": {"value": 29.99},
            "sku": {"value": "ABC123"},
        })
        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )
        # No confidence data at all
        cm = result.get("confidence_metrics")
        if cm is not None:
            assert cm["overall"]["auroc"]["value"] is None
            assert cm["coverage"]["fields_with_confidence"] == 0
