"""
Tests for the Rich Value Pattern.

A rich value is a dict with a "_value" key plus optional metadata keys
(_confidence, _bbox, etc.). The RichValueHelper unwraps these during
from_json(), extracting the value for the model field and storing
metadata separately.

These tests verify that:
- Rich values with only "_value" (no metadata) are unwrapped correctly
- Rich values with _confidence are unwrapped and confidence is stored
- Rich values with non-confidence metadata are unwrapped (future _bbox etc.)
- Plain values still work unchanged
- Mixed rich/plain values work in the same model
- Nested and list structures handle rich values correctly
- Confidence is optional and its absence doesn't break anything
"""

from typing import List

import pytest

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
        assert RichValueHelper._is_rich_value({"_value": "Widget", "_confidence": 0.9})

    def test_value_only_is_rich(self):
        """A dict with just '_value' is treated as a rich value."""
        assert RichValueHelper._is_rich_value({"_value": "Widget"})

    def test_value_plus_bbox_is_rich(self):
        """Future metadata types are detected as rich values."""
        assert RichValueHelper._is_rich_value({"_value": "Widget", "_bbox": [0.1, 0.2, 0.3, 0.4]})

    def test_value_plus_multiple_metadata_is_rich(self):
        assert RichValueHelper._is_rich_value({
            "_value": "Widget", "_confidence": 0.9, "_bbox": [0.1, 0.2, 0.3, 0.4]
        })

    def test_no_value_key_is_not_rich(self):
        assert not RichValueHelper._is_rich_value({"name": "Widget", "_confidence": 0.9})

    def test_plain_string_is_not_rich(self):
        assert not RichValueHelper._is_rich_value("Widget")

    def test_plain_number_is_not_rich(self):
        assert not RichValueHelper._is_rich_value(42)

    def test_empty_dict_is_not_rich(self):
        assert not RichValueHelper._is_rich_value({})


# ── Unwrapping tests ──

class TestRichValueUnwrapping:
    def test_value_with_confidence_unwraps(self):
        data = {"name": {"_value": "Widget", "_confidence": 0.9}, "price": 29.99}
        unwrapped, confidences, _extras = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget", "price": 29.99}
        assert confidences == {"name": 0.9}

    def test_value_only_unwraps_no_confidence(self):
        """Rich value with just '_value' unwraps but produces no confidence entry."""
        data = {"name": {"_value": "Widget"}, "price": 29.99}
        unwrapped, confidences, _extras = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget", "price": 29.99}
        assert confidences == {}

    def test_value_with_bbox_only_unwraps_no_confidence(self):
        """Rich value with _bbox but no _confidence produces no confidence entry."""
        data = {"name": {"_value": "Widget", "_bbox": [0.1, 0.2, 0.3, 0.4]}}
        unwrapped, confidences, _extras = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget"}
        assert confidences == {}

    def test_value_with_confidence_and_bbox_extracts_confidence(self):
        """When both _confidence and _bbox are present, confidence is extracted."""
        data = {"name": {"_value": "Widget", "_confidence": 0.9, "_bbox": [0.1, 0.2, 0.3, 0.4]}}
        unwrapped, confidences, _extras = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Widget"}
        assert confidences == {"name": 0.9}

    def test_plain_values_pass_through(self):
        data = {"name": "Widget", "price": 29.99}
        unwrapped, confidences, _extras = RichValueHelper.process_rich_values(data)
        assert unwrapped == data
        assert confidences == {}

    def test_nested_rich_values(self):
        data = {
            "name": {"_value": "Jane", "_confidence": 0.95},
            "address": {
                "street": {"_value": "123 Main", "_confidence": 0.85},
                "city": "Boston",
            },
        }
        unwrapped, confidences, _extras = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"name": "Jane", "address": {"street": "123 Main", "city": "Boston"}}
        assert confidences == {"name": 0.95, "address.street": 0.85}

    def test_list_rich_values(self):
        data = {
            "items": [
                {"_value": "Widget", "_confidence": 0.9},
                {"_value": "Gadget"},
                "PlainItem",
            ]
        }
        unwrapped, confidences, _extras = RichValueHelper.process_rich_values(data)
        assert unwrapped == {"items": ["Widget", "Gadget", "PlainItem"]}
        assert confidences == {"items[0]": 0.9}


# ── from_json integration tests ──

class TestFromJsonRichValues:
    def test_confidence_rich_values(self):
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        assert pred.name == "Widget"
        assert pred.price == 29.99
        assert pred.get_field_confidence("name") == 0.9
        assert pred.get_field_confidence("price") == 0.8

    def test_value_only_rich_values(self):
        """Rich values without confidence unwrap correctly, no confidence stored."""
        pred = Product.from_json({
            "name": {"_value": "Widget"},
            "price": {"_value": 29.99},
            "sku": "ABC123",
        })
        assert pred.name == "Widget"
        assert pred.price == 29.99
        assert pred.sku == "ABC123"
        assert pred.get_field_confidence("name") is None
        assert pred.get_field_confidence("price") is None
        assert pred.get_all_confidences() == {}

    def test_bbox_only_rich_values(self):
        """Rich values with _bbox but no _confidence work correctly."""
        pred = Product.from_json({
            "name": {"_value": "Widget", "_bbox": [0.1, 0.2, 0.3, 0.4]},
            "price": 29.99,
            "sku": "ABC123",
        })
        assert pred.name == "Widget"
        assert pred.get_field_confidence("name") is None

    def test_mixed_rich_and_plain(self):
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,
            "sku": {"_value": "ABC123"},
        })
        assert pred.name == "Widget"
        assert pred.price == 29.99
        assert pred.sku == "ABC123"
        assert pred.get_field_confidence("name") == 0.9
        assert pred.get_field_confidence("price") is None
        assert pred.get_field_confidence("sku") is None

    def test_nested_model_rich_values_without_confidence(self):
        pred = Customer.from_json({
            "name": {"_value": "Jane"},
            "address": {
                "street": {"_value": "123 Main"},
                "city": {"_value": "Boston", "_confidence": 0.85},
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
                    "name": {"_value": "Widget", "_confidence": 0.9},
                    "price": {"_value": 29.99},
                    "sku": "ABC",
                },
                {
                    "name": {"_value": "Gadget", "_bbox": [0.1, 0.2, 0.3, 0.4]},
                    "price": 49.99,
                    "sku": {"_value": "DEF", "_confidence": 0.7},
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
    pytestmark = pytest.mark.filterwarnings(
        "ignore:Single-document confidence metrics:UserWarning"
    )

    def test_compare_with_value_only_rich_values(self):
        """Comparison works when predictions use value-only rich values."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget"},
            "price": {"_value": 29.99},
            "sku": {"_value": "ABC123"},
        })
        result = gt.compare_with(pred)
        assert result["overall_score"] > 0.9

    def test_confidence_metrics_with_partial_confidence(self):
        """Confidence metrics work when only some fields have confidence."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99},  # value-only, no confidence
            "sku": {"_value": "ABC123", "_confidence": 0.8},
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
            "name": {"_value": "Widget"},
            "price": {"_value": 29.99},
            "sku": {"_value": "ABC123"},
        })
        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )
        # No confidence data at all
        cm = result.get("confidence_metrics")
        if cm is not None:
            assert cm["overall"]["auroc"]["value"] is None
            assert cm["coverage"]["fields_with_confidence"] == 0


# ── Old format rejection tests ──

class TestLegacyRichValueShim:
    """The pre-underscore {"value", "confidence"} shape still unwraps for one
    release, with a DeprecationWarning emitted per occurrence."""

    def test_legacy_value_confidence_detected(self):
        assert RichValueHelper._is_legacy_rich_value(
            {"value": "Widget", "confidence": 0.9}
        )

    def test_legacy_not_detected_when_new_shape_present(self):
        """If _value is present, the new-shape detector wins."""
        d = {"_value": "Widget", "_confidence": 0.9}
        assert RichValueHelper._is_rich_value(d)
        assert not RichValueHelper._is_legacy_rich_value(d)

    def test_legacy_format_unwraps_with_deprecation_warning(self):
        """Legacy dicts unwrap and emit a DeprecationWarning naming the field."""
        data = {"name": {"value": "Widget", "confidence": 0.9}}
        with pytest.warns(DeprecationWarning, match="name"):
            unwrapped, confidences, extras = RichValueHelper.process_rich_values(
                data
            )
        assert unwrapped == {"name": "Widget"}
        assert confidences == {"name": 0.9}
        assert extras == {}

    def test_legacy_format_from_json_roundtrip(self):
        """Legacy shape flows through from_json end-to-end."""
        with pytest.warns(DeprecationWarning):
            pred = Product.from_json({
                "name": {"value": "Widget", "confidence": 0.9},
                "price": 29.99,
                "sku": "ABC123",
            })
        assert pred.name == "Widget"
        assert pred.get_field_confidence("name") == 0.9

    def test_legacy_value_only_is_not_detected(self):
        """A dict with ``value`` but no ``confidence`` is plain user data.

        The legacy shim used to treat this as a rich value and silently
        unwrap it, which would discard sibling keys like ``currency`` in
        ``{"currency": "USD", "value": 100}`` while emitting a misleading
        DeprecationWarning. Both keys must be present for the shim to
        fire.
        """
        import warnings

        data = {"name": {"value": "Widget"}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an exception
            unwrapped, confidences, extras = RichValueHelper.process_rich_values(
                data
            )
        # The dict survives intact — no unwrapping, no warning.
        assert unwrapped == {"name": {"value": "Widget"}}
        assert confidences == {}
        assert extras == {}

    def test_value_with_sibling_keys_is_not_detected(self):
        """Plain user dicts with a ``value`` field plus siblings round-trip.

        Regression guard: the loose detector previously dropped the
        ``currency`` sibling on the floor while warning about deprecation.
        """
        import warnings

        data = {"price": {"currency": "USD", "value": 100}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            unwrapped, confidences, extras = RichValueHelper.process_rich_values(
                data
            )
        assert unwrapped == {"price": {"currency": "USD", "value": 100}}
        assert confidences == {}
        assert extras == {}

    def test_legacy_value_and_confidence_still_unwraps(self):
        """Regression guard: the legacy shim still fires when both keys are
        present, so existing JSONL corpora continue to round-trip cleanly."""
        data = {"name": {"value": "Widget", "confidence": 0.9}}
        with pytest.warns(DeprecationWarning, match="name"):
            unwrapped, confidences, extras = RichValueHelper.process_rich_values(
                data
            )
        assert unwrapped == {"name": "Widget"}
        assert confidences == {"name": 0.9}
        assert extras == {}


class TestProcessConfidenceDeprecationShim:
    """from_json(process_confidence=...) still works for one release."""

    def test_process_confidence_alias_emits_deprecation(self):
        """Passing the legacy kwarg emits DeprecationWarning and still works."""
        with pytest.warns(DeprecationWarning, match="process_confidence"):
            pred = Product.from_json(
                {
                    "name": {"_value": "Widget", "_confidence": 0.9},
                    "price": 29.99,
                    "sku": "ABC123",
                },
                process_confidence=True,
            )
        assert pred.name == "Widget"
        assert pred.get_field_confidence("name") == 0.9

    def test_process_confidence_false_skips_unwrapping(self):
        """process_confidence=False behaves like process_rich_values=False.

        With unwrapping off no rich-value metadata is collected. We verify
        the alias is threaded through by confirming both the kwarg is
        accepted and that get_all_confidences() reports no entries.
        """
        with pytest.warns(DeprecationWarning):
            pred = Product.from_json(
                {"name": "Widget", "price": 29.99, "sku": "ABC123"},
                process_confidence=False,
            )
        assert pred.name == "Widget"
        # No rich values in, no confidences extracted.
        assert pred.get_all_confidences() == {}

    def test_unknown_kwarg_still_raises(self):
        """Unexpected kwargs are still rejected with TypeError."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Product.from_json(
                {"name": "Widget", "price": 29.99, "sku": "ABC123"},
                bogus_kwarg=True,
            )


class TestReservedDunderNames:
    """``from_json`` rejects payloads that try to inject library-managed
    metadata via the ``__stickler_*`` namespace. Without this guard,
    ``extra: "allow"`` would let user data silently shadow the library's
    own state."""

    def test_raw_json_key_rejected(self):
        with pytest.raises(ValueError, match="__stickler_raw_json__"):
            Product.from_json(
                {
                    "name": "Widget",
                    "price": 29.99,
                    "sku": "ABC",
                    "__stickler_raw_json__": "injected",
                }
            )

    def test_field_confidences_key_rejected(self):
        with pytest.raises(ValueError, match="__stickler_field_confidences__"):
            Product.from_json(
                {
                    "name": "Widget",
                    "price": 29.99,
                    "sku": "ABC",
                    "__stickler_field_confidences__": {"name": 0.9},
                }
            )

    def test_field_extras_key_rejected(self):
        with pytest.raises(ValueError, match="__stickler_field_extras__"):
            Product.from_json(
                {
                    "name": "Widget",
                    "price": 29.99,
                    "sku": "ABC",
                    "__stickler_field_extras__": {},
                }
            )


# ── Extras tests ──

class TestExtras:
    def test_extras_stored_on_instance(self):
        """Underscore-prefixed metadata keys are stored as extras."""
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9, "_handwritten": True, "_source": "page 3"},
            "price": 29.99,
            "sku": "ABC123",
        })
        assert pred.name == "Widget"
        assert pred.get_field_confidence("name") == 0.9
        extras = pred.get_field_extras("name")
        assert extras is not None
        assert extras["_handwritten"] is True
        assert extras["_source"] == "page 3"

    def test_no_extras_returns_none(self):
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,
            "sku": "ABC123",
        })
        # name has confidence but no extras
        assert pred.get_field_extras("name") is None
        # price has no rich value at all
        assert pred.get_field_extras("price") is None

    def test_get_all_extras(self):
        pred = Product.from_json({
            "name": {"_value": "Widget", "_handwritten": True},
            "price": {"_value": 29.99, "_ocr_engine": "tesseract"},
            "sku": "ABC123",
        })
        all_extras = pred.get_all_extras()
        assert "name" in all_extras
        assert all_extras["name"]["_handwritten"] is True
        assert "price" in all_extras
        assert all_extras["price"]["_ocr_engine"] == "tesseract"
        assert "sku" not in all_extras

    def test_extras_with_underscore_prefixed_keys(self):
        """Underscore-prefixed keys that aren't _value or _confidence go to extras."""
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9, "_bbox": [0.1, 0.2, 0.3, 0.4]},
            "price": 29.99,
            "sku": "ABC123",
        })
        extras = pred.get_field_extras("name")
        assert extras is not None
        assert extras["_bbox"] == [0.1, 0.2, 0.3, 0.4]

    def test_no_extras_when_no_rich_values(self):
        pred = Product(name="Widget", price=29.99, sku="ABC123")
        assert pred.get_all_extras() == {}
        assert pred.get_field_extras("name") is None

    def test_non_prefixed_key_emits_warning(self):
        """Non-underscore-prefixed keys inside a rich value emit a UserWarning."""
        with pytest.warns(UserWarning, match="Non-prefixed key 'handwritten'"):
            Product.from_json({
                "name": {"_value": "Widget", "handwritten": True},
                "price": 29.99,
                "sku": "ABC123",
            })
