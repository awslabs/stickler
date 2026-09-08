

"""Comprehensive coverage for non-rolled-up nested metrics.

Ensures that nested StructuredModel fields behave correctly:
- Object-level metrics count objects, not nested field rollups
- Nested field details are preserved for debugging
- Confusion matrix counts are bounded by max objects being compared

Historically these models passed ``aggregate=False`` explicitly. That
parameter is deprecated and has no effect (see issue #226) -- the behavior
asserted here is simply how nested metrics work, so the argument was dropped
rather than the tests changed.
"""

from typing import List, Optional

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# ========== SIMPLE NESTED MODELS ==========


class SimpleContact(StructuredModel):
    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class SimpleOwner(StructuredModel):
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    contact: Optional[SimpleContact] = ComparableField(
        default=None,
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0,
    )


# ========== DOUBLE NESTED MODELS ==========


class Address(StructuredModel):
    street: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    city: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)


class DetailedContact(StructuredModel):
    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    address: Address = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0,
    )


class DetailedOwner(StructuredModel):
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    contact: DetailedContact = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0,
    )


# ========== LIST MODELS ==========


class Product(StructuredModel):
    match_threshold = 1.0  # Moved from List[Product] field

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    price: float = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Order(StructuredModel):
    order_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    products: List[Product] = ComparableField(
        weight=1.0,
    )


# ========== TEST CASES ==========


def test_simple_contact_non_matching_aggregate_false():
    """Test basic aggregate=False with non-matching contact object."""

    true_owner = SimpleOwner(
        **{
            "name": "John Doe",
            "contact": {"phone": "555-1234"},  # No email
        }
    )

    pred_owner = SimpleOwner(
        **{
            "name": "John Doe",
            "contact": {
                "phone": "555-9999",  # Different phone = FD
                "email": "john@example.com",  # Extra email = FA at nested level
            },
        }
    )

    result = true_owner.compare_with(pred_owner, include_confusion_matrix=True)
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print("\nSimple non-matching case:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # CRITICAL: aggregate=False means count OBJECTS, not nested field rollups
    assert contact_cm["tp"] == 0, (
        f"Expected 0 TP (contact object didn't match), got {contact_cm['tp']}"
    )
    assert contact_cm["fa"] == 0, (
        f"Expected 0 FA (1v1 object comparison), got {contact_cm['fa']}"
    )
    assert contact_cm["fd"] == 1, (
        f"Expected 1 FD (contact object didn't meet threshold), got {contact_cm['fd']}"
    )
    assert contact_cm["fp"] == 1, f"Expected 1 FP (fa + fd), got {contact_cm['fp']}"

    # Verify nested field details are still available for debugging
    assert "fields" in result["confusion_matrix"]["fields"]["contact"]
    contact_fields = result["confusion_matrix"]["fields"]["contact"]["fields"]
    assert "phone" in contact_fields, "Phone field details should be available"
    assert "email" in contact_fields, "Email field details should be available"


def test_simple_contact_matching_aggregate_false():
    """Test basic aggregate=False with matching contact object."""

    true_owner = SimpleOwner(
        **{
            "name": "John Doe",
            "contact": {"phone": "555-1234", "email": "john@example.com"},
        }
    )

    pred_owner = SimpleOwner(
        **{
            "name": "John Doe",
            "contact": {
                "phone": "555-1234",  # Same phone
                "email": "john@example.com",  # Same email
            },
        }
    )

    result = true_owner.compare_with(pred_owner, include_confusion_matrix=True)
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print("\nSimple matching case:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # Perfect match should be 1 TP at object level
    assert contact_cm["tp"] == 1, (
        f"Expected 1 TP (contact object matched), got {contact_cm['tp']}"
    )
    assert contact_cm["fa"] == 0, f"Expected 0 FA, got {contact_cm['fa']}"
    assert contact_cm["fd"] == 0, f"Expected 0 FD, got {contact_cm['fd']}"
    assert contact_cm["fp"] == 0, f"Expected 0 FP, got {contact_cm['fp']}"


def test_simple_contact_partial_match_aggregate_false():
    """Test aggregate=False with partial match (some fields match, some don't)."""

    true_owner = SimpleOwner(
        **{
            "name": "John Doe",
            "contact": {"phone": "555-1234", "email": "john@example.com"},
        }
    )

    pred_owner = SimpleOwner(
        **{
            "name": "John Doe",
            "contact": {
                "phone": "555-1234",  # Same phone (match)
                "email": "different@example.com",  # Different email (no match)
            },
        }
    )

    result = true_owner.compare_with(pred_owner, include_confusion_matrix=True)
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print("\nPartial match case:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # With 50% similarity (1 of 2 fields match), this should be below threshold=1.0
    # So it's a False Discovery at object level
    assert contact_cm["tp"] == 0, (
        f"Expected 0 TP (partial match below threshold), got {contact_cm['tp']}"
    )
    assert contact_cm["fa"] == 0, f"Expected 0 FA, got {contact_cm['fa']}"
    assert contact_cm["fd"] == 1, (
        f"Expected 1 FD (partial match below threshold), got {contact_cm['fd']}"
    )
    assert contact_cm["fp"] == 1, f"Expected 1 FP, got {contact_cm['fp']}"


def test_double_nested_aggregate_false():
    """Test aggregate=False with double nesting (owner->contact->address)."""

    true_owner = DetailedOwner(
        **{
            "name": "Jane Smith",
            "contact": {
                "phone": "555-1111",
                "email": "jane@example.com",
                "address": {"street": "123 Main St", "city": "Springfield"},
            },
        }
    )

    pred_owner = DetailedOwner(
        **{
            "name": "Jane Smith",
            "contact": {
                "phone": "555-2222",  # Different phone
                "email": "jane@example.com",  # Same email
                "address": {
                    "street": "456 Oak Ave",  # Different street
                    "city": "Springfield",  # Same city
                },
            },
        }
    )

    result = true_owner.compare_with(pred_owner, include_confusion_matrix=True)
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print("\nDouble nested case:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # Contact object level should still be bounded by 1 (single object comparison)
    # Even though it has nested address and multiple field mismatches
    assert contact_cm["tp"] == 0, (
        f"Expected 0 TP (contact object didn't fully match), got {contact_cm['tp']}"
    )
    assert contact_cm["fa"] == 0, (
        f"Expected 0 FA (1v1 object comparison), got {contact_cm['fa']}"
    )
    assert contact_cm["fd"] == 1, (
        f"Expected 1 FD (single contact object), got {contact_cm['fd']}"
    )
    assert contact_cm["fp"] == 1, (
        f"Expected 1 FP (bounded by single object), got {contact_cm['fp']}"
    )

    # Verify nested structure is preserved
    assert "address" in result["confusion_matrix"]["fields"]["contact"]["fields"]
    address_cm = result["confusion_matrix"]["fields"]["contact"]["fields"]["address"][
        "overall"
    ]

    print(
        f"Address overall: tp={address_cm['tp']}, fa={address_cm['fa']}, fd={address_cm['fd']}, fp={address_cm['fp']}"
    )

    # Address should also be bounded by single object
    assert address_cm["tp"] == 0, (
        f"Expected 0 TP (address object didn't fully match), got {address_cm['tp']}"
    )
    assert address_cm["fa"] == 0, (
        f"Expected 0 FA (1v1 address comparison), got {address_cm['fa']}"
    )
    assert address_cm["fd"] == 1, (
        f"Expected 1 FD (single address object), got {address_cm['fd']}"
    )
    assert address_cm["fp"] == 1, (
        f"Expected 1 FP (bounded by single address), got {address_cm['fp']}"
    )


def test_list_aggregate_false():
    """Test aggregate=False with List[StructuredModel] field."""

    true_order = Order(
        **{
            "order_id": "ORD-001",
            "products": [
                {"name": "Widget A", "price": 10.99},
                {"name": "Widget B", "price": 15.50},
            ],
        }
    )

    pred_order = Order(
        **{
            "order_id": "ORD-001",
            "products": [
                {"name": "Widget A", "price": 10.99},  # Perfect match
                {
                    "name": "Widget X",
                    "price": 15.50,
                },  # Name mismatch but similar structure
                {"name": "Widget C", "price": 20.00},  # Extra product
            ],
        }
    )

    result = true_order.compare_with(pred_order, include_confusion_matrix=True)
    products_cm = result["confusion_matrix"]["fields"]["products"]["overall"]

    print("\nList aggregate=False case:")
    print(
        f"Products overall: tp={products_cm['tp']}, fa={products_cm['fa']}, fd={products_cm['fd']}, fp={products_cm['fp']}"
    )

    # With list comparison: 2 true products vs 3 predicted products
    # Hungarian matching gives us: 1 TP ("Widget A" exact) + 1 FD ("Widget B"/"Widget X" below 1.0 threshold) + 1 FA (extra "Widget C")
    # Product.match_threshold = 1.0 means only perfect matches count as TP
    expected_tp = 1  # Only "Widget A" (perfect match 1.000 ≥ 1.0 threshold)
    expected_fd = 1  # "Widget B"/"Widget X" (similarity 0.938 < 1.0 threshold)
    expected_fa = 1  # One unmatched prediction ("Widget C")
    expected_fp = expected_fd + expected_fa

    assert products_cm["tp"] == expected_tp, (
        f"Expected {expected_tp} TP (objects matched), got {products_cm['tp']}"
    )
    assert products_cm["fd"] == expected_fd, (
        f"Expected {expected_fd} FD (partial matches below threshold), got {products_cm['fd']}"
    )
    assert products_cm["fa"] == expected_fa, (
        f"Expected {expected_fa} FA (unmatched predictions), got {products_cm['fa']}"
    )
    assert products_cm["fp"] == expected_fp, (
        f"Expected {expected_fp} FP (total false positives), got {products_cm['fp']}"
    )


def test_two_identically_configured_nested_fields_agree():
    """Two nested fields with the same config produce the same metric shape.

    Previously this compared an ``aggregate=True`` field against an
    ``aggregate=False`` one. With the parameter gone the two definitions are
    identical, so what is left worth asserting is that identical configuration
    gives identical treatment -- each nested object counted once, at the object
    level, independent of the other field alongside it.
    """

    class MixedModel(StructuredModel):
        simple_field: str = ComparableField(
            comparator=ExactComparator(), threshold=1.0, weight=1.0
        )
        first_contact: SimpleContact = ComparableField(
            comparator=ExactComparator(), threshold=1.0, weight=1.0
        )
        second_contact: SimpleContact = ComparableField(
            comparator=ExactComparator(), threshold=1.0, weight=1.0
        )

    true_model = MixedModel(
        **{
            "simple_field": "test",
            "first_contact": {"phone": "555-1111"},
            "second_contact": {"phone": "555-2222"},
        }
    )

    pred_model = MixedModel(
        **{
            "simple_field": "test",
            "first_contact": {
                "phone": "555-9999",  # Different phone
                "email": "test@example.com",  # Extra email
            },
            "second_contact": {
                "phone": "555-8888",  # Different phone
                "email": "test2@example.com",  # Extra email
            },
        }
    )

    result = true_model.compare_with(pred_model, include_confusion_matrix=True)
    fields = result["confusion_matrix"]["fields"]

    first = {k: fields["first_contact"]["overall"][k] for k in ("tp", "fa", "fd", "fp")}
    second = {
        k: fields["second_contact"]["overall"][k] for k in ("tp", "fa", "fd", "fp")
    }

    # Same configuration, same data shape, so the same verdict.
    assert first == second

    # Each nested object is counted once: one FD, bounded by the single object.
    assert first == {"tp": 0, "fa": 0, "fd": 1, "fp": 1}


def test_null_handling_aggregate_false():
    """Test aggregate=False behavior with null/None values."""

    # Case 1: Both null
    true_owner_null = SimpleOwner(**{"name": "Test", "contact": None})
    pred_owner_null = SimpleOwner(**{"name": "Test", "contact": None})

    result = true_owner_null.compare_with(
        pred_owner_null, include_confusion_matrix=True
    )
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print("\nNull handling - both null:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # Both null should be True Negative
    assert contact_cm["tn"] == 1, f"Expected 1 TN (both null), got {contact_cm['tn']}"
    assert (
        contact_cm["tp"]
        + contact_cm["fa"]
        + contact_cm["fd"]
        + contact_cm["fp"]
        + contact_cm["fn"]
        == 0
    )

    # Case 2: GT null, pred non-null
    pred_owner_present = SimpleOwner(
        **{
            "name": "Test",
            "contact": {"phone": "555-1234", "email": "test@example.com"},
        }
    )

    result = true_owner_null.compare_with(
        pred_owner_present, include_confusion_matrix=True
    )
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print("Null handling - GT null, pred present:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # Should be False Alarm (single object level)
    assert contact_cm["fa"] == 1, (
        f"Expected 1 FA (GT null, pred present), got {contact_cm['fa']}"
    )
    assert contact_cm["fp"] == 1, f"Expected 1 FP, got {contact_cm['fp']}"

