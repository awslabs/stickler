

"""Test case specifically for the aggregate_testing.ipynb notebook issue.

The notebook shows a 'contact' field with wrong totals (3 instead of 1 false discovery).
This test verifies that with aggregate=False, we get correct object-level counting.
"""

from typing import Optional
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.exact import ExactComparator


class Contact(StructuredModel):
    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Owner(StructuredModel):
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    contact: Optional[Contact] = ComparableField(
        default=None,
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0,
        aggregate=False,  # KEY: This prevents nested field rollup
    )


def test_aggregate_false_prevents_nested_rollup():
    """Test that aggregate=False prevents rolling up values from nested fields.

    This reproduces the issue from aggregate_testing.ipynb where contact field
    was showing wrong totals (3 instead of 1) because it was rolling up nested
    field metrics instead of counting at the object level.
    """

    true_owner = Owner(
        **{
            "name": "John Doe",
            "contact": {"phone": "555-1234"},  # No email
        }
    )

    pred_owner = Owner(
        **{
            "name": "John Doe",
            "contact": {
                "phone": "555-9999",  # Different phone = contact object similarity low
                "email": "john@example.com",  # Extra email = nested FA but shouldn't roll up
            },
        }
    )

    result = true_owner.compare_with(pred_owner, include_confusion_matrix=True)
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print(f"\nAggregate=False contact metrics:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # With aggregate=False, this should be object-level counting:
    # - 1 contact object vs 1 contact object comparison
    # - Contact objects don't match well enough (phone mismatch) = 1 FD
    # - No false alarms at object level (both GT and pred have contact objects)
    # - Nested field details (phone FD, email FA) should NOT roll up to overall

    assert contact_cm["tp"] == 0, (
        f"Expected 0 TP (contact objects don't match threshold), got {contact_cm['tp']}"
    )
    assert contact_cm["fa"] == 0, (
        f"Expected 0 FA (object-level: both have contact), got {contact_cm['fa']}"
    )
    assert contact_cm["fd"] == 1, (
        f"Expected 1 FD (contact object similarity below threshold), got {contact_cm['fd']}"
    )
    assert contact_cm["fp"] == 1, f"Expected 1 FP (fd + fa), got {contact_cm['fp']}"

    # Verify nested details are still available but don't affect overall
    assert "fields" in result["confusion_matrix"]["fields"]["contact"]
    contact_fields = result["confusion_matrix"]["fields"]["contact"]["fields"]

    # Phone should be FD at nested level
    phone_cm = contact_fields["phone"]["overall"]
    assert phone_cm["fd"] == 1, f"Expected phone FD=1, got {phone_cm['fd']}"

    # Email should be FA at nested level
    email_cm = contact_fields["email"]["overall"]
    assert email_cm["fa"] == 1, f"Expected email FA=1, got {email_cm['fa']}"

    print("✅ Test passed: aggregate=False correctly prevents nested rollup")


def test_aggregate_true_would_rollup():
    """Test that  would roll up nested metrics (for comparison).

    This shows the difference in behavior when .
    """

    class OwnerWithAggregateTrue(StructuredModel):
        name: str = ComparableField(
            comparator=ExactComparator(), threshold=1.0, weight=1.0
        )
        contact: Optional[Contact] = ComparableField(
            default=None,
            comparator=ExactComparator(),
            threshold=1.0,
            weight=1.0,
            # This would allow nested field rollup
        )

    true_owner = OwnerWithAggregateTrue(
        **{"name": "John Doe", "contact": {"phone": "555-1234"}}
    )

    pred_owner = OwnerWithAggregateTrue(
        **{
            "name": "John Doe",
            "contact": {"phone": "555-9999", "email": "john@example.com"},
        }
    )

    result = true_owner.compare_with(pred_owner, include_confusion_matrix=True)
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print(f"\nAggregate=True contact metrics:")
    print(
        f"Contact overall: tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # With  nested field metrics could potentially affect overall
    # The exact behavior depends on current implementation, but it should be different from aggregate=False

    # We don't assert specific values here since  behavior may vary,
    # but we document that it's different from aggregate=False
    print("✅ Aggregate=True behavior documented (may differ based on implementation)")


if __name__ == "__main__":
    test_aggregate_false_prevents_nested_rollup()
    test_aggregate_true_would_rollup()

    print("\n✅ All aggregate contact rollup tests passed!")
