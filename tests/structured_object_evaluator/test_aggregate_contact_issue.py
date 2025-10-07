# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""Test to investigate aggregate contact totals issue from the notebook."""

from typing import Optional
from pprint import pprint

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
    id: int = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    contact: Contact = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0,
        aggregate=False,  # Let's test with aggregate=False first
    )


def test_contact_object_level_metrics_not_rollup():
    """Test that contact object-level metrics are NOT rolled up from nested fields."""

    # Create the test data from the notebook
    true_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {"phone": "555-689-1234"},  # email is None/missing
        }
    )

    pred_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {
                "phone": "666-689-1234",  # false discovery (different from true)
                "email": "sjohnson@example.com",  # false alarm (not in true)
            },
        }
    )

    # Verify contact similarity and threshold
    contact_similarity = true_owner.contact.compare(pred_owner.contact)
    contact_threshold = true_owner._get_comparison_info("contact").threshold

    # Get the comparison result
    result = true_owner.compare_with(
        pred_owner, include_confusion_matrix=True, add_derived_metrics=False
    )

    # Extract contact-level metrics
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    # CRITICAL ASSERTION: Contact object-level should be based on object similarity, not nested field rollup
    # Since contact_similarity=0.0 and threshold=1.0, this should be 1 FD (false discovery)

    print(
        f"\nDEBUG: Contact similarity={contact_similarity}, threshold={contact_threshold}"
    )
    print(
        f"DEBUG: Contact metrics - tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # EXPECTED: Contact as object didn't match (similarity=0.0 < threshold=1.0) = 1 FD
    # CRITICAL: With 1 contact object vs 1 contact object, max errors = 1
    expected_contact_fd = 1  # The single contact object didn't match
    expected_contact_fa = 0  # No false alarms (both sides have 1 contact)
    expected_contact_fp = 1  # fp = fa + fd = 0 + 1 = 1
    expected_contact_tp = 0  # No true positives
    expected_contact_tn = 0  # No true negatives
    expected_contact_fn = 0  # No false negatives

    # CURRENT BUG: These will fail because it's rolling up nested field metrics (fd=1, fa=1, fp=2)
    # instead of treating this as a single object comparison (fd=1, fa=0, fp=1)
    assert contact_cm["fd"] == expected_contact_fd, (
        f"Expected contact fd=1 (1 object didn't match), got fd={contact_cm['fd']}"
    )
    assert contact_cm["fa"] == expected_contact_fa, (
        f"Expected contact fa=0 (1v1 comparison, no false alarms), got fa={contact_cm['fa']}"
    )
    assert contact_cm["fp"] == expected_contact_fp, (
        f"Expected contact fp=1 (max 1 for singleton), got fp={contact_cm['fp']}"
    )
    assert contact_cm["tp"] == expected_contact_tp, (
        f"Expected contact tp=0, got tp={contact_cm['tp']}"
    )
    assert contact_cm["fn"] == expected_contact_fn, (
        f"Expected contact fn=0 (1v1 comparison, no missing objects), got fn={contact_cm['fn']}"
    )


def test_contact_nested_fields_still_available():
    """Test that nested field details are still available for debugging."""

    true_owner = Owner(
        **{"id": 1501, "name": "Sarah Johnson", "contact": {"phone": "555-689-1234"}}
    )

    pred_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {"phone": "666-689-1234", "email": "sjohnson@example.com"},
        }
    )

    result = true_owner.compare_with(
        pred_owner, include_confusion_matrix=True, add_derived_metrics=False
    )

    # Verify nested field details are present
    contact_fields = result["confusion_matrix"]["fields"]["contact"]["fields"]

    assert "phone" in contact_fields, "Phone field details should be available"
    assert "email" in contact_fields, "Email field details should be available"

    # Check nested field metrics are correct
    phone_metrics = contact_fields["phone"]["overall"]
    email_metrics = contact_fields["email"]["overall"]

    # Phone: different values = 1 FD
    assert phone_metrics["fd"] == 1, (
        f"Phone should have 1 FD, got {phone_metrics['fd']}"
    )
    assert phone_metrics["fp"] == 1, (
        f"Phone should have 1 FP, got {phone_metrics['fp']}"
    )

    # Email: None vs value = 1 FA
    assert email_metrics["fa"] == 1, (
        f"Email should have 1 FA, got {email_metrics['fa']}"
    )
    assert email_metrics["fp"] == 1, (
        f"Email should have 1 FP, got {email_metrics['fp']}"
    )


def test_contact_matching_case():
    """Test the case where contact objects DO match."""

    true_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {"phone": "555-689-1234", "email": "sarah@example.com"},
        }
    )

    pred_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {
                "phone": "555-689-1234",  # matches
                "email": "sarah@example.com",  # matches
            },
        }
    )

    # Verify contact similarity meets threshold
    contact_similarity = true_owner.contact.compare(pred_owner.contact)
    contact_threshold = true_owner._get_comparison_info("contact").threshold

    result = true_owner.compare_with(
        pred_owner, include_confusion_matrix=True, add_derived_metrics=False
    )
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print(
        f"\nMatching case: Contact similarity={contact_similarity}, threshold={contact_threshold}"
    )
    print(
        f"Contact metrics - tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # Since both fields match exactly, contact similarity should be 1.0 >= threshold=1.0 = TP
    assert contact_cm["tp"] == 1, (
        f"Expected contact tp=1 (contact matched), got tp={contact_cm['tp']}"
    )
    assert contact_cm["fp"] == 0, (
        f"Expected contact fp=0 (no false positives), got fp={contact_cm['fp']}"
    )
    assert contact_cm["fa"] == 0, (
        f"Expected contact fa=0 (no false alarms), got fa={contact_cm['fa']}"
    )
    assert contact_cm["fd"] == 0, (
        f"Expected contact fd=0 (no false discoveries), got fd={contact_cm['fd']}"
    )


def test_contact_partial_match_case():
    """Test case where contact has partial match (some fields match, some don't)."""

    true_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {"phone": "555-689-1234", "email": "sarah@example.com"},
        }
    )

    pred_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {
                "phone": "555-689-1234",  # matches
                "email": "different@example.com",  # doesn't match
            },
        }
    )

    # Check contact similarity
    contact_similarity = true_owner.contact.compare(pred_owner.contact)
    contact_threshold = true_owner._get_comparison_info("contact").threshold

    result = true_owner.compare_with(
        pred_owner, include_confusion_matrix=True, add_derived_metrics=False
    )
    contact_cm = result["confusion_matrix"]["fields"]["contact"]["overall"]

    print(
        f"\nPartial match case: Contact similarity={contact_similarity}, threshold={contact_threshold}"
    )
    print(
        f"Contact metrics - tp={contact_cm['tp']}, fa={contact_cm['fa']}, fd={contact_cm['fd']}, fp={contact_cm['fp']}"
    )

    # With 50% match (1 of 2 fields), similarity should be 0.5 < threshold=1.0 = FD
    assert contact_similarity == 0.5, (
        f"Expected contact similarity=0.5, got {contact_similarity}"
    )
    assert contact_cm["fd"] == 1, (
        f"Expected contact fd=1 (partial match < threshold), got fd={contact_cm['fd']}"
    )
    assert contact_cm["fp"] == 1, f"Expected contact fp=1, got fp={contact_cm['fp']}"


def test_correct_behavior_demonstration():
    """Demonstrate what the correct behavior should be."""
    print("\n=== CORRECT BEHAVIOR DEMONSTRATION ===")

    # Create the same test data
    true_owner = Owner(
        **{"id": 1501, "name": "Sarah Johnson", "contact": {"phone": "555-689-1234"}}
    )

    pred_owner = Owner(
        **{
            "id": 1501,
            "name": "Sarah Johnson",
            "contact": {"phone": "666-689-1234", "email": "sjohnson@example.com"},
        }
    )

    # Test contact similarity
    contact_similarity = true_owner.contact.compare(pred_owner.contact)
    contact_threshold = true_owner._get_comparison_info("contact").threshold

    print(f"Contact similarity score: {contact_similarity}")
    print(f"Contact threshold: {contact_threshold}")
    print(f"Contact matches threshold: {contact_similarity >= contact_threshold}")

    print(f"\nFor aggregate=False, contact field overall should be:")
    if contact_similarity >= contact_threshold:
        print("tp=1, fa=0, fd=0, fp=0, tn=0, fn=0")
    else:
        print(
            "tp=0, fa=0, fd=1, fp=1, tn=0, fn=0  (1 false discovery - contact object didn't match)"
        )

    print(f"\nNested field details (for debugging) should show:")
    print("phone: fd=1 (phone values don't match)")
    print("email: fa=1 (email in pred but not in true)")
    print("But these should NOT roll up to contact overall when aggregate=False")


if __name__ == "__main__":
    test_aggregate_contact_debug()
