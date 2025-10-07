"""
Test module for evaluating double-nested objects in StructuredModelEvaluator.

This test verifies that metrics are correctly calculated for objects with multiple
levels of nesting (e.g., A contains B contains C).
"""

import pytest
from typing import List, Optional

from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    NonMatchField,
    NonMatchType,
    StructuredModelEvaluator,
)
from stickler.comparators.levenshtein import LevenshteinComparator


# Define test models with double nesting
class ContactInfo(StructuredModel):
    """Test model for contact information."""

    email: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    phone: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )


class Address(StructuredModel):
    """Test model for an address with contact info (double nesting)."""

    street: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    city: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    state: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    zip_code: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )
    contact_info: Optional[ContactInfo] = ComparableField()  # Double-nested field


class Person(StructuredModel):
    """Test model for a person."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )
    age: int = ComparableField(threshold=1.0)  # Exact match for age
    address: Address = ComparableField()
    phone_numbers: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )


def test_double_nested_metrics():
    """Test that metrics are properly calculated for double nested objects."""
    # Create ground truth data with double nesting
    gt_contact_info = ContactInfo(email="contact@example.com", phone="555-987-6543")

    gt_address = Address(
        street="123 Main Street",
        city="New York",
        state="NY",
        zip_code="10001",
        contact_info=gt_contact_info,
    )

    gt_person = Person(
        name="John Smith",
        age=30,
        address=gt_address,
        phone_numbers=["555-123-4567", "555-765-4321"],
    )

    # Create prediction with various types of mismatches in the double-nested object
    pred_contact_info = ContactInfo(
        email="contact@exmple.com",  # Typo (missing 'a')
        phone=None,  # Missing value - should be a false negative
    )

    pred_address = Address(
        street="123 Main St",  # Abbreviation - should match with high score
        city="New Yrok",  # Typo - might match with lower score
        state="New York",  # Wrong format - likely won't match
        zip_code=None,  # Missing field - false negative
        contact_info=pred_contact_info,  # Double-nested object with issues
    )

    pred_person = Person(
        name="Jon Smith",  # Missing 'h' - should still match with good score
        age=31,  # Wrong age - false discovery
        address=pred_address,
        phone_numbers=[
            "555-123-4567",
            "555-765-9999",
            "555-111-2222",
        ],  # Extra phone number - false alarm
    )

    # Create evaluator with non-match documentation enabled
    evaluator = StructuredModelEvaluator(threshold=0.8, document_non_matches=True)

    # Clear any existing non-match documents
    evaluator.clear_non_match_documents()

    # Evaluate prediction against ground truth
    result = evaluator.evaluate(gt_person, pred_person)

    # Verify non-matches were documented for double-nested fields
    assert len(result["non_matches"]) > 0, "Expected non-matches to be documented"

    # Check specifically for the double-nested fields
    double_nested_non_matches = [
        nm
        for nm in result["non_matches"]
        if nm["field_path"].startswith("address.contact_info.")
    ]

    # There should be at least one non-match in the double-nested field (false negative for phone)
    assert len(double_nested_non_matches) > 0, (
        "Expected non-matches in double-nested fields"
    )

    # The false negative for contact_info.phone should be documented
    phone_non_matches = [
        nm
        for nm in double_nested_non_matches
        if nm["field_path"] == "address.contact_info.phone"
    ]
    assert len(phone_non_matches) == 1, (
        "Expected a non-match for address.contact_info.phone"
    )
    assert phone_non_matches[0]["non_match_type"] == NonMatchType.FALSE_NEGATIVE, (
        "Expected FALSE_NEGATIVE for contact_info.phone"
    )

    # Verify metrics for nested fields in confusion matrix
    confusion_matrix = result["confusion_matrix"]

    # Access double-nested fields
    address_fields = confusion_matrix["fields"]["address"].get("fields", {})
    assert "contact_info" in address_fields, "Missing metrics for contact_info field"

    contact_info_fields = address_fields["contact_info"]["fields"]
    assert "email" in contact_info_fields, (
        "Missing metrics for double-nested email field"
    )
    assert "phone" in contact_info_fields, (
        "Missing metrics for double-nested phone field"
    )

    # Verify metrics for the phone field - need to check available keys
    phone_metrics = contact_info_fields["phone"]

    # Add debugging to see what keys are actually available
    print(f"Available phone metrics keys: {list(phone_metrics.keys())}")

    # Check either directly or in overall depending on structure
    if "fn" in phone_metrics:
        assert phone_metrics["fn"] == 1
    elif "overall" in phone_metrics and "fn" in phone_metrics["overall"]:
        assert phone_metrics["overall"]["fn"] == 1
    else:
        assert False, (
            f"Expected 'fn' or 'overall.fn' in phone metrics, got keys: {list(phone_metrics.keys())}"
        )

    # Verify the overall counts reflect object-level metrics (not field rollups)
    overall = confusion_matrix.get("overall", {})
    # With object-level counting, all objects (Person, Address, ContactInfo) are present
    # so there should be no object-level false negatives, even if individual fields don't match
    assert overall["fn"] == 0, (
        f"Expected overall fn=0 since all objects are present (object-level counting, not field rollup)"
    )

    # Print summary for demonstration purposes
    print("\nDouble-nested object test results:")
    print(f"Confusion matrix fields: {list(confusion_matrix['fields'].keys())}")
    print(f"Overall metrics: {overall}")
    print("Non-match documents for double-nested fields:")
    for nm in double_nested_non_matches:
        print(f"  {nm['field_path']}: {nm['non_match_type']}")


def test_double_nested_null_contact_info():
    """Test handling of null values in double nested objects."""
    # Create ground truth with null value for the double-nested object
    gt_address = Address(
        street="123 Main Street",
        city="New York",
        state="NY",
        zip_code="10001",
        contact_info=None,  # Null double-nested object
    )

    gt_person = Person(
        name="John Smith", age=30, address=gt_address, phone_numbers=["555-123-4567"]
    )

    # Create prediction with a value for the double-nested object (should be a false alarm)
    pred_contact_info = ContactInfo(email="contact@example.com", phone="555-987-6543")

    pred_address = Address(
        street="123 Main Street",
        city="New York",
        state="NY",
        zip_code="10001",
        contact_info=pred_contact_info,  # Present when it should be null
    )

    pred_person = Person(
        name="John Smith", age=30, address=pred_address, phone_numbers=["555-123-4567"]
    )

    # Create evaluator
    evaluator = StructuredModelEvaluator(threshold=0.8, document_non_matches=True)
    evaluator.clear_non_match_documents()

    # Evaluate
    result = evaluator.evaluate(gt_person, pred_person)

    # There should be a false alarm for contact_info
    contact_info_non_matches = [
        nm for nm in result["non_matches"] if nm["field_path"] == "address.contact_info"
    ]

    assert len(contact_info_non_matches) == 1, (
        "Expected a non-match for address.contact_info"
    )
    assert contact_info_non_matches[0]["non_match_type"] == NonMatchType.FALSE_ALARM, (
        "Expected FALSE_ALARM for contact_info"
    )

    # Check confusion matrix metrics - access through hierarchical structure
    address_fields = result["confusion_matrix"]["fields"]["address"]["fields"]
    assert "contact_info" in address_fields, "Missing metrics for contact_info field"

    # The false alarm count for contact_info should be 1
    contact_info_metrics = address_fields["contact_info"]

    # Add debugging to see what keys are actually available
    print(f"Available contact_info metrics keys: {list(contact_info_metrics.keys())}")

    # Check either directly or in overall depending on structure
    if "fa" in contact_info_metrics:
        assert contact_info_metrics["fa"] == 1, "Expected fa=1 for contact_info"
        assert contact_info_metrics["fp"] == 1, "Expected fp=1 for contact_info"
    elif "overall" in contact_info_metrics and "fa" in contact_info_metrics["overall"]:
        assert contact_info_metrics["overall"]["fa"] == 1, (
            "Expected overall.fa=1 for contact_info"
        )
        assert contact_info_metrics["overall"]["fp"] == 1, (
            "Expected overall.fp=1 for contact_info"
        )
    else:
        assert False, (
            f"Expected 'fa'/'fp' or 'overall.fa'/'overall.fp' in metrics, got keys: {list(contact_info_metrics.keys())}"
        )
