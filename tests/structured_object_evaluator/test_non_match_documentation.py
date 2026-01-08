"""
Test module for non-match documentation feature in compare_with().

This test demonstrates how to use and benefit from the non-match documentation
functionality, which captures detailed information about false positives and false negatives.
"""

from typing import List, Optional

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator import (
    ComparableField,
    NonMatchType,
    StructuredModel,
)


# Define test models
class Address(StructuredModel):
    """Test model for an address."""
    
    match_threshold = 0.8

    street: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    city: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    state: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    zip_code: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )


class Person(StructuredModel):
    """Test model for a person."""
    
    match_threshold = 0.8

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )
    age: int = ComparableField(threshold=1.0)  # Exact match for age
    address: Address = ComparableField()
    phone_numbers: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )


def test_document_non_matches():
    """Test that non-matches are properly documented."""
    # Create ground truth data
    gt_address = Address(
        street="123 Main Street", city="New York", state="NY", zip_code="10001"
    )
    gt_person = Person(
        name="John Smith",
        age=30,
        address=gt_address,
        phone_numbers=["555-123-4567", "555-765-4321"],
    )

    # Create prediction with various types of mismatches
    pred_address = Address(
        street="123 Main St",  # Abbreviation - should match with high score
        city="New Yrok",  # Typo - might match with lower score
        state="New York",  # Wrong format - likely won't match
        zip_code=None,  # Missing field - false negative
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

    # Evaluate prediction against ground truth with non-match documentation
    result = gt_person.compare_with(pred_person, include_confusion_matrix=True, document_non_matches=True)

    # Verify non-matches were documented
    assert len(result["non_matches"]) > 0, "Expected non-matches to be documented"

    # Find false discoveries (both values present but don't match)
    false_discoveries = [
        nm
        for nm in result["non_matches"]
        if nm["non_match_type"] == NonMatchType.FALSE_DISCOVERY
    ]
    assert len(false_discoveries) > 0, "Expected false discoveries to be documented"

    # Find false negatives (ground truth present, prediction missing)
    false_negatives = [
        nm
        for nm in result["non_matches"]
        if nm["non_match_type"] == NonMatchType.FALSE_NEGATIVE
    ]
    assert len(false_negatives) > 0, "Expected false negatives to be documented"

    # Find false alarms (ground truth missing, prediction present)
    false_alarms = [
        nm
        for nm in result["non_matches"]
        if nm["non_match_type"] == NonMatchType.FALSE_ALARM
    ]
    assert len(false_alarms) > 0, "Expected false alarms to be documented"

    # Print summary for demonstration purposes
    print("\nNon-match documentation test example:")
    print(f"Total non-matches: {len(result['non_matches'])}")
    print(f"False discoveries: {len(false_discoveries)}")
    print(f"False negatives: {len(false_negatives)}")
    print(f"False alarms: {len(false_alarms)}")

    # Demonstrate basic access to non-match fields
    for nm in result["non_matches"]:
        print(f"  {nm['field_path']}: {nm['non_match_type']}")

    # Verify the non-matches are included in the evaluation result
    assert "non_matches" in result, "Expected non_matches in evaluation result"


def test_non_match_documentation_disabled():
    """Test that no non-matches are documented when the feature is disabled."""
    # Create simple test data
    gt = Address(street="123 Main Street", city="New York", state="NY", zip_code=None)
    pred = Address(street="123 Main St", city="New Yrok", state="N.Y.", zip_code=None)

    # Evaluate with non-match documentation disabled
    result = gt.compare_with(pred, include_confusion_matrix=True, document_non_matches=False)

    # Verify no non-matches were documented
    assert "non_matches" not in result or len(result.get("non_matches", [])) == 0, (
        "Expected no non-matches to be documented"
    )
