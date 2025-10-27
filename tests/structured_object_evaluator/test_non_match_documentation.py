"""
Test module for non-match documentation feature in StructuredModelEvaluator.

This test demonstrates how to use and benefit from the non-match documentation
functionality, which captures detailed information about false positives and false negatives.
"""

from typing import List, Optional

from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    NonMatchType,
    StructuredModelEvaluator,
)
from stickler.comparators.levenshtein import LevenshteinComparator


# Define test models
class Address(StructuredModel):
    """Test model for an address."""

    street: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    city: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    state: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)
    zip_code: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )


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

    # Create evaluator with non-match documentation enabled
    evaluator = StructuredModelEvaluator(threshold=0.8, document_non_matches=True)

    # Clear any existing non-match documents
    evaluator.clear_non_match_documents()

    # Evaluate prediction against ground truth
    result = evaluator.evaluate(gt_person, pred_person)

    # Verify non-matches were documented
    assert len(result["non_matches"]) > 0, "Expected non-matches to be documented"

    # Find false discoveries (both values present but don't match)
    false_discoveries = [
        nm
        for nm in evaluator.non_match_documents
        if nm.non_match_type == NonMatchType.FALSE_DISCOVERY
    ]
    assert len(false_discoveries) > 0, "Expected false discoveries to be documented"

    # Find false negatives (ground truth present, prediction missing)
    false_negatives = [
        nm
        for nm in evaluator.non_match_documents
        if nm.non_match_type == NonMatchType.FALSE_NEGATIVE
    ]
    assert len(false_negatives) > 0, "Expected false negatives to be documented"

    # Find false alarms (ground truth missing, prediction present)
    false_alarms = [
        nm
        for nm in evaluator.non_match_documents
        if nm.non_match_type == NonMatchType.FALSE_ALARM
    ]
    assert len(false_alarms) > 0, "Expected false alarms to be documented"

    # Print summary for demonstration purposes
    print("\nNon-match documentation test example:")
    print(f"Total non-matches: {len(evaluator.non_match_documents)}")
    print(f"False discoveries: {len(false_discoveries)}")
    print(f"False negatives: {len(false_negatives)}")
    print(f"False alarms: {len(false_alarms)}")

    # Demonstrate basic access to non-match fields
    for nm in evaluator.non_match_documents:
        print(f"  {nm.field_path}: {nm.non_match_type}")

    # Verify the non-matches are included in the evaluation result
    assert "non_matches" in result, "Expected non_matches in evaluation result"
    assert len(result["non_matches"]) == len(evaluator.non_match_documents), (
        "Non-match count mismatch"
    )


def test_non_match_documentation_disabled():
    """Test that no non-matches are documented when the feature is disabled."""
    # Create simple test data
    gt = Address(street="123 Main Street", city="New York", state="NY", zip_code=None)
    pred = Address(street="123 Main St", city="New Yrok", state="N.Y.", zip_code=None)

    # Create evaluator with non-match documentation disabled
    evaluator = StructuredModelEvaluator(document_non_matches=False)

    # Clear any existing non-match documents
    evaluator.clear_non_match_documents()

    # Evaluate
    evaluator.evaluate(gt, pred)

    # Verify no non-matches were documented
    assert len(evaluator.non_match_documents) == 0, (
        "Expected no non-matches to be documented"
    )
