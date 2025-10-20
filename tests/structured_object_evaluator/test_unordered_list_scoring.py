"""
Test to validate that unordered list items in StructuredModel are matched correctly
using Hungarian matching rather than strict ordering.

This test is migrated from the anls_star_lib directory to ensure test coverage
is maintained during the refactor.
"""

from typing import List, Optional

from stickler.structured_object_evaluator import StructuredModel, ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator


class SimpleItem(StructuredModel):
    """Simple model for list items."""

    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)

    value: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7
    )


class ListContainer(StructuredModel):
    """Container model with a list of items."""

    items: List[SimpleItem] = ComparableField()


def test_unordered_list_scoring():
    """Test that lists with the same items but in different order should match perfectly."""
    # Create ground truth with items in one order
    gt = ListContainer(
        items=[
            SimpleItem(name="Item A", value="Value 1"),
            SimpleItem(name="Item B", value="Value 2"),
            SimpleItem(name="Item C", value="Value 3"),
        ]
    )

    # Create prediction with items in a different order
    pred = ListContainer(
        items=[
            SimpleItem(name="Item B", value="Value 2"),
            SimpleItem(name="Item C", value="Value 3"),
            SimpleItem(name="Item A", value="Value 1"),
        ]
    )

    # Perform direct comparison
    result = gt.compare_with(pred)

    # With Hungarian matching, this should be a perfect match with score 1.0
    assert result["field_scores"]["items"] == 1.0, (
        "Lists with same items in different order should match perfectly"
    )

    # Overall score should also be 1.0
    assert result["overall_score"] == 1.0


def test_unordered_list_with_partial_matches():
    """Test list scoring with items that only partially match."""
    # Create ground truth
    gt = ListContainer(
        items=[
            SimpleItem(name="Item A", value="Value 1"),
            SimpleItem(name="Item B", value="Value 2"),
            SimpleItem(name="Item C", value="Value 3"),
        ]
    )

    # Create prediction with significantly different values and names to ensure partial matches
    pred = ListContainer(
        items=[
            SimpleItem(name="Item B", value="Value 2"),  # Perfect match
            SimpleItem(
                name="Different Item", value="Completely Different String"
            ),  # Partial match
            SimpleItem(
                name="Another Item", value="Totally New Value XYZ"
            ),  # Partial match
        ]
    )

    # Perform direct comparison
    result = gt.compare_with(pred)

    # Since there are partial matches, the score should be between 0 and 1
    items_score = result["field_scores"]["items"]
    assert 0 < items_score < 1, (
        f"Lists with partially matching items should have score between 0 and 1, got {items_score}"
    )


def test_unordered_list_with_different_lengths():
    """Test Hungarian matching when lists have different lengths."""
    # Create ground truth with 3 items
    gt = ListContainer(
        items=[
            SimpleItem(name="Item A", value="Value 1"),
            SimpleItem(name="Item B", value="Value 2"),
            SimpleItem(name="Item C", value="Value 3"),
        ]
    )

    # Create prediction with 2 items (missing one)
    pred = ListContainer(
        items=[
            SimpleItem(name="Item C", value="Value 3"),
            SimpleItem(name="Item A", value="Value 1"),
        ]
    )

    # Perform direct comparison
    result = gt.compare_with(pred)

    # The score should be approximately 2/3 = ~0.67 (since 2 out of 3 match perfectly)
    items_score = result["field_scores"]["items"]
    assert 0.65 < items_score < 0.7, (
        f"Lists with 2/3 matching items should have score around 0.67, got {items_score}"
    )

    # Now test the opposite case: prediction has more items than ground truth
    gt2 = ListContainer(
        items=[
            SimpleItem(name="Item A", value="Value 1"),
            SimpleItem(name="Item B", value="Value 2"),
        ]
    )

    pred2 = ListContainer(
        items=[
            SimpleItem(name="Item A", value="Value 1"),
            SimpleItem(name="Item B", value="Value 2"),
            SimpleItem(name="Item C", value="Value 3"),
        ]
    )

    # Perform direct comparison
    result2 = gt2.compare_with(pred2)

    # The score should be approximately 2/3 = ~0.67 (since 2 out of 3 match perfectly)
    items_score2 = result2["field_scores"]["items"]
    assert 0.65 < items_score2 < 0.7, (
        f"Lists with prediction having extra item should have score around 0.67, got {items_score2}"
    )


def test_empty_list_handling():
    """Test handling of empty lists in both ground truth and prediction."""
    # Create ground truth and prediction with empty lists
    gt_empty = ListContainer(items=[])
    pred_empty = ListContainer(items=[])

    # Create ground truth and prediction with non-empty lists
    gt_full = ListContainer(
        items=[
            SimpleItem(name="Item A", value="Value 1"),
            SimpleItem(name="Item B", value="Value 2"),
        ]
    )

    # Test empty vs empty (should be perfect match)
    result1 = gt_empty.compare_with(pred_empty)
    assert result1["field_scores"]["items"] == 1.0

    # Test full vs empty (should be 0 score)
    result2 = gt_full.compare_with(pred_empty)
    assert result2["field_scores"]["items"] == 0.0

    # Test empty vs full (should be 0 score)
    result3 = gt_empty.compare_with(gt_full)
    assert result3["field_scores"]["items"] == 0.0


def test_list_with_complex_nested_items():
    """Test lists containing complex structured model items."""

    # Define a nested model for testing
    class NestedItem(StructuredModel):
        name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
        details: SimpleItem = ComparableField()

    # Define a container for nested items
    class ComplexContainer(StructuredModel):
        items: List[NestedItem] = ComparableField()

    # Create ground truth with nested items
    gt = ComplexContainer(
        items=[
            NestedItem(
                name="Complex A", details=SimpleItem(name="Detail 1", value="Value X")
            ),
            NestedItem(
                name="Complex B", details=SimpleItem(name="Detail 2", value="Value Y")
            ),
        ]
    )

    # Create prediction with same nested items in different order
    pred = ComplexContainer(
        items=[
            NestedItem(
                name="Complex B", details=SimpleItem(name="Detail 2", value="Value Y")
            ),
            NestedItem(
                name="Complex A", details=SimpleItem(name="Detail 1", value="Value X")
            ),
        ]
    )

    # Perform direct comparison
    result = gt.compare_with(pred)

    # With perfect matching but different order, should be 1.0
    assert result["field_scores"]["items"] == 1.0
    assert result["overall_score"] == 1.0
