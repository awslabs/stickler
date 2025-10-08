"""
Test the three possible leaf conditions for comparison in a structured object.
1. A value (ComparableField)
2. A List[ComparableField]
3. A nested object (StructuredModel)
"""

import pytest
from typing import Dict, Any, List, Optional

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


# 1. Test for a simple value (ComparableField)
def test_comparable_field_value():
    """Test comparison of a simple value field."""

    class SimpleValueModel(StructuredModel):
        """Model with a simple string value field."""

        name: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
        )

    # Test with exact match
    gt = SimpleValueModel(name="John Doe")
    pred = SimpleValueModel(name="John Doe")

    evaluator = StructuredModelEvaluator(threshold=0.5)
    results = evaluator.evaluate(gt, pred)

    # Check that exact matches get perfect score
    assert results["fields"]["name"]["anls_score"] == 1.0
    assert results["overall"]["anls_score"] == 1.0

    # Test with slightly different value (similar but not exact)
    pred_similar = SimpleValueModel(name="Jon Doe")
    results_similar = evaluator.evaluate(gt, pred_similar)

    # Should be high but not perfect score
    assert 0.7 <= results_similar["fields"]["name"]["anls_score"] < 1.0
    assert 0.7 <= results_similar["overall"]["anls_score"] < 1.0

    # Test with completely different value
    pred_different = SimpleValueModel(name="Jane Smith")
    results_different = evaluator.evaluate(gt, pred_different)

    # Should be below threshold
    assert results_different["fields"]["name"]["anls_score"] < 0.7


# 2. Test for a list of values (List[ComparableField])
def test_list_of_comparable_fields():
    """Test comparison of a field containing a list of values."""

    class ListFieldModel(StructuredModel):
        """Model with a field containing a list of strings."""

        tags: List[str] = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
        )

    # Need to properly separate strings in lists with commas
    gt = ListFieldModel(tags=["red", "blue", "green"])
    pred = ListFieldModel(tags=["red", "blue", "green"])

    # Use evaluator instead of direct comparison for consistency with other tests
    evaluator = StructuredModelEvaluator(threshold=0.5)
    results = evaluator.evaluate(gt, pred)

    # Check that exact matches get perfect score
    assert results["overall"]["anls_score"] == 1.0

    # Test with same items but different order
    # The implementation now uses Hungarian matching for all list types, making order irrelevant
    pred_reordered = ListFieldModel(tags=["green", "red", "blue"])
    results_reordered = evaluator.evaluate(gt, pred_reordered)

    # Should get perfect score as order doesn't matter (using Hungarian matching)
    assert results_reordered["overall"]["anls_score"] == 1.0

    # Test with different length lists
    pred_missing = ListFieldModel(tags=["red", "blue"])  # Missing "green"
    results_missing = evaluator.evaluate(gt, pred_missing)

    # Score should be lower due to missing item
    assert results_missing["overall"]["anls_score"] < 1.0


# 3. Test for a nested structured model (StructuredModel)
def test_nested_structured_model():
    """Test comparison of a field containing a nested StructuredModel."""

    class AddressModel(StructuredModel):
        """Nested model for representing an address."""

        street: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
        )
        city: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
        )

    class PersonModel(StructuredModel):
        """Model with a nested StructuredModel field."""

        name: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
        )
        address: AddressModel = ComparableField(
            comparator=LevenshteinComparator(),  # This won't be used directly for structured models
            threshold=0.7,
            weight=1.0,
        )

    # Test with exact match
    gt_address = AddressModel(street="123 Main St", city="Springfield")
    gt = PersonModel(name="John Doe", address=gt_address)

    pred_address = AddressModel(street="123 Main St", city="Springfield")
    pred = PersonModel(name="John Doe", address=pred_address)

    evaluator = StructuredModelEvaluator(threshold=0.5)
    results = evaluator.evaluate(gt, pred)

    # Check that exact matches get perfect score
    assert results["fields"]["name"]["anls_score"] == 1.0
    assert results["fields"]["address"]["anls_score"] == 1.0  # Nested model comparison
    assert results["overall"]["anls_score"] == 1.0

    # Test with differences in the nested model
    pred_address_diff = AddressModel(
        street="123 Main St", city="Shelbyville"
    )  # Different city
    pred_diff = PersonModel(name="John Doe", address=pred_address_diff)

    results_diff = evaluator.evaluate(gt, pred_diff)

    # The name should still match perfectly
    assert results_diff["fields"]["name"]["anls_score"] == 1.0

    # The address field should have a lower score due to the city difference
    assert results_diff["fields"]["address"]["anls_score"] < 1.0

    # Overall score should be between the two field scores
    assert results_diff["overall"]["anls_score"] < 1.0


# 4. Test for a field containing a list of structured models
def test_list_of_structured_models():
    """Test comparison of a field containing a list of StructuredModels."""

    class ItemModel(StructuredModel):
        """Simple model for items in a list."""

        match_threshold = 0.7

        name: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
        )
        quantity: int = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
        )

    class ShoppingListModel(StructuredModel):
        """Model with a list of nested StructuredModels."""

        items: List[ItemModel] = ComparableField(
            comparator=LevenshteinComparator(),  # Used for comparing individual items
            weight=1.0,
        )

    # Create ground truth with a list of items
    gt_items = [
        ItemModel(name="apple", quantity=5),
        ItemModel(name="banana", quantity=3),
        ItemModel(name="orange", quantity=2),
    ]
    gt = ShoppingListModel(items=gt_items)

    # Use evaluator instead of direct comparison for consistency with other tests
    evaluator = StructuredModelEvaluator(threshold=0.5)

    # Test with exact match
    pred_items_exact = [
        ItemModel(name="apple", quantity=5),
        ItemModel(name="banana", quantity=3),
        ItemModel(name="orange", quantity=2),
    ]
    pred_exact = ShoppingListModel(items=pred_items_exact)
    results_exact = evaluator.evaluate(gt, pred_exact)

    # Check that exact matches get perfect score
    assert results_exact["overall"]["anls_score"] == 1.0

    # Test with same items but in different order
    pred_items_reordered = [
        ItemModel(name="orange", quantity=2),
        ItemModel(name="apple", quantity=5),
        ItemModel(name="banana", quantity=3),
    ]
    pred_reordered = ShoppingListModel(items=pred_items_reordered)
    results_reordered = evaluator.evaluate(gt, pred_reordered)

    # Should still get perfect score as order doesn't matter
    assert results_reordered["overall"]["anls_score"] == 1.0

    # Test with item differences
    pred_items_diff = [
        ItemModel(name="apple", quantity=4),  # Different quantity
        ItemModel(name="pear", quantity=3),  # Different name
        ItemModel(name="orange", quantity=2),
    ]
    pred_diff = ShoppingListModel(items=pred_items_diff)
    results_diff = evaluator.evaluate(gt, pred_diff)

    # Score should be lower due to differences
    assert results_diff["overall"]["anls_score"] < 1.0

    # Test with missing items
    pred_items_missing = [
        ItemModel(name="apple", quantity=5),
        ItemModel(name="orange", quantity=2),
    ]  # Missing banana
    pred_missing = ShoppingListModel(items=pred_items_missing)
    results_missing = evaluator.evaluate(gt, pred_missing)

    # Score should be lower due to missing item
    assert results_missing["overall"]["anls_score"] < 1.0
