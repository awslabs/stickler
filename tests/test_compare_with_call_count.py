"""Test that compare_with doesn't unnecessarily loop/call comparators multiple times."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from stickler import StructuredModel, ComparableField
from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator


def test_compare_with_single_call_per_field():
    """Test that compare_with calls each field's comparator exactly once."""

    class SimpleModel(StructuredModel):
        """Simple model for testing."""

        name: str = ComparableField(threshold=0.7)
        description: str = ComparableField(threshold=0.7)

    # Create models
    gt = SimpleModel(name="John", description="A person")
    pred = SimpleModel(name="Jon", description="A human")

    # Patch the compare method on LevenshteinComparator to track calls
    with patch.object(
        LevenshteinComparator, "compare", wraps=LevenshteinComparator().compare
    ) as mock_compare:
        # Perform comparison
        result = gt.compare_with(pred)

        # Verify the comparator was called exactly twice (once for each field)
        assert mock_compare.call_count == 2, (
            f"Comparator called {mock_compare.call_count} times, expected 2"
        )

        # Verify the correct values were passed
        calls = mock_compare.call_args_list
        call_values = [(call[0][0], call[0][1]) for call in calls]

        # Should have calls for both fields
        assert ("John", "Jon") in call_values or ("Jon", "John") in call_values
        assert ("A person", "A human") in call_values or (
            "A human",
            "A person",
        ) in call_values


def test_compare_with_confusion_matrix_no_extra_calls():
    """Test that enabling confusion matrix doesn't cause extra comparator calls."""

    class SimpleModel(StructuredModel):
        """Simple model for testing."""

        name: str = ComparableField(threshold=0.7)
        description: str = ComparableField(threshold=0.7)

    # Create models
    gt = SimpleModel(name="Alice", description="A developer")
    pred = SimpleModel(name="Alice", description="A programmer")

    # Patch the compare method to track calls
    with patch.object(
        LevenshteinComparator, "compare", wraps=LevenshteinComparator().compare
    ) as mock_compare:
        # Perform comparison with confusion matrix enabled
        result = gt.compare_with(pred, include_confusion_matrix=True)

        # Verify the comparator was still called exactly twice
        assert mock_compare.call_count == 2, (
            f"Comparator called {mock_compare.call_count} times, expected 2"
        )


def test_compare_with_all_options_no_extra_calls():
    """Test that enabling all options doesn't cause extra comparator calls."""

    class SimpleModel(StructuredModel):
        """Simple model for testing."""

        name: str = ComparableField(threshold=0.7)
        description: str = ComparableField(threshold=0.7)

    # Create models
    gt = SimpleModel(name="Bob", description="An engineer")
    pred = SimpleModel(name="Robert", description="A software engineer")

    # Patch the compare method to track calls
    with patch.object(
        LevenshteinComparator, "compare", wraps=LevenshteinComparator().compare
    ) as mock_compare:
        # Perform comparison with all options enabled
        result = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_non_matches=True,
            evaluator_format=True,
            add_derived_metrics=True,
        )

        # Verify the comparator was still called exactly twice
        assert mock_compare.call_count == 2, (
            f"Comparator called {mock_compare.call_count} times, expected 2"
        )


def test_multiple_compare_with_calls_increment_correctly():
    """Test that multiple separate compare_with calls increment counters correctly."""

    class SimpleModel(StructuredModel):
        """Simple model for testing."""

        name: str = ComparableField(threshold=0.7)
        description: str = ComparableField(threshold=0.7)

    # Create models
    gt = SimpleModel(name="Charlie", description="A manager")
    pred1 = SimpleModel(name="Charles", description="A supervisor")
    pred2 = SimpleModel(name="Chuck", description="A leader")

    # Patch the compare method to track calls
    with patch.object(
        LevenshteinComparator, "compare", wraps=LevenshteinComparator().compare
    ) as mock_compare:
        # First comparison
        result1 = gt.compare_with(pred1)
        assert mock_compare.call_count == 2, (
            f"After first comparison: {mock_compare.call_count} calls, expected 2"
        )

        # Second comparison should add 2 more calls
        result2 = gt.compare_with(pred2)
        assert mock_compare.call_count == 4, (
            f"After second comparison: {mock_compare.call_count} calls, expected 4"
        )


def test_nested_model_comparisons_single_call():
    """Test that nested StructuredModel comparisons don't cause excessive calls."""

    class Address(StructuredModel):
        street: str = ComparableField(threshold=0.8)
        city: str = ComparableField(threshold=0.8)

    class Person(StructuredModel):
        name: str = ComparableField(threshold=0.7)
        address: Address = ComparableField(threshold=0.9)

    # Create models with nested structure
    gt = Person(name="John Doe", address=Address(street="123 Main St", city="Anytown"))
    pred = Person(
        name="Jon Doe", address=Address(street="123 Main Street", city="Anytown")
    )

    # Patch the compare method to track calls
    with patch.object(
        LevenshteinComparator, "compare", wraps=LevenshteinComparator().compare
    ) as mock_compare:
        # Perform comparison
        result = gt.compare_with(pred)

        # Should have calls for:
        # 1. Person.name comparison
        # 2. Address.street comparison (within nested address comparison)
        # 3. Address.city comparison (within nested address comparison)
        # There might be a few additional calls due to the nested comparison logic,
        # but it should be reasonable (not exponential)
        assert mock_compare.call_count <= 10, (
            f"Comparator called {mock_compare.call_count} times, should be <= 10"
        )
        assert mock_compare.call_count >= 3, (
            f"Comparator called {mock_compare.call_count} times, should be >= 3"
        )


def test_list_field_comparisons_reasonable_calls():
    """Test that list field comparisons don't cause excessive calls."""

    class SimpleModel(StructuredModel):
        name: str = ComparableField(threshold=0.7)
        tags: list = ComparableField(threshold=0.8)  # List of strings

    # Create models with list fields
    gt = SimpleModel(name="Product", tags=["electronics", "gadget"])
    pred = SimpleModel(name="Product", tags=["electronic", "gadgets"])

    # Patch the compare method to track calls
    with patch.object(
        LevenshteinComparator, "compare", wraps=LevenshteinComparator().compare
    ) as mock_compare:
        # Perform comparison
        result = gt.compare_with(pred)

        # Should have calls for:
        # 1. name field comparison
        # 2-5. List comparisons (Hungarian matching between list items)
        # The exact number depends on the Hungarian algorithm implementation,
        # but it should be reasonable (not exponential)
        assert mock_compare.call_count <= 10, (
            f"Comparator called {mock_compare.call_count} times, should be <= 10"
        )
        assert mock_compare.call_count >= 1, (
            f"Comparator called {mock_compare.call_count} times, should be >= 1"
        )
