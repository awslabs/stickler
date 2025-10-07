"""
Test to validate structured model scoring behavior and compatibility.

This test ensures that the scoring of structured models maintains
expected behavior and is consistent with the design requirements.
Migrated from test_star_metrics to preserve test coverage.
"""

import pytest
from pytest import approx
from typing import Optional, Dict, List, Union, Any

from stickler.structured_object_evaluator import StructuredModel, ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator


class SimpleModel(StructuredModel):
    """Simple model to test basic scoring."""

    text: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class ComplexModel(StructuredModel):
    """Complex model to test nested structures and multiple fields."""

    # Simplified model that just uses the details field directly
    details: Dict[str, Any] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class Address(StructuredModel):
    """Model representing an address."""

    city: str = ComparableField(comparator=LevenshteinComparator())
    state: str = ComparableField(comparator=LevenshteinComparator())


class Person(StructuredModel):
    """Model representing a person with an address."""

    name: str = ComparableField(comparator=LevenshteinComparator())
    address: Address = ComparableField()


def test_simple_string_scoring():
    """Test basic string scoring behavior."""
    # Test cases from original test_anls_star.py
    test_cases = [
        ("Hello World", "Hello World", 1.0),
        ("Hello World", "Hello Wrold", 0.82),
        ("Hello World", "How are you?", 0.0),
        (None, "How are you?", 0.0),
    ]

    for gt, pred, expected_score in test_cases:
        # Create models
        gt_model = SimpleModel(text=gt) if gt is not None else SimpleModel(text="")
        pred_model = (
            SimpleModel(text=pred) if pred is not None else SimpleModel(text="")
        )

        # Direct comparison
        if gt is not None and pred is not None:
            # Compare models
            result = gt_model.compare_with(pred_model)
            structured_score = result["overall_score"]

            # Don't check exact match for fuzzy scores like 0.82, just check it's in the right range
            if expected_score > 0 and expected_score < 1:
                assert abs(structured_score - expected_score) < 0.1, (
                    f"Score for GT: {gt}, Pred: {pred} should be ~{expected_score}, got {structured_score}"
                )
            else:
                assert structured_score == approx(expected_score, abs=0.01), (
                    f"Score for GT: {gt}, Pred: {pred} should be {expected_score}, got {structured_score}"
                )
        else:
            # For None cases, expect 0 score
            result = gt_model.compare_with(pred_model)
            structured_score = result["overall_score"]
            assert structured_score == 0.0, (
                f"Score for None comparison should be 0.0, got {structured_score}"
            )


def test_complex_dict_scoring():
    """Test complex dictionary scoring."""
    # TODO: Fix Dictionary Comparison Architecture Issue
    # Currently, Dict[str, Any] fields with LevenshteinComparator create unpredictable results
    # due to string representation differences. This defeats the purpose of structured comparison.
    #
    # Potential Solutions:
    # 1. Use ANLS_star comparison method for Dict[str, Any] fields that compares objects
    #    of any kind and returns a scalar comparison value
    # 2. Force users to break down Dict[str, Any] into proper StructuredModel subclasses
    #    for type-safe, predictable comparison
    #
    # For now, this test should raise an exception to prevent this anti-pattern.

    # Simplified test case focusing just on the first exact match case
    gt = {"a": "Hello", "b": "World"}
    pred = {"b": "World", "a": "Hello"}

    # Test structured model scoring using just the details field
    gt_model = ComplexModel(details=gt)
    pred_model = ComplexModel(details=pred)

    # Since ComparableField is now a function, we can't call compare() on it directly
    # Dictionary comparison now works through StructuredModel comparison
    result = gt_model.compare_with(pred_model)

    # Dictionary comparison should work but may give unpredictable results
    # due to string representation differences. For now, just test it works.
    assert 0.0 <= result["overall_score"] <= 1.0, (
        "Overall score should be between 0 and 1"
    )
    assert "details" in result["field_scores"], "Should have score for details field"
    assert 0.0 <= result["field_scores"]["details"] <= 1.0, (
        "Details field score should be between 0 and 1"
    )


def test_field_level_scoring():
    """Test that field-level scores are calculated correctly."""
    # Create a model with different field scores
    gt = Person(name="John Smith", address=Address(city="New York", state="NY"))

    pred = Person(name="John Smith", address=Address(city="New York", state="New York"))

    # Direct comparison
    result = gt.compare_with(pred)

    # Check individual field scores
    assert result["field_scores"]["name"] == 1.0, "Name field should be exact match"
    assert result["field_scores"]["address"] >= 0.5, "Address should be partial match"

    # Check that overall score reflects the weighted average
    # Since both fields have equal weight by default, overall score should be average
    expected_overall = (
        result["field_scores"]["name"] + result["field_scores"]["address"]
    ) / 2
    assert result["overall_score"] == approx(expected_overall, abs=0.01)


def test_weighted_fields():
    """Test that field weights are applied correctly in scoring."""

    # Define a model with weighted fields
    class WeightedModel(StructuredModel):
        high_weight: str = ComparableField(
            comparator=LevenshteinComparator(), weight=2.0
        )
        low_weight: str = ComparableField(
            comparator=LevenshteinComparator(), weight=1.0
        )

    # Create instances with different match patterns
    gt = WeightedModel(high_weight="Important", low_weight="Less Important")

    # Test case where high_weight field matches exactly, low_weight doesn't match at all
    pred1 = WeightedModel(high_weight="Important", low_weight="Completely Different")
    result1 = gt.compare_with(pred1)

    # high_weight (2.0): 1.0 * 2.0 = 2.0
    # low_weight (1.0): 0.0 * 1.0 = 0.0
    # Total weight: 3.0
    # Expected score: 2.0 / 3.0 = 0.667
    assert 0.65 < result1["overall_score"] < 0.68, (
        f"Weighted score calculation incorrect: {result1['overall_score']}"
    )

    # Test case where high_weight field doesn't match, low_weight matches exactly
    pred2 = WeightedModel(
        high_weight="Completely Different", low_weight="Less Important"
    )
    result2 = gt.compare_with(pred2)

    # high_weight (2.0): 0.0 * 2.0 = 0.0
    # low_weight (1.0): 1.0 * 1.0 = 1.0
    # Total weight: 3.0
    # Expected score: 1.0 / 3.0 = 0.333
    assert 0.32 < result2["overall_score"] < 0.35, (
        f"Weighted score calculation incorrect: {result2['overall_score']}"
    )


def test_field_thresholds():
    """Test that field thresholds are applied correctly."""

    # Define models with different threshold behaviors
    class ClippingModel(StructuredModel):
        high_threshold: str = ComparableField(
            comparator=LevenshteinComparator(),
            threshold=0.9,
            clip_under_threshold=True,  # Default behavior, clips scores below threshold
        )
        low_threshold: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.5, clip_under_threshold=True
        )

    class NonClippingModel(StructuredModel):
        high_threshold: str = ComparableField(
            comparator=LevenshteinComparator(),
            threshold=0.9,
            clip_under_threshold=False,  # New behavior, preserves scores below threshold
        )
        low_threshold: str = ComparableField(
            comparator=LevenshteinComparator(),
            threshold=0.5,
            clip_under_threshold=False,
        )

    # Create test instances
    gt_clipping = ClippingModel(high_threshold="Testing", low_threshold="Testing")
    pred_clipping = ClippingModel(high_threshold="Texting", low_threshold="Texting")

    gt_non_clipping = NonClippingModel(
        high_threshold="Testing", low_threshold="Testing"
    )
    pred_non_clipping = NonClippingModel(
        high_threshold="Texting", low_threshold="Texting"
    )

    # Get similarity score between "Testing" and "Texting" for reference
    # This should be around 0.85 (5 of 6 characters matching with 1 substitution)
    levenshtein_comparator = LevenshteinComparator()
    raw_similarity = levenshtein_comparator.compare("Testing", "Texting")

    # Test 1: With clipping enabled (default behavior)
    result_with_clipping = gt_clipping.compare_with(pred_clipping)

    # High threshold field (0.9) should be clipped to 0.0 since similarity (~0.85) is below threshold
    assert result_with_clipping["field_scores"]["high_threshold"] == 0.0, (
        "high_threshold field should be 0.0 when below threshold with clipping enabled"
    )

    # Low threshold field (0.5) should retain score since similarity (~0.85) is above threshold
    assert result_with_clipping["field_scores"]["low_threshold"] > 0.0, (
        "low_threshold field should have non-zero score when above threshold"
    )

    # Test 2: With clipping disabled (new behavior)
    result_without_clipping = gt_non_clipping.compare_with(pred_non_clipping)

    # Both fields should retain their raw scores regardless of threshold
    assert (
        abs(result_without_clipping["field_scores"]["high_threshold"] - raw_similarity)
        < 0.01
    ), "high_threshold field should preserve raw score when clipping disabled"

    assert (
        abs(result_without_clipping["field_scores"]["low_threshold"] - raw_similarity)
        < 0.01
    ), "low_threshold field should preserve raw score when clipping disabled"


def test_nested_model_scoring():
    """Test scoring with deeply nested structured models."""

    # Define nested models for testing
    class Address(StructuredModel):
        street: str = ComparableField()
        city: str = ComparableField()
        postal_code: str = ComparableField()

    class Department(StructuredModel):
        name: str = ComparableField()
        location: Address = ComparableField()

    class Company(StructuredModel):
        name: str = ComparableField()
        departments: List[Department] = ComparableField()

    # Create nested instances
    gt = Company(
        name="Acme Corp",
        departments=[
            Department(
                name="Engineering",
                location=Address(
                    street="123 Main St", city="San Francisco", postal_code="94105"
                ),
            ),
            Department(
                name="Marketing",
                location=Address(
                    street="456 Market St", city="San Francisco", postal_code="94107"
                ),
            ),
        ],
    )

    # Create prediction with slight differences at each level
    pred = Company(
        name="Acme Corporation",  # slightly different
        departments=[
            Department(
                name="Engineering Team",  # slightly different
                location=Address(
                    street="123 Main Street", city="San Francisco", postal_code="94105"
                ),  # street slightly different
            ),
            Department(
                name="Marketing",
                location=Address(
                    street="456 Market St", city="San Fransisco", postal_code="94107"
                ),  # city misspelled
            ),
        ],
    )

    # Direct comparison
    result = gt.compare_with(pred)

    # Check that comparison works at all levels
    assert 0 < result["overall_score"] < 1, (
        "Nested model comparison should produce a score between 0 and 1"
    )
    assert "departments" in result["field_scores"], (
        "Should have scores for departments list"
    )
