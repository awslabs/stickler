# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

"""Simple test for StructuredModel compare_with functionality."""

import pytest
from typing import Optional
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


class SimpleTestModel(StructuredModel):
    """Simple model for testing StructuredModel functionality."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0,
    )

    age: Optional[int] = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=0.5,
    )


def test_structured_model_compare_with_exact_match():
    """Test compare_with with exact match."""
    model1 = SimpleTestModel(name="John Doe", age=30)
    model2 = SimpleTestModel(name="John Doe", age=30)

    result = model1.compare_with(model2)

    # Check result structure
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Check field scores
    assert "name" in result["field_scores"]
    assert "age" in result["field_scores"]

    # Should be perfect match
    assert result["field_scores"]["name"] == 1.0
    assert result["field_scores"]["age"] == 1.0
    assert result["overall_score"] == 1.0
    assert result["all_fields_matched"] is True

    print("✓ Exact match test passed")
    print(f"Result: {result}")


def test_structured_model_compare_with_partial_match():
    """Test compare_with with partial match."""
    model1 = SimpleTestModel(name="John Doe", age=30)
    model2 = SimpleTestModel(name="John Smith", age=30)

    result = model1.compare_with(model2)

    # Check result structure
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Age should match, name should have some similarity
    assert result["field_scores"]["age"] == 1.0
    assert 0.0 <= result["field_scores"]["name"] <= 1.0

    # Overall score should use weighted average calculation
    # name has weight 1.0, age has weight 0.5
    name_score = result["field_scores"]["name"]
    age_score = result["field_scores"]["age"]
    expected_score = (name_score * 1.0 + age_score * 0.5) / 1.5
    assert abs(result["overall_score"] - expected_score) < 0.001

    print("✓ Partial match test passed")
    print(f"Result: {result}")


def test_structured_model_compare_with_no_match():
    """Test compare_with with no match."""
    model1 = SimpleTestModel(name="John Doe", age=30)
    model2 = SimpleTestModel(name="Jane Smith", age=25)

    result = model1.compare_with(model2)

    # Check result structure
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Fields should not match
    assert result["field_scores"]["age"] == 0.0  # ExactComparator with different values

    print("✓ No match test passed")
    print(f"Result: {result}")


def test_structured_model_compare_scalar():
    """Test compare method (scalar output)."""
    model1 = SimpleTestModel(name="John Doe", age=30)
    model2 = SimpleTestModel(name="John Doe", age=30)

    score = model1.compare(model2)

    # Should return a scalar score
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 1.0  # Perfect match

    print("✓ Scalar compare test passed")
    print(f"Score: {score}")


if __name__ == "__main__":
    test_structured_model_compare_with_exact_match()
    test_structured_model_compare_with_partial_match()
    test_structured_model_compare_with_no_match()
    test_structured_model_compare_scalar()
    print("All tests passed!")
