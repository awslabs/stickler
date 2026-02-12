"""Tests for the anls_score utility functions."""

from pydantic import Field

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator import (
    ComparableField,
    StructuredModel,
    anls_score,
    compare_json,
    compare_structured_models,
)


class Person(StructuredModel):
    """Simple person model for testing."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    age: int = Field()


class TestANLSScore:
    """Test cases for the anls_score function."""

    def test_anls_score_exact_match(self):
        """Test anls_score with exact matches."""
        # Test with strings
        assert anls_score("hello", "hello") == 1.0

        # Test with numbers
        assert anls_score(42, 42) == 1.0

        # Test with lists
        assert anls_score(["a", "b", "c"], ["a", "b", "c"]) == 1.0

        # Test with dictionaries
        assert anls_score({"a": 1, "b": 2}, {"a": 1, "b": 2}) == 1.0

    def test_anls_score_similar_match(self):
        """Test anls_score with similar but not exact matches."""
        # Test with similar strings
        score = anls_score("hello", "helo")
        assert score > 0.0
        assert score < 1.0

        # Test with lists with one different item
        score = anls_score(["a", "b", "c"], ["a", "b", "d"])
        assert score > 0.0
        assert score < 1.0

        # Test with dictionaries with one different value
        score = anls_score({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert score > 0.0
        assert score < 1.0

    def test_anls_score_return_gt(self):
        """Test anls_score with return_gt=True."""
        # Test with strings
        score, closest_gt = anls_score("hello", "helo", return_gt=True)
        assert closest_gt == "hello"

        # Test with lists
        score, closest_gt = anls_score(["a", "b", "c"], ["a", "b"], return_gt=True)
        assert closest_gt == ["a", "b", "c"]

        # Test with dictionaries
        score, closest_gt = anls_score({"a": 1, "b": 2}, {"a": 1}, return_gt=True)
        assert closest_gt == {"a": 1, "b": 2}

    def test_anls_score_return_key_scores(self):
        """Test anls_score with return_key_scores=True."""
        # Test with dictionaries
        score, closest_gt, key_scores = anls_score(
            {"a": 1, "b": 2}, {"a": 1, "b": 3}, return_gt=True, return_key_scores=True
        )
        assert closest_gt == {"a": 1, "b": 2}
        assert "a" in key_scores
        assert "b" in key_scores
        assert key_scores["a"].score == 1.0
        assert key_scores["b"].score == 0.0


class TestCompareStructuredModels:
    """Test cases for the compare_structured_models function."""

    def test_compare_structured_models(self):
        """Test compare_structured_models function."""
        # Create two Person models
        person1 = Person(name="John Doe", age=30)
        person2 = Person(name="John Doe", age=30)

        # Compare them
        result = compare_structured_models(person1, person2)

        # Check the result
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"]
        assert result["field_scores"]["name"] == 1.0
        assert result["field_scores"]["age"] == 1.0


class TestCompareJSON:
    """Test cases for the compare_json function."""

    def test_compare_json(self):
        """Test compare_json function."""
        # Create two JSON objects
        json1 = {"name": "John Doe", "age": 30}
        json2 = {"name": "John Doe", "age": 30}

        # Compare them using the Person model
        result = compare_json(json1, json2, Person)

        # Check the result
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"]
        assert result["field_scores"]["name"] == 1.0
        assert result["field_scores"]["age"] == 1.0

    def test_compare_json_with_error(self):
        """Test compare_json function with invalid JSON."""
        # Create an invalid JSON object (missing required field)
        json1 = {"name": "John Doe"}  # Missing age
        json2 = {"name": "John Doe", "age": 30}

        # Compare them using the Person model
        result = compare_json(json1, json2, Person)

        # Check that an error was returned
        assert "error" in result
        assert result["overall_score"] == 0.0
        assert result["field_scores"] == {}
        assert not result["all_fields_matched"]
