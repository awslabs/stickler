"""Tests for the anls_score utility functions."""

import unittest

from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    anls_score,
    compare_structured_models,
    compare_json,
)
from stickler.comparators.levenshtein import LevenshteinComparator
from pydantic import Field


class Person(StructuredModel):
    """Simple person model for testing."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    age: int = Field()


class TestANLSScore(unittest.TestCase):
    """Test cases for the anls_score function."""

    def test_anls_score_exact_match(self):
        """Test anls_score with exact matches."""
        # Test with strings
        self.assertEqual(anls_score("hello", "hello"), 1.0)

        # Test with numbers
        self.assertEqual(anls_score(42, 42), 1.0)

        # Test with lists
        self.assertEqual(anls_score(["a", "b", "c"], ["a", "b", "c"]), 1.0)

        # Test with dictionaries
        self.assertEqual(anls_score({"a": 1, "b": 2}, {"a": 1, "b": 2}), 1.0)

    def test_anls_score_similar_match(self):
        """Test anls_score with similar but not exact matches."""
        # Test with similar strings
        score = anls_score("hello", "helo")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

        # Test with lists with one different item
        score = anls_score(["a", "b", "c"], ["a", "b", "d"])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

        # Test with dictionaries with one different value
        score = anls_score({"a": 1, "b": 2}, {"a": 1, "b": 3})
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_anls_score_return_gt(self):
        """Test anls_score with return_gt=True."""
        # Test with strings
        score, closest_gt = anls_score("hello", "helo", return_gt=True)
        self.assertEqual(closest_gt, "hello")

        # Test with lists
        score, closest_gt = anls_score(["a", "b", "c"], ["a", "b"], return_gt=True)
        self.assertEqual(closest_gt, ["a", "b", "c"])

        # Test with dictionaries
        score, closest_gt = anls_score({"a": 1, "b": 2}, {"a": 1}, return_gt=True)
        self.assertEqual(closest_gt, {"a": 1, "b": 2})

    def test_anls_score_return_key_scores(self):
        """Test anls_score with return_key_scores=True."""
        # Test with dictionaries
        score, closest_gt, key_scores = anls_score(
            {"a": 1, "b": 2}, {"a": 1, "b": 3}, return_gt=True, return_key_scores=True
        )
        self.assertEqual(closest_gt, {"a": 1, "b": 2})
        self.assertIn("a", key_scores)
        self.assertIn("b", key_scores)
        self.assertEqual(key_scores["a"].score, 1.0)
        self.assertEqual(key_scores["b"].score, 0.0)


class TestCompareStructuredModels(unittest.TestCase):
    """Test cases for the compare_structured_models function."""

    def test_compare_structured_models(self):
        """Test compare_structured_models function."""
        # Create two Person models
        person1 = Person(name="John Doe", age=30)
        person2 = Person(name="John Doe", age=30)

        # Compare them
        result = compare_structured_models(person1, person2)

        # Check the result
        self.assertEqual(result["overall_score"], 1.0)
        self.assertTrue(result["all_fields_matched"])
        self.assertEqual(result["field_scores"]["name"], 1.0)
        self.assertEqual(result["field_scores"]["age"], 1.0)


class TestCompareJSON(unittest.TestCase):
    """Test cases for the compare_json function."""

    def test_compare_json(self):
        """Test compare_json function."""
        # Create two JSON objects
        json1 = {"name": "John Doe", "age": 30}
        json2 = {"name": "John Doe", "age": 30}

        # Compare them using the Person model
        result = compare_json(json1, json2, Person)

        # Check the result
        self.assertEqual(result["overall_score"], 1.0)
        self.assertTrue(result["all_fields_matched"])
        self.assertEqual(result["field_scores"]["name"], 1.0)
        self.assertEqual(result["field_scores"]["age"], 1.0)

    def test_compare_json_with_error(self):
        """Test compare_json function with invalid JSON."""
        # Create an invalid JSON object (missing required field)
        json1 = {"name": "John Doe"}  # Missing age
        json2 = {"name": "John Doe", "age": 30}

        # Compare them using the Person model
        result = compare_json(json1, json2, Person)

        # Check that an error was returned
        self.assertIn("error", result)
        self.assertEqual(result["overall_score"], 0.0)
        self.assertEqual(result["field_scores"], {})
        self.assertFalse(result["all_fields_matched"])


if __name__ == "__main__":
    unittest.main()
