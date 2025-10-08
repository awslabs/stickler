"""Tests for consistency between different evaluation methods."""

import unittest
from typing import Any, Dict, List, Optional

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


class TestConsistency(unittest.TestCase):
    """Test cases for consistency between different evaluation methods."""

    def test_anls_score_vs_compare_structured_models(self):
        """Test consistency between anls_score and compare_structured_models."""
        # Create two Person models
        person1 = Person(name="John Doe", age=30)
        person2 = Person(name="John Doe", age=30)

        # Compare using anls_score
        anls_result = anls_score(person1, person2)

        # Compare using compare_structured_models
        compare_result = compare_structured_models(person1, person2)

        # Check that the results are consistent
        self.assertEqual(anls_result, compare_result["overall_score"])

    def test_anls_score_vs_compare_json(self):
        """Test consistency between anls_score and compare_json."""
        # Create JSON objects
        json1 = {"name": "John Doe", "age": 30}
        json2 = {"name": "John Doe", "age": 30}

        # Compare using anls_score with dictionaries
        anls_result = anls_score(json1, json2)

        # Compare using compare_json
        compare_result = compare_json(json1, json2, Person)

        # Check that the results are consistent
        self.assertAlmostEqual(anls_result, compare_result["overall_score"], places=4)

    def test_compare_structured_models_vs_compare_json(self):
        """Test consistency between compare_structured_models and compare_json."""
        # Create Person model and JSON object
        person = Person(name="John Doe", age=30)
        json_obj = {"name": "John Doe", "age": 30}

        # Compare using compare_structured_models
        model_result = compare_structured_models(person, person)

        # Compare using compare_json
        json_result = compare_json(json_obj, json_obj, Person)

        # Check that the results are consistent
        self.assertEqual(model_result["overall_score"], json_result["overall_score"])
        self.assertEqual(
            model_result["all_fields_matched"], json_result["all_fields_matched"]
        )
        self.assertEqual(model_result["field_scores"], json_result["field_scores"])

    def test_anls_score_with_different_return_options(self):
        """Test consistency of anls_score with different return options."""
        # Create two Person models
        person1 = Person(name="John Doe", age=30)
        person2 = Person(name="John Doe", age=30)

        # Compare with different return options
        score1 = anls_score(person1, person2)
        score2, _ = anls_score(person1, person2, return_gt=True)

        # For structured models, we need to use the structured_anls_score function
        # which has different return behavior
        result = compare_structured_models(person1, person2)
        score3 = result["overall_score"]

        # Check that the scores are consistent
        self.assertEqual(score1, score2)
        self.assertEqual(score1, score3)

    def test_compare_structured_models_field_scores(self):
        """Test consistency of field scores in compare_structured_models."""
        # Create two Person models with one field different
        person1 = Person(name="John Doe", age=30)
        person2 = Person(name="John Doe", age=31)  # Different age

        # Compare them
        result = compare_structured_models(person1, person2)

        # Check that the field scores are consistent with the overall score
        self.assertEqual(result["field_scores"]["name"], 1.0)  # Exact match

        # Note: The age field is not a ComparableField, so it's compared using default logic
        # which may not give exactly 0.0 for different values
        self.assertLess(result["field_scores"]["age"], 1.0)  # Should be less than 1.0

        # Check that the overall score is between the field scores
        self.assertGreaterEqual(
            result["overall_score"], min(result["field_scores"].values())
        )
        self.assertLessEqual(
            result["overall_score"], max(result["field_scores"].values())
        )


if __name__ == "__main__":
    unittest.main()
