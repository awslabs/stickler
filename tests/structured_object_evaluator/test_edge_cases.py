"""Tests for edge cases in the structured object evaluator."""

import unittest
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    anls_score,
)
from stickler.comparators.levenshtein import LevenshteinComparator


class Person(StructuredModel):
    """Simple person model for testing."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    age: Optional[int] = Field(None)
    email: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases in the structured object evaluator."""

    def test_null_values(self):
        """Test handling of null values."""
        # Test with None values in both objects
        person1 = Person(name="John Doe", age=None, email=None)
        person2 = Person(name="John Doe", age=None, email=None)

        result = person1.compare_with(person2)
        self.assertEqual(result["overall_score"], 1.0)
        self.assertTrue(result["all_fields_matched"])

        # Test with None value in one object
        person3 = Person(name="John Doe", age=30, email=None)
        person4 = Person(name="John Doe", age=None, email=None)

        result = person3.compare_with(person4)
        self.assertLess(result["overall_score"], 1.0)
        self.assertFalse(result["all_fields_matched"])

    def test_empty_strings(self):
        """Test handling of empty strings."""
        # Test with empty strings in both objects
        person1 = Person(name="John Doe", age=30, email="")
        person2 = Person(name="John Doe", age=30, email="")

        result = person1.compare_with(person2)
        self.assertEqual(result["overall_score"], 1.0)
        self.assertTrue(result["all_fields_matched"])

        # Test with empty string in one object
        person3 = Person(name="John Doe", age=30, email="john@example.com")
        person4 = Person(name="John Doe", age=30, email="")

        result = person3.compare_with(person4)
        self.assertLess(result["overall_score"], 1.0)
        self.assertFalse(result["all_fields_matched"])

    def test_different_types(self):
        """Test handling of different types."""
        # Test with different types using anls_score
        score = anls_score("John Doe", 123)
        self.assertEqual(score, 0.0)

        # Test with different types in dictionaries
        score = anls_score({"name": "John Doe"}, {"name": 123})
        self.assertEqual(score, 0.0)

        # Test with different types in lists
        score = anls_score(["John", "Doe"], [1, 2])
        self.assertEqual(score, 0.0)

    def test_missing_fields(self):
        """Test handling of missing fields in dictionaries."""
        # Test with missing field in one dictionary
        score = anls_score({"name": "John Doe", "age": 30}, {"name": "John Doe"})
        self.assertLess(score, 1.0)

        # Test with missing field in both dictionaries
        score = anls_score({"name": "John Doe"}, {"name": "John Doe"})
        self.assertEqual(score, 1.0)

    def test_extra_fields(self):
        """Test handling of extra fields in dictionaries."""
        # Test with extra field in one dictionary
        score = anls_score({"name": "John Doe"}, {"name": "John Doe", "age": 30})
        self.assertLess(score, 1.0)

    def test_empty_containers(self):
        """Test handling of empty containers."""
        # Test with empty dictionaries
        self.assertEqual(anls_score({}, {}), 1.0)

        # Test with empty lists
        self.assertEqual(anls_score([], []), 1.0)

        # Test with empty dictionary and non-empty dictionary
        self.assertEqual(anls_score({}, {"name": "John Doe"}), 0.0)

        # Test with empty list and non-empty list
        self.assertEqual(anls_score([], ["John Doe"]), 0.0)

    def test_special_characters(self):
        """Test handling of special characters."""
        # Test with special characters in strings
        score = anls_score("John & Doe's", "John & Doe's")
        self.assertEqual(score, 1.0)

        # Test with special characters in dictionaries
        score = anls_score({"name": "John & Doe's"}, {"name": "John & Doe's"})
        self.assertEqual(score, 1.0)

    def test_unicode_characters(self):
        """Test handling of Unicode characters."""
        # Test with Unicode characters in strings
        score = anls_score("Jöhn Döé", "Jöhn Döé")
        self.assertEqual(score, 1.0)

        # Test with Unicode characters in dictionaries
        score = anls_score({"name": "Jöhn Döé"}, {"name": "Jöhn Döé"})
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
