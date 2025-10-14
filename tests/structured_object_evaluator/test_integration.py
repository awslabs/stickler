"""Integration tests for the structured object evaluator."""

import unittest
from typing import List

from pydantic import Field

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    anls_score,
    compare_structured_models,
    compare_json,
)


class Address(StructuredModel):
    """Address model for testing."""

    street: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    city: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    state: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class Person(StructuredModel):
    """Person model with nested Address for testing."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )
    age: int = Field()
    address: Address


class Company(StructuredModel):
    """Company model with nested Person and list fields for testing."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )
    founded_year: int = Field()
    employees: List[Person]


class TestIntegration(unittest.TestCase):
    """Integration tests for the structured object evaluator."""

    def test_nested_models_comparison(self):
        """Test comparison of nested models."""
        # Create two Person models with nested Address
        person1 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", state="NY"),
        )

        person2 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", state="NY"),
        )

        # Compare them using compare_structured_models
        result = compare_structured_models(person1, person2)

        # Check the result
        self.assertEqual(result["overall_score"], 1.0)
        self.assertTrue(result["all_fields_matched"])
        self.assertEqual(result["field_scores"]["name"], 1.0)
        self.assertEqual(result["field_scores"]["age"], 1.0)
        self.assertEqual(result["field_scores"]["address"], 1.0)

    def test_list_field_comparison(self):
        """Test comparison of models with list fields."""
        # Create two Company models with list of Person
        company1 = Company(
            name="Acme Inc",
            founded_year=2000,
            employees=[
                Person(
                    name="John Doe",
                    age=30,
                    address=Address(street="123 Main St", city="New York", state="NY"),
                ),
                Person(
                    name="Jane Smith",
                    age=25,
                    address=Address(street="456 Oak Ave", city="Boston", state="MA"),
                ),
            ],
        )

        company2 = Company(
            name="Acme Inc",
            founded_year=2000,
            employees=[
                Person(
                    name="Jane Smith",
                    age=25,
                    address=Address(street="456 Oak Ave", city="Boston", state="MA"),
                ),
                Person(
                    name="John Doe",
                    age=30,
                    address=Address(street="123 Main St", city="New York", state="NY"),
                ),
            ],
        )

        # Compare them using compare_structured_models
        result = compare_structured_models(company1, company2)

        # Check the result
        self.assertEqual(result["overall_score"], 1.0)
        self.assertTrue(result["all_fields_matched"])
        self.assertEqual(result["field_scores"]["name"], 1.0)
        self.assertEqual(result["field_scores"]["founded_year"], 1.0)
        self.assertEqual(result["field_scores"]["employees"], 1.0)

    def test_anls_score_with_structured_models(self):
        """Test anls_score with structured models."""
        # Create two Person models
        person1 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", state="NY"),
        )

        person2 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", state="NY"),
        )

        # Calculate ANLS score
        score = anls_score(person1, person2)

        # Check the result
        self.assertEqual(score, 1.0)

        # Test with return_gt and return_key_scores
        score, closest_gt, key_scores = anls_score(
            person1, person2, return_gt=True, return_key_scores=True
        )

        # Check the result
        self.assertEqual(score, 1.0)
        # The closest_gt is now the original Person object
        self.assertEqual(closest_gt.name, "John Doe")
        self.assertEqual(closest_gt.age, 30)
        self.assertEqual(closest_gt.address.street, "123 Main St")
        self.assertIsInstance(key_scores, dict)

    def test_compare_json_with_nested_structure(self):
        """Test compare_json with nested structure."""
        # Create two JSON objects with nested structure
        json1 = {
            "name": "John Doe",
            "age": 30,
            "address": {"street": "123 Main St", "city": "New York", "state": "NY"},
        }

        json2 = {
            "name": "John Doe",
            "age": 30,
            "address": {"street": "123 Main St", "city": "New York", "state": "NY"},
        }

        # Compare them using compare_json
        result = compare_json(json1, json2, Person)

        # Check the result
        self.assertEqual(result["overall_score"], 1.0)
        self.assertTrue(result["all_fields_matched"])
        self.assertEqual(result["field_scores"]["name"], 1.0)
        self.assertEqual(result["field_scores"]["age"], 1.0)
        self.assertEqual(result["field_scores"]["address"], 1.0)


if __name__ == "__main__":
    unittest.main()
