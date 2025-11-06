"""Tests for ModelFactory.create_model_from_fields() functionality.

This module tests the new create_model_from_fields() method that accepts
pre-converted Pydantic field definitions.
"""

import pytest
from typing import List
from pydantic import ValidationError

from stickler.structured_object_evaluator.models.model_factory import ModelFactory
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.numeric import NumericComparator


class TestCreateModelFromFields:
    """Test ModelFactory.create_model_from_fields() method."""

    def test_basic_model_creation(self):
        """Test creating a basic model with pre-converted fields."""
        field_definitions = {
            "name": (
                str,
                ComparableField(
                    comparator=LevenshteinComparator(),
                    threshold=0.8,
                    weight=2.0,
                ),
            ),
            "age": (
                int,
                ComparableField(
                    comparator=NumericComparator(),
                    threshold=0.9,
                    default=0,
                ),
            ),
        }

        PersonClass = ModelFactory.create_model_from_fields(
            model_name="Person",
            field_definitions=field_definitions,
            match_threshold=0.75,
            base_class=StructuredModel,
        )

        # Verify class properties
        assert PersonClass.__name__ == "Person"
        assert issubclass(PersonClass, StructuredModel)
        assert PersonClass.match_threshold == 0.75

        # Verify fields exist
        assert "name" in PersonClass.model_fields
        assert "age" in PersonClass.model_fields

        # Test instance creation
        person = PersonClass(name="Alice", age=30)
        assert person.name == "Alice"
        assert person.age == 30

    def test_model_with_required_and_optional_fields(self):
        """Test model with mix of required and optional fields."""
        field_definitions = {
            "id": (
                str,
                ComparableField(
                    comparator=ExactComparator(),
                    threshold=1.0,
                    # No default = required
                ),
            ),
            "description": (
                str,
                ComparableField(
                    comparator=LevenshteinComparator(),
                    threshold=0.7,
                    default="",  # Has default = optional
                ),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="TestModel",
            field_definitions=field_definitions,
        )

        # Can create with just required field
        instance1 = TestClass(id="123")
        assert instance1.id == "123"
        assert instance1.description == ""

        # Can create with all fields
        instance2 = TestClass(id="456", description="Test description")
        assert instance2.id == "456"
        assert instance2.description == "Test description"

    def test_model_with_list_fields(self):
        """Test model with List type fields."""
        field_definitions = {
            "tags": (
                List[str],
                ComparableField(
                    comparator=LevenshteinComparator(),
                    threshold=0.6,
                    default=[],
                ),
            ),
            "scores": (
                List[float],
                ComparableField(
                    comparator=NumericComparator(),
                    threshold=0.8,
                    default=[],
                ),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="DataModel",
            field_definitions=field_definitions,
        )

        instance = TestClass(tags=["python", "testing"], scores=[0.9, 0.85])
        assert instance.tags == ["python", "testing"]
        assert instance.scores == [0.9, 0.85]

    def test_comparison_functionality_inherited(self):
        """Test that comparison functionality is properly inherited."""
        field_definitions = {
            "value1": (
                str,
                ComparableField(
                    comparator=LevenshteinComparator(),
                    threshold=0.8,
                ),
            ),
            "value2": (
                int,
                ComparableField(
                    comparator=NumericComparator(),
                    threshold=0.9,
                    default=0,
                ),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="ComparisonTest",
            field_definitions=field_definitions,
            match_threshold=0.7,
        )

        # Create two instances
        obj1 = TestClass(value1="hello", value2=100)
        obj2 = TestClass(value1="hello", value2=100)

        # Test comparison
        result = obj1.compare_with(obj2)
        assert "overall_score" in result
        assert result["overall_score"] == 1.0

    def test_default_match_threshold(self):
        """Test that default match_threshold is applied."""
        field_definitions = {
            "field1": (
                str,
                ComparableField(comparator=LevenshteinComparator()),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="DefaultThreshold",
            field_definitions=field_definitions,
            # Not specifying match_threshold, should use default 0.7
        )

        assert TestClass.match_threshold == 0.7

    def test_custom_match_threshold(self):
        """Test setting custom match_threshold."""
        field_definitions = {
            "field1": (
                str,
                ComparableField(comparator=LevenshteinComparator()),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="CustomThreshold",
            field_definitions=field_definitions,
            match_threshold=0.85,
        )

        assert TestClass.match_threshold == 0.85

    def test_base_class_default(self):
        """Test that StructuredModel is used as default base class."""
        field_definitions = {
            "field1": (
                str,
                ComparableField(comparator=LevenshteinComparator()),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="DefaultBase",
            field_definitions=field_definitions,
            # Not specifying base_class
        )

        assert issubclass(TestClass, StructuredModel)

    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        field_definitions = {
            "field1": (
                str,
                ComparableField(comparator=LevenshteinComparator()),
            ),
        }

        # Invalid: not a string
        with pytest.raises(ValueError, match="valid Python identifier"):
            ModelFactory.create_model_from_fields(
                model_name=123,
                field_definitions=field_definitions,
            )

        # Invalid: not a valid identifier (contains space)
        with pytest.raises(ValueError, match="valid Python identifier"):
            ModelFactory.create_model_from_fields(
                model_name="Invalid Name",
                field_definitions=field_definitions,
            )

        # Invalid: starts with number
        with pytest.raises(ValueError, match="valid Python identifier"):
            ModelFactory.create_model_from_fields(
                model_name="123Model",
                field_definitions=field_definitions,
            )

    def test_invalid_match_threshold(self):
        """Test error handling for invalid match_threshold values."""
        field_definitions = {
            "field1": (
                str,
                ComparableField(comparator=LevenshteinComparator()),
            ),
        }

        # Too low
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ModelFactory.create_model_from_fields(
                model_name="Test",
                field_definitions=field_definitions,
                match_threshold=-0.1,
            )

        # Too high
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ModelFactory.create_model_from_fields(
                model_name="Test",
                field_definitions=field_definitions,
                match_threshold=1.5,
            )

        # Not a number
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ModelFactory.create_model_from_fields(
                model_name="Test",
                field_definitions=field_definitions,
                match_threshold="invalid",
            )

    def test_invalid_field_definitions_type(self):
        """Test error handling for invalid field_definitions type."""
        # Not a dictionary
        with pytest.raises(ValueError, match="must be a dictionary"):
            ModelFactory.create_model_from_fields(
                model_name="Test",
                field_definitions="invalid",
            )

        # Empty dictionary
        with pytest.raises(ValueError, match="at least one field"):
            ModelFactory.create_model_from_fields(
                model_name="Test",
                field_definitions={},
            )

    def test_invalid_field_definition_structure(self):
        """Test error handling for malformed field definitions."""
        # Not a tuple
        with pytest.raises(ValueError, match="must be a tuple"):
            ModelFactory.create_model_from_fields(
                model_name="Test",
                field_definitions={
                    "field1": "not a tuple",
                },
            )

        # Tuple with wrong number of elements
        with pytest.raises(ValueError, match="must be a tuple"):
            ModelFactory.create_model_from_fields(
                model_name="Test",
                field_definitions={
                    "field1": (str,),  # Only 1 element
                },
            )

    def test_pydantic_validation_works(self):
        """Test that Pydantic validation is properly enforced."""
        from pydantic import Field
        
        field_definitions = {
            "required_field": (
                str,
                ComparableField(
                    comparator=LevenshteinComparator(),
                    default=...,  # Ellipsis means required in Pydantic
                ),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="ValidationTest",
            field_definitions=field_definitions,
        )

        # Should raise ValidationError when required field is missing
        with pytest.raises(ValidationError):
            TestClass()

    def test_field_with_description_and_examples(self):
        """Test that Pydantic field metadata is preserved."""
        field_definitions = {
            "email": (
                str,
                ComparableField(
                    comparator=LevenshteinComparator(),
                    threshold=0.9,
                    description="User email address",
                    examples=["user@example.com"],
                ),
            ),
        }

        TestClass = ModelFactory.create_model_from_fields(
            model_name="MetadataTest",
            field_definitions=field_definitions,
        )

        # Verify field exists and has metadata
        assert "email" in TestClass.model_fields
        field_info = TestClass.model_fields["email"]
        assert field_info.description == "User email address"
        assert field_info.examples == ["user@example.com"]

    def test_multiple_models_independent(self):
        """Test that creating multiple models doesn't cause interference."""
        field_defs_1 = {
            "name": (
                str,
                ComparableField(comparator=LevenshteinComparator()),
            ),
        }

        field_defs_2 = {
            "value": (
                int,
                ComparableField(comparator=NumericComparator()),
            ),
        }

        Model1 = ModelFactory.create_model_from_fields(
            model_name="Model1",
            field_definitions=field_defs_1,
            match_threshold=0.6,
        )

        Model2 = ModelFactory.create_model_from_fields(
            model_name="Model2",
            field_definitions=field_defs_2,
            match_threshold=0.8,
        )

        # Verify they are independent
        assert Model1.__name__ == "Model1"
        assert Model2.__name__ == "Model2"
        assert Model1.match_threshold == 0.6
        assert Model2.match_threshold == 0.8
        assert "name" in Model1.model_fields
        assert "name" not in Model2.model_fields
        assert "value" in Model2.model_fields
        assert "value" not in Model1.model_fields
