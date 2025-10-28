"""Integration tests for JsonSchemaFieldConverter.convert_properties_to_fields()."""

import pytest
from typing import List
from pydantic import Field
from pydantic_core import PydanticUndefined

from stickler.structured_object_evaluator.models.json_schema_field_converter import (
    JsonSchemaFieldConverter,
)
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.comparators.exact import ExactComparator


class TestConvertPropertiesToFields:
    """Integration tests for convert_properties_to_fields method."""

    def test_convert_all_primitive_types(self):
        """Test converting all primitive JSON Schema types."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"},
            },
            "required": ["name", "age"],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check all fields are present
        assert "name" in field_definitions
        assert "age" in field_definitions
        assert "price" in field_definitions
        assert "active" in field_definitions

        # Check types
        assert field_definitions["name"][0] == str
        assert field_definitions["age"][0] == int
        assert field_definitions["price"][0] == float
        assert field_definitions["active"][0] == bool

        # Check required vs optional (via is_required)
        name_field = field_definitions["name"][1]
        age_field = field_definitions["age"][1]
        price_field = field_definitions["price"][1]
        active_field = field_definitions["active"][1]

        assert name_field.is_required()  # Required
        assert age_field.is_required()  # Required
        assert not price_field.is_required()  # Optional
        assert not active_field.is_required()  # Optional

    def test_convert_with_default_comparators(self):
        """Test that default comparators are assigned correctly."""
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "amount": {"type": "number"},
                "flag": {"type": "boolean"},
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check comparator types through metadata stored as function attributes
        text_field = field_definitions["text"][1]
        count_field = field_definitions["count"][1]
        amount_field = field_definitions["amount"][1]
        flag_field = field_definitions["flag"][1]

        # Access comparator instances from json_schema_extra function attributes
        text_comparator = text_field.json_schema_extra._comparator_instance
        count_comparator = count_field.json_schema_extra._comparator_instance
        amount_comparator = amount_field.json_schema_extra._comparator_instance
        flag_comparator = flag_field.json_schema_extra._comparator_instance

        # Verify comparator types
        assert isinstance(text_comparator, LevenshteinComparator)
        assert isinstance(count_comparator, NumericComparator)
        assert isinstance(amount_comparator, NumericComparator)
        assert isinstance(flag_comparator, ExactComparator)

    def test_convert_with_custom_extensions(self):
        """Test converting properties with x-aws-stickler-* extensions."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "ExactComparator",
                    "x-aws-stickler-threshold": 0.9,
                    "x-aws-stickler-weight": 2.0,
                    "x-aws-stickler-clip-under-threshold": False,
                    "x-aws-stickler-aggregate": True,
                },
                "age": {
                    "type": "integer",
                    "x-aws-stickler-threshold": 0.8,
                    "x-aws-stickler-weight": 1.5,
                },
            },
            "required": ["name"],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check name field extensions via function attributes
        name_field = field_definitions["name"][1]
        
        assert isinstance(name_field.json_schema_extra._comparator_instance, ExactComparator)
        assert name_field.json_schema_extra._threshold == 0.9
        assert name_field.json_schema_extra._weight == 2.0
        assert name_field.json_schema_extra._clip_under_threshold is False
        assert name_field.json_schema_extra._aggregate is True

        # Check age field extensions
        age_field = field_definitions["age"][1]
        
        assert age_field.json_schema_extra._threshold == 0.8
        assert age_field.json_schema_extra._weight == 1.5

    def test_convert_with_pydantic_metadata(self):
        """Test that Pydantic metadata (description, examples) is preserved."""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "User email address",
                    "examples": ["user@example.com"],
                },
                "score": {
                    "type": "number",
                    "description": "User score",
                    "default": 0.0,
                },
            },
            "required": ["email"],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check email field metadata
        email_field = field_definitions["email"][1]
        assert email_field.description == "User email address"
        assert email_field.examples == ["user@example.com"]

        # Check score field metadata
        score_field = field_definitions["score"][1]
        assert score_field.description == "User score"
        assert score_field.default == 0.0

    def test_convert_with_arrays_of_primitives(self):
        """Test converting array properties with primitive elements."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "scores": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check types are List[primitive]
        tags_type = field_definitions["tags"][0]
        scores_type = field_definitions["scores"][0]

        # Verify they are List types
        assert hasattr(tags_type, "__origin__")
        assert tags_type.__origin__ == list
        assert tags_type.__args__[0] == str

        assert hasattr(scores_type, "__origin__")
        assert scores_type.__origin__ == list
        assert scores_type.__args__[0] == float

    def test_convert_empty_properties(self):
        """Test converting empty properties dictionary."""
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Should return empty dictionary
        assert field_definitions == {}

    def test_convert_preserves_field_order(self):
        """Test that field order is preserved from schema."""
        schema = {
            "type": "object",
            "properties": {
                "first": {"type": "string"},
                "second": {"type": "integer"},
                "third": {"type": "boolean"},
                "fourth": {"type": "number"},
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check that all fields are present
        field_names = list(field_definitions.keys())
        assert field_names == ["first", "second", "third", "fourth"]

    def test_convert_with_mixed_required_optional(self):
        """Test converting with mix of required and optional fields."""
        schema = {
            "type": "object",
            "properties": {
                "required1": {"type": "string"},
                "optional1": {"type": "string"},
                "required2": {"type": "integer"},
                "optional2": {"type": "integer"},
            },
            "required": ["required1", "required2"],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check required fields are marked as required
        assert field_definitions["required1"][1].is_required()
        assert field_definitions["required2"][1].is_required()

        # Check optional fields are not required
        assert not field_definitions["optional1"][1].is_required()
        assert not field_definitions["optional2"][1].is_required()

    def test_convert_with_array_extensions(self):
        """Test converting array with custom comparator extensions."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-aws-stickler-comparator": "ExactComparator",
                    "x-aws-stickler-threshold": 0.95,
                    "x-aws-stickler-weight": 1.5,
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check that extensions are applied via function attributes
        tags_field = field_definitions["tags"][1]
        
        assert isinstance(tags_field.json_schema_extra._comparator_instance, ExactComparator)
        assert tags_field.json_schema_extra._threshold == 0.95
        assert tags_field.json_schema_extra._weight == 1.5

    def test_convert_returns_correct_structure(self):
        """Test that convert_properties_to_fields returns the correct structure for create_model()."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Verify structure: Dict[str, Tuple[Type, Field]]
        assert isinstance(field_definitions, dict)
        
        for field_name, field_def in field_definitions.items():
            assert isinstance(field_name, str)
            assert isinstance(field_def, tuple)
            assert len(field_def) == 2
            # First element is a type
            assert isinstance(field_def[0], type)
            # Second element is a Pydantic FieldInfo
            from pydantic.fields import FieldInfo
            assert isinstance(field_def[1], FieldInfo)
