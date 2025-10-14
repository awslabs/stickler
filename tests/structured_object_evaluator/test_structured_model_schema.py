"""
Test the JSON schema serialization of StructuredModel classes.
This verifies that structured models can be correctly serialized to JSON schema
with all comparison metadata intact.
"""

import json
from typing import List, Optional


from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator


class SimpleTestModel(StructuredModel):
    """A simple model with a single field for testing."""

    text: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class ComplexTestModel(StructuredModel):
    """A more complex model with multiple fields and different configurations."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )

    description: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.5, weight=0.5
    )

    tags: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.6, weight=1.0
    )


class NestedTestModel(StructuredModel):
    """A model with a nested StructuredModel field."""

    title: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )

    simple: SimpleTestModel = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


def test_simple_model_schema():
    """Test that a simple model can be serialized to JSON schema with comparison metadata."""
    # Get the JSON schema for the simple model
    schema = SimpleTestModel.model_json_schema()

    # Check that the schema has the expected structure
    assert "properties" in schema
    assert "text" in schema["properties"]

    # Check that the text property has comparison metadata
    text_props = schema["properties"]["text"]
    assert "x-comparison" in text_props

    # Check that comparison metadata has the expected fields
    comp_info = text_props["x-comparison"]
    assert comp_info["comparator_type"] == "LevenshteinComparator"
    assert comp_info["threshold"] == 0.7
    assert comp_info["weight"] == 1.0


def test_complex_model_schema():
    """Test that a complex model can be serialized to JSON schema with comparison metadata for all fields."""
    # Get the JSON schema for the complex model
    schema = ComplexTestModel.model_json_schema()

    # Check that the schema has the expected structure
    assert "properties" in schema
    assert all(
        field in schema["properties"] for field in ["id", "name", "description", "tags"]
    )

    # Check that each property has comparison metadata
    fields = {
        "id": {"threshold": 0.9, "weight": 2.0},
        "name": {"threshold": 0.7, "weight": 1.0},
        "description": {"threshold": 0.5, "weight": 0.5},
        "tags": {"threshold": 0.6, "weight": 1.0},
    }

    for field_name, expected_values in fields.items():
        field_props = schema["properties"][field_name]
        assert "x-comparison" in field_props

        comp_info = field_props["x-comparison"]
        assert comp_info["threshold"] == expected_values["threshold"]
        assert comp_info["weight"] == expected_values["weight"]


def test_nested_model_schema():
    """Test that a model with nested StructuredModel fields can be serialized to JSON schema."""
    # Get the JSON schema for the nested model
    schema = NestedTestModel.model_json_schema()

    # Check that the schema has the expected structure
    assert "properties" in schema
    assert "title" in schema["properties"]
    assert "simple" in schema["properties"]

    # Check that title has comparison metadata
    title_props = schema["properties"]["title"]
    assert "x-comparison" in title_props
    title_comp = title_props["x-comparison"]
    assert title_comp["threshold"] == 0.8
    assert title_comp["weight"] == 1.5

    # Check that simple field has a reference to the SimpleTestModel schema
    simple_props = schema["properties"]["simple"]
    assert (
        "$ref" in simple_props or "allOf" in simple_props
    )  # Pydantic might use either format


def test_schema_serialization():
    """Test that the schema can be serialized to JSON without errors."""
    # Get the schema and ensure it can be converted to JSON string
    schema = ComplexTestModel.model_json_schema()
    json_string = json.dumps(schema)

    # Verify it can be parsed back
    parsed_schema = json.loads(json_string)
    assert parsed_schema["properties"]["id"]["x-comparison"]["threshold"] == 0.9

    # Test with nested model as well
    nested_schema = NestedTestModel.model_json_schema()
    nested_json = json.dumps(nested_schema)
    parsed_nested = json.loads(nested_json)


def test_schema_validation_compatibility():
    """Test that the schema is compatible with JSON Schema validators."""
    # Get the schema for a model
    schema = SimpleTestModel.model_json_schema()

    # Check that the schema has required JSON Schema fields
    # Pydantic doesn't include $schema by default, but does include other required fields
    assert "title" in schema
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema

    # We can add the $schema field manually if needed for external validators
    schema_with_uri = schema.copy()
    schema_with_uri["$schema"] = "http://json-schema.org/draft-07/schema#"

    # Verify that the schema is valid JSON and can be serialized/deserialized
    schema_json = json.dumps(schema_with_uri)
    parsed_schema = json.loads(schema_json)
    assert parsed_schema["$schema"] == "http://json-schema.org/draft-07/schema#"
