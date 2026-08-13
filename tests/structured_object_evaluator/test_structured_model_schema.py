"""
Test the JSON schema serialization of StructuredModel classes.

``model_json_schema()`` describes the model's *shape* for external consumers
(e.g. tool specs sent to an LLM), so it must NOT carry comparison metadata
(issue #188). The metadata is still attached to every field and reachable via
``json_schema_extra`` and the deliberate export path ``to_json_schema()``.
"""

import json
from typing import Any, Dict, List, Optional

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def _comparison_metadata(model_cls, field_name: str) -> Dict[str, Any]:
    """Read a field's comparison metadata the way the engine does.

    The metadata lives on the field's ``json_schema_extra`` callable; it is
    deliberately not rendered into ``model_json_schema()``.
    """
    extra: Dict[str, Any] = {}
    model_cls.model_fields[field_name].json_schema_extra(extra)
    return extra["x-comparison"]


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

    # Comparison metadata must NOT leak into the rendered schema (issue #188).
    text_props = schema["properties"]["text"]
    assert "x-comparison" not in text_props

    # It is still attached to the field, read the way the engine reads it.
    comp_info = _comparison_metadata(SimpleTestModel, "text")
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
        # Not rendered (issue #188), but still configured on the field.
        assert "x-comparison" not in schema["properties"][field_name]

        comp_info = _comparison_metadata(ComplexTestModel, field_name)
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

    # title's comparison config stays off the rendered schema but on the field
    assert "x-comparison" not in schema["properties"]["title"]
    title_comp = _comparison_metadata(NestedTestModel, "title")
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

    # Verify it can be parsed back, and that no comparison metadata leaked
    # anywhere in the document (issue #188).
    parsed_schema = json.loads(json_string)
    assert "x-comparison" not in json_string
    assert parsed_schema["properties"]["id"]["type"] == "string"

    # Test with nested model as well
    nested_schema = NestedTestModel.model_json_schema()
    nested_json = json.dumps(nested_schema)
    json.loads(nested_json)


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
