"""
Test the JSON schema serialization of StructuredModel classes.

``model_json_schema()`` describes the model's *shape* for external consumers
(e.g. tool specs sent to an LLM), so it must NOT carry comparison metadata
(issue #188). The metadata is still attached to every field and reachable via
``json_schema_extra`` and the deliberate export path ``to_json_schema()``.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic.json_schema import GenerateJsonSchema

from stickler.comparators.levenshtein import LevenshteinComparator

# Import from structured_object_evaluator instead of anls_star_lib
from stickler.structured_object_evaluator import ComparableField, StructuredModel


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
    """The rendered schema describes the shape; comparison config stays off it."""
    # Get the JSON schema for the simple model
    schema = SimpleTestModel.model_json_schema()

    # Check that the schema has the expected structure
    assert "properties" in schema
    assert "text" in schema["properties"]

    # Comparison metadata must NOT leak into the rendered schema (issue #188):
    # it is evaluation config, not part of the shape, and it would ride into
    # tool specs sent to LLMs.
    text_props = schema["properties"]["text"]
    assert "x-comparison" not in text_props

    # The metadata is still attached to the field and readable the way the
    # comparison engine reads it.
    comp_info = _comparison_metadata(SimpleTestModel, "text")
    assert comp_info["comparator_type"] == "LevenshteinComparator"
    assert comp_info["threshold"] == 0.7
    assert comp_info["weight"] == 1.0


def test_complex_model_schema():
    """Every field keeps its metadata off the rendered schema but readable."""
    # Get the JSON schema for the complex model
    schema = ComplexTestModel.model_json_schema()

    # Check that the schema has the expected structure
    assert "properties" in schema
    assert all(
        field in schema["properties"] for field in ["id", "name", "description", "tags"]
    )

    fields = {
        "id": {"threshold": 0.9, "weight": 2.0},
        "name": {"threshold": 0.7, "weight": 1.0},
        "description": {"threshold": 0.5, "weight": 0.5},
        "tags": {"threshold": 0.6, "weight": 1.0},
    }

    for field_name, expected_values in fields.items():
        # Not rendered (issue #188)...
        assert "x-comparison" not in schema["properties"][field_name]

        # ...but still configured on the field.
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
    assert "x-comparison" not in nested_json


def test_required_follows_annotation():
    """`required` derives from the annotation, not ComparableField's default.

    ComparableField assigns ``default=None`` so construction tolerates
    partial predictions, but that is a runtime concern: for schema purposes
    ``shipment_id: str`` is required and ``notes: Optional[str]`` is not
    (issue #188). A field with a real default stays optional.
    """

    class RequirednessModel(StructuredModel):
        must_have: str = ComparableField()
        may_skip: Optional[str] = ComparableField(default=None)
        has_default: str = ComparableField(default="fallback")

    schema = RequirednessModel.model_json_schema()

    assert schema.get("required") == ["must_have"]
    # The required field renders its bare type, with no contradictory
    # default:null and no null-widening.
    must_have = schema["properties"]["must_have"]
    assert must_have["type"] == "string"
    assert "default" not in must_have
    # The genuinely optional field keeps its nullable rendering and default.
    may_skip = schema["properties"]["may_skip"]
    assert {"type": "null"} in may_skip.get("anyOf", [])
    # The defaulted field keeps its default and stays optional.
    assert schema["properties"]["has_default"]["default"] == "fallback"

    # Schema requiredness must not leak into runtime: partial construction
    # still works, because the comparison engine builds models from
    # prediction JSON that may omit fields.
    instance = RequirednessModel.from_json({"may_skip": "x"})
    assert instance.must_have is None


def test_nested_model_required_in_defs():
    """Nested models rendered into $defs get the same required treatment."""
    schema = NestedTestModel.model_json_schema()

    simple_def = schema["$defs"]["SimpleTestModel"]
    assert simple_def.get("required") == ["text"]


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


def test_model_defaults():
    """Test that models with default values maintain proper schema information."""

    # Define a model with default values
    class DefaultTestModel(StructuredModel):
        required_field: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.8
        )
        optional_field: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.7, weight=1.5
        )
        defaulted_field: str = Field(
            "default value", description="A field with default value"
        )

    # Get the schema and check it
    schema = DefaultTestModel.model_json_schema()

    assert "properties" in schema
    assert all(
        field in schema["properties"]
        for field in ["required_field", "optional_field", "defaulted_field"]
    )

    # required_field's comparison config stays off the rendered schema but on
    # the field (issue #188).
    assert "x-comparison" not in schema["properties"]["required_field"]
    assert _comparison_metadata(DefaultTestModel, "required_field")["threshold"] == 0.8

    # Check that the default value is present
    default_props = schema["properties"]["defaulted_field"]
    assert "default" in default_props
    assert default_props["default"] == "default value"

    # Check that description is properly set
    assert "description" in default_props
    assert default_props["description"] == "A field with default value"


# ---------------------------------------------------------------------------
# schema_generator composition (#188)
# ---------------------------------------------------------------------------
#
# `schema_generator` is a documented public parameter of
# `BaseModel.model_json_schema()`. The annotation-driven requiredness rule is a
# mixin overriding only `field_is_required`, so a caller's generator must be
# composed with rather than deferred to -- `setdefault` would leave theirs in
# place and silently drop `required` entirely, which is the bug #188 fixes.


class _CallerGenerator(GenerateJsonSchema):
    """A caller's own generator, standing in for a framework's."""

    def generate(self, schema, mode="validation"):
        rendered = super().generate(schema, mode=mode)
        rendered["x-caller-marker"] = True
        return rendered


def test_caller_supplied_generator_keeps_the_requiredness_rule():
    """The fix must survive a caller passing their own generator."""
    schema = SimpleTestModel.model_json_schema(schema_generator=_CallerGenerator)

    assert schema.get("required") == ["text"], (
        "a caller's generator must not silently drop the requiredness derivation"
    )


def test_caller_supplied_generator_is_still_applied():
    """Composition, not replacement: the caller's customisation survives too."""
    schema = SimpleTestModel.model_json_schema(schema_generator=_CallerGenerator)

    assert schema.get("x-caller-marker") is True


def test_generator_already_carrying_the_rule_is_used_unchanged():
    """No pointless synthesised subclass when the caller already has the mixin."""
    from stickler.structured_object_evaluator.models.structured_model import (
        _AnnotationDrivenJsonSchema,
        _compose_schema_generator,
    )

    class Already(_AnnotationDrivenJsonSchema):
        pass

    assert _compose_schema_generator(Already) is Already
    assert _compose_schema_generator(None) is _AnnotationDrivenJsonSchema
