"""Round-trip and error tests for the ``extra_fields`` leak (issue #214).

``model_json_schema()`` used to emit the internal ``extra_fields`` property as
a propertyless ``{"type": "object", "additionalProperties": true}``, which
``from_json_schema()`` then rejected. So the natural thing to try -- export a
model and re-import it -- failed with an error that named an internal field the
user never wrote.

These tests pin three things:

1. ``model_json_schema()`` no longer emits ``extra_fields`` (top level or in
   nested ``$defs``).
2. Schemas exported by *older* versions, which still carry the leaked
   ``extra_fields``, re-import (backward compatibility).
3. A genuine object schema with no ``properties`` fails with an error that
   explains the real constraint rather than implying the schema is malformed.
"""

from typing import List

import pytest

from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)


class _Inner(StructuredModel):
    x: str = ComparableField()


class _Outer(StructuredModel):
    a: str = ComparableField()
    inner: _Inner = ComparableField()
    items: List[_Inner] = ComparableField()


class TestExtraFieldsNotLeaked:
    """model_json_schema() must not advertise the internal extra_fields."""

    def test_top_level_schema_omits_extra_fields(self):
        schema = _Outer.model_json_schema()
        assert "extra_fields" not in schema.get("properties", {})

    def test_nested_defs_omit_extra_fields(self):
        schema = _Outer.model_json_schema()
        for name, definition in schema.get("$defs", {}).items():
            assert "extra_fields" not in definition.get("properties", {}), (
                f"$defs.{name} still carries the internal extra_fields property"
            )

    def test_extra_fields_removed_from_required(self):
        # Pydantic never marks extra_fields required (it has a default), but
        # guard against a future change that would leave it in `required`
        # pointing at a now-absent property.
        schema = _Outer.model_json_schema()
        assert "extra_fields" not in schema.get("required", [])

    def test_real_field_properties_still_present(self):
        # Stripping extra_fields must not disturb the user's own fields.
        schema = _Outer.model_json_schema()
        assert set(schema["properties"]) == {"a", "inner", "items"}


class TestModelJsonSchemaRoundTrips:
    """model_json_schema() output must re-import through from_json_schema()."""

    def test_flat_model_round_trips(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        model = StructuredModel.from_json_schema(schema)

        # The exact reproduction from the issue.
        rebuilt = StructuredModel.from_json_schema(model.model_json_schema())

        assert "a" in rebuilt.model_fields

    def test_nested_model_round_trips(self):
        rebuilt = StructuredModel.from_json_schema(_Outer.model_json_schema())

        assert {"a", "inner", "items"} <= set(rebuilt.model_fields)

    def test_round_trip_preserves_comparison_behavior(self):
        # All fields required on purpose: an optional field exports as
        # ``anyOf[T, null]`` (issue #159), which only re-imports once anyOf
        # support (issue #198) lands. That round trip is a separate concern
        # from the extra_fields leak this test isolates.
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["name", "score"],
            }
        )
        rebuilt = StructuredModel.from_json_schema(model.model_json_schema())

        data_a = {"name": "hello", "score": 1.0}
        data_b = {"name": "hallo", "score": 1.0}
        original = model.from_json(data_a).compare_with(model.from_json(data_b))
        after = rebuilt.from_json(data_a).compare_with(rebuilt.from_json(data_b))

        assert original["overall_score"] == after["overall_score"]


class TestBackwardCompatibleImport:
    """Schemas exported by older versions still carry the leaked extra_fields."""

    def test_old_export_with_leaked_extra_fields_imports(self):
        old_export = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "extra_fields": {
                    "additionalProperties": True,
                    "title": "Extra Fields",
                    "type": "object",
                },
            },
        }
        model = StructuredModel.from_json_schema(old_export)

        user_fields = [f for f in model.model_fields if f != "extra_fields"]
        assert user_fields == ["a"]

    def test_user_field_named_extra_fields_with_properties_is_kept(self):
        # A user field that happens to be named extra_fields but declares real
        # properties is a genuine nested object, not the internal leak. It must
        # survive import as a field.
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "extra_fields": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                },
            },
        }
        model = StructuredModel.from_json_schema(schema)

        assert "extra_fields" in model.model_fields

    def test_user_primitive_field_named_extra_fields_is_kept(self):
        schema = {
            "type": "object",
            "properties": {"extra_fields": {"type": "string"}},
        }
        model = StructuredModel.from_json_schema(schema)

        assert "extra_fields" in model.model_fields


class TestPropertylessObjectError:
    """A valid-but-unmodelable object schema gets an explanatory error."""

    def test_propertyless_object_names_the_real_constraint(self):
        with pytest.raises(ValueError) as exc_info:
            StructuredModel.from_json_schema(
                {"type": "object", "additionalProperties": True}
            )

        message = str(exc_info.value)
        # It must explain there are no fields to compare, not imply the schema
        # is malformed (it is valid JSON Schema).
        assert "properties" in message
        assert "compare" in message or "fields" in message
