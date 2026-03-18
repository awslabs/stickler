"""Unit tests for schema_builder.py — schema construction and file export.

Tests focus on the pure logic: internal field-to-schema conversion and
file export. Streamlit widget rendering is not tested here (requires a
running Streamlit app context).
"""

import json
from pathlib import Path

import pytest

from stickler.annotator.schema_builder import SchemaBuilder, _build_field_schema
from stickler.structured_object_evaluator import StructuredModel


# ---------------------------------------------------------------------------
# _build_field_schema — primitive types
# ---------------------------------------------------------------------------


class TestBuildFieldSchemaPrimitives:
    """Primitive field types produce correct JSON Schema snippets."""

    @pytest.mark.parametrize("ftype", ["string", "number", "integer", "boolean"])
    def test_primitive_type(self, ftype: str):
        result = _build_field_schema({"name": "x", "type": ftype})
        assert result == {"type": ftype}


# ---------------------------------------------------------------------------
# _build_field_schema — object type
# ---------------------------------------------------------------------------


class TestBuildFieldSchemaObject:
    """Nested object fields produce correct JSON Schema."""

    def test_empty_object(self):
        result = _build_field_schema({"name": "addr", "type": "object", "properties": []})
        assert result == {"type": "object", "properties": {}}

    def test_object_with_sub_fields(self):
        field = {
            "name": "address",
            "type": "object",
            "properties": [
                {"name": "street", "type": "string"},
                {"name": "zip", "type": "integer"},
            ],
        }
        result = _build_field_schema(field)
        assert result == {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "zip": {"type": "integer"},
            },
            "required": ["street", "zip"],
        }


# ---------------------------------------------------------------------------
# _build_field_schema — array type
# ---------------------------------------------------------------------------


class TestBuildFieldSchemaArray:
    """Array fields produce correct JSON Schema."""

    def test_array_of_strings(self):
        field = {"name": "tags", "type": "array", "items_type": "string"}
        result = _build_field_schema(field)
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_array_of_numbers(self):
        field = {"name": "scores", "type": "array", "items_type": "number"}
        result = _build_field_schema(field)
        assert result == {"type": "array", "items": {"type": "number"}}

    def test_array_of_objects(self):
        field = {
            "name": "items",
            "type": "array",
            "items_type": "object",
            "items_properties": [
                {"name": "desc", "type": "string"},
                {"name": "qty", "type": "integer"},
            ],
        }
        result = _build_field_schema(field)
        assert result == {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "desc": {"type": "string"},
                    "qty": {"type": "integer"},
                },
                "required": ["desc", "qty"],
            },
        }

    def test_array_of_objects_empty(self):
        field = {
            "name": "items",
            "type": "array",
            "items_type": "object",
            "items_properties": [],
        }
        result = _build_field_schema(field)
        assert result == {
            "type": "array",
            "items": {"type": "object", "properties": {}},
        }


# ---------------------------------------------------------------------------
# SchemaBuilder._fields_to_schema
# ---------------------------------------------------------------------------


class TestFieldsToSchema:
    """_fields_to_schema produces valid root-level JSON Schema."""

    def test_single_string_field(self):
        fields = [{"name": "name", "type": "string"}]
        schema = SchemaBuilder._fields_to_schema(fields)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["required"] == ["name"]

    def test_multiple_fields(self):
        fields = [
            {"name": "vendor", "type": "string"},
            {"name": "total", "type": "number"},
            {"name": "paid", "type": "boolean"},
        ]
        schema = SchemaBuilder._fields_to_schema(fields)
        assert set(schema["properties"].keys()) == {"vendor", "total", "paid"}
        assert set(schema["required"]) == {"vendor", "total", "paid"}

    def test_nested_object_field(self):
        fields = [
            {
                "name": "address",
                "type": "object",
                "properties": [{"name": "city", "type": "string"}],
            }
        ]
        schema = SchemaBuilder._fields_to_schema(fields)
        addr = schema["properties"]["address"]
        assert addr["type"] == "object"
        assert "city" in addr["properties"]

    def test_compatible_with_structured_model(self):
        """The output schema must be accepted by StructuredModel.from_json_schema()."""
        fields = [
            {"name": "name", "type": "string"},
            {"name": "age", "type": "integer"},
            {"name": "active", "type": "boolean"},
        ]
        schema = SchemaBuilder._fields_to_schema(fields)
        model_cls = StructuredModel.from_json_schema(schema)
        assert issubclass(model_cls, StructuredModel)
        assert "name" in model_cls.model_fields
        assert "age" in model_cls.model_fields
        assert "active" in model_cls.model_fields

    def test_nested_object_compatible_with_structured_model(self):
        fields = [
            {"name": "vendor", "type": "string"},
            {
                "name": "address",
                "type": "object",
                "properties": [
                    {"name": "street", "type": "string"},
                    {"name": "zip", "type": "string"},
                ],
            },
        ]
        schema = SchemaBuilder._fields_to_schema(fields)
        model_cls = StructuredModel.from_json_schema(schema)
        assert issubclass(model_cls, StructuredModel)
        assert "vendor" in model_cls.model_fields
        assert "address" in model_cls.model_fields

    def test_array_field_compatible_with_structured_model(self):
        fields = [
            {"name": "tags", "type": "array", "items_type": "string"},
        ]
        schema = SchemaBuilder._fields_to_schema(fields)
        model_cls = StructuredModel.from_json_schema(schema)
        assert issubclass(model_cls, StructuredModel)
        assert "tags" in model_cls.model_fields

    def test_array_of_objects_compatible_with_structured_model(self):
        fields = [
            {
                "name": "line_items",
                "type": "array",
                "items_type": "object",
                "items_properties": [
                    {"name": "description", "type": "string"},
                    {"name": "amount", "type": "number"},
                ],
            },
        ]
        schema = SchemaBuilder._fields_to_schema(fields)
        model_cls = StructuredModel.from_json_schema(schema)
        assert issubclass(model_cls, StructuredModel)
        assert "line_items" in model_cls.model_fields


# ---------------------------------------------------------------------------
# SchemaBuilder.export_to_file
# ---------------------------------------------------------------------------


class TestExportToFile:
    """export_to_file writes valid, indented JSON."""

    def test_writes_json_file(self, tmp_path: Path):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        out = tmp_path / "schema.json"
        SchemaBuilder().export_to_file(schema, out)

        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == schema

    def test_uses_4_space_indent(self, tmp_path: Path):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        out = tmp_path / "schema.json"
        SchemaBuilder().export_to_file(schema, out)

        text = out.read_text(encoding="utf-8")
        # 4-space indent means properties key is indented 4 spaces
        assert '    "type"' in text

    def test_trailing_newline(self, tmp_path: Path):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        out = tmp_path / "schema.json"
        SchemaBuilder().export_to_file(schema, out)

        text = out.read_text(encoding="utf-8")
        assert text.endswith("\n")

    def test_creates_parent_directories(self, tmp_path: Path):
        out = tmp_path / "sub" / "dir" / "schema.json"
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        SchemaBuilder().export_to_file(schema, out)
        assert out.exists()

    def test_round_trip_byte_identical(self, tmp_path: Path):
        """Export then read back produces identical JSON."""
        schema = SchemaBuilder._fields_to_schema([
            {"name": "name", "type": "string"},
            {"name": "count", "type": "integer"},
        ])
        out = tmp_path / "schema.json"
        SchemaBuilder().export_to_file(schema, out)

        text = out.read_text(encoding="utf-8")
        expected = json.dumps(schema, indent=4, sort_keys=False) + "\n"
        assert text == expected

    def test_accepts_string_path(self, tmp_path: Path):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        out = str(tmp_path / "schema.json")
        SchemaBuilder().export_to_file(schema, out)
        assert Path(out).exists()
