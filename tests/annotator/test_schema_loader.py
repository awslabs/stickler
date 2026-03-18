"""Unit tests for schema_loader.py — loading schemas from files, imports, and builder dicts."""

import json
from pathlib import Path

import pytest

from stickler.annotator.schema_loader import SchemaLoader
from stickler.structured_object_evaluator import StructuredModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "amount": {"type": "number"},
    },
    "required": ["name"],
}

SCHEMA_WITH_EXTENSIONS = {
    "type": "object",
    "x-aws-stickler-model-name": "Invoice",
    "x-aws-stickler-match-threshold": 0.85,
    "properties": {
        "vendor": {
            "type": "string",
            "x-aws-stickler-comparator": "LevenshteinComparator",
            "x-aws-stickler-threshold": 0.9,
            "x-aws-stickler-weight": 2.0,
        },
        "total": {"type": "number"},
    },
    "required": ["vendor"],
}

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["street"],
        },
    },
    "required": ["name"],
}


# ---------------------------------------------------------------------------
# from_json_schema_file
# ---------------------------------------------------------------------------


class TestFromJsonSchemaFile:
    """Tests for SchemaLoader.from_json_schema_file()."""

    def test_loads_simple_schema(self, tmp_path: Path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SIMPLE_SCHEMA))

        raw, model_cls = SchemaLoader.from_json_schema_file(schema_file)

        assert raw == SIMPLE_SCHEMA
        assert isinstance(model_cls, type)
        assert issubclass(model_cls, StructuredModel)

    def test_model_has_correct_fields(self, tmp_path: Path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SIMPLE_SCHEMA))

        _, model_cls = SchemaLoader.from_json_schema_file(schema_file)

        assert "name" in model_cls.model_fields
        assert "amount" in model_cls.model_fields

    def test_preserves_stickler_extensions_in_raw_schema(self, tmp_path: Path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SCHEMA_WITH_EXTENSIONS))

        raw, _ = SchemaLoader.from_json_schema_file(schema_file)

        assert raw["x-aws-stickler-match-threshold"] == 0.85
        assert raw["properties"]["vendor"]["x-aws-stickler-weight"] == 2.0

    def test_nested_schema(self, tmp_path: Path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(NESTED_SCHEMA))

        raw, model_cls = SchemaLoader.from_json_schema_file(schema_file)

        assert "name" in model_cls.model_fields
        assert "address" in model_cls.model_fields

    def test_accepts_string_path(self, tmp_path: Path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SIMPLE_SCHEMA))

        raw, model_cls = SchemaLoader.from_json_schema_file(str(schema_file))
        assert raw == SIMPLE_SCHEMA

    def test_nonexistent_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            SchemaLoader.from_json_schema_file(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        schema_file = tmp_path / "bad.json"
        schema_file.write_text("{not valid json!!!")

        with pytest.raises(ValueError, match="Invalid JSON"):
            SchemaLoader.from_json_schema_file(schema_file)

    def test_non_object_json_raises(self, tmp_path: Path):
        schema_file = tmp_path / "array.json"
        schema_file.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="JSON object"):
            SchemaLoader.from_json_schema_file(schema_file)

    def test_invalid_schema_raises(self, tmp_path: Path):
        schema_file = tmp_path / "bad_schema.json"
        schema_file.write_text(json.dumps({"type": "object"}))

        with pytest.raises(ValueError, match="Invalid schema"):
            SchemaLoader.from_json_schema_file(schema_file)


# ---------------------------------------------------------------------------
# from_pydantic_import
# ---------------------------------------------------------------------------


class TestFromPydanticImport:
    """Tests for SchemaLoader.from_pydantic_import()."""

    def test_no_dot_in_path_raises(self):
        with pytest.raises(ValueError, match="dotted path"):
            SchemaLoader.from_pydantic_import("NoDots")

    def test_nonexistent_module_raises(self):
        with pytest.raises(ImportError, match="Could not import"):
            SchemaLoader.from_pydantic_import("nonexistent.module.MyModel")

    def test_nonexistent_class_raises(self):
        with pytest.raises(ImportError, match="has no attribute"):
            SchemaLoader.from_pydantic_import(
                "stickler.structured_object_evaluator.NoSuchClass"
            )

    def test_non_structured_model_raises(self):
        with pytest.raises(TypeError, match="not a StructuredModel subclass"):
            SchemaLoader.from_pydantic_import("pathlib.Path")

    def test_imports_structured_model_subclass(self):
        """Import StructuredModel itself (base class) — should work as identity."""
        schema, cls = SchemaLoader.from_pydantic_import(
            "stickler.structured_object_evaluator.StructuredModel"
        )
        assert isinstance(schema, dict)
        assert issubclass(cls, StructuredModel)


# ---------------------------------------------------------------------------
# from_builder_schema
# ---------------------------------------------------------------------------


class TestFromBuilderSchema:
    """Tests for SchemaLoader.from_builder_schema()."""

    def test_simple_schema(self):
        raw, model_cls = SchemaLoader.from_builder_schema(SIMPLE_SCHEMA)

        assert raw is SIMPLE_SCHEMA  # same dict object returned
        assert issubclass(model_cls, StructuredModel)
        assert "name" in model_cls.model_fields

    def test_schema_with_extensions(self):
        raw, model_cls = SchemaLoader.from_builder_schema(SCHEMA_WITH_EXTENSIONS)

        assert raw["x-aws-stickler-match-threshold"] == 0.85
        assert issubclass(model_cls, StructuredModel)

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            SchemaLoader.from_builder_schema([1, 2, 3])  # type: ignore[arg-type]

    def test_invalid_schema_raises(self):
        with pytest.raises(ValueError, match="Invalid schema from builder"):
            SchemaLoader.from_builder_schema({"type": "object"})
