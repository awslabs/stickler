"""Unit tests for JSON Schema validation utilities."""

import pytest
from jsonschema.exceptions import SchemaError, ValidationError

from stickler.structured_object_evaluator.utils.json_schema_validator import (
    validate_json_schema,
    validate_instance_against_schema,
)


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""
    
    def test_valid_basic_schema(self):
        """Test validation of a basic valid JSON Schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        # Should not raise any exception
        validate_json_schema(schema)
    
    def test_valid_schema_with_required(self):
        """Test validation of schema with required fields."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name", "email"]
        }
        validate_json_schema(schema)
    
    def test_valid_schema_with_nested_objects(self):
        """Test validation of schema with nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }
        validate_json_schema(schema)
    
    def test_valid_schema_with_arrays(self):
        """Test validation of schema with array types."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        validate_json_schema(schema)
    
    def test_valid_schema_with_definitions(self):
        """Test validation of schema with definitions."""
        schema = {
            "type": "object",
            "properties": {
                "person": {"$ref": "#/definitions/Person"}
            },
            "definitions": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        validate_json_schema(schema)
    
    def test_valid_schema_with_defs(self):
        """Test validation of schema with $defs (draft 2019-09+)."""
        schema = {
            "type": "object",
            "properties": {
                "person": {"$ref": "#/$defs/Person"}
            },
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        validate_json_schema(schema)
    
    def test_valid_schema_with_extensions(self):
        """Test validation of schema with x-aws-stickler-* extensions."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "TestModel",
            "x-aws-stickler-match-threshold": 0.8,
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-threshold": 0.9,
                    "x-aws-stickler-weight": 2.0
                }
            }
        }
        # Extensions should be allowed (JSON Schema allows additional properties)
        validate_json_schema(schema)
    
    def test_invalid_schema_not_dict(self):
        """Test that non-dictionary input raises ValueError."""
        with pytest.raises(ValueError, match="Schema must be a dictionary"):
            validate_json_schema("not a dict")
    
    def test_invalid_schema_empty_dict(self):
        """Test that empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="Schema cannot be empty"):
            validate_json_schema({})
    
    def test_invalid_schema_bad_type(self):
        """Test that invalid type value raises SchemaError."""
        schema = {
            "type": "invalid_type"
        }
        with pytest.raises(SchemaError):
            validate_json_schema(schema)
    
    def test_invalid_schema_bad_structure(self):
        """Test that malformed schema structure raises SchemaError."""
        schema = {
            "type": "object",
            "properties": "should be a dict not a string"
        }
        with pytest.raises(SchemaError):
            validate_json_schema(schema)
    
    def test_invalid_schema_bad_items(self):
        """Test that invalid items definition raises SchemaError."""
        schema = {
            "type": "array",
            "items": "should be a schema not a string"
        }
        with pytest.raises(SchemaError):
            validate_json_schema(schema)


class TestValidateInstanceAgainstSchema:
    """Tests for validate_instance_against_schema function."""
    
    def test_valid_instance_basic(self):
        """Test validation of valid instance against basic schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        instance = {"name": "Alice", "age": 30}
        # Should not raise any exception
        validate_instance_against_schema(instance, schema)
    
    def test_valid_instance_with_required_fields(self):
        """Test validation when all required fields are present."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name", "email"]
        }
        instance = {"name": "Alice", "email": "alice@example.com"}
        validate_instance_against_schema(instance, schema)
    
    def test_valid_instance_with_nested_objects(self):
        """Test validation of nested object instance."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }
        instance = {
            "name": "Alice",
            "address": {
                "street": "123 Main St",
                "city": "Boston"
            }
        }
        validate_instance_against_schema(instance, schema)
    
    def test_valid_instance_with_arrays(self):
        """Test validation of array instance."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        instance = {"tags": ["python", "testing", "json-schema"]}
        validate_instance_against_schema(instance, schema)
    
    def test_invalid_instance_wrong_type(self):
        """Test that instance with wrong type raises ValidationError."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"}
            }
        }
        instance = {"age": "not an integer"}
        with pytest.raises(ValidationError):
            validate_instance_against_schema(instance, schema)
    
    def test_invalid_instance_missing_required(self):
        """Test that missing required field raises ValidationError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        instance = {"age": 30}  # Missing required 'name'
        with pytest.raises(ValidationError):
            validate_instance_against_schema(instance, schema)
    
    def test_invalid_instance_wrong_array_item_type(self):
        """Test that wrong array item type raises ValidationError."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        instance = {"tags": ["valid", 123, "another"]}  # 123 is not a string
        with pytest.raises(ValidationError):
            validate_instance_against_schema(instance, schema)
    
    def test_valid_instance_with_additional_properties(self):
        """Test that additional properties are allowed by default."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        instance = {"name": "Alice", "extra": "allowed"}
        # Should not raise - additional properties allowed by default
        validate_instance_against_schema(instance, schema)
