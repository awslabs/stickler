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


class TestRefResolution:
    """Tests for $ref resolution functionality."""

    def test_resolve_ref_from_definitions(self):
        """Test resolving $ref from definitions."""
        schema = {
            "type": "object",
            "definitions": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                }
            },
            "properties": {
                "home": {"$ref": "#/definitions/Address"},
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Should successfully resolve and create nested model
        assert "home" in field_definitions
        home_type = field_definitions["home"][0]
        
        # Verify it's a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import StructuredModel
        assert issubclass(home_type, StructuredModel)

    def test_resolve_ref_from_defs(self):
        """Test resolving $ref from $defs (JSON Schema draft 2019-09+)."""
        schema = {
            "type": "object",
            "$defs": {
                "Contact": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                    },
                }
            },
            "properties": {
                "contact": {"$ref": "#/$defs/Contact"},
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Should successfully resolve and create nested model
        assert "contact" in field_definitions
        contact_type = field_definitions["contact"][0]
        
        # Verify it's a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import StructuredModel
        assert issubclass(contact_type, StructuredModel)

    def test_resolve_ref_not_found_in_definitions(self):
        """Test error when $ref references non-existent definition."""
        schema = {
            "type": "object",
            "definitions": {
                "Address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                }
            },
            "properties": {
                "home": {"$ref": "#/definitions/NonExistent"},
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "Reference '#/definitions/NonExistent' not found" in str(exc_info.value)
        assert "Available: ['Address']" in str(exc_info.value)

    def test_resolve_ref_unsupported_format(self):
        """Test error when $ref uses unsupported format."""
        schema = {
            "type": "object",
            "properties": {
                "external": {"$ref": "http://example.com/schema.json#/Address"},
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "Unsupported $ref format" in str(exc_info.value)
        assert "Only '#/definitions/' and '#/$defs/' references are supported" in str(exc_info.value)

    def test_resolve_ref_in_array_items(self):
        """Test resolving $ref in array items."""
        schema = {
            "type": "object",
            "definitions": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                }
            },
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Item"},
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Should successfully resolve and create List[StructuredModel]
        assert "items" in field_definitions
        items_type = field_definitions["items"][0]
        
        # Verify it's a List type
        assert hasattr(items_type, "__origin__")
        assert items_type.__origin__ == list
        
        # Verify element is a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import StructuredModel
        assert issubclass(items_type.__args__[0], StructuredModel)


class TestNestedObjectHandling:
    """Tests for nested object handling."""

    def test_convert_nested_object(self):
        """Test converting nested object creates StructuredModel."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check nested model was created
        assert "person" in field_definitions
        person_type = field_definitions["person"][0]
        
        # Verify it's a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import StructuredModel
        assert issubclass(person_type, StructuredModel)
        
        # Verify nested model has correct fields
        assert "name" in person_type.model_fields
        assert "age" in person_type.model_fields

    def test_nested_object_with_extensions(self):
        """Test nested object with x-aws-stickler-* extensions."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "x-aws-stickler-weight": 2.0,
                    "x-aws-stickler-aggregate": True,
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check extensions are applied to the field
        address_field = field_definitions["address"][1]
        assert address_field.json_schema_extra._weight == 2.0
        assert address_field.json_schema_extra._aggregate is True

    def test_deeply_nested_objects(self):
        """Test deeply nested object structures."""
        schema = {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "location": {
                                    "type": "object",
                                    "properties": {
                                        "lat": {"type": "number"},
                                        "lon": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Should successfully create deeply nested structure
        assert "company" in field_definitions
        company_type = field_definitions["company"][0]
        
        from stickler.structured_object_evaluator.models.structured_model import StructuredModel
        assert issubclass(company_type, StructuredModel)
        
        # Verify nested fields exist
        assert "name" in company_type.model_fields
        assert "address" in company_type.model_fields


class TestArrayHandling:
    """Tests for array handling with both primitives and objects."""

    def test_array_of_objects(self):
        """Test converting array of objects creates List[StructuredModel]."""
        schema = {
            "type": "object",
            "properties": {
                "employees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                        },
                    },
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check array type
        assert "employees" in field_definitions
        employees_type = field_definitions["employees"][0]
        
        # Verify it's a List type
        assert hasattr(employees_type, "__origin__")
        assert employees_type.__origin__ == list
        
        # Verify element is a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import StructuredModel
        element_type = employees_type.__args__[0]
        assert issubclass(element_type, StructuredModel)
        
        # Verify element model has correct fields
        assert "name" in element_type.model_fields
        assert "role" in element_type.model_fields

    def test_array_of_objects_with_extensions(self):
        """Test array of objects with custom extensions."""
        schema = {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "x-aws-stickler-weight": 1.5,
                    "x-aws-stickler-threshold": 0.8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                        },
                    },
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check extensions are applied
        products_field = field_definitions["products"][1]
        assert products_field.json_schema_extra._weight == 1.5
        assert products_field.json_schema_extra._threshold == 0.8

    def test_array_of_primitives_all_types(self):
        """Test arrays of all primitive types."""
        schema = {
            "type": "object",
            "properties": {
                "strings": {"type": "array", "items": {"type": "string"}},
                "integers": {"type": "array", "items": {"type": "integer"}},
                "numbers": {"type": "array", "items": {"type": "number"}},
                "booleans": {"type": "array", "items": {"type": "boolean"}},
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Check all array types
        assert field_definitions["strings"][0].__args__[0] == str
        assert field_definitions["integers"][0].__args__[0] == int
        assert field_definitions["numbers"][0].__args__[0] == float
        assert field_definitions["booleans"][0].__args__[0] == bool


class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_invalid_json_type(self):
        """Test error when JSON Schema type is unsupported."""
        schema = {
            "type": "object",
            "properties": {
                "data": {"type": "null"},  # null type not supported for fields
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "Unsupported JSON Schema type: null" in str(exc_info.value)
        assert "Supported types:" in str(exc_info.value)

    def test_invalid_threshold_value(self):
        """Test error when threshold is out of range."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 1.5,  # Invalid: > 1.0
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "x-aws-stickler-threshold must be a number between 0.0 and 1.0" in str(exc_info.value)
        assert "1.5" in str(exc_info.value)

    def test_invalid_threshold_negative(self):
        """Test error when threshold is negative."""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "x-aws-stickler-threshold": -0.5,  # Invalid: < 0.0
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "x-aws-stickler-threshold must be a number between 0.0 and 1.0" in str(exc_info.value)

    def test_invalid_weight_value(self):
        """Test error when weight is not positive."""
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "x-aws-stickler-weight": -1.0,  # Invalid: not positive
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "x-aws-stickler-weight must be a positive number" in str(exc_info.value)
        assert "-1.0" in str(exc_info.value)

    def test_invalid_weight_zero(self):
        """Test error when weight is zero."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "x-aws-stickler-weight": 0,  # Invalid: not positive
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "x-aws-stickler-weight must be a positive number" in str(exc_info.value)

    def test_invalid_comparator_name(self):
        """Test error when comparator name is not registered."""
        schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "x-aws-stickler-comparator": "NonExistentComparator",
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "Invalid x-aws-stickler-comparator 'NonExistentComparator'" in str(exc_info.value)

    def test_invalid_clip_under_threshold_type(self):
        """Test error when clip-under-threshold is not boolean."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-clip-under-threshold": "yes",  # Invalid: not boolean
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "x-aws-stickler-clip-under-threshold must be a boolean" in str(exc_info.value)
        assert "str" in str(exc_info.value)

    def test_invalid_aggregate_type(self):
        """Test error when aggregate is not boolean."""
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "x-aws-stickler-aggregate": 1,  # Invalid: not boolean
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        assert "x-aws-stickler-aggregate must be a boolean" in str(exc_info.value)
        assert "int" in str(exc_info.value)

    def test_error_includes_field_path(self):
        """Test that errors include field path for context."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "x-aws-stickler-threshold": 2.0,  # Invalid
                        },
                    },
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        with pytest.raises(ValueError) as exc_info:
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
        
        # Error should include the nested field path
        error_msg = str(exc_info.value)
        assert "user.email" in error_msg or "field 'user.email'" in error_msg

    def test_missing_items_in_array(self):
        """Test handling of array without items specification."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    # Missing "items" - should default to empty dict
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        
        # Should handle gracefully - items defaults to {} which has no type
        # This will raise an error when trying to map the type
        with pytest.raises(ValueError):
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )
