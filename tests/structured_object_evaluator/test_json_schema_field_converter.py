"""Integration tests for JsonSchemaFieldConverter.convert_properties_to_fields()."""


import typing

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.json_schema_field_converter import (
    JsonSchemaFieldConverter,
)


def _unwrap_optional(field_type):
    """Strip ``Optional[...]`` (i.e. ``Union[X, None]``) down to the inner type.

    Non-required JSON-Schema fields are annotated as ``Optional[T]`` so their
    ``None`` default is valid (issue #149). Required fields stay bare. This
    helper lets type assertions target the underlying ``T`` regardless of the
    optional wrapper.
    """
    if typing.get_origin(field_type) is typing.Union:
        args = [a for a in typing.get_args(field_type) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return field_type


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

        # Check types. Required fields keep a bare annotation; optional fields
        # are widened to Optional[...] (issue #149) so their None default is
        # valid, so unwrap before comparing the underlying type.
        assert field_definitions["name"][0] is str  # required -> bare
        assert field_definitions["age"][0] is int  # required -> bare
        assert _unwrap_optional(field_definitions["price"][0]) is float  # optional
        assert _unwrap_optional(field_definitions["active"][0]) is bool  # optional

        # Check required vs optional (via is_required)
        name_field = field_definitions["name"][1]
        age_field = field_definitions["age"][1]
        price_field = field_definitions["price"][1]
        active_field = field_definitions["active"][1]

        assert name_field.is_required()  # Required
        assert age_field.is_required()  # Required
        assert not price_field.is_required()  # Optional
        assert not active_field.is_required()  # Optional

    def test_not_required_fields_are_optional_annotations(self):
        """Non-required fields get Optional[...] annotations; required stay bare.

        Regression for issue #149: an optional field has ``default=None`` but
        must also be annotated as ``Optional[...]`` so that None is valid when
        the rich-value path round-trips through from_json(...).model_dump().
        Covers all three sites: primitive, nested object, and array.
        """
        schema = {
            "type": "object",
            "properties": {
                # Required primitive -> bare
                "req_str": {"type": "string"},
                # Optional primitive -> Optional[str]
                "opt_str": {"type": "string"},
                # Optional nested object -> Optional[NestedModel]
                "opt_obj": {
                    "type": "object",
                    "properties": {"inner": {"type": "string"}},
                },
                # Optional array -> Optional[List[str]]
                "opt_arr": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["req_str"],
        }

        converter = JsonSchemaFieldConverter(schema)
        field_definitions = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        # Required field stays a bare annotation.
        req_type = field_definitions["req_str"][0]
        assert req_type is str
        assert typing.get_origin(req_type) is not typing.Union

        # Each optional field is wrapped in Optional[...] (Union[X, None]).
        for name, inner_check in [
            ("opt_str", lambda t: t is str),
            (
                "opt_obj",
                lambda t: (
                    __import__(
                        "stickler.structured_object_evaluator.models.structured_model",
                        fromlist=["StructuredModel"],
                    ).StructuredModel
                    in t.__mro__
                ),
            ),
            ("opt_arr", lambda t: typing.get_origin(t) is list),
        ]:
            field_type = field_definitions[name][0]
            assert typing.get_origin(field_type) is typing.Union, name
            assert type(None) in typing.get_args(field_type), name
            assert inner_check(_unwrap_optional(field_type)), name

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
        assert name_field.json_schema_extra._aggregate is False  # aggregate param is deprecated; auto-aggregation is used

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

        # Check types are List[primitive]. Both fields are optional (not in
        # required), so they are widened to Optional[List[...]] (issue #149);
        # unwrap to inspect the List.
        tags_type = _unwrap_optional(field_definitions["tags"][0])
        scores_type = _unwrap_optional(field_definitions["scores"][0])

        # Verify they are List types
        assert hasattr(tags_type, "__origin__")
        assert tags_type.__origin__ is list
        assert tags_type.__args__[0] is str

        assert hasattr(scores_type, "__origin__")
        assert scores_type.__origin__ is list
        assert scores_type.__args__[0] is float

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
            # First element is a type. Optional fields are wrapped in
            # Optional[...] (issue #149), so unwrap before the type check.
            assert isinstance(_unwrap_optional(field_def[0]), type)
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

        # Should successfully resolve and create nested model.
        # 'home' is optional, so the annotation is Optional[NestedModel]
        # (issue #149); unwrap to inspect the model class.
        assert "home" in field_definitions
        home_type = _unwrap_optional(field_definitions["home"][0])

        # Verify it's a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )
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

        # Should successfully resolve and create nested model.
        # 'contact' is optional, so the annotation is Optional[NestedModel]
        # (issue #149); unwrap to inspect the model class.
        assert "contact" in field_definitions
        contact_type = _unwrap_optional(field_definitions["contact"][0])

        # Verify it's a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )
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

        # Should successfully resolve and create List[StructuredModel].
        # 'items' is optional, so the annotation is Optional[List[...]]
        # (issue #149); unwrap to inspect the List.
        assert "items" in field_definitions
        items_type = _unwrap_optional(field_definitions["items"][0])

        # Verify it's a List type
        assert hasattr(items_type, "__origin__")
        assert items_type.__origin__ is list

        # Verify element is a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )
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

        # Check nested model was created. 'person' is optional, so the
        # annotation is Optional[NestedModel] (issue #149); unwrap it.
        assert "person" in field_definitions
        person_type = _unwrap_optional(field_definitions["person"][0])

        # Verify it's a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )
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
        assert address_field.json_schema_extra._aggregate is False  # aggregate param is deprecated; auto-aggregation is used

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

        # Should successfully create deeply nested structure. 'company' is
        # optional, so the annotation is Optional[NestedModel] (issue #149).
        assert "company" in field_definitions
        company_type = _unwrap_optional(field_definitions["company"][0])

        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )
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

        # Check array type. 'employees' is optional, so the annotation is
        # Optional[List[...]] (issue #149); unwrap to inspect the List.
        assert "employees" in field_definitions
        employees_type = _unwrap_optional(field_definitions["employees"][0])

        # Verify it's a List type
        assert hasattr(employees_type, "__origin__")
        assert employees_type.__origin__ is list

        # Verify element is a StructuredModel subclass
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )
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

        # Check all array types. All fields are optional, so each annotation
        # is Optional[List[...]] (issue #149); unwrap to inspect the List.
        assert _unwrap_optional(field_definitions["strings"][0]).__args__[0] is str
        assert _unwrap_optional(field_definitions["integers"][0]).__args__[0] is int
        assert _unwrap_optional(field_definitions["numbers"][0]).__args__[0] is float
        assert _unwrap_optional(field_definitions["booleans"][0]).__args__[0] is bool


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


class TestNullableTypeListForm:
    """Tests for the JSON Schema ``type: ["X", "null"]`` nullable idiom."""

    def test_nullable_string_optional_field(self):
        """``["string", "null"]`` on an optional field accepts both str and None."""
        schema = {
            "type": "object",
            "properties": {"description": {"type": ["string", "null"]}},
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        fields = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        from typing import Union, get_args, get_origin

        field_type, _ = fields["description"]
        # Optional[str] is Union[str, None]
        assert get_origin(field_type) is Union
        assert str in get_args(field_type)
        assert type(None) in get_args(field_type)

    def test_nullable_integer_required_field(self):
        """``["integer", "null"]`` on a required field still accepts None."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": ["integer", "null"]}},
            "required": ["count"],
        }

        converter = JsonSchemaFieldConverter(schema)
        fields = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        from typing import Union, get_args, get_origin

        field_type, _ = fields["count"]
        assert get_origin(field_type) is Union
        assert int in get_args(field_type)
        assert type(None) in get_args(field_type)

    def test_nullable_array_items(self):
        """Array ``items`` may use the ``[type, null]`` idiom for nullable elements."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": ["string", "null"]},
                },
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        fields = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        from typing import Union, get_args, get_origin

        field_type, _ = fields["tags"]
        # The field is not required, so the outer annotation is
        # Optional[List[Optional[str]]] (issue #149); unwrap to inspect the List.
        (inner,) = get_args(_unwrap_optional(field_type))
        assert get_origin(inner) is Union
        assert str in get_args(inner)
        assert type(None) in get_args(inner)

    def test_nullable_end_to_end_accepts_none(self):
        """A required nullable field accepts None where a plain field would reject it."""
        from pydantic import ValidationError

        from stickler import StructuredModel

        NullableModel = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"description": {"type": ["string", "null"]}},
                "required": ["description"],
            }
        )
        # Required and nullable: None is a valid value, not a missing field.
        assert NullableModel(description=None).description is None
        # Nullable does not make the field optional: omitting it still errors.
        with pytest.raises(ValidationError):
            NullableModel()

        PlainModel = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            }
        )
        # Without the ["X", "null"] wrap, None must be rejected.
        with pytest.raises(ValidationError):
            PlainModel(description=None)

    def test_nullable_round_trips_through_json_schema(self):
        """Optional fields survive to_json_schema -> from_json_schema as ["X", "null"]."""
        from stickler import StructuredModel

        Model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"description": {"type": ["string", "null"]}},
                "required": ["description"],
            }
        )
        schema = Model.to_json_schema()
        assert schema["properties"]["description"]["type"] == ["string", "null"]

        Rebuilt = StructuredModel.from_json_schema(schema)
        assert Rebuilt(description=None).description is None

    def test_optional_field_round_trips_as_explicitly_nullable(self):
        """An optional-but-not-nullable field gains an explicit "null" on export.

        This pins the interaction between the #149 widening (an optional field
        is annotated Optional[T] so its None default validates) and the #127
        exporter (an Optional[T] primitive is emitted as ["X", "null"]): a
        property declared only as {"type": "string"} and omitted from
        ``required`` round-trips out as ["string", "null"].

        This is intentional, not incidental. Under the explicit-null contract
        documented in docs/dynamic-models.md, "absent from required" means the
        value may be None, and the exported schema now says so out loud rather
        than leaving it implicit. The round trip is idempotent and lossless in
        meaning -- rebuilding from the exported schema yields a model that
        accepts exactly the same values.
        """
        from stickler import StructuredModel

        Model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"nickname": {"type": "string"}},
                "required": [],
            }
        )

        schema = Model.to_json_schema()
        assert schema["properties"]["nickname"]["type"] == ["string", "null"]
        assert "nickname" not in schema.get("required", [])

        # Idempotent: exporting the rebuilt model produces the same schema, and
        # the accepted value set is unchanged.
        Rebuilt = StructuredModel.from_json_schema(schema)
        assert Rebuilt.to_json_schema()["properties"]["nickname"]["type"] == [
            "string",
            "null",
        ]
        assert Rebuilt(nickname=None).nickname is None
        assert Rebuilt(nickname="ada").nickname == "ada"
        assert Model(nickname=None).nickname is None

    def test_nullable_object_array_element_accepts_none(self):
        """An array of nullable objects accepts None as a list element."""
        from stickler import StructuredModel

        Model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": ["object", "null"],
                            "properties": {"id": {"type": "string"}},
                            "required": ["id"],
                        },
                    }
                },
                "required": ["items"],
            }
        )
        assert Model(items=[None]).items == [None]

    def test_nullable_object_array_compare_with_none_element(self):
        """compare_with on a list holding None elements does not crash."""
        from stickler import StructuredModel

        Model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {
                            "type": ["object", "null"],
                            "properties": {"n": {"type": "string"}},
                            "required": ["n"],
                        },
                    }
                },
                "required": ["people"],
            }
        )

        gt = Model(people=[{"n": "alice"}, None])
        same = Model(people=[{"n": "alice"}, None])
        mismatch = Model(people=[{"n": "alice"}, {"n": "bob"}])

        # Identical None placement scores higher than a None-vs-model mismatch.
        assert same.compare_with(gt)["overall_score"] == pytest.approx(1.0)
        assert mismatch.compare_with(gt)["overall_score"] < 1.0

    def test_nullable_object_array_compare_with_none_element_list_subfield(self):
        """compare_with does not crash when a None-holding list's elements
        carry a list-valued sub-field (exercises the is_simple_list branch)."""
        from stickler import StructuredModel

        Model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {
                            "type": ["object", "null"],
                            "properties": {
                                "n": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["n"],
                        },
                    }
                },
                "required": ["people"],
            }
        )

        gt = Model(people=[{"n": "alice", "tags": ["x"]}, None])
        same = Model(people=[{"n": "alice", "tags": ["x"]}, None])
        assert same.compare_with(gt)["overall_score"] == pytest.approx(1.0)

    def test_nullable_object_array_compare_with_none_element_evaluator_format(self):
        """The evaluator_format=True path also tolerates None list elements."""
        from stickler import StructuredModel

        Model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {
                            "type": ["object", "null"],
                            "properties": {"n": {"type": "string"}},
                            "required": ["n"],
                        },
                    }
                },
                "required": ["people"],
            }
        )

        gt = Model(people=[{"n": "alice"}, None])
        same = Model(people=[{"n": "alice"}, None])
        # Must not raise; produces a metrics dict in evaluator format.
        result = same.compare_with(gt, evaluator_format=True)
        assert "overall" in result

    @pytest.mark.parametrize(
        "bad_type",
        [
            [],
            ["null"],
            ["null", "null"],
            ["string", "integer"],
            [["string"], "null"],
            [123, "null"],
        ],
    )
    def test_invalid_list_form_type_rejected(self, bad_type):
        """List-form ``type`` values that are not ['<type>', 'null'] are rejected."""
        schema = {
            "type": "object",
            "properties": {"value": {"type": bad_type}},
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        with pytest.raises(ValueError, match="Unsupported JSON Schema type"):
            converter.convert_properties_to_fields(
                schema["properties"], schema["required"]
            )

    def test_nullable_outer_object(self):
        """Outer-level {"type": ["object", "null"]} field wraps model in Optional."""
        from typing import Union, get_args, get_origin

        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": ["object", "null"],
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                }
            },
            "required": ["nested"],
        }

        converter = JsonSchemaFieldConverter(schema)
        fields = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        field_type, _ = fields["nested"]
        assert get_origin(field_type) is Union
        assert type(None) in get_args(field_type)

        from stickler import StructuredModel
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel as _SM,
        )

        Model = StructuredModel.from_json_schema(schema)
        assert Model(nested=None).nested is None
        non_null = [t for t in get_args(field_type) if t is not type(None)]
        assert len(non_null) == 1
        assert isinstance(non_null[0], type) and issubclass(non_null[0], _SM)

    def test_nullable_outer_array(self):
        """Outer-level {"type": ["array", "null"]} field wraps list in Optional."""
        from typing import Union, get_args, get_origin

        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                }
            },
            "required": [],
        }

        converter = JsonSchemaFieldConverter(schema)
        fields = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )

        field_type, _ = fields["tags"]
        assert get_origin(field_type) is Union
        assert type(None) in get_args(field_type)
        inner_types = [t for t in get_args(field_type) if t is not type(None)]
        assert len(inner_types) == 1
        assert get_origin(inner_types[0]) is list

        from stickler import StructuredModel

        Model = StructuredModel.from_json_schema(schema)
        assert Model(tags=None).tags is None

    def test_nullable_reversed_order(self):
        """["null", "string"] (null first) is equivalent to ["string", "null"]."""
        from typing import Union, get_args, get_origin

        schema = {
            "type": "object",
            "properties": {"value": {"type": ["null", "string"]}},
            "required": ["value"],
        }
        converter = JsonSchemaFieldConverter(schema)
        fields = converter.convert_properties_to_fields(
            schema["properties"], schema["required"]
        )
        field_type, _ = fields["value"]
        assert get_origin(field_type) is Union
        assert type(None) in get_args(field_type)
        assert str in get_args(field_type)
