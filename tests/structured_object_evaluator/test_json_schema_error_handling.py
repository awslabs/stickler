"""Comprehensive error handling tests for JSON Schema conversion.

This module tests all error cases for the JSON Schema to StructuredModel conversion,
including validation of x-aws-stickler-* extensions and field path tracking.
"""

import pytest
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.json_schema_field_converter import (
    JsonSchemaFieldConverter,
)


class TestThresholdValidation:
    """Test validation of x-aws-stickler-threshold extension."""

    def test_threshold_below_zero_raises_error(self):
        """Test that threshold below 0.0 raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": -0.1
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-threshold must be a number between 0.0 and 1.0"):
            StructuredModel.from_json_schema(schema)

    def test_threshold_above_one_raises_error(self):
        """Test that threshold above 1.0 raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 1.5
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-threshold must be a number between 0.0 and 1.0"):
            StructuredModel.from_json_schema(schema)

    def test_threshold_non_numeric_raises_error(self):
        """Test that non-numeric threshold raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": "0.5"
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-threshold must be a number between 0.0 and 1.0"):
            StructuredModel.from_json_schema(schema)

    def test_threshold_exactly_zero_is_valid(self):
        """Test that threshold of exactly 0.0 is valid."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 0.0
                }
            }
        }
        
        # Should not raise
        Model = StructuredModel.from_json_schema(schema)
        assert Model is not None

    def test_threshold_exactly_one_is_valid(self):
        """Test that threshold of exactly 1.0 is valid."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 1.0
                }
            }
        }
        
        # Should not raise
        Model = StructuredModel.from_json_schema(schema)
        assert Model is not None


class TestWeightValidation:
    """Test validation of x-aws-stickler-weight extension."""

    def test_weight_zero_raises_error(self):
        """Test that weight of 0 raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-weight": 0
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-weight must be a positive number"):
            StructuredModel.from_json_schema(schema)

    def test_weight_negative_raises_error(self):
        """Test that negative weight raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-weight": -1.5
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-weight must be a positive number"):
            StructuredModel.from_json_schema(schema)

    def test_weight_non_numeric_raises_error(self):
        """Test that non-numeric weight raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-weight": "2.0"
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-weight must be a positive number"):
            StructuredModel.from_json_schema(schema)

    def test_weight_positive_is_valid(self):
        """Test that positive weight is valid."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-weight": 2.5
                }
            }
        }
        
        # Should not raise
        Model = StructuredModel.from_json_schema(schema)
        assert Model is not None

    def test_weight_very_small_positive_is_valid(self):
        """Test that very small positive weight is valid."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-weight": 0.001
                }
            }
        }
        
        # Should not raise
        Model = StructuredModel.from_json_schema(schema)
        assert Model is not None


class TestComparatorValidation:
    """Test validation of x-aws-stickler-comparator extension."""

    def test_invalid_comparator_name_raises_error(self):
        """Test that invalid comparator name raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "NonExistentComparator"
                }
            }
        }
        
        with pytest.raises(ValueError, match="Unknown comparator.*NonExistentComparator"):
            StructuredModel.from_json_schema(schema)

    def test_invalid_comparator_shows_valid_options(self):
        """Test that error message includes valid comparator names."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "InvalidComparator"
                }
            }
        }
        
        with pytest.raises(ValueError, match="Available:"):
            StructuredModel.from_json_schema(schema)

    def test_valid_comparator_names_work(self):
        """Test that valid comparator names work correctly."""
        valid_comparators = [
            "LevenshteinComparator",
            "ExactComparator",
            "NumericComparator",
        ]
        
        for comparator_name in valid_comparators:
            schema = {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "x-aws-stickler-comparator": comparator_name
                    }
                }
            }
            
            # Should not raise
            Model = StructuredModel.from_json_schema(schema)
            assert Model is not None


class TestBooleanExtensionValidation:
    """Test validation of boolean x-aws-stickler-* extensions."""

    def test_clip_under_threshold_non_boolean_raises_error(self):
        """Test that non-boolean clip-under-threshold raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-clip-under-threshold": "true"
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-clip-under-threshold must be a boolean"):
            StructuredModel.from_json_schema(schema)

    def test_clip_under_threshold_integer_raises_error(self):
        """Test that integer clip-under-threshold raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-clip-under-threshold": 1
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-clip-under-threshold must be a boolean"):
            StructuredModel.from_json_schema(schema)

    def test_aggregate_non_boolean_raises_error(self):
        """Test that non-boolean aggregate raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-aggregate": "false"
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-aggregate must be a boolean"):
            StructuredModel.from_json_schema(schema)

    def test_aggregate_integer_raises_error(self):
        """Test that integer aggregate raises ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-aggregate": 0
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-aggregate must be a boolean"):
            StructuredModel.from_json_schema(schema)

    def test_boolean_true_is_valid(self):
        """Test that boolean True is valid."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-clip-under-threshold": True,
                    "x-aws-stickler-aggregate": True
                }
            }
        }
        
        # Should not raise
        Model = StructuredModel.from_json_schema(schema)
        assert Model is not None

    def test_boolean_false_is_valid(self):
        """Test that boolean False is valid."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-clip-under-threshold": False,
                    "x-aws-stickler-aggregate": False
                }
            }
        }
        
        # Should not raise
        Model = StructuredModel.from_json_schema(schema)
        assert Model is not None


class TestFieldPathInErrors:
    """Test that field paths are included in error messages."""

    def test_top_level_field_path_in_error(self):
        """Test that top-level field name appears in error message."""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "x-aws-stickler-threshold": 2.0
                }
            }
        }
        
        with pytest.raises(ValueError, match="field 'email'"):
            StructuredModel.from_json_schema(schema)

    def test_nested_field_path_in_error(self):
        """Test that nested field path appears in error message."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "x-aws-stickler-threshold": 2.0
                        }
                    }
                }
            }
        }
        
        with pytest.raises(ValueError, match="user\\.email"):
            StructuredModel.from_json_schema(schema)

    def test_deeply_nested_field_path_in_error(self):
        """Test that deeply nested field path appears in error message."""
        schema = {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "object",
                            "properties": {
                                "zipcode": {
                                    "type": "string",
                                    "x-aws-stickler-weight": -1
                                }
                            }
                        }
                    }
                }
            }
        }
        
        with pytest.raises(ValueError, match="company\\.address\\.zipcode"):
            StructuredModel.from_json_schema(schema)

    def test_array_field_path_in_error(self):
        """Test that array field path appears in error message."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "x-aws-stickler-comparator": "InvalidComparator"
                            }
                        }
                    }
                }
            }
        }
        
        with pytest.raises(ValueError, match="items\\[\\]"):
            StructuredModel.from_json_schema(schema)


class TestMultipleErrorScenarios:
    """Test error handling in complex scenarios."""

    def test_multiple_invalid_extensions_first_error_raised(self):
        """Test that first error is raised when multiple extensions are invalid."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 2.0,  # Invalid
                    "x-aws-stickler-weight": -1  # Also invalid
                }
            }
        }
        
        # Should raise error for threshold (processed first)
        with pytest.raises(ValueError, match="x-aws-stickler-threshold"):
            StructuredModel.from_json_schema(schema)

    def test_error_in_nested_object_with_valid_parent(self):
        """Test error in nested object when parent is valid."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "x-aws-stickler-weight": 2.0,  # Valid
                    "properties": {
                        "email": {
                            "type": "string",
                            "x-aws-stickler-threshold": 1.5  # Invalid
                        }
                    }
                }
            }
        }
        
        with pytest.raises(ValueError, match="x-aws-stickler-threshold"):
            StructuredModel.from_json_schema(schema)

    def test_error_in_array_items_with_valid_array(self):
        """Test error in array items when array itself is valid."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "x-aws-stickler-weight": 1.5,  # Valid
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "x-aws-stickler-comparator": "BadComparator"  # Invalid
                            }
                        }
                    }
                }
            }
        }
        
        with pytest.raises(ValueError, match="Unknown comparator.*BadComparator"):
            StructuredModel.from_json_schema(schema)


class TestEdgeCases:
    """Test edge cases in error handling."""

    def test_threshold_boundary_values(self):
        """Test threshold at exact boundary values."""
        # Test 0.0 - should work
        schema1 = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-threshold": 0.0}
            }
        }
        Model1 = StructuredModel.from_json_schema(schema1)
        assert Model1 is not None
        
        # Test 1.0 - should work
        schema2 = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-threshold": 1.0}
            }
        }
        Model2 = StructuredModel.from_json_schema(schema2)
        assert Model2 is not None
        
        # Test just below 0.0 - should fail
        schema3 = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-threshold": -0.0001}
            }
        }
        with pytest.raises(ValueError):
            StructuredModel.from_json_schema(schema3)
        
        # Test just above 1.0 - should fail
        schema4 = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-threshold": 1.0001}
            }
        }
        with pytest.raises(ValueError):
            StructuredModel.from_json_schema(schema4)

    def test_weight_boundary_values(self):
        """Test weight at boundary values."""
        # Test very small positive - should work
        schema1 = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-weight": 0.0001}
            }
        }
        Model1 = StructuredModel.from_json_schema(schema1)
        assert Model1 is not None
        
        # Test exactly 0 - should fail
        schema2 = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-weight": 0}
            }
        }
        with pytest.raises(ValueError):
            StructuredModel.from_json_schema(schema2)
        
        # Test negative - should fail
        schema3 = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-weight": -0.0001}
            }
        }
        with pytest.raises(ValueError):
            StructuredModel.from_json_schema(schema3)

    def test_empty_string_comparator_raises_error(self):
        """Test that empty string comparator raises error."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": ""
                }
            }
        }
        
        with pytest.raises(ValueError):
            StructuredModel.from_json_schema(schema)


class TestErrorMessageQuality:
    """Test that error messages are clear and helpful."""

    def test_threshold_error_includes_value(self):
        """Test that threshold error includes the invalid value."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 1.5
                }
            }
        }
        
        with pytest.raises(ValueError, match="1.5"):
            StructuredModel.from_json_schema(schema)

    def test_weight_error_includes_value(self):
        """Test that weight error includes the invalid value."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-weight": -2.0
                }
            }
        }
        
        with pytest.raises(ValueError, match="-2.0"):
            StructuredModel.from_json_schema(schema)

    def test_comparator_error_includes_name(self):
        """Test that comparator error includes the invalid name."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "MyCustomComparator"
                }
            }
        }
        
        with pytest.raises(ValueError, match="MyCustomComparator"):
            StructuredModel.from_json_schema(schema)

    def test_boolean_error_includes_type(self):
        """Test that boolean error includes the actual type received."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-aggregate": "yes"
                }
            }
        }
        
        with pytest.raises(ValueError, match="str"):
            StructuredModel.from_json_schema(schema)
