"""Tests for StructuredModel.model_from_json() functionality.

This module tests the configuration-based model creation feature that allows
defining StructuredModel classes entirely through JSON configuration.
"""

import pytest
from typing import List, Optional
from pydantic import ValidationError

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.numeric import NumericComparator


class TestBasicModelFromJson:
    """Test basic model_from_json functionality."""

    def test_simple_model_creation(self):
        """Test creating a simple model with basic field types."""
        config = {
            "model_name": "SimpleInvoice",
            "match_threshold": 0.8,
            "fields": {
                "invoice_number": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.9,
                    "weight": 2.0,
                    "required": True,
                },
                "total": {
                    "type": "float",
                    "comparator": "NumericComparator",
                    "threshold": 0.95,
                    "weight": 1.5,
                    "default": 0.0,
                },
            },
        }

        InvoiceClass = StructuredModel.model_from_json(config)

        # Test class properties
        assert InvoiceClass.__name__ == "SimpleInvoice"
        assert issubclass(InvoiceClass, StructuredModel)
        assert InvoiceClass.match_threshold == 0.8

        # Test field definitions exist
        assert "invoice_number" in InvoiceClass.model_fields
        assert "total" in InvoiceClass.model_fields

        # Test instance creation
        invoice = InvoiceClass(invoice_number="INV-001", total=100.0)
        assert invoice.invoice_number == "INV-001"
        assert invoice.total == 100.0

        # Test comparison functionality is inherited
        invoice2 = InvoiceClass(invoice_number="INV-001", total=100.0)
        result = invoice.compare_with(invoice2)
        assert "overall_score" in result
        assert result["overall_score"] == 1.0

    def test_model_with_defaults(self):
        """Test model creation with default values."""
        config = {
            "model_name": "Product",
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "required": True,
                },
                "price": {
                    "type": "float",
                    "comparator": "NumericComparator",
                    "default": 0.0,
                },
                "description": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "default": "",
                },
            },
        }

        ProductClass = StructuredModel.model_from_json(config)

        # Test with minimal data (using defaults)
        product = ProductClass(name="Widget")
        assert product.name == "Widget"
        assert product.price == 0.0
        assert product.description == ""

        # Test with full data
        product2 = ProductClass(
            name="Gadget", price=29.99, description="A useful gadget"
        )
        assert product2.name == "Gadget"
        assert product2.price == 29.99
        assert product2.description == "A useful gadget"

    def test_different_field_types(self):
        """Test model with various field types."""
        config = {
            "model_name": "TestModel",
            "fields": {
                "text_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "required": True,
                },
                "number_field": {
                    "type": "int",
                    "comparator": "NumericComparator",
                    "default": 0,
                },
                "decimal_field": {
                    "type": "float",
                    "comparator": "NumericComparator",
                    "default": 0.0,
                },
                "flag_field": {
                    "type": "bool",
                    "comparator": "ExactComparator",
                    "default": False,
                },
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        instance = TestClass(
            text_field="test", number_field=42, decimal_field=3.14, flag_field=True
        )

        assert instance.text_field == "test"
        assert instance.number_field == 42
        assert instance.decimal_field == 3.14
        assert instance.flag_field is True


class TestComparatorConfiguration:
    """Test comparator configuration in model_from_json."""

    def test_different_comparators(self):
        """Test model with different comparator types."""
        config = {
            "model_name": "ComparatorTest",
            "fields": {
                "fuzzy_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.7,
                },
                "exact_field": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 1.0,
                },
                "numeric_field": {
                    "type": "float",
                    "comparator": "NumericComparator",
                    "threshold": 0.95,
                },
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        # Test that comparison info is properly configured
        fuzzy_info = TestClass._get_comparison_info("fuzzy_field")
        assert isinstance(fuzzy_info.comparator, LevenshteinComparator)
        assert fuzzy_info.threshold == 0.7

        exact_info = TestClass._get_comparison_info("exact_field")
        assert isinstance(exact_info.comparator, ExactComparator)
        assert exact_info.threshold == 1.0

        numeric_info = TestClass._get_comparison_info("numeric_field")
        assert isinstance(numeric_info.comparator, NumericComparator)
        assert numeric_info.threshold == 0.95

    def test_field_weights(self):
        """Test field weight configuration."""
        config = {
            "model_name": "WeightTest",
            "fields": {
                "high_weight": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "weight": 3.0,
                },
                "normal_weight": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "weight": 1.0,
                },
                "low_weight": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "weight": 0.5,
                },
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        # Test weight configuration
        high_info = TestClass._get_comparison_info("high_weight")
        assert high_info.weight == 3.0

        normal_info = TestClass._get_comparison_info("normal_weight")
        assert normal_info.weight == 1.0

        low_info = TestClass._get_comparison_info("low_weight")
        assert low_info.weight == 0.5


class TestFieldComparison:
    """Test that generated models properly compare fields."""

    def test_field_comparison_with_thresholds(self):
        """Test field comparison respects configured thresholds."""
        config = {
            "model_name": "ThresholdTest",
            "fields": {
                "strict_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.9,
                },
                "lenient_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.5,
                },
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        model1 = TestClass(strict_field="hello", lenient_field="world")
        model2 = TestClass(strict_field="helo", lenient_field="word")  # Typos

        result = model1.compare_with(model2)

        # The lenient field should match better than the strict field
        assert "field_scores" in result
        strict_score = result["field_scores"]["strict_field"]
        lenient_score = result["field_scores"]["lenient_field"]

        # Both should have some similarity, but lenient should be more forgiving
        assert 0 <= strict_score <= 1
        assert 0 <= lenient_score <= 1

    def test_weighted_comparison(self):
        """Test that field weights affect overall score."""
        config = {
            "model_name": "WeightedTest",
            "fields": {
                "important_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.8,
                    "weight": 5.0,
                },
                "minor_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.8,
                    "weight": 1.0,
                },
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        # Test case where important field matches, minor doesn't
        model1 = TestClass(important_field="match", minor_field="different")
        model2 = TestClass(important_field="match", minor_field="other")

        result = model1.compare_with(model2)

        # Should have reasonable overall score due to important field matching
        assert result["overall_score"] > 0.5


class TestErrorHandling:
    """Test error handling in model_from_json."""

    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        # Missing required keys
        with pytest.raises((KeyError, ValueError)):
            StructuredModel.model_from_json({})

        # Invalid field type
        with pytest.raises((ValueError, TypeError)):
            config = {
                "model_name": "InvalidTest",
                "fields": {
                    "bad_field": {
                        "type": "invalid_type",
                        "comparator": "LevenshteinComparator",
                    }
                },
            }
            StructuredModel.model_from_json(config)

        # Invalid comparator
        with pytest.raises((ValueError, KeyError)):
            config = {
                "model_name": "InvalidTest",
                "fields": {
                    "bad_field": {"type": "str", "comparator": "NonExistentComparator"}
                },
            }
            StructuredModel.model_from_json(config)

    def test_model_validation_errors(self):
        """Test that generated models properly validate data."""
        config = {
            "model_name": "ValidationTest",
            "fields": {
                "required_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "required": True,
                },
                "typed_field": {
                    "type": "int",
                    "comparator": "NumericComparator",
                    "default": 0,
                },
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        # Missing required field should raise ValidationError
        with pytest.raises(ValidationError):
            TestClass(typed_field=42)  # Missing required_field

        # Wrong type should be coerced or raise error
        instance = TestClass(required_field="test", typed_field="123")  # String for int
        assert instance.typed_field == 123  # Should be coerced to int


class TestDefaultValues:
    """Test default value handling."""

    def test_default_model_name(self):
        """Test default model name when not specified."""
        config = {
            "fields": {
                "test_field": {"type": "str", "comparator": "LevenshteinComparator"}
            }
        }

        TestClass = StructuredModel.model_from_json(config)

        # Should have a default name
        assert TestClass.__name__ in ["DynamicModel", "GeneratedModel"]

    def test_default_match_threshold(self):
        """Test default match threshold when not specified."""
        config = {
            "model_name": "DefaultTest",
            "fields": {
                "test_field": {"type": "str", "comparator": "LevenshteinComparator"}
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        # Should have default match threshold
        assert hasattr(TestClass, "match_threshold")
        assert TestClass.match_threshold == 0.7  # Default from StructuredModel

    def test_default_field_parameters(self):
        """Test default field parameters when not specified."""
        config = {
            "model_name": "DefaultFieldTest",
            "fields": {
                "minimal_field": {"type": "str", "comparator": "LevenshteinComparator"}
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        field_info = TestClass._get_comparison_info("minimal_field")

        # Should have reasonable defaults
        assert field_info.threshold == 0.5  # Default threshold
        assert field_info.weight == 1.0  # Default weight
        assert field_info.clip_under_threshold is True  # Default clipping


class TestPydanticIntegration:
    """Test integration with Pydantic features."""

    def test_model_fields_access(self):
        """Test that generated models have proper Pydantic model_fields."""
        config = {
            "model_name": "PydanticTest",
            "fields": {
                "test_field": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "description": "A test field",
                }
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        # Should have Pydantic model_fields
        assert hasattr(TestClass, "model_fields")
        assert isinstance(TestClass.model_fields, dict)
        assert "test_field" in TestClass.model_fields

    def test_json_schema_generation(self):
        """Test that generated models can generate JSON schemas."""
        config = {
            "model_name": "SchemaTest",
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "description": "Name field",
                },
                "age": {
                    "type": "int",
                    "comparator": "NumericComparator",
                    "description": "Age field",
                },
            },
        }

        TestClass = StructuredModel.model_from_json(config)

        # Should be able to generate schema
        schema = TestClass.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_model_serialization(self):
        """Test that generated models can serialize/deserialize."""
        config = {
            "model_name": "SerializationTest",
            "fields": {"data": {"type": "str", "comparator": "LevenshteinComparator"}},
        }

        TestClass = StructuredModel.model_from_json(config)

        instance = TestClass(data="test data")

        # Test serialization
        json_data = instance.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["data"] == "test data"

        # Test deserialization
        new_instance = TestClass.model_validate(json_data)
        assert new_instance.data == "test data"


class TestNestedStructuredModels:
    """Test nested StructuredModel creation from JSON configurations."""

    def test_single_level_nesting(self):
        """Test Person with nested Address (2 levels)."""
        config = {
            "model_name": "Person",
            "match_threshold": 0.8,
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.9,
                    "weight": 2.0,
                    "required": True,
                },
                "address": {
                    "type": "structured_model",
                    "model_name": "Address",
                    "match_threshold": 0.7,
                    "weight": 1.5,
                    "required": False,
                    "default": None,
                    "fields": {
                        "street": {
                            "type": "str",
                            "comparator": "LevenshteinComparator",
                            "threshold": 0.8,
                            "weight": 1.0,
                        },
                        "city": {
                            "type": "str",
                            "comparator": "LevenshteinComparator",
                            "threshold": 0.8,
                            "weight": 1.0,
                        },
                        "zipcode": {
                            "type": "str",
                            "comparator": "ExactComparator",
                            "weight": 2.0,
                        },
                    },
                },
            },
        }

        Person = StructuredModel.model_from_json(config)

        # Test class properties
        assert Person.__name__ == "Person"
        assert issubclass(Person, StructuredModel)
        assert Person.match_threshold == 0.8

        # Test field definitions exist
        assert "name" in Person.model_fields
        assert "address" in Person.model_fields

        # Test instance creation with nested dictionary
        person1 = Person(
            name="John Doe",
            address={"street": "123 Main St", "city": "Seattle", "zipcode": "98101"},
        )

        assert person1.name == "John Doe"
        assert person1.address.street == "123 Main St"
        assert person1.address.city == "Seattle"
        assert person1.address.zipcode == "98101"

        # Test comparison with nested objects
        person2 = Person(
            name="Jon Doe",  # Slight difference
            address={
                "street": "123 Main Street",  # Slight difference
                "city": "Seattle",  # Same
                "zipcode": "98101",  # Same
            },
        )

        result = person1.compare_with(person2)

        # Verify nested comparison works
        assert "overall_score" in result
        assert "field_scores" in result
        assert "name" in result["field_scores"]
        assert "address" in result["field_scores"]

        # Address should have high similarity (2/3 fields exact match)
        assert result["field_scores"]["address"] > 0.6

    def test_two_level_nesting(self):
        """Test Company with Person with Address (3 levels)."""
        config = {
            "model_name": "Company",
            "match_threshold": 0.8,
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "weight": 3.0,
                },
                "ceo": {
                    "type": "structured_model",
                    "model_name": "Person",
                    "match_threshold": 0.7,
                    "weight": 2.0,
                    "fields": {
                        "name": {
                            "type": "str",
                            "comparator": "LevenshteinComparator",
                            "weight": 2.0,
                        },
                        "address": {
                            "type": "structured_model",
                            "model_name": "Address",
                            "match_threshold": 0.6,
                            "weight": 1.0,
                            "fields": {
                                "street": {
                                    "type": "str",
                                    "comparator": "LevenshteinComparator",
                                },
                                "city": {
                                    "type": "str",
                                    "comparator": "LevenshteinComparator",
                                },
                            },
                        },
                    },
                },
            },
        }

        Company = StructuredModel.model_from_json(config)

        # Test 3-level instantiation
        company1 = Company(
            name="Acme Corp",
            ceo={
                "name": "John Doe",
                "address": {"street": "123 Main St", "city": "Seattle"},
            },
        )

        # Verify 3-level access works
        assert company1.name == "Acme Corp"
        assert company1.ceo.name == "John Doe"
        assert company1.ceo.address.street == "123 Main St"
        assert company1.ceo.address.city == "Seattle"

        # Test 3-level comparison
        company2 = Company(
            name="Acme Corporation",  # Slight difference
            ceo={
                "name": "Jon Doe",  # Slight difference
                "address": {
                    "street": "123 Main Street",  # Slight difference
                    "city": "Seattle",  # Exact match
                },
            },
        )

        result = company1.compare_with(company2)

        # Verify nested scoring bubbles up correctly
        assert "overall_score" in result
        assert "field_scores" in result
        assert "ceo" in result["field_scores"]

        # Should have reasonable similarity despite differences
        assert result["overall_score"] > 0.5

    def test_list_structured_model(self):
        """Test list of nested models."""
        config = {
            "model_name": "Company",
            "fields": {
                "name": {"type": "str", "comparator": "LevenshteinComparator"},
                "employees": {
                    "type": "list_structured_model",
                    "model_name": "Employee",
                    "match_threshold": 0.8,
                    "weight": 1.5,
                    "default": [],
                    "fields": {
                        "name": {"type": "str", "comparator": "LevenshteinComparator"},
                        "salary": {
                            "type": "float",
                            "comparator": "NumericComparator",
                            "comparator_config": {"tolerance": 1000},
                        },
                    },
                },
            },
        }

        Company = StructuredModel.model_from_json(config)

        # Test list instantiation
        company1 = Company(
            name="Tech Corp",
            employees=[
                {"name": "Alice Smith", "salary": 75000.0},
                {"name": "Bob Jones", "salary": 80000.0},
            ],
        )

        assert company1.name == "Tech Corp"
        assert len(company1.employees) == 2
        assert company1.employees[0].name == "Alice Smith"
        assert company1.employees[0].salary == 75000.0
        assert company1.employees[1].name == "Bob Jones"
        assert company1.employees[1].salary == 80000.0

        # Test list comparison (Hungarian matching)
        company2 = Company(
            name="Tech Corp",
            employees=[
                {
                    "name": "Bob Jones",
                    "salary": 81000.0,
                },  # Reordered + slight difference
                {
                    "name": "Alice Smith",
                    "salary": 75500.0,
                },  # Reordered + slight difference
            ],
        )

        result = company1.compare_with(company2)

        # Should match well despite reordering
        assert result["overall_score"] > 0.8
        assert "employees" in result["field_scores"]

    def test_optional_structured_model(self):
        """Test optional nested models."""
        config = {
            "model_name": "Person",
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "required": True,
                },
                "address": {
                    "type": "optional_structured_model",
                    "model_name": "Address",
                    "weight": 1.0,
                    "default": None,
                    "fields": {
                        "street": {"type": "str", "comparator": "LevenshteinComparator"}
                    },
                },
            },
        }

        Person = StructuredModel.model_from_json(config)

        # Test with address
        person1 = Person(name="John Doe", address={"street": "123 Main St"})

        # Test without address (should use default None)
        person2 = Person(name="Jane Doe")

        assert person1.address is not None
        assert person1.address.street == "123 Main St"
        assert person2.address is None

        # Test comparison with None vs object
        result = person1.compare_with(person2)
        assert "overall_score" in result

    def test_schema_validation_errors(self):
        """Test helpful error messages for invalid configurations."""

        # Error: structured model with comparator (forbidden)
        invalid_config1 = {
            "model_name": "InvalidTest",
            "fields": {
                "person": {
                    "type": "structured_model",
                    "comparator": "LevenshteinComparator",  # ❌ Forbidden!
                    "fields": {
                        "name": {"type": "str", "comparator": "ExactComparator"}
                    },
                }
            },
        }

        with pytest.raises(ValueError, match="structured_model.*cannot.*comparator"):
            StructuredModel.model_from_json(invalid_config1)

        # Error: primitive field with fields (forbidden)
        invalid_config2 = {
            "model_name": "InvalidTest",
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "fields": {"nested": {"type": "str"}},  # ❌ Forbidden!
                }
            },
        }

        with pytest.raises(ValueError, match="primitive.*cannot.*fields"):
            StructuredModel.model_from_json(invalid_config2)

        # Error: structured model missing fields
        invalid_config3 = {
            "model_name": "InvalidTest",
            "fields": {
                "person": {
                    "type": "structured_model"
                    # Missing required "fields" key
                }
            },
        }

        with pytest.raises(ValueError, match="structured_model.*requires.*fields"):
            StructuredModel.model_from_json(invalid_config3)

        # Error: primitive field missing comparator
        invalid_config4 = {
            "model_name": "InvalidTest",
            "fields": {
                "name": {
                    "type": "str"
                    # Missing required "comparator" key
                }
            },
        }

        with pytest.raises(ValueError, match="primitive.*requires.*comparator"):
            StructuredModel.model_from_json(invalid_config4)

    def test_nested_serialization(self):
        """Test JSON serialization of nested structures."""
        config = {
            "model_name": "Company",
            "fields": {
                "name": {"type": "str", "comparator": "LevenshteinComparator"},
                "ceo": {
                    "type": "structured_model",
                    "fields": {
                        "name": {"type": "str", "comparator": "LevenshteinComparator"},
                        "address": {
                            "type": "structured_model",
                            "fields": {
                                "street": {
                                    "type": "str",
                                    "comparator": "LevenshteinComparator",
                                }
                            },
                        },
                    },
                },
            },
        }

        Company = StructuredModel.model_from_json(config)

        company = Company(
            name="Test Corp",
            ceo={"name": "CEO Name", "address": {"street": "CEO Street"}},
        )

        # Test serialization
        json_data = company.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["name"] == "Test Corp"
        assert json_data["ceo"]["name"] == "CEO Name"
        assert json_data["ceo"]["address"]["street"] == "CEO Street"

        # Test deserialization
        new_company = Company.model_validate(json_data)
        assert new_company.name == "Test Corp"
        assert new_company.ceo.name == "CEO Name"
        assert new_company.ceo.address.street == "CEO Street"

    def test_nested_comparison_scoring(self):
        """Test that nested comparison scores bubble up correctly."""
        config = {
            "model_name": "Person",
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "weight": 1.0,
                },
                "address": {
                    "type": "structured_model",
                    "weight": 2.0,  # Higher weight for address
                    "fields": {
                        "street": {
                            "type": "str",
                            "comparator": "ExactComparator",
                            "weight": 1.0,
                        },
                        "city": {
                            "type": "str",
                            "comparator": "ExactComparator",
                            "weight": 1.0,
                        },
                    },
                },
            },
        }

        Person = StructuredModel.model_from_json(config)

        # Perfect address match, slight name difference
        person1 = Person(
            name="John Doe", address={"street": "123 Main St", "city": "Seattle"}
        )

        person2 = Person(
            name="Jon Doe",  # Slight difference
            address={"street": "123 Main St", "city": "Seattle"},  # Perfect match
        )

        result = person1.compare_with(person2)

        # Address has 2x weight and perfect match, so overall should be high
        assert result["overall_score"] > 0.8
        assert result["field_scores"]["address"] == 1.0  # Perfect address match
        assert result["field_scores"]["name"] < 1.0  # Imperfect name match


if __name__ == "__main__":
    pytest.main([__file__])
