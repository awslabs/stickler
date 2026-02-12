"""Tests for StructuredModel export methods.

This module tests the to_json_schema() and to_stickler_config() methods
that export StructuredModel configurations for serialization.
"""

from typing import List, Optional

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class SimpleProduct(StructuredModel):
    """Simple model for testing basic export."""
    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.8,
        weight=2.0,
        default=...
    )
    price: float = ComparableField(
        comparator=NumericComparator(),
        threshold=0.95,
        weight=1.5,
        default=...
    )


class NestedModel(StructuredModel):
    """Model with nested StructuredModel for testing recursive export."""
    title: str = ComparableField(threshold=0.8)
    product: SimpleProduct = ComparableField()


class ListModel(StructuredModel):
    """Model with List[StructuredModel] for testing list export."""
    name: str = ComparableField(threshold=0.8)
    products: List[SimpleProduct] = ComparableField()


def test_to_json_schema_basic():
    """Test exporting simple model to JSON Schema format."""
    schema = SimpleProduct.to_json_schema()
    
    # Check basic structure
    assert schema["type"] == "object"
    assert schema["x-aws-stickler-model-name"] == "SimpleProduct"
    assert "properties" in schema
    assert "required" in schema
    
    # Check name field
    assert "name" in schema["properties"]
    name_prop = schema["properties"]["name"]
    assert name_prop["type"] == "string"
    assert name_prop["x-aws-stickler-comparator"] == "LevenshteinComparator"
    assert name_prop["x-aws-stickler-threshold"] == 0.8
    assert name_prop["x-aws-stickler-weight"] == 2.0
    
    # Check price field
    assert "price" in schema["properties"]
    price_prop = schema["properties"]["price"]
    assert price_prop["type"] == "number"
    assert price_prop["x-aws-stickler-comparator"] == "NumericComparator"
    assert price_prop["x-aws-stickler-threshold"] == 0.95
    assert price_prop["x-aws-stickler-weight"] == 1.5
    
    # Check required fields
    assert "name" in schema["required"]
    assert "price" in schema["required"]


def test_to_stickler_config_basic():
    """Test exporting simple model to Stickler config format."""
    config = SimpleProduct.to_stickler_config()
    
    # Check basic structure
    assert config["model_name"] == "SimpleProduct"
    assert "fields" in config
    
    # Check name field
    assert "name" in config["fields"]
    name_field = config["fields"]["name"]
    assert name_field["type"] == "str"
    assert name_field["comparator"] == "LevenshteinComparator"
    assert name_field["threshold"] == 0.8
    assert name_field["weight"] == 2.0
    assert name_field["required"] is True
    
    # Check price field
    assert "price" in config["fields"]
    price_field = config["fields"]["price"]
    assert price_field["type"] == "float"
    assert price_field["comparator"] == "NumericComparator"
    assert price_field["threshold"] == 0.95
    assert price_field["weight"] == 1.5
    assert price_field["required"] is True


def test_to_json_schema_nested():
    """Test exporting model with nested StructuredModel."""
    schema = NestedModel.to_json_schema()
    
    # Check basic structure
    assert schema["x-aws-stickler-model-name"] == "NestedModel"
    
    # Check nested product field
    assert "product" in schema["properties"]
    product_prop = schema["properties"]["product"]
    
    # Nested model should be recursively exported
    assert product_prop["type"] == "object"
    assert product_prop["x-aws-stickler-model-name"] == "SimpleProduct"
    assert "name" in product_prop["properties"]
    assert "price" in product_prop["properties"]


def test_to_stickler_config_nested():
    """Test exporting model with nested StructuredModel to Stickler config."""
    config = NestedModel.to_stickler_config()
    
    # Check basic structure
    assert config["model_name"] == "NestedModel"
    
    # Check nested product field
    assert "product" in config["fields"]
    product_field = config["fields"]["product"]
    
    # Nested model should use "structured_model" type
    assert product_field["type"] == "structured_model"
    assert "fields" in product_field
    assert "name" in product_field["fields"]
    assert "price" in product_field["fields"]


def test_to_json_schema_list():
    """Test exporting model with List[StructuredModel]."""
    schema = ListModel.to_json_schema()
    
    # Check list field
    assert "products" in schema["properties"]
    products_prop = schema["properties"]["products"]
    
    # Should be array type with nested items schema
    assert products_prop["type"] == "array"
    assert "items" in products_prop
    
    # Items should be the nested model schema
    items_schema = products_prop["items"]
    assert items_schema["type"] == "object"
    assert items_schema["x-aws-stickler-model-name"] == "SimpleProduct"


def test_to_stickler_config_list():
    """Test exporting model with List[StructuredModel] to Stickler config."""
    config = ListModel.to_stickler_config()
    
    # Check list field
    assert "products" in config["fields"]
    products_field = config["fields"]["products"]
    
    # Should use "list_structured_model" type
    assert products_field["type"] == "list_structured_model"
    assert "fields" in products_field
    assert "name" in products_field["fields"]
    assert "price" in products_field["fields"]


def test_export_preserves_metadata():
    """Test that all comparison metadata is preserved in export."""
    
    class DetailedModel(StructuredModel):
        field1: str = ComparableField(
            threshold=0.75,
            weight=1.5,
            clip_under_threshold=False,
            aggregate=True
        )
    
    # Test JSON Schema export
    schema = DetailedModel.to_json_schema()
    field_prop = schema["properties"]["field1"]
    assert field_prop["x-aws-stickler-threshold"] == 0.75
    assert field_prop["x-aws-stickler-weight"] == 1.5
    assert field_prop["x-aws-stickler-clip-under-threshold"] is False
    assert field_prop["x-aws-stickler-aggregate"] is True
    
    # Test Stickler config export
    config = DetailedModel.to_stickler_config()
    field_config = config["fields"]["field1"]
    assert field_config["threshold"] == 0.75
    assert field_config["weight"] == 1.5
    assert field_config["clip_under_threshold"] is False
    assert field_config["aggregate"] is True


class OptionalFieldModel(StructuredModel):
    """Model with Optional fields for testing export."""
    required_name: str = ComparableField(threshold=0.8, default=...)
    optional_note: Optional[str] = ComparableField(threshold=0.6, default=None)
    optional_product: Optional[SimpleProduct] = ComparableField(default=None)
    optional_products: Optional[List[SimpleProduct]] = ComparableField(default=None)


def test_to_json_schema_optional_fields():
    """Test that Optional fields export correctly and are not in required list."""
    schema = OptionalFieldModel.to_json_schema()

    # Required field is in required list
    assert "required_name" in schema["required"]

    # Optional fields are NOT in required list
    assert "optional_note" not in schema["required"]
    assert "optional_product" not in schema["required"]
    assert "optional_products" not in schema["required"]

    # Optional[str] unwraps to string type
    assert schema["properties"]["optional_note"]["type"] == "string"

    # Optional[StructuredModel] unwraps to nested object schema
    product_prop = schema["properties"]["optional_product"]
    assert product_prop["type"] == "object"
    assert "name" in product_prop["properties"]
    assert "price" in product_prop["properties"]

    # Optional[List[StructuredModel]] unwraps to array with nested items
    products_prop = schema["properties"]["optional_products"]
    assert products_prop["type"] == "array"
    assert products_prop["items"]["type"] == "object"
    assert "name" in products_prop["items"]["properties"]


def test_to_stickler_config_optional_fields():
    """Test that Optional fields export correctly in Stickler config format."""
    config = OptionalFieldModel.to_stickler_config()

    # Optional[str] unwraps to str type
    assert config["fields"]["optional_note"]["type"] == "str"

    # Optional[StructuredModel] unwraps to structured_model type
    product_field = config["fields"]["optional_product"]
    assert product_field["type"] == "structured_model"
    assert "name" in product_field["fields"]
    assert "price" in product_field["fields"]

    # Optional[List[StructuredModel]] unwraps to list_structured_model type
    products_field = config["fields"]["optional_products"]
    assert products_field["type"] == "list_structured_model"
    assert "name" in products_field["fields"]
