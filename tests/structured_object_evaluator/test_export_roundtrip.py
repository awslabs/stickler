"""Round-trip tests for StructuredModel export and import.

This module tests that models can be exported and re-imported while
preserving their comparison behavior.
"""

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from typing import List


class Product(StructuredModel):
    """Test model for round-trip testing."""
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
    in_stock: bool = ComparableField(threshold=1.0, default=...)


def test_json_schema_roundtrip():
    """Test export to JSON Schema and re-import produces equivalent model."""
    # Export
    schema = Product.to_json_schema()
    
    # Re-import
    ReconstructedProduct = StructuredModel.from_json_schema(schema)
    
    # Create instances
    p1 = Product(name="Laptop", price=999.99, in_stock=True)
    p2 = ReconstructedProduct(name="Laptop", price=999.99, in_stock=True)
    
    # Compare with self - should be perfect match
    result1 = p1.compare_with(p1)
    result2 = p2.compare_with(p2)
    
    assert result1["overall_score"] == 1.0
    assert result2["overall_score"] == 1.0
    assert result1["overall_score"] == result2["overall_score"]


def test_stickler_config_roundtrip():
    """Test export to Stickler config and re-import produces equivalent model."""
    # Export
    config = Product.to_stickler_config()
    
    # Re-import
    ReconstructedProduct = StructuredModel.model_from_json(config)
    
    # Create instances
    p1 = Product(name="Laptop", price=999.99, in_stock=True)
    p2 = ReconstructedProduct(name="Laptop", price=999.99, in_stock=True)
    
    # Compare with self - should be perfect match
    result1 = p1.compare_with(p1)
    result2 = p2.compare_with(p2)
    
    assert result1["overall_score"] == 1.0
    assert result2["overall_score"] == 1.0
    assert result1["overall_score"] == result2["overall_score"]


def test_json_schema_roundtrip_comparison_behavior():
    """Test that comparison behavior is preserved after round-trip."""
    # Export and re-import
    schema = Product.to_json_schema()
    ReconstructedProduct = StructuredModel.from_json_schema(schema)
    
    # Create test instances with slight differences
    p1_orig = Product(name="Laptop", price=999.99, in_stock=True)
    p2_orig = Product(name="Laptop Pro", price=999.99, in_stock=True)
    
    p1_recon = ReconstructedProduct(name="Laptop", price=999.99, in_stock=True)
    p2_recon = ReconstructedProduct(name="Laptop Pro", price=999.99, in_stock=True)
    
    # Compare - should get same scores
    result_orig = p1_orig.compare_with(p2_orig)
    result_recon = p1_recon.compare_with(p2_recon)
    
    # Overall scores should match
    assert abs(result_orig["overall_score"] - result_recon["overall_score"]) < 0.01
    
    # Field scores should match
    assert abs(result_orig["field_scores"]["name"] - result_recon["field_scores"]["name"]) < 0.01
    assert result_orig["field_scores"]["price"] == result_recon["field_scores"]["price"]
    assert result_orig["field_scores"]["in_stock"] == result_recon["field_scores"]["in_stock"]


def test_stickler_config_roundtrip_comparison_behavior():
    """Test that comparison behavior is preserved after Stickler config round-trip."""
    # Export and re-import
    config = Product.to_stickler_config()
    ReconstructedProduct = StructuredModel.model_from_json(config)
    
    # Create test instances with slight differences
    p1_orig = Product(name="Laptop", price=999.99, in_stock=True)
    p2_orig = Product(name="Laptop Pro", price=999.99, in_stock=True)
    
    p1_recon = ReconstructedProduct(name="Laptop", price=999.99, in_stock=True)
    p2_recon = ReconstructedProduct(name="Laptop Pro", price=999.99, in_stock=True)
    
    # Compare - should get same scores
    result_orig = p1_orig.compare_with(p2_orig)
    result_recon = p1_recon.compare_with(p2_recon)
    
    # Overall scores should match
    assert abs(result_orig["overall_score"] - result_recon["overall_score"]) < 0.01
    
    # Field scores should match
    assert abs(result_orig["field_scores"]["name"] - result_recon["field_scores"]["name"]) < 0.01
    assert result_orig["field_scores"]["price"] == result_recon["field_scores"]["price"]
    assert result_orig["field_scores"]["in_stock"] == result_recon["field_scores"]["in_stock"]


def test_nested_model_roundtrip():
    """Test round-trip with nested StructuredModels."""
    
    class Order(StructuredModel):
        order_id: str = ComparableField(threshold=1.0, default=...)
        product: Product = ComparableField(default=...)
    
    # Export and re-import via JSON Schema
    schema = Order.to_json_schema()
    ReconstructedOrder = StructuredModel.from_json_schema(schema)
    
    # Create instances
    o1 = Order(
        order_id="ORD-001",
        product=Product(name="Laptop", price=999.99, in_stock=True)
    )
    o2 = ReconstructedOrder(
        order_id="ORD-001",
        product=ReconstructedOrder.model_fields["product"].annotation(
            name="Laptop", price=999.99, in_stock=True
        )
    )
    
    # Compare
    result1 = o1.compare_with(o1)
    result2 = o2.compare_with(o2)
    
    assert result1["overall_score"] == 1.0
    assert result2["overall_score"] == 1.0


def test_list_model_roundtrip():
    """Test round-trip with List[StructuredModel]."""
    
    class Cart(StructuredModel):
        cart_id: str = ComparableField(threshold=1.0, default=...)
        products: List[Product] = ComparableField(default=...)
    
    # Export and re-import via Stickler config
    config = Cart.to_stickler_config()
    ReconstructedCart = StructuredModel.model_from_json(config)
    
    # Create instances
    c1 = Cart(
        cart_id="CART-001",
        products=[
            Product(name="Laptop", price=999.99, in_stock=True),
            Product(name="Mouse", price=29.99, in_stock=True)
        ]
    )
    
    # Get the reconstructed Product class
    ReconstructedProduct = ReconstructedCart.model_fields["products"].annotation.__args__[0]
    
    c2 = ReconstructedCart(
        cart_id="CART-001",
        products=[
            ReconstructedProduct(name="Laptop", price=999.99, in_stock=True),
            ReconstructedProduct(name="Mouse", price=29.99, in_stock=True)
        ]
    )
    
    # Compare
    result1 = c1.compare_with(c1)
    result2 = c2.compare_with(c2)
    
    assert result1["overall_score"] == 1.0
    assert result2["overall_score"] == 1.0


def test_multiple_roundtrips():
    """Test that multiple export/import cycles don't degrade the model."""
    original_schema = Product.to_json_schema()
    
    # First round-trip
    Model1 = StructuredModel.from_json_schema(original_schema)
    schema1 = Model1.to_json_schema()
    
    # Second round-trip
    Model2 = StructuredModel.from_json_schema(schema1)
    schema2 = Model2.to_json_schema()
    
    # Schemas should be identical
    assert schema1 == schema2
    
    # Comparison behavior should be identical
    p_orig = Product(name="Test", price=100.0, in_stock=True)
    p1 = Model1(name="Test", price=100.0, in_stock=True)
    p2 = Model2(name="Test", price=100.0, in_stock=True)
    
    result_orig = p_orig.compare_with(p_orig)
    result1 = p1.compare_with(p1)
    result2 = p2.compare_with(p2)
    
    assert result_orig["overall_score"] == result1["overall_score"] == result2["overall_score"] == 1.0
