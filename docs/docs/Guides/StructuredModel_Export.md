# StructuredModel Export Guide

## Overview

Stickler provides three methods for working with model schemas:

1. **`to_json_schema()`** - Export with `x-aws-stickler-*` extensions (round-trip compatible)
2. **`to_stickler_config()`** - Export to custom Stickler JSON format (round-trip compatible)
3. **`model_json_schema()`** - Export with `x-comparison` metadata (Pydantic standard)

### Common Use Case: Export, Customize, Re-import

A typical workflow is to create a model with default comparators, export the configuration, customize the thresholds and weights, then re-import:

```python
from stickler import StructuredModel, ComparableField

# Step 1: Create model with defaults
class Product(StructuredModel):
    name: str = ComparableField(default=...)
    price: float = ComparableField(default=...)
    description: str = ComparableField(default=...)

# Step 2: Export to get default configuration
config = Product.to_stickler_config()

# Step 3: Customize thresholds and weights
config["fields"]["name"]["threshold"] = 0.9  # Stricter matching
config["fields"]["name"]["weight"] = 2.0     # More important
config["fields"]["description"]["threshold"] = 0.6  # More lenient

# Step 4: Re-import customized configuration
CustomProduct = StructuredModel.model_from_json(config)

# Now use the customized model
product1 = CustomProduct(name="Laptop", price=999.99, description="High-end laptop")
product2 = CustomProduct(name="Laptop Pro", price=999.99, description="Premium laptop")
result = product1.compare_with(product2)
```

This approach is useful when you want to:
- Start with sensible defaults and fine-tune later
- Share configurations across teams
- Version control your comparison logic
- A/B test different threshold configurations

## Export Methods

### to_json_schema()

Export your StructuredModel as a standard JSON Schema with Stickler-specific extensions. This format works with `from_json_schema()` for round-trip serialization.

**Why use this?** When you need a standards-compliant format that works with existing JSON Schema tools and validators.

```python
from stickler import StructuredModel, ComparableField

class Product(StructuredModel):
    name: str = ComparableField(threshold=0.8, weight=2.0, default=...)
    price: float = ComparableField(threshold=0.95, default=...)

# Export to JSON Schema
schema = Product.to_json_schema()

# Save to file
import json
with open('product_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

# Later, re-import from the saved file
with open('product_schema.json', 'r') as f:
    schema = json.load(f)
ReconstructedProduct = StructuredModel.from_json_schema(schema)
```

**Output Format:**
```json
{
  "type": "object",
  "x-aws-stickler-model-name": "Product",
  "properties": {
    "name": {
      "type": "string",
      "x-aws-stickler-threshold": 0.8,
      "x-aws-stickler-weight": 2.0
    },
    "price": {
      "type": "number",
      "x-aws-stickler-threshold": 0.95
    }
  },
  "required": ["name", "price"]
}
```

### to_stickler_config()

Export your StructuredModel as a custom Stickler configuration. This format is more concise and easier to edit manually. It works with `model_from_json()` for round-trip serialization.

**Why use this?** When you want a simpler format that's easier to read and edit by hand.

```python
# Export to Stickler config
config = Product.to_stickler_config()

# Save to file
with open('product_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Later, re-import from the saved file
with open('product_config.json', 'r') as f:
    config = json.load(f)
ReconstructedProduct = StructuredModel.model_from_json(config)
```

**Output Format:**
```json
{
  "model_name": "Product",
  "fields": {
    "name": {
      "type": "str",
      "threshold": 0.8,
      "weight": 2.0,
      "required": true
    },
    "price": {
      "type": "float",
      "threshold": 0.95,
      "required": true
    }
  }
}
```

## When to Use Each Format

Here's a quick guide to help you choose the right export method:

### x-aws-stickler-* Extensions (to_json_schema)
- **Use when**: Working with standard JSON Schema tooling
- **Benefits**: Industry standard, works with OpenAPI and AsyncAPI
- **Round-trip**: ✅ Yes (via `from_json_schema()`)
- **Best for**: Sharing schemas with external systems, API documentation

### Custom Stickler Config (to_stickler_config)
- **Use when**: Working exclusively within Stickler
- **Benefits**: More concise, easier to read and edit manually
- **Round-trip**: ✅ Yes (via `model_from_json()`)
- **Best for**: Configuration files, manual editing, version control

### x-comparison Metadata (model_json_schema)
- **Use when**: Need Pydantic-standard JSON Schema
- **Benefits**: Compatible with Pydantic tooling, includes all Pydantic metadata
- **Round-trip**: ❌ No (different format, not compatible with import methods)
- **Best for**: Runtime introspection, Pydantic-specific use cases

## Nested Models

Both export methods handle nested StructuredModels recursively:

```python
class Address(StructuredModel):
    street: str = ComparableField(threshold=0.8, default=...)
    city: str = ComparableField(threshold=0.9, default=...)

class Customer(StructuredModel):
    name: str = ComparableField(threshold=0.8, default=...)
    address: Address = ComparableField(default=...)

# Export includes nested model configuration
schema = Customer.to_json_schema()
# schema["properties"]["address"] contains the full Address schema

config = Customer.to_stickler_config()
# config["fields"]["address"]["fields"] contains Address fields
```

## Lists of StructuredModels

Lists are exported with their element schemas:

```python
from typing import List

class Order(StructuredModel):
    order_id: str = ComparableField(threshold=1.0, default=...)
    products: List[Product] = ComparableField(default=...)

# JSON Schema export
schema = Order.to_json_schema()
# schema["properties"]["products"]["type"] == "array"
# schema["properties"]["products"]["items"] contains Product schema

# Stickler config export
config = Order.to_stickler_config()
# config["fields"]["products"]["type"] == "list_structured_model"
# config["fields"]["products"]["fields"] contains Product fields
```

## Round-trip Examples

### JSON Schema Round-trip

```python
# Original model
class Product(StructuredModel):
    name: str = ComparableField(threshold=0.8, weight=2.0, default=...)
    price: float = ComparableField(threshold=0.95, default=...)

# Export
schema = Product.to_json_schema()

# Re-import
ReconstructedProduct = StructuredModel.from_json_schema(schema)

# Verify behavior is preserved
p1 = Product(name="Laptop", price=999.99)
p2 = ReconstructedProduct(name="Laptop", price=999.99)

result1 = p1.compare_with(p1)
result2 = p2.compare_with(p2)

assert result1["overall_score"] == result2["overall_score"]  # Both 1.0
```

### Stickler Config Round-trip

```python
# Export
config = Product.to_stickler_config()

# Re-import
ReconstructedProduct = StructuredModel.model_from_json(config)

# Comparison behavior is identical
p1 = Product(name="Laptop", price=999.99)
p2 = ReconstructedProduct(name="Laptop", price=999.99)

result1 = p1.compare_with(p1)
result2 = p2.compare_with(p2)

assert result1["overall_score"] == result2["overall_score"]  # Both 1.0
```

## model_json_schema() Explained

The `model_json_schema()` method is inherited from Pydantic and extended by Stickler to include comparison metadata. Unlike the export methods above, this is primarily for runtime introspection and Pydantic tooling integration.

**Important:** This method is NOT for round-trip serialization. Use `to_json_schema()` or `to_stickler_config()` instead if you need to export and re-import models.

### When to Use This

Use `model_json_schema()` when you need to:

1. **Generate API documentation** - Create OpenAPI schemas with comparison metadata
2. **Validate data** - Use with standard JSON Schema validators
3. **Inspect models at runtime** - Programmatically examine model structure
4. **Integrate with Pydantic tools** - Work with tools expecting Pydantic schemas

### Example

```python
from stickler import StructuredModel, ComparableField

class Product(StructuredModel):
    name: str = ComparableField(threshold=0.8, default=...)
    price: float = ComparableField(threshold=0.95, default=...)

# Get Pydantic-compatible schema
schema = Product.model_json_schema()

# Use for validation
from jsonschema import validate
data = {"name": "Laptop", "price": 999.99}
validate(instance=data, schema=schema)

# Inspect comparison config
name_comparison = schema["properties"]["name"]["x-comparison"]
print(f"Name threshold: {name_comparison['threshold']}")
```

### Output Format

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "x-comparison": {
        "comparator_type": "LevenshteinComparator",
        "comparator_name": "levenshtein",
        "comparator_config": {},
        "threshold": 0.8,
        "weight": 2.0
      }
    },
    "price": {
      "type": "number",
      "x-comparison": {
        "comparator_type": "NumericComparator",
        "threshold": 0.95
      }
    }
  }
}
```

### Why Not Use for Round-Trip?

The `x-comparison` format includes runtime objects and is optimized for Pydantic compatibility, not serialization. It's designed for introspection and tooling integration, not for saving and loading model configurations.

**For saving and loading models, always use:**
- `to_json_schema()` + `from_json_schema()`, or
- `to_stickler_config()` + `model_from_json()`

## Format Comparison

| Feature | to_json_schema() | to_stickler_config() | model_json_schema() |
|---------|------------------|----------------------|---------------------|
| Format | JSON Schema + x-aws-stickler-* | Custom Stickler JSON | JSON Schema + x-comparison |
| Round-trip | ✅ Yes | ✅ Yes | ❌ No |
| Standard | JSON Schema draft-07 | Custom | Pydantic JSON Schema |
| Import method | `from_json_schema()` | `model_from_json()` | N/A |
| Use case | Interoperability | Stickler-specific | Pydantic tooling |

## Best Practices

### Choosing an Export Format

- **Use `to_json_schema()`** when:
    - Integrating with OpenAPI/AsyncAPI
    - Working with JSON Schema validators
    - Need industry-standard format
    - Sharing schemas with external systems

- **Use `to_stickler_config()`** when:
    - Working exclusively within Stickler
    - Need more concise format
    - Manually editing configurations
    - Prefer simpler structure

- **Use `model_json_schema()`** when:
    - Generating API documentation
    - Using Pydantic tooling
    - Need runtime introspection
    - Not doing round-trip serialization

### Saving and Loading

```python
import json

# Save JSON Schema
schema = Product.to_json_schema()
with open('product_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

# Load and reconstruct
with open('product_schema.json', 'r') as f:
    schema = json.load(f)
Product = StructuredModel.from_json_schema(schema)

# Save Stickler config
config = Product.to_stickler_config()
with open('product_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Load and reconstruct
with open('product_config.json', 'r') as f:
    config = json.load(f)
Product = StructuredModel.model_from_json(config)
```

### Version Control

Store exported schemas in version control to track model evolution:

```bash
# Export current model
python -c "from myapp.models import Product; import json; \
    json.dump(Product.to_json_schema(), open('schemas/product_v1.json', 'w'), indent=2)"

# Commit to version control
git add schemas/product_v1.json
git commit -m "Add Product model schema v1"
```

## See Also

- [StructuredModel Dynamic Creation](StructuredModel_Dynamic_Creation.md) - Import methods
- [StructuredModel Advanced Functionality](StructuredModel_Advanced_Functionality.md) - Comparison features
- [JSON Schema Extensions Reference](../README.md#json-schema-extensions-x-aws-stickler--complete-reference) - Full extension documentation
