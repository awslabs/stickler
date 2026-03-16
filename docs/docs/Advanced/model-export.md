---
title: Model Export
---

# Model Export

Stickler provides three methods for exporting model schemas. Two support round-trip serialization (export then re-import); the third is for runtime introspection only.

## Export Methods at a Glance

| Method | Format | Round-trip | Import via | Best for |
|--------|--------|-----------|------------|----------|
| `to_json_schema()` | JSON Schema + `x-aws-stickler-*` | Yes | `from_json_schema()` | Interoperability, OpenAPI |
| `to_stickler_config()` | Custom Stickler JSON | Yes | `model_from_json()` | Hand-editing, version control |
| `model_json_schema()` | Pydantic JSON Schema + `x-comparison` | No | N/A | API docs, runtime inspection |

## Common Workflow: Export, Customize, Re-import

```python
from stickler import StructuredModel, ComparableField

class Product(StructuredModel):
    name: str = ComparableField(default=...)
    price: float = ComparableField(default=...)

# Export defaults
config = Product.to_stickler_config()

# Customize
config["fields"]["name"]["threshold"] = 0.9
config["fields"]["name"]["weight"] = 2.0

# Re-import
CustomProduct = StructuredModel.model_from_json(config)
```

This is useful for starting with sensible defaults and then tuning thresholds, sharing configurations across teams, or A/B testing comparison strategies.

## to_json_schema()

Exports a standard JSON Schema document with `x-aws-stickler-*` extensions. Compatible with JSON Schema tooling and validators.

```python
schema = Product.to_json_schema()
```

**Output:**

```json
{
  "type": "object",
  "x-aws-stickler-model-name": "Product",
  "properties": {
    "name": {
      "type": "string",
      "x-aws-stickler-comparator": "LevenshteinComparator",
      "x-aws-stickler-threshold": 0.8,
      "x-aws-stickler-weight": 2.0
    },
    "price": {
      "type": "number",
      "x-aws-stickler-comparator": "NumericComparator",
      "x-aws-stickler-threshold": 0.95
    }
  },
  "required": ["name", "price"]
}
```

## to_stickler_config()

Exports a concise Stickler-specific format that is easy to read and edit by hand.

```python
config = Product.to_stickler_config()
```

**Output:**

```json
{
  "model_name": "Product",
  "fields": {
    "name": {
      "type": "str",
      "comparator": "LevenshteinComparator",
      "threshold": 0.8,
      "weight": 2.0,
      "required": true
    },
    "price": {
      "type": "float",
      "comparator": "NumericComparator",
      "threshold": 0.95,
      "required": true
    }
  }
}
```

## model_json_schema()

Inherited from Pydantic, extended by Stickler to include `x-comparison` metadata. This method is **not** round-trip compatible -- use it for API documentation and runtime introspection only.

```python
schema = Product.model_json_schema()
# schema["properties"]["name"]["x-comparison"]["threshold"]  -> 0.8
```

## Nested Models and Lists

Both round-trip methods handle nested `StructuredModel` classes and `List[StructuredModel]` fields recursively:

```python
from typing import List

class LineItem(StructuredModel):
    match_threshold = 0.8
    product: str = ComparableField(default=...)
    quantity: int = ComparableField(default=...)

class Order(StructuredModel):
    order_id: str = ComparableField(threshold=1.0, default=...)
    items: List[LineItem] = ComparableField(default=...)

# JSON Schema export includes the LineItem schema under "items"
schema = Order.to_json_schema()

# Stickler config includes LineItem fields under items.fields
config = Order.to_stickler_config()
```

## Round-Trip Examples

### JSON Schema

```python
schema = Product.to_json_schema()
Reconstructed = StructuredModel.from_json_schema(schema)

p1 = Product(name="Laptop", price=999.99)
p2 = Reconstructed(name="Laptop", price=999.99)

r1 = p1.compare_with(p1)
r2 = p2.compare_with(p2)
assert r1["overall_score"] == r2["overall_score"]
```

### Stickler Config

```python
config = Product.to_stickler_config()
Reconstructed = StructuredModel.model_from_json(config)

# Comparison behavior is identical
```

## Saving and Loading

```python
import json

# Save
with open('product_schema.json', 'w') as f:
    json.dump(Product.to_json_schema(), f, indent=2)

# Load
with open('product_schema.json') as f:
    schema = json.load(f)
Product = StructuredModel.from_json_schema(schema)
```

## See Also

- [Dynamic Model Creation](dynamic-models.md) -- the import side (`from_json_schema`, `model_from_json`)
