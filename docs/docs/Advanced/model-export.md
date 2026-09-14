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
| `model_json_schema()` | Pydantic JSON Schema | No | N/A | API docs, LLM tool specs, runtime inspection |

Both round-trip methods are also reachable from a plain Pydantic `BaseModel` that was never written
as a `StructuredModel`, by converting first:

```python
StructuredModel.from_pydantic(Invoice).to_json_schema()
```

`from_pydantic()` infers a comparator and threshold per field from the Python type *and* the field
name, so the exported schema is a full configuration rather than a bare shape — `invoice_id` comes
out as `ExactComparator` at `1.0`, `customer_name` as `LevenshteinComparator` at `0.85`, `notes` as
`FuzzyComparator` at `0.6`. This is the supported way to turn a Pydantic model into a schema with
`x-aws-stickler-*` extensions; `Invoice.model_json_schema()` gives you the shape with none of them.
See [Choosing a Configuration Path](../Getting-Started/choosing-a-configuration-path.md).

## Common Workflow: Export, Customize, Re-import

```python
from stickler import (
    ComparableField,
    LevenshteinComparator,
    NumericComparator,
    StructuredModel,
)

class Product(StructuredModel):
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0, default=...
    )
    price: float = ComparableField(
        comparator=NumericComparator(), threshold=0.95, default=...
    )

# Export the configuration as written
config = Product.to_stickler_config()

# Customize
config["fields"]["name"]["threshold"] = 0.9
config["fields"]["name"]["weight"] = 3.0

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
      "x-aws-stickler-comparator-config": {"normalize": true},
      "x-aws-stickler-threshold": 0.8,
      "x-aws-stickler-weight": 2.0,
      "x-aws-stickler-clip-under-threshold": true
    },
    "price": {
      "type": "number",
      "x-aws-stickler-comparator": "NumericComparator",
      "x-aws-stickler-threshold": 0.95,
      "x-aws-stickler-weight": 1.0,
      "x-aws-stickler-clip-under-threshold": true
    }
  },
  "required": ["name", "price"],
  "x-aws-stickler-match-threshold": 0.7
}
```

The export is complete, not minimal: every field carries its weight, clip flag, and comparator
configuration even where those were left at their defaults, and the root carries its match
threshold. This is what makes the round trip exact — the re-imported model cannot drift if a default
changes in a later release. `price` has no `x-aws-stickler-comparator-config` only because
`NumericComparator()` was constructed with no arguments to record. Fields declared `Optional` export
as `{"type": ["string", "null"]}`; see [Optional fields and `null`](dynamic-models.md#optional-fields-and-null).

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
      "comparator_config": {"normalize": true},
      "threshold": 0.8,
      "weight": 2.0,
      "clip_under_threshold": true,
      "required": true
    },
    "price": {
      "type": "float",
      "comparator": "NumericComparator",
      "threshold": 0.95,
      "weight": 1.0,
      "clip_under_threshold": true,
      "required": true
    }
  },
  "match_threshold": 0.7
}
```

The same completeness applies here, and `required` is explicit per field rather than collected into
a list as JSON Schema does.

## model_json_schema()

Inherited from Pydantic. Describes the model's *shape* only: no comparison
configuration is emitted, so the output matches what an equivalent plain
`BaseModel` would produce. That is what makes a configured `StructuredModel`
usable directly as an LLM structured-output schema.

This method is **not** round-trip compatible. Use `to_json_schema()` or
`to_stickler_config()` when you need the configuration back.

Feeding its output to `from_json_schema()` parses without error — `Optional`
fields render as `anyOf: [{"type": ...}, {"type": "null"}]` and are read
correctly since [#198](https://github.com/awslabs/stickler/pull/198) — but the
result misleads. The rebuilt model is named `DynamicModel` and every field
carries default comparators, thresholds and weights, because the shape alone
does not describe them
([#214](https://github.com/awslabs/stickler/issues/214)). A field configured
`ExactComparator` at threshold `1.0` and weight `3.0` comes back as
`LevenshteinComparator` at `0.5` and `1.0`, so the round trip scores differently
while raising nothing to tell you.

```python
schema = Product.model_json_schema()
# schema["properties"]["name"]["type"]  -> "string"

# For comparison configuration, read the field or use the export methods:
Product.model_fields["name"].json_schema_extra  # carries the config
Product.to_json_schema()                        # x-aws-stickler-* extensions
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
