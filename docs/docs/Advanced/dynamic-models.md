---
title: Dynamic Model Creation
---

# Dynamic Model Creation

Stickler can create `StructuredModel` classes at runtime from JSON configuration, enabling model definition without writing Python class code. This is useful for configuration-driven evaluation, A/B testing of comparison strategies, and integration with external systems that produce JSON Schema.

## Two Creation Methods

| Method | Input Format | Best For |
|--------|-------------|----------|
| `from_json_schema()` | Standard JSON Schema with `x-aws-stickler-*` extensions | Interoperability, external tooling |
| `model_from_json()` | Custom Stickler JSON configuration | Concise hand-edited configs |

Both produce fully functional `StructuredModel` classes with comparison capabilities, nested hierarchies, custom comparators, and Hungarian matching for lists.

## Method 1: JSON Schema (Recommended)

### Basic Example

```python
from stickler import StructuredModel

product_schema = {
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
        },
        "in_stock": {
            "type": "boolean"
        }
    },
    "required": ["name", "price"]
}

Product = StructuredModel.from_json_schema(product_schema)

gt = Product(name="Laptop", price=999.99, in_stock=True)
pred = Product(name="Laptop Pro", price=999.99, in_stock=True)

result = gt.compare_with(pred)
```

### Nested Objects and Arrays

Nested JSON Schema objects become nested `StructuredModel` classes automatically. Arrays of objects use Hungarian matching:

```python
order_schema = {
    "type": "object",
    "x-aws-stickler-model-name": "Order",
    "properties": {
        "order_id": {
            "type": "string",
            "x-aws-stickler-comparator": "ExactComparator"
        },
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-aws-stickler-threshold": 0.8},
                "email": {"type": "string", "x-aws-stickler-comparator": "ExactComparator"}
            }
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "x-aws-stickler-threshold": 0.8},
                    "quantity": {"type": "integer"},
                    "price": {"type": "number", "x-aws-stickler-threshold": 0.95}
                },
                "required": ["product", "quantity", "price"]
            }
        }
    },
    "required": ["order_id"]
}

Order = StructuredModel.from_json_schema(order_schema)
```

### JSON Schema Extension Reference

The `x-aws-stickler-*` extensions control comparison behavior on each property:

| Extension | Purpose | Example |
|-----------|---------|---------|
| `x-aws-stickler-comparator` | Comparison algorithm | `"LevenshteinComparator"` |
| `x-aws-stickler-threshold` | Match threshold (0.0--1.0) | `0.8` |
| `x-aws-stickler-weight` | Field importance weight | `2.0` |
| `x-aws-stickler-clip-under-threshold` | Zero out scores below threshold | `true` |
| `x-aws-stickler-aggregate` | Include in aggregate metrics | `true` |
| `x-aws-stickler-model-name` | Class name (object-level) | `"Invoice"` |
| `x-aws-stickler-match-threshold` | Hungarian match threshold (object-level) | `0.75` |

Default comparators are assigned by JSON Schema type when no extension is specified:

| JSON Schema Type | Default Comparator | Default Threshold |
|------------------|-------------------|-------------------|
| `string` | LevenshteinComparator | 0.5 |
| `number` / `integer` | NumericComparator | 0.5 |
| `boolean` | ExactComparator | 1.0 |
| `array` (objects) | Hungarian matching | 0.7 |
| `object` | Recursive comparison | 0.7 |

For the complete reference, see the [Evaluation](../Evaluation/README.md) page.

---

## Method 2: Custom Stickler Configuration

For cases where JSON Schema is unnecessary, `model_from_json()` accepts a more concise format.

### Basic Example

```python
from stickler import StructuredModel

person_config = {
    "model_name": "Person",
    "fields": {
        "name": {
            "type": "str",
            "comparator": "LevenshteinComparator",
            "threshold": 0.8,
            "weight": 1.0,
            "required": True
        },
        "age": {
            "type": "int",
            "comparator": "NumericComparator",
            "threshold": 0.9,
            "weight": 0.5
        },
        "email": {
            "type": "str",
            "comparator": "ExactComparator",
            "threshold": 1.0,
            "required": False,
            "default": None
        }
    }
}

Person = StructuredModel.model_from_json(person_config)
```

### Configuration Schema

**Top level:**

```json
{
    "model_name": "string",
    "match_threshold": 0.7,
    "fields": { ... }
}
```

**Primitive fields:**

```json
{
    "type": "str|int|float|bool|list|dict",
    "comparator": "ComparatorName",
    "comparator_config": {},
    "threshold": 0.8,
    "weight": 1.0,
    "required": true,
    "default": null,
    "aggregate": false,
    "clip_under_threshold": true
}
```

**Nested model fields** use `"type": "structured_model"` with a nested `"fields"` object. **List of models** use `"type": "list_structured_model"`. **Optional models** use `"type": "optional_structured_model"`.

### Nested Model Example

```python
company_config = {
    "model_name": "Company",
    "fields": {
        "name": {
            "type": "str",
            "comparator": "LevenshteinComparator",
            "threshold": 0.8,
            "weight": 2.0
        },
        "employees": {
            "type": "list_structured_model",
            "weight": 1.0,
            "match_threshold": 0.7,
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.8
                },
                "department": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 1.0
                }
            }
        }
    }
}

Company = StructuredModel.model_from_json(company_config)
```

## Loading from Files

```python
import json
from stickler import StructuredModel

# JSON Schema
with open('schema.json') as f:
    schema = json.load(f)
Model = StructuredModel.from_json_schema(schema)

# Stickler config
with open('config.json') as f:
    config = json.load(f)
Model = StructuredModel.model_from_json(config)
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `"Unknown type"` | Unsupported type string | Use one of: `str`, `int`, `float`, `bool`, `list`, `dict`, `structured_model`, `list_structured_model` |
| `"Missing comparator"` | Primitive field without comparator | Add a `"comparator"` key |
| `"Invalid threshold"` | Threshold outside 0.0--1.0 | Use a value between 0.0 and 1.0 |
| Nested model errors | Invalid nested `fields` config | Validate nested config independently |

## See Also

- [Model Export](model-export.md) -- exporting models back to JSON Schema or Stickler config
- [Hungarian Matching](hungarian-matching.md) -- how list fields are compared
