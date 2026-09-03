---
title: Dynamic Model Creation
---

# Dynamic Model Creation

Stickler can create `StructuredModel` classes at runtime from JSON configuration, enabling model definition without writing Python class code. This is useful for configuration-driven evaluation, A/B testing of comparison strategies, and integration with external systems that produce JSON Schema.

## Three Creation Methods

| Method | Input Format | Infers comparators? | Best For |
|--------|-------------|---------------------|----------|
| `from_pydantic()` | A Pydantic `BaseModel` class | Yes — from the Python type *and* the field name | You already have a model and want sensible defaults |
| `from_json_schema()` | Standard JSON Schema with `x-aws-stickler-*` extensions | Type only, coarsely — every `string` becomes Levenshtein at `0.5` | Interoperability, external tooling, config files |
| `model_from_json()` | Custom Stickler JSON configuration | No — a primitive field without a `comparator` is an error | Concise hand-edited configs |

All three produce fully functional `StructuredModel` classes with comparison capabilities, nested hierarchies, custom comparators, and Hungarian matching for lists. They differ only in how much you have to say.

The two inferring methods do not agree with each other, and the gap is wide enough to change a score: see [Choosing a Configuration Path](../Getting-Started/choosing-a-configuration-path.md) before picking one.

## Method 1: JSON Schema

Recommended when your configuration lives outside Python — a file under version control, a value fetched at runtime, or a schema produced by another system. If your starting point is a Pydantic class already in the codebase, use `from_pydantic()` (Method 3) and skip writing a schema at all.

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
| `x-aws-stickler-model-name` | Class name (object-level) | `"Invoice"` |
| `x-aws-stickler-match-threshold` | Hungarian match threshold (object-level) | `0.75` |

Default comparators are assigned by JSON Schema type when no extension is specified:

| JSON Schema Type | Default Comparator | Default Threshold |
|------------------|-------------------|-------------------|
| `string` | LevenshteinComparator | 0.5 |
| `number` / `integer` | NumericComparator | 0.5 |
| `boolean` | ExactComparator | 0.5 |
| `array` (objects) | Hungarian matching | 0.7 |
| `object` | Recursive comparison | 0.7 |

For the complete reference, see the [Evaluation](../Guides/Evaluation/README.md) page.

### Optional fields and `null`

A property that is **not** listed in `required` (or that declares an explicit
`"default": null`) is built as an optional field: its Python annotation is
widened to `Optional[T]` and its default is `None`. This keeps the model usable
through every construction path — in particular `from_json(..., process_rich_values=True)`,
which round-trips nested objects and would otherwise re-validate the `None`
default against a non-nullable annotation and raise.

Two consequences are worth noting:

- An explicit `null` is **accepted** for an optional typed field (e.g. a JSON
  payload `{"note": null}` for an optional `string`), even though strict JSON
  Schema treats `null` as invalid for `"type": "string"`. This is deliberate:
  model predictions often emit explicit nulls, and these should score as a value
  rather than crash evaluation.
- `model_json_schema()` reflects this by emitting `{"anyOf": [{"type": "string"}, {"type": "null"}]}`
  for such fields.
- `to_json_schema()` likewise makes the nullability explicit, emitting
  `{"type": ["string", "null"]}`. So an optional property declared only as
  `{"type": "string"}` gains an explicit `"null"` on export that the input schema
  never stated. This is intended: "absent from `required`" already means the value
  may be `None`, and the exported schema now says so out loud instead of leaving it
  implicit. The round trip is idempotent — re-importing the exported schema and
  exporting again yields the same document, and accepts the same values.

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

## Method 3: From a Pydantic Class

When the model already exists in Python, `from_pydantic()` builds the evaluator from it — no schema, no config, no comparators named anywhere:

```python
from typing import Optional
from pydantic import BaseModel
from stickler import StructuredModel

class Product(BaseModel):
    sku: str
    name: str
    price: float
    notes: Optional[str] = None

Model = StructuredModel.from_pydantic(Product)   # class is named ProductEval
```

Each field is given a comparator and threshold from its type and its name:

| Field | Inferred | Why |
|-------|----------|-----|
| `sku` | `ExactComparator` @ 1.0 | identifier token — a SKU differing in case may be a different SKU |
| `name` | `LevenshteinComparator` @ 0.85 | name token — tolerates typos, not substitutions |
| `price` | `NumericComparator` @ 0.95 | `float` with a relative tolerance |
| `notes` | `FuzzyComparator` @ 0.6 | free-text token — word order should not matter |

Instances compare like any other `StructuredModel`, and `stickler.evaluate(gt, pred)` reaches the same inference without building the class yourself. Use `to_json_schema()` on the result to export those decisions as a schema — the bridge between this method and Method 1:

```python
Model.to_json_schema()   # comparators, thresholds, ["string", "null"] for Optional
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

Every `model_from_json()` failure raises `ValueError`, and all but one are prefixed `Invalid field configuration:`. The message names the offending field and lists the valid values.

| Message | Cause | Fix |
|-------|-------|-----|
| `Configuration must contain 'fields' key` | Top-level `fields` missing | Wrap the field definitions in `{"fields": {...}}` |
| `Field 'x' missing required 'type' parameter` | Field definition has no `type` | Add a `"type"` key |
| `Unknown type: 'x'. Available types: [...]` | Unsupported type string | Use a listed type; the message enumerates them |
| `Field 'x' with primitive type 'str' requires a 'comparator'` | Primitive field without a comparator | Add a `"comparator"` key — this method infers nothing |
| `Field 'x' threshold must be between 0.0 and 1.0, got 5.0` | Threshold outside the range | Use a value in 0.0--1.0 |
| `Unknown comparator: 'x'. Available: [...]` | Comparator name not registered | Use a class name from the list; `BERTComparator` and `LLMComparator` appear only when the `[bert]` and `[llm]` extras are installed |
| `Field 'x' with type 'structured_model' requires a 'fields' configuration` | Nested model without its own `fields` | Add the nested `fields` block |

## See Also

- [Model Export](model-export.md) -- exporting models back to JSON Schema or Stickler config
- [Hungarian Matching](hungarian-matching.md) -- how list fields are compared
