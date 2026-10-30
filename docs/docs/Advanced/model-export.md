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

A plain Pydantic `BaseModel` field, or a list of them, is scored as one object and exports as a
free-form object (`{"type": "object"}`, `"dict"` in a Stickler config). It re-imports as `dict` with
its comparison settings, but its own field types are not exported, so the re-imported field does
not coerce values. A list of lists exports as nested arrays; `to_stickler_config()` writes it as
`List[List]`, without the inner element type. A plain model inside a nested list comes back as a
`dict`, which can score slightly differently there. `to_stickler_config()` has no type for a union
with a model arm (`Union[Card, Bank]`, `Union[Plain, str]`) or for a `StructuredModel` inside a
nested list (`List[List[Card]]`), so both methods raise for these, to keep one rule
([#349](https://github.com/awslabs/stickler/issues/349)).

A `Dict` field exports as `{"type": "object"}`, with `additionalProperties` for a primitive value
type, and as `"Dict[str, int]"` in a Stickler config (`"dict"` when a key or value type has no config
name). Through JSON Schema it re-imports as `dict`, so the value type is not enforced. A
`default_factory` of `list`, `dict` or `set` on a list or mapping field exports as the value it
produces (`[]`, `{}`, a set as a list) and re-imports as that default. Any other factory, or one on
a field exported as a scalar such as `Set[str]`, exports no default and re-imports as `None`.

A Stickler config has no nullable flag: `model_from_json()` makes a field `Optional` only when its
default is `None`. So on an `Optional` field `to_stickler_config()` writes no empty `[]` or `{}`
default, from a factory or written literally. The field re-imports nullable with a `None` default,
which scores the same as the original's `[]` or `{}`, though the rebuilt instance holds `None`
(`model_dump()` shows `null`). This covers an empty list or mapping default only. Any other default
on an `Optional` field, such as `Optional[List[int]] = [1]` or `Optional[str] = ""`, is written and
re-imports non-nullable, rejecting `None`. A `List[Model]` field is the exception: a non-empty
literal (`[Sub(...)]`), a tuple, or a factory other than `list`, `dict` or `set` is not written, so
the field re-imports as `Optional` with a `None` default and accepts `None`, which scores the same
as `[]`. That holds whether or not the original was `Optional`.

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
threshold. `price` has no `x-aws-stickler-comparator-config` only because
`NumericComparator()` was constructed with no arguments to record. Fields declared `Optional` export
as `{"type": ["string", "null"]}`; see [Optional fields and `null`](dynamic-models.md#optional-fields-and-null).
Re-importing an export preserves comparison settings. A field that is not required (a bare `T`
without `default=...`) is currently re-imported as nullable `Optional[T]`
([#189](https://github.com/awslabs/stickler/issues/189)).

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

A list of mappings (`List[Dict[str, str]]`, `List[Mapping[...]]`) exports free-form object items:
`items: {"type": "object"}` from `to_json_schema()`, `List[Dict]` from `to_stickler_config()`. An
optional element exports as `["object", "null"]` and `List[Optional[Dict]]`. They re-import as
`List[dict]` and `List[Dict]` respectively (`List[Optional[dict]]` and `List[Optional[Dict]]` for an
optional element) with the comparator, threshold, weight and clip the field resolves, declared or
default, so it scores as the original did. The value type is not exported, so the re-imported
field does not coerce its values.

A model built with `from_pydantic()` exports the same way, with two differences. Its element accepts
any value, so a malformed prediction is scored rather than rejected; `to_stickler_config()` keeps
that by writing `List[Any]`, but the `to_json_schema()` re-import validates each element as a dict,
so a prediction element that is not a mapping raises instead of scoring `0.0`. An `Optional` written
inside `Annotated` (`List[Annotated[Optional[Dict], ...]]`) loses its `null`; write
`Optional[Annotated[...]]` instead.

A mapping field that is not in a list, and a list whose element is a union such as
`List[Union[Dict, str]]`, still export as a string ([#349](https://github.com/awslabs/stickler/issues/349)).

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
