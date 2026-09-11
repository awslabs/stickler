---
title: Dynamic Model Creation
---

# Dynamic Model Creation

Stickler can create `StructuredModel` classes at runtime from JSON configuration, enabling model definition without writing Python class code. This is useful for configuration-driven evaluation, A/B testing of comparison strategies, and integration with external systems that produce JSON Schema.

## Three Creation Methods

| Method | Input Format | Infers comparators? | Best For |
|--------|-------------|---------------------|----------|
| `from_pydantic()` | A Pydantic `BaseModel` class | Yes — from the Python type *and* the field name | You already have a model and want sensible defaults |
| `from_json_schema()` | Standard JSON Schema with `x-aws-stickler-*` extensions | From structure, never names — a plain `string` becomes Levenshtein at `0.5` | Interoperability, external tooling, config files |
| `model_from_json()` | Custom Stickler JSON configuration | Only on request — a primitive field without a `comparator` is an error unless you set `infer_unspecified_fields` or `"comparator": "auto"` | Concise hand-edited configs |

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
| `x-aws-stickler-threshold` | Match threshold (0.0--1.0). Read on a scalar, an object, and an array of scalars; **ignored on an array of models** -- an array whose `items` declare `properties` -- where the element class's `x-aws-stickler-match-threshold` is the gate. Read, not ignored, on an array of free-form `{"type": "object"}` items, which become `List[dict]` | `0.8` |
| `x-aws-stickler-weight` | Field importance weight | `2.0` |
| `x-aws-stickler-clip-under-threshold` | Zero out scores below threshold | `true` |
| `x-aws-stickler-model-name` | Class name (object-level) | `"Invoice"` |
| `x-aws-stickler-match-threshold` | Hungarian match threshold (object-level) | `0.75` |
| `x-aws-stickler-infer-unspecified` | Infer comparators for properties that name none (object-level). See [Inferring unspecified fields](#inferring-unspecified-fields) | `true` |

With no extension specified, the property is parsed to a strict Python annotation and the comparator
is chosen from that annotation, which is then widened back to the JSON value type so an invalid
extraction scores `0.0` instead of raising. So `format`, `enum` and `const` participate in the
choice, while field names never do:

| Property | Default Comparator | Default Threshold |
|------------------|-------------------|-------------------|
| `string` | LevenshteinComparator | 0.5 |
| `number` / `integer` | NumericComparator | 0.5 |
| `boolean` | ExactComparator | 0.5 |
| `string` + `format: date` | DateComparator | 1.0 |
| `string` + `format: date-time` | DateComparator | 1.0 |
| `string` + `format: time` | ExactComparator | 1.0 |
| `string` + `format: uri` | ExactComparator | 1.0 |
| `string` + `format: uuid` | ExactComparator | 1.0 |
| `string` + `enum: [...]` | ExactComparator | 1.0 |
| `const` | ExactComparator | 1.0 |
| `array` (objects) | Hungarian matching | 0.5, pairing elements at 0.7 |
| `array` (primitives) | the element's row, applied per element | the element's |
| `object` | Recursive comparison | 0.7 |
| `object` with no `properties` | ExactComparator | 1.0 |

Five `format` values are special-cased, and they are exactly the ones the schema
parser maps to a distinct Python type: `date`, `date-time`, `time`, `uri` and
`uuid`. The three with no comparator of their own (`time`, `uri`, `uuid`) get
`ExactComparator` at 1.0, the fallback for a type with no row of its own -- **not**
the `string` row. Every other `format`, `email` and `ipv4` among them, is parsed as
plain `str` and does fall to the `string` row.

These defaults are **not** the same as the ones `stickler.evaluate()` infers, which
are tuned per type: a `number` gets `0.95` there rather than `0.5`. See
[Inferring unspecified fields](#inferring-unspecified-fields) to opt a
schema-driven model into the tuned set.

Because the annotation is widened after the comparator is chosen, `model_fields` shows
`Optional[str]` for every row above, including the `format` and `enum` ones —
`to_json_schema()["properties"]` is where you read back what was actually chosen.

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
    "infer_unspecified_fields": false,
    "fields": { ... }
}
```

`comparator` is **required** on every primitive field unless
`infer_unspecified_fields` is true or the field sets `"comparator": "auto"`. See
[Inferring unspecified fields](#inferring-unspecified-fields).

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

## Inferring unspecified fields

A large extraction schema usually has a handful of fields whose comparison rules
matter to you -- the invoice ID must match exactly, the total needs numeric
tolerance -- and a long tail where you just want something sensible. Set
`infer_unspecified_fields` and Stickler chooses for the rest, using the same
inference `stickler.evaluate()` uses.

### Turning it on

Configure the fields you care about, and leave the others bare:

```json
{
    "model_name": "Invoice",
    "infer_unspecified_fields": true,
    "fields": {
        "invoice_id": {"type": "str", "comparator": "ExactComparator", "threshold": 1.0},
        "total":      {"type": "float"},
        "paid":       {"type": "bool"},
        "vendor":     {"type": "str"}
    }
}
```

To infer only certain fields, set `"comparator": "auto"` on those instead:

```json
{
    "fields": {
        "invoice_id": {"type": "str", "comparator": "ExactComparator"},
        "total":      {"type": "float", "comparator": "auto"}
    }
}
```

The JSON Schema path uses `x-aws-stickler-infer-unspecified` at the object level,
and `x-aws-stickler-comparator: "auto"` on a property:

```json
{
    "type": "object",
    "x-aws-stickler-model-name": "Invoice",
    "x-aws-stickler-infer-unspecified": true,
    "properties": {
        "invoice_id": {"type": "string", "x-aws-stickler-comparator": "ExactComparator"},
        "total": {"type": "number"},
        "issued": {"type": "string", "format": "date"}
    }
}
```

A field-level setting always wins over the model-level flag, both ways: `"auto"`
infers one field in an otherwise explicit config, and naming a comparator pins one
field in an otherwise inferred config.

A nested model can scope inference to its own subtree, in either direction, without
affecting its parent or siblings: `infer_unspecified_fields` on a
`structured_model` field, or `x-aws-stickler-infer-unspecified` on a nested object
in a JSON Schema.

`"auto"` is for a scalar or a list of scalars. A nested object is compared
recursively and an array of objects by Hungarian matching, so neither has a
comparator for inference to choose, and
`x-aws-stickler-comparator: "auto"` on one is an error naming
`x-aws-stickler-infer-unspecified` as the key that does work there.

### Loading a config and checking what you got

Load the config, then ask the model what it decided. `explain()` returns one row per
field with the comparator, threshold and where each came from:

```python
import json
import stickler
from stickler import StructuredModel

with open("invoice_eval.json") as handle:
    config = json.load(handle)

Invoice = StructuredModel.model_from_json(config)

for name, row in stickler.eval_for(Invoice).explain().items():
    print(f"{name:12} {row['comparator']:22} {row['threshold']:<6} {row['source']}")
```

```
invoice_id   ExactComparator        1.0    explicit
total        NumericComparator      0.95   name-token
paid         ExactComparator        1.0    type
vendor       LevenshteinComparator  0.85   name-token
```

`source` is `explicit` for a field you configured. Anything else is a field Stickler
chose, and names what drove the choice: `type` from the declared type alone,
`name-token` when the field's name refined it. A name that matched a rule the
declared type cannot support reports `type`, because the type default is what
shipped -- `row['why']` records the refusal. These are the same values
`stickler.evaluate()` reports, so the two paths read alike.

`row['why']` gives the full reasoning:

```python
stickler.eval_for(Invoice).explain()["total"]["why"]
# ['type:float -> NumericComparator(rel_tol=0.001)@0.95',
#  'name-token:total -> NumericComparator@0.95']
```

Check this before a first evaluation run. It is the fastest way to catch a field you
meant to configure and misspelled, since a misspelled key means the field gets
inferred rather than what you intended.

Use `Invoice.to_stickler_config()` if you want the resolved configuration back as
JSON, for example to commit the fully-expanded version once you are happy with it.

!!! warning "The export is lossy for a `dict` field, so the round trip does not hold"

    `to_stickler_config()` writes a mapping field as `"type": "str"`:

    ```json
    {"type": "str", "comparator": "ANLSStarComparator", "threshold": 0.7, ...}
    ```

    Re-importing that config gives a model whose field is a `str`, which then
    rejects dict input with `Input should be a valid string`. The comparator and
    threshold survive; the type does not.

    This is not specific to inference -- an explicitly configured `dict` field
    exports the same way -- but a bare `{"type": "dict"}` only became legal to write
    with this feature, so following the advice above can now reach it. Until the
    export is fixed, either keep `"type": "dict"` by hand in the committed config,
    or treat the export as a readable record rather than a re-importable one.

### What inference chooses

Both the comparator and the threshold, matching `stickler.evaluate()`:

| field | without the flag | with it |
|---|---|---|
| `total: float` | NumericComparator @ 0.5 | NumericComparator @ 0.95 |
| `paid: bool` | ExactComparator @ 0.5 | ExactComparator @ 1.0 |
| `issued` (`format: date`) | DateComparator @ 1.0 | DateComparator @ 0.95 |
| `vendor: str` | LevenshteinComparator @ 0.5 | LevenshteinComparator @ 0.85 |
| `amounts: List[float]` | NumericComparator @ 0.5 | NumericComparator @ 0.95 |
| `meta: dict` | ExactComparator @ 1.0 | ANLSStarComparator @ the model's `match_threshold` |

The "without the flag" column applies to `from_json_schema()`. With
`model_from_json()`, a primitive field that names no comparator is an error unless
you opt in.

A list is inferred from its **element** type, because that is what the comparator is
applied to. Declare the element: `{"type": "List[float]"}` in a Stickler config, or
`{"type": "array", "items": {"type": "number"}}` in a JSON Schema.

!!! warning "`{"type": "list"}` names no element type"

    A bare `list` gives inference nothing to work with, so it falls to the answer
    for any type with no scalar form: `ExactComparator` at threshold 1.0. Elements
    are still compared one by one, so a list with two of three elements right still
    scores 0.67 -- what you lose is the element comparator. `[1.0]` against
    `[1.0000001]` scores **0.0**, where `{"type": "List[float]"}` scores 1.0; a
    string element off by one character scores 0.0 where `{"type": "List[str]"}`
    scores 0.83.

    Write the element type out. `List[str]`, `List[int]`, `List[float]` and
    `List[bool]` all resolve.

### Partly-configured fields

Name some parameters and leave others out, and only the missing ones are inferred:

```json
{"total": {"type": "float", "threshold": 0.99}}
```

```
comparator   NumericComparator   inferred
threshold    0.99                yours
```

So you can keep a threshold you tuned while still getting a comparator that suits
the type. A `comparator_config` works the same way: it is merged over the inferred
one, so setting `absolute_tolerance` keeps the `relative_tolerance` inference chose.
Both paths do this, from the same input --
`x-aws-stickler-comparator-config` beside `x-aws-stickler-comparator: "auto"`
merges exactly as `comparator_config` beside `"comparator": "auto"` does.

Naming a **comparator** is different, and pins the threshold too:

```json
{"total": {"type": "float", "comparator": "ExactComparator"}}
```

```
comparator   ExactComparator     yours
threshold    0.5                 the ordinary default, not inference's 0.95
```

A threshold only means something next to the metric that produced the score: 0.85
is one thing on edit distance and another on numeric tolerance. Inference's
threshold belongs to the comparator inference would have picked, so it is not
carried over onto one you chose instead. Set the threshold yourself when you name a
comparator.

### It is off by default

Enabling it changes scores for any field you left unspecified: a `float` compared as
text starts being compared as a number, which is usually what you want but is still
a change to your reported metrics. Turn it on deliberately, check `explain()`, and
re-baseline.

!!! note "Date fields and `model_from_json()`"

    `model_from_json()` accepts `str`, `int`, `float`, `bool`, `list` and `dict`;
    there is no `date` type. A date field declared as `str` there does **not** get
    `DateComparator` -- inference will not apply a comparator its declared type
    cannot support, and says so in `why`:

    ```
    ['type:str -> LevenshteinComparator@0.7',
     'name-token-unused:issued_date matched DateComparator but type str is
      incompatible; keeping type default']
    ```

    `source` for such a field is `type`, not `name-token`: the name matched a rule,
    but the type default is what shipped.

    Name the comparator explicitly (`"comparator": "DateComparator"`), or use the
    JSON Schema path with `{"type": "string", "format": "date"}`, which carries the
    date semantics in the type.

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
| `Field 'x' with primitive type 'str' requires a 'comparator'` | Primitive field without a comparator | Add a `"comparator"` key, or set `"infer_unspecified_fields": true` (see [Inferring unspecified fields](#inferring-unspecified-fields)) |
| `Field 'x' threshold must be between 0.0 and 1.0, got 5.0` | Threshold outside the range | Use a value in 0.0--1.0 |
| `Unknown comparator: 'x'. Available: [...]` | Comparator name not registered | Use a class name from the list; `BERTComparator` and `LLMComparator` appear only when the `[bert]` and `[llm]` extras are installed |
| `Field 'x' with type 'structured_model' requires a 'fields' configuration` | Nested model without its own `fields` | Add the nested `fields` block |

## See Also

- [Model Export](model-export.md) -- exporting models back to JSON Schema or Stickler config
- [Hungarian Matching](hungarian-matching.md) -- how list fields are compared
