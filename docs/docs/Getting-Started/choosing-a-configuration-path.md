---
title: Choosing a Configuration Path
---

# Choosing a Configuration Path

Stickler has two ways to tell it how to compare your fields, and they are independent. They take
different inputs, infer different comparators, and score the same data differently. Neither accepts
the other's input, so this is a choice you make once per project.

| | Inference | JSON Schema |
|---|---|---|
| **Entry points** | `stickler.evaluate`, `stickler.eval_for`, `StructuredModel.from_pydantic` | `StructuredModel.from_json_schema` |
| **Input** | a live Pydantic `BaseModel` **class** | a schema **dict** |
| **Use it when** | you already have a model — an agent's `response_model`, an API contract | your config lives outside Python, or is edited without a deploy |

## What each one infers

| | Inference | JSON Schema |
|---|---|---|
| From the type | yes — `bool`/`Enum` → Exact, `int` → Numeric(exact), `float` → Numeric(rel_tol 0.001), `date` → Date, `str` → Levenshtein | yes, much coarser — a lookup on the declared `"type"` alone: `string` → Levenshtein, `number`/`integer` → Numeric, `boolean` → Exact. A type outside those four raises |
| From the field name | yes — `*_id` → Exact(case-sensitive), `notes` → Fuzzy(token_set), `total` → Numeric(rel_tol), `phone` → Phone | no |
| `"format": "date"` | n/a — a `date` annotation is already Date | **not read** — the property is a `string`, so it gets Levenshtein @ 0.5. `enum`, `const` and every other `format` are ignored the same way |
| Default threshold | per comparator, 0.6–1.0 | `0.5`, for every type |
| Introspection | `.explain()`, with a provenance trail per field | none; round-trip the class through `to_json_schema()` to read back what it chose |
| Takes the other's input | no — `eval_for({...})` raises `TypeError` | no — `from_json_schema(MyModel)` raises `ValueError` |

Weights are `1.0` on both paths. Inference never picks `SemanticComparator`, `BERTComparator`, or
`LLMComparator` — those need models or credentials, so they are explicit-only.

The schema path keeps no separate record of what it chose, so read it back off the exported schema.
This is worth doing on any schema whose strings are not free text, because `format` and `enum` are
silently ignored:

```python
s = {"type": "object", "properties": {
    "d": {"type": "string", "format": "date"},
    "e": {"type": "string", "enum": ["ALPHA", "BETA"]},
    "plain": {"type": "string"}}}
props = StructuredModel.from_json_schema(s).to_json_schema()["properties"]
{k: (v["x-aws-stickler-comparator"], v["x-aws-stickler-threshold"]) for k, v in props.items()}
# {'d':     ('LevenshteinComparator', 0.5),
#  'e':     ('LevenshteinComparator', 0.5),
#  'plain': ('LevenshteinComparator', 0.5)}
```

## The divergence is not academic

Six fields, one dataset, both paths. Same names, same types, no `x-aws-stickler-*` extensions:

```python
class Invoice(BaseModel):
    invoice_id: str
    customer_name: str
    notes: Optional[str] = None
    issue_date: date
    total: float
    quantity: int
```

| Field | Inference | JSON Schema |
|---|---|---|
| `invoice_id` | `ExactComparator` @ 1.0 | `LevenshteinComparator` @ 0.5 |
| `customer_name` | `LevenshteinComparator` @ 0.85 | `LevenshteinComparator` @ 0.5 |
| `notes` | `FuzzyComparator` @ 0.6 | `LevenshteinComparator` @ 0.5 |
| `issue_date` | `DateComparator` @ 0.95 | `LevenshteinComparator` @ 0.5 |
| `total` | `NumericComparator` @ 0.95 | `NumericComparator` @ 0.5 |
| `quantity` | `NumericComparator` @ 1.0 | `NumericComparator` @ 0.5 |

Three of six comparators and **six of six** thresholds differ. The schema here is
`Invoice.model_json_schema()`, so `issue_date` carries `"format": "date"` — and the schema path
ignores it, scoring the field as a string. That is worth a whole field whenever the two sides
serialize differently: on `"2024-01-05"` against `"Jan 5, 2024"`, inference scores **1.0** where the
schema path scores **0.0**. The scored pair below holds `issue_date` identical, so it happens to
contribute `1.0` on both paths and is not one of the fields that diverges there — which is exactly
how this class of error stays invisible in a spot check.
`customer_name` is the subtle case: the same comparator on both paths,
separated only by the threshold. Scoring one realistic prediction
(`"inv-001"` for `"INV-001"`, `"Acme Corp."` for `"Acme Corporation"`, `"friday delivery"` for
`"deliver by friday"`) gives **0.646** by inference and **0.760** by schema — and the two disagree
in *opposite directions* on three fields:

| Field | Inference | JSON Schema | Why |
|---|---|---|---|
| `invoice_id` | 0.000 | 1.000 | Exact is case-sensitive for identifiers; Levenshtein normalizes case first |
| `notes` | 0.875 | 0.000 | Fuzzy `token_set_ratio` ignores word order; Levenshtein scores the reorder below 0.5 and clips |
| `customer_name` | 0.000 | 0.562 | same comparator, but 0.562 falls under inference's 0.85 threshold and is clipped |

Neither number is wrong. Inference encodes what the field *means* — an ID that differs in case may
be a different ID, and reordered free text usually is not an error. The schema path reads only the
declared type: it cannot tell an identifier from a date from a paragraph of notes, since all three
are `"string"`. That is what `x-aws-stickler-comparator` is for. Pick the path that matches the
question you are asking, and do not compare scores across the two.

## Moving between them

`StructuredModel.from_pydantic(cls).to_json_schema()` exports inference's decisions *as* a schema —
comparators, thresholds, nullable fields as `["string", "null"]`, and the `required` list. Use it to
start from a Pydantic model and hand the result to a config-driven pipeline, or to see the schema
that would reproduce what inference chose.

## Next

- [Ultra Quick Start](ultra-quick-start.md) — the inference path end to end
- [Dynamic Model Creation](../Advanced/dynamic-models.md) — the JSON Schema path in full
- [Model Export](../Advanced/model-export.md) — round-tripping models and schemas
- [Thresholds and Metrics](thresholds-and-metrics.md) — what a threshold changes once one is chosen
