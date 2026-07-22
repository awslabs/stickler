# `stickler.auto` — zero-config evaluation of vanilla pydantic models

This package lets a caller evaluate structured output **without** defining a
`StructuredModel`, a JSON schema, or any `x-aws-stickler-*` annotation. It is the
implementation behind the top-level `stickler.evaluate` / `stickler.eval_for`.

```python
import stickler
# `pred` is a plain pydantic instance, e.g. a Strands agent response_model
pred = agent.structured_output(Invoice, "Extract the invoice: ...")
result = stickler.evaluate(ground_truth, pred)
result.overall_score, result.precision, result.recall, result.f1, result.field_scores
result.explain()   # per-field {comparator, threshold, weight, source, why}
```

## Why it exists

The two pre-existing entry points fail on real pydantic models:

- `StructuredModel.from_json_schema(Model.model_json_schema())` **crashes** on
  `Optional`/`Union` (`anyOf` with no `type`), and silently degrades enums and
  `datetime` to bare strings.
- Unannotated `StructuredModel` fields fall through
  `configuration_helper.py`'s type-blind default and get compared with
  `LevenshteinComparator` — wrong for `float`, `bool`, `date`.

`auto` walks the **live** `cls.model_fields` (keeping every python-type signal)
and infers a real per-field comparator, so "unconfigured" already means
"reasonably configured."

## Architecture

Three modules, one pipeline:

| Module | Responsibility |
|---|---|
| `inference.py` | `infer_field_config(name, field_info)` → `InferredSpec`. Pure. The type table, name-token table, Optional-unwrap, and the comparator availability gate. |
| `builder.py` | `structured_model_for(cls)` → a cached `StructuredModel` subclass. Walks `model_fields`, recurses into nested models / lists, delegates class creation to `ModelFactory.create_model_from_fields`. |
| `facade.py` | `evaluate`, `eval_for`, `EvalSpec`, `EvalResult`. Normalizes instances via `model_dump(mode="json")`, runs `compare_with`, flattens the result. |

Pipeline: `model_fields` → infer `InferredSpec` per field → `ComparableField`
tuples → `create_model_from_fields` (cached) → `from_json(model_dump(mode="json"))`
for gt & pred → `compare_with(include_confusion_matrix=True)` → `EvalResult`.

A single `compare_with(include_confusion_matrix=True, add_derived_metrics=True)`
call yields both `field_scores`/`overall_score` **and** precision/recall/f1/accuracy
(under `confusion_matrix.overall.derived.cm_*`), so no second comparison is needed.

## Inference precedence (per field, first match wins)

1. **Override** — a `ComparableField` in `overrides={...}` is used verbatim.
2. **Type signal** (always on, safe) — after unwrapping `Optional`/`Union[X, None]`:

   | Python type | Comparator | Threshold | clip |
   |---|---|---|---|
   | `bool` | Exact | 1.0 | yes |
   | `Enum` / `Literal` | Exact | 1.0 | yes |
   | `int` | Numeric (exact) | 1.0 | yes |
   | `float` | Numeric (`relative_tolerance=0.001`) | 0.95 | yes |
   | `date` / `datetime` | Date | 0.95 | yes |
   | `str` | Levenshtein | 0.7 | yes |
   | nested `BaseModel` | recurse → child shadow model | 0.9 | yes |
   | `List[BaseModel]` | Hungarian object matching | (element `match_threshold`) | — |
   | `List[primitive]` | element comparator | element | yes |
   | anything else | Levenshtein (str wire) | 0.7 | yes |

3. **Name-token refinement** (comparator/threshold on by default; **weight only
   when `weight_hints=True`**). Tokens are matched as whole words in the snake/
   camelCase-split field name. Highlights: `id/sku/code`→Exact (w 3.0),
   `amount/price/total`→Numeric\@0.95 (w 2.5), `email/url/zip`→Exact,
   `name/vendor/customer`→Levenshtein\@0.85, `address`→Fuzzy(token_sort),
   `notes/description/summary`→Fuzzy(token_set)\@0.6 with `clip=False` (partial
   credit). Ambiguous tokens (`type/state/number/key/level`) are intentionally
   omitted so they keep the safe type default.

Every decision is recorded in `InferredSpec.provenance` and surfaced by
`EvalResult.explain()` / `EvalSpec.explain()`.

## Honesty rules

- **Weights default to 1.0.** Business-criticality is not encoded in a vanilla
  model; guessing it silently would skew precision/recall. Opt in with
  `weight_hints=True`.
- **Semantic / BERT / LLM are never auto-selected.** They need embeddings/creds
  and would surprise. A field only gets them via explicit override.
- **Unavailable comparators degrade.** If `rapidfuzz` (Fuzzy) or
  `python-dateutil` (Date) is missing, the field falls back to Levenshtein/Exact
  and the degrade is recorded in provenance.

## Wire-form normalization

Instances are compared as `model_dump(mode="json")`: `date`/`datetime` become
ISO strings and enums become their values, matching the wire types the inferred
comparators expect. Shadow fields for those types are therefore declared as
`str`. `mode="json"` (not plain `model_dump()`) is mandatory — a native `date`
would otherwise score 0.

## Known limitations

- **Directly self-referential models** (a field whose type is the class itself)
  degrade to a generic structured comparison at that edge; pydantic cannot build
  a truly recursive dynamic class in one pass.
- **`List[BaseModel]` semantics** use Hungarian matching (order-independent),
  which differs from `anls_score`'s ordering-based path — this is intentional and
  documented so scores are explainable.
- **Extending later.** The `InferredSpec` config dict is the single contract; an
  agentic configurator (future) would be just another producer of that dict,
  seeded by this deterministic inference as its prior. Not implemented here.
