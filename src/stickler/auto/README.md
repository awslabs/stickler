# `stickler.auto`: zero-config evaluation of vanilla pydantic models

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

The two pre-existing entry points lose information from real pydantic models:

- `StructuredModel.from_json_schema(Model.model_json_schema())` supports
  nullable `Optional` schemas, but not general multi-type unions, and silently
  degrades enums and `datetime` to bare strings.
- Unannotated `StructuredModel` fields fall through
  `configuration_helper.py`'s type-blind default and get compared with
  `LevenshteinComparator`, which is wrong for `float`, `bool`, `date`.

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

The builder is also exposed as **`StructuredModel.from_pydantic(cls)`**,
sibling to `from_json_schema()` / `model_from_json()`. That is the
integration point with the rest of stickler: the returned class is an
ordinary `StructuredModel`, so `BulkStructuredModelEvaluator`, HTML reports,
`to_stickler_config()` (export → edit → rebuild), and every other consumer
work on it unchanged. One caveat on that round-trip: the exported config
carries comparators, thresholds, and weights but not the `BeforeValidator`
wire-normalizers, so a rebuilt class needs JSON-form input
(`model_dump(mode="json")`) and cannot round-trip `dict` or `set` fields. `evaluate()`/`eval_for()` are conveniences layered on
top for the single-pair case; there is no bulk-specific API in this package
by design.

Pipeline: `model_fields` → infer `InferredSpec` per field → `ComparableField`
tuples → `create_model_from_fields` (cached) → `from_json(model_dump(mode="json"))`
for gt & pred → `compare_with(include_confusion_matrix=True)` → `EvalResult`.

A single `compare_with(include_confusion_matrix=True, add_derived_metrics=True)`
call yields both `field_scores`/`overall_score` **and** precision/recall/f1/accuracy
(under `confusion_matrix.overall.derived.cm_*`), so no second comparison is needed.

## Inference precedence (per field, first match wins)

1. **Type signal** (always on, safe), after unwrapping `Optional[X]` /
   `Union[X, None]` / PEP 604 `X | None`:

   | Python type | Comparator | Threshold | clip |
   |---|---|---|---|
   | `bool` | Exact | 1.0 | yes |
   | `Enum` / `Literal` | Exact | 1.0 | yes |
   | `int` | Numeric (exact) | 1.0 | yes |
   | `float` | Numeric (`relative_tolerance=0.001`) | 0.95 | yes |
   | `datetime` | Date (`tolerance=1min`, so timestamps hours apart do not match) | 0.95 | yes |
   | `date` | Date (same calendar day) | 0.95 | yes |
   | `str` | Levenshtein | 0.7 | yes |
   | nested `BaseModel` | recurse → child shadow model | `match_threshold` | **no** (partial credit, consistent with the same object inside a list) |
   | nested `StructuredModel` | used as configured, never re-inferred | (its own) | (its own) |
   | `List[BaseModel]` | Hungarian object matching | (element `match_threshold`) | n/a |
   | `List[primitive]` | element comparator | element | yes |
   | `dict` / `tuple` / `set` / `Any` / multi-arm unions | Exact over a canonical JSON string (sorted keys; sets also sort elements) | 1.0 | n/a (Exact is all-or-nothing) |

2. **Name-token refinement**, gated on type compatibility (comparator/threshold
   on by default; **weight only when `weight_hints=True`**). Tokens are matched
   as whole words in the snake/camelCase-split field name; trailing digits are
   shed (`isbn13`→`isbn`) and simple plurals singularized (`ids`→`id`).
   Highlights: `id/sku/code`→Exact (w 3.0), `amount/price/total`→Numeric\@0.95
   (w 2.5, numeric fields only), `email/url`→Exact,
   `zip/postal/postcode`→Exact (case-sensitive), `phone`→Phone\@1.0,
   `name/vendor/customer`→Levenshtein\@0.85,
   `date/due/created/...`→Date\@0.95 (date/datetime fields only),
   `address`→Fuzzy(token_sort), `notes/description/summary`→Fuzzy(token_set)\@0.6
   with `clip=False` (partial credit). Ambiguous tokens
   (`type/state/number/key/level/num`) are intentionally omitted so they keep
   the safe type default.

   **The type gates the token.** A rule whose comparator cannot parse the
   field's actual type never fires: `amount: str` keeps Levenshtein,
   `has_due_date: bool` keeps Exact, and the skipped refinement is recorded in
   provenance. Without the gate, a Date/Numeric comparator applied to
   unparseable values silently scores identical strings 0.0, which is the worst
   failure mode a metrics API can have. `bool`/`Enum`/`Literal` are never
   token-refined at all (nothing sharpens an exact match).

   **The gate is on the declared type, not the value.** `phone: str` passes the
   gate, so the gate cannot catch a *value* that fails to parse -- which is how
   `PhoneComparator` came to score identical unparseable numbers 0.0 through
   this path (#258). Any token rule whose comparator can reject an individual
   value therefore needs a same-value fallback of its own; the type gate is not
   enough on its own.

Every decision is recorded in `InferredSpec.provenance` and surfaced by
`EvalResult.explain()` / `EvalSpec.explain()`.

## Honesty rules

- **Weights default to 1.0.** Business-criticality is not encoded in a vanilla
  model; guessing it silently would skew precision/recall. Opt in with
  `weight_hints=True`.
- **Semantic / BERT / LLM are never auto-selected.** They need embeddings/creds
  and would surprise as a silent default.
- **Unavailable comparators degrade.** If `rapidfuzz` (Fuzzy) or
  `python-dateutil` (Date) is missing, the field falls back to Levenshtein/Exact
  and the degrade is recorded in provenance.

## Wire-form normalization

Instances are normalized before comparison: `date`/`datetime` become ISO
strings and enums become their values, matching the wire types the inferred
comparators expect. Fields whose JSON form is not a string scalar (`dict`,
`tuple`, `set`, `Any`, multi-arm unions, `IntEnum`, int `Literal`, `Decimal`)
canonicalize to a deterministic form (via `pydantic_core.to_jsonable_python`
then sorted-key JSON), so key order, container spelling, and dump mode never
affect scores. Both `model_dump()` (native `date`/`Decimal`/`set` objects) and
`model_dump(mode="json")` (already-serialized) validate and score identically,
so `Model.from_json(instance.model_dump())` is safe either way.

Nullability of a shadow field comes from the **source annotation**
(`Optional[X]` / `X | None` / `Any`), not from required-ness: a required
`Optional[str]` accepts `None` as a value, and omitting it is still an error
in the user's own model.

## Unsupported shapes (loud errors, not crashes)

These raise `TypeError` with an actionable message at `eval_for` time:

- **Recursive models** (self-referential or mutually recursive): a shadow
  class would need itself as a field type before it exists.
- **Empty models** (no fields to score).
- **A field named `extra_fields`** (reserved by the comparison engine).

## Known limitations

- **`List[BaseModel]` semantics** use Hungarian matching (order-independent),
  which differs from `anls_score`'s ordering-based path. This is intentional and
  documented so scores are explainable.
- **Inference heuristics may be tuned across releases**, which can change
  scores for unconfigured models. Benchmark against a pinned stickler version,
  or graduate to an explicit `StructuredModel` (whose configuration this layer
  never overrides) for long-lived comparisons.
- **Extending later.** The `InferredSpec` config dict is the single contract; an
  agentic configurator (future) would be just another producer of that dict,
  seeded by this deterministic inference as its prior. Not implemented here.
