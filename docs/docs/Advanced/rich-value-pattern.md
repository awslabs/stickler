---
title: Rich Value Pattern
---

# Rich Value Pattern

The Rich Value Pattern lets fields carry metadata alongside their actual values. Instead of just `"Widget"`, a field can be `{"_value": "Widget", "_confidence": 0.95}`, wrapping the value with confidence scores, bounding boxes, or any custom metadata your pipeline produces.

## The Convention

A rich value is any JSON dict with a `_value` key. The underscore prefix prevents collision with user data.

```json
{
  "invoice_number": {"_value": "INV-2024-001", "_confidence": 0.97},
  "vendor": {"_value": "Acme Corp", "_bbox": [0.1, 0.2, 0.3, 0.4]},
  "total": {"_value": 1247.50},
  "notes": "Delivered to front door"
}
```

- `_value` (required): the actual field value, unwrapped into the model field
- `_confidence` (optional): model confidence score, consumed by the confidence evaluation module
- Any other keys (optional): stored as extras, accessible via `get_field_extras()`
- Plain values (no wrapper): still work, no change needed

## How It Works

`from_json()` walks the JSON tree. When it finds a dict with `_value`, it:

1. Extracts `_value` as the field value for the model
2. Extracts `_confidence` (if present) into the confidence accessor
3. Stores everything else as extras for that field path

After unwrapping, the model behaves like a normal pydantic model. `model_dump()`, validation, and comparison all work on the unwrapped values.

```python
from stickler import StructuredModel, ComparableField

class Invoice(StructuredModel):
    invoice_number: str = ComparableField()
    vendor: str = ComparableField()
    total: float = ComparableField()

pred = Invoice.from_json({
    "invoice_number": {"_value": "INV-001", "_confidence": 0.97},
    "vendor": {"_value": "Acme Corp", "_handwritten": True, "_source_page": 2},
    "total": 1247.50
})

# Values are unwrapped
pred.invoice_number          # "INV-001"
pred.vendor                  # "Acme Corp"
pred.total                   # 1247.50

# Confidence is accessible
pred.get_field_confidence("invoice_number")  # 0.97
pred.get_field_confidence("total")           # None (no rich value)

# Extras are accessible
pred.get_field_extras("vendor")              # {"_handwritten": True, "_source_page": 2}
pred.get_all_extras()                        # {"vendor": {"_handwritten": True, "_source_page": 2}}
```

## Nested Objects and Lists

Rich values work at any nesting depth. Paths use dot notation for nested objects and bracket notation for list items.

```json
{
  "customer": {
    "name": {"_value": "Jane Smith", "_confidence": 0.96},
    "address": {
      "street": {"_value": "123 Main St", "_confidence": 0.85},
      "city": "Boston"
    }
  },
  "items": [
    {
      "product": {"_value": "Laptop", "_confidence": 0.89},
      "price": {"_value": 1299.99, "_bbox": [0.3, 0.4, 0.5, 0.1]}
    }
  ]
}
```

```python
pred.get_field_confidence("customer.address.street")  # 0.85
pred.get_field_extras("items[0].price")               # {"_bbox": [0.3, 0.4, 0.5, 0.1]}
```

## Wrapping Null Values

Rich values support explicit null predictions. This distinguishes "I predicted null" from "the field is missing entirely":

```json
{
  "notes": {"_value": null, "_confidence": 0.3}
}
```

The model field gets `None` as its value, and the confidence score is preserved.

## Value-Only Wrapping

You can wrap a value without any metadata. This is useful when some fields in your pipeline use rich values and you want a consistent format:

```json
{
  "name": {"_value": "Widget"},
  "price": {"_value": 29.99},
  "sku": "ABC123"
}
```

All three fields produce the same model values. The first two are unwrapped from rich values; the third is a plain value.

## What Gets Stored Where

**All keys inside a rich value wrapper must use an underscore prefix.** This is enforced with a warning at parse time. Non-prefixed keys will still be stored in extras, but a `UserWarning` is emitted to help you catch the issue early.

!!! note "Reserved namespace"
    Underscore-prefixed keys (`_*`) inside a rich value wrapper are a reserved namespace owned by stickler. Any dict containing a `_value` key is treated as a wrapper, and stickler may extract additional `_`-prefixed keys (`_confidence`, `_bbox`, `_source_span`, ...) into typed accessors as the schema grows. Don't ship `{"_value": x}` payloads where `_value` is meant to be user data — wrap once at the boundary, or use a different key name.

| Key in rich value | Where it goes | How to access it |
|---|---|---|
| `_value` | Model field value | `pred.field_name` |
| `_confidence` | Confidence store | `pred.get_field_confidence("field_name")` |
| `_bbox`, `_handwritten`, etc. | Extras store | `pred.get_field_extras("field_name")` |

## Detection Rule

A dict is a rich value if and only if it contains a `_value` key. The underscore prefix is chosen because:

- It's conventionally "private/system" in Python and JSON
- It's extremely unlikely to appear as a real data key
- It's visually distinct from user data keys

Dicts without `_value` are treated as regular nested data and passed through to pydantic unchanged.

### Deprecated legacy shape

The pre-rename `{"value": ..., "confidence": ...}` shape is still
recognized for one release so existing JSONL corpora keep their
confidence data on upgrade. Loading such a payload emits a
`DeprecationWarning` naming the offending field path and still unwraps
the value the same way as the new form. The legacy shape will be
removed in the next release — migrate payloads to `_value`/`_confidence`
as soon as you can. See the [CHANGELOG](../../../CHANGELOG.md) for the
full migration guide.

## Integration with Confidence Evaluation

The confidence evaluation module consumes `_confidence` values from rich values. See [Confidence Metrics](confidence-metrics.md) for AUROC, Brier Score, ECE, and Error Capture at Review Budget metrics.

## JSONL Serialization

When a prediction is created via `from_json()` with rich values, the original JSON is stored on the model instance. When `compare_with()` runs, this raw JSON is included in the result as `prediction_raw`. This enables the map/reduce pattern: compare individual documents, save results to JSONL, and later aggregate them via `update_from_comparison_result()` with full confidence (and future bbox) metric support.

See [Bulk Evaluation: Map/Reduce](../Guides/Evaluation/bulk-evaluation.md#mapreduce-single-doc-compare-bulk-aggregate) for the full pattern.

## Future Metadata Types

The pattern is designed for extensibility. When new metadata types are supported (bounding boxes for MAP evaluation, source spans for provenance tracking), they'll be promoted from extras to dedicated accessors (e.g., `get_field_bbox()`). Users who already store these in extras get a natural upgrade path.

## See Also

- [Confidence Metrics](confidence-metrics.md): evaluating confidence calibration quality
- [Confidence Evaluation Guide](../Guides/Evaluation/confidence-evaluation.md): practical guide for using confidence metrics
