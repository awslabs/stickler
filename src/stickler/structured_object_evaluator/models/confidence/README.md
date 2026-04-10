# Confidence Evaluation Module

Measures how well a model's self-reported confidence scores correlate with actual prediction correctness.

This module consumes confidence data extracted by the **Rich Value Pattern**. A rich value is any JSON dict with a `"value"` key (e.g., `{"value": "Widget", "confidence": 0.95}`). The `RichValueHelper` unwraps these during `from_json()`, and this module evaluates the confidence slice of that metadata. Confidence is optional in rich values; fields without it are silently skipped by this module.

## Architecture

```
confidence/
├── __init__.py       # Public API re-exports
├── metrics.py        # ConfidenceMetric base class + AUROC, Brier, ECE implementations
├── calculator.py     # ConfidenceCalculator: extracts pairs, runs metrics
└── README.md
```

`ConfidenceCalculator` is the orchestrator. It joins `field_comparisons` (from `compare_with`) with confidence data (from `from_json`) to produce `ConfidencePair` objects keyed by field path, then runs configured metrics at overall and per-field levels.

## ConfidencePair

```python
class ConfidencePair(BaseModel):
    is_match: bool      # Did the field cross its ComparableField threshold?
    confidence: float   # Model's self-reported confidence (from JSON)
    similarity: float   # Raw comparator similarity score (0.0–1.0)
```

`is_match` is threshold-gated: `raw_score >= ComparableField.threshold`. This means a field with similarity 0.65 and threshold 0.7 is `is_match=False`, even though it's partially correct. The `similarity` field preserves the continuous score for future metrics that want to correlate confidence with degree of correctness rather than binary match.

## Adding a New Metric

```python
from stickler.structured_object_evaluator.models.confidence.metrics import (
    ConfidenceMetric, ConfidencePairs,
)

class MyMetric(ConfidenceMetric):
    @property
    def name(self) -> str:
        return "my_metric"

    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        if not pairs:
            return {"value": None}
        # Your computation here using p.is_match, p.confidence, p.similarity
        value = ...
        return {"value": value}  # Add extra keys for structured data (bins, curves, etc.)
```

Then pass it in:

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import BulkStructuredModelEvaluator

evaluator = BulkStructuredModelEvaluator(
    target_schema=Invoice,
    confidence_metrics=[AUROCMetric(), MyMetric()]
)
```

## Result Structure

```python
{
    "overall": {
        "auroc": {"value": 0.85},
        "brier_score": {"value": 0.12},
        "ece": {"value": 0.08, "bins": [...]},
    },
    "fields": {
        "vendor": {"auroc": {"value": 0.78}, ...},
        "total": {"auroc": {"value": None}, ...},  # None = single class
    },
    "coverage": {
        "fields_with_confidence": 200,
        "fields_total": 400,
        "ratio": 0.5
    }
}
```

Each metric returns `{"value": float | None}` with optional extra keys. `None` means the metric couldn't be computed (no data, single class for AUROC, etc.).

## Coverage

Not every field has a confidence score. Coverage tracks how many fields were evaluated vs. total fields compared. A ratio of 0.5 means half your fields lack confidence data, and the metrics only reflect the half that has it.

## Field Path Conventions

- Flat fields: `vendor`, `total`
- Nested objects: `address.street`, `contact.address.city`
- List items: `items[0].product`, `items[1].price`

List indices are prediction-side indices. Hungarian matching may reorder items, but the confidence paths always use the original prediction JSON indices.

## State Management

`BulkStructuredModelEvaluator` accumulates keyed pairs across documents. These survive:
- `get_state()` / `load_state()` for checkpointing
- `merge_state()` for distributed evaluation across workers

Coverage counters are also serialized and merged.

## Future Work

- Similarity-based metrics: correlate confidence with continuous similarity rather than binary match
- Hierarchical grouping: compute metrics for field groups (e.g., all `address.*` fields)
- Per-field-type aggregation: metrics grouped by comparator type
- Calibration curve visualization utilities
