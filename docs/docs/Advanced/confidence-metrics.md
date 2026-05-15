---
title: Confidence Metrics
---

# Confidence Metrics

Stickler's confidence module measures how well a model's self-reported confidence scores correlate with actual prediction correctness. It consumes confidence data from the **Rich Value Pattern** and supports pluggable metrics (AUROC, Brier Score, ECE), per-field breakdowns, and coverage tracking.

## Rich Value Pattern

A rich value is any JSON dict with a `"_value"` key. Everything else is metadata. Confidence is one type of metadata, but it's optional. This pattern also supports bounding boxes, source spans, and other future metadata types.

### Standard Format (no metadata)

```json
{
  "name": "Widget",
  "price": 29.99
}
```

### Rich Value with Confidence

```json
{
  "name": {"_value": "Widget", "_confidence": 0.95},
  "price": {"_value": 29.99, "_confidence": 0.8}
}
```

### Rich Value without Confidence

```json
{
  "name": {"_value": "Widget", "_bbox": [0.1, 0.2, 0.3, 0.4]},
  "price": {"_value": 29.99}
}
```

### Mixed Format

Fields with and without rich values can coexist:

```json
{
  "name": {"_value": "Widget", "_confidence": 0.95},
  "price": 29.99,
  "sku": {"_value": "ABC123", "_bbox": [0.1, 0.2, 0.3, 0.4]}
}
```

### Nested Structures

Rich values work with nested objects and arrays:

```json
{
  "customer": {
    "name": {"_value": "John Doe", "_confidence": 0.92},
    "address": {
      "street": {"_value": "123 Main St", "_confidence": 0.85},
      "city": "New York"
    }
  },
  "items": [
    {
      "product": {"_value": "Laptop", "_confidence": 0.89},
      "price": {"_value": 1299.99, "_confidence": 0.76}
    }
  ]
}
```

## Usage

### Creating Models with Confidence Data

```python
from stickler import StructuredModel, ComparableField

class Product(StructuredModel):
    name: str = ComparableField()
    price: float = ComparableField()
    sku: str = ComparableField()

ground_truth = Product(name="Widget Pro", price=29.99, sku="ABC123")

prediction = Product.from_json({
    "name": {"_value": "Widget Pro", "_confidence": 0.95},
    "price": {"_value": 29.99, "_confidence": 0.8},
    "sku": {"_value": "XYZ789", "_confidence": 0.3}
})
```

### Single-Document Confidence Metrics

```python
result = ground_truth.compare_with(
    prediction,
    add_confidence_metrics=True,
    document_field_comparisons=True
)

# Structured result with overall, per-field, and coverage
print(result["confidence_metrics"]["overall"])
# {"auroc": {"value": 1.0}}

print(result["confidence_metrics"]["fields"])
# {"name": {"auroc": {"value": None}}, "price": {"auroc": {"value": None}}, ...}

print(result["confidence_metrics"]["coverage"])
# {"fields_with_confidence": 3, "fields_total": 3, "ratio": 1.0}
```

Both `add_confidence_metrics=True` and `document_field_comparisons=True` are required.

### Bulk Evaluation (Recommended)

Per-document AUROC with 3-5 fields is noisy and often returns `None` (single class). Dataset-level metrics are statistically meaningful:

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.confidence import (
    AUROCMetric, BrierScoreMetric, ECEMetric,
)

evaluator = BulkStructuredModelEvaluator(
    target_schema=Product,
    confidence_metrics=[AUROCMetric(), BrierScoreMetric(), ECEMetric(n_bins=10)]
)

for gt_json, pred_json in dataset:
    gt = Product(**gt_json)
    pred = Product.from_json(pred_json)
    evaluator.update(gt, pred)

results = evaluator.compute()
print(results.confidence_metrics["overall"]["auroc"]["value"])
print(results.confidence_metrics["coverage"]["ratio"])
```

### Accessing Confidence Scores

```python
# Individual field
prediction.get_field_confidence("name")        # 0.95

# Nested field (dot notation)
prediction.get_field_confidence("address.street")

# Array element (bracket notation)
prediction.get_field_confidence("items[0].product")

# All confidence scores
all_conf = prediction.get_all_confidences()    # dict of path -> float
```

## AUROC Calculation

AUROC treats confidence evaluation as a binary classification problem:

- **Positive class**: fields where prediction matches ground truth
- **Negative class**: fields where prediction does not match
- **Score**: the model's confidence value

| AUROC Range | Interpretation |
|-------------|---------------|
| 0.7 - 1.0 | Well calibrated. Confidence correlates with correctness. |
| ~0.5 | Random. Confidence provides no signal. |
| < 0.5 | Inversely calibrated. Confidence correlates with errors. |

AUROC returns `None` when all predictions match (no negative class) or all fail (no positive class).

## Error Capture at Review Budget

AUROC tells you confidence is useful. Error Capture at Review Budget tells you *how* useful in practical terms.

The question: "If I review X% of my data (lowest confidence first), what percentage of errors do I catch?"

```python
from stickler.structured_object_evaluator.models.confidence import (
    AUROCMetric, ErrorCaptureAtBudgetMetric,
)

evaluator = BulkStructuredModelEvaluator(
    target_schema=Product,
    confidence_metrics=[AUROCMetric(), ErrorCaptureAtBudgetMetric(budgets=[0.10, 0.30, 0.50])]
)

for gt, pred in dataset:
    evaluator.update(gt, pred)

results = evaluator.compute()
ecab = results.confidence_metrics["overall"]["error_capture_at_budget"]

for budget, data in ecab["budgets"].items():
    print(f"Review {budget:.0%} of data: catch {data['pct_errors_caught']:.0%} of errors "
          f"({data['gain']:.1f}x vs random)")
```

Example output:
```
Review 10% of data: catch 55% of errors (5.5x vs random)
Review 30% of data: catch 89% of errors (3.0x vs random)
Review 50% of data: catch 97% of errors (1.9x vs random)
```

The `gain` at each budget level is the ratio of errors caught by confidence-guided review vs. random sampling at the same review effort. A gain of 5.5x at 10% budget means reviewing the bottom 10% by confidence finds 5.5 times more errors than reviewing a random 10%.

## Coverage

Not every field has a confidence score. Coverage tells you how much of your data is being evaluated:

```python
cov = results.confidence_metrics["coverage"]
print(f"{cov['fields_with_confidence']}/{cov['fields_total']} ({cov['ratio']:.0%})")
```

Fields without confidence are silently skipped by the confidence module.

## Adding Custom Metrics

Subclass `ConfidenceMetric` and implement `name` and `compute()`:

```python
from stickler.structured_object_evaluator.models.confidence import (
    ConfidenceMetric, ConfidencePairs,
)

class ConfidenceSimilarityCorrelation(ConfidenceMetric):
    @property
    def name(self) -> str:
        return "conf_sim_correlation"

    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        if len(pairs) < 2:
            return {"value": None}
        from scipy.stats import pearsonr
        confs = [p.confidence for p in pairs]
        sims = [p.similarity for p in pairs]
        corr, pvalue = pearsonr(confs, sims)
        return {"value": corr, "pvalue": pvalue}
```

Each `ConfidencePair` has three fields:

- `is_match` (bool): whether the field crossed its `ComparableField` threshold
- `confidence` (float): the model's self-reported confidence from JSON
- `similarity` (float): the raw comparator similarity score (0.0 to 1.0)

## Custom Accumulators

A `ConfidenceMetric` plugs into the existing confidence pipeline. If you need to evaluate something the confidence pipeline can't model — e.g. mean Average Precision over bounding boxes, source-span attribution quality, anything that consumes its own metadata key from the rich value pattern — implement a `PostComparisonAccumulator` instead.

`BulkStructuredModelEvaluator` holds a list of accumulators. Each one sees every comparison result and the prediction's raw JSON, accumulates its own state, and produces its own block of aggregate metrics at `compute()` time. The built-in `ConfidenceAccumulator` is one of them; yours runs alongside it.

```python
from typing import Any, Dict, Optional
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.confidence.accumulator import (
    ConfidenceAccumulator,
)
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)


class FieldCountAccumulator(PostComparisonAccumulator):
    """Toy accumulator: counts total fields seen across all documents."""

    def __init__(self) -> None:
        self.reset()

    @property
    def name(self) -> str:
        # Keys this accumulator's block in compute().accumulator_metrics.
        # Must be unique across the accumulator list.
        return "field_count"

    def reset(self) -> None:
        self._total = 0

    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        self._total += len(comparison_result.get("field_comparisons", []))

    def compute(self) -> Optional[Dict[str, Any]]:
        return {"total_fields": self._total} if self._total else None

    def get_state(self) -> Dict[str, Any]:
        return {"total": self._total}

    def load_state(self, state: Dict[str, Any]) -> None:
        self._total = int(state.get("total", 0))

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        self._total += int(other_state.get("total", 0))


# Pass an explicit accumulators list. ConfidenceAccumulator only runs if
# you include it — `accumulators=` and `confidence_metrics=` are mutually
# exclusive.
evaluator = BulkStructuredModelEvaluator(
    target_schema=Product,
    accumulators=[
        ConfidenceAccumulator(),
        FieldCountAccumulator(),
    ],
)

for gt, pred in dataset:
    evaluator.update(gt, pred)

result = evaluator.compute()
print(result.confidence_metrics["overall"])           # from ConfidenceAccumulator
print(result.accumulator_metrics["field_count"])      # from FieldCountAccumulator
```

### Interface contract

| Method | Purpose |
|--------|---------|
| `name` | Unique key in `compute().accumulator_metrics`. Two accumulators sharing a name raise `ValueError` at constructor time. |
| `reset()` | Clear accumulated state. Called from `BulkStructuredModelEvaluator.reset()`. |
| `accumulate(comparison_result, prediction_raw)` | Process one document. Called once per `update()` / `update_from_comparison_result()`. |
| `compute()` | Return aggregate metrics, or `None` if no data was seen. |
| `get_state()` / `load_state()` / `merge_state()` | Required for checkpointing and distributed evaluation. State must be JSON-serializable. |

### What's in `comparison_result` and `prediction_raw`

- `comparison_result["field_comparisons"]` — list of per-field rows with `actual_key`, `match`, `score`. Available when `compare_with()` was called with `document_field_comparisons=True` (which `BulkStructuredModelEvaluator` always does internally).
- `comparison_result["confusion_matrix"]` — overall and per-field TP/FP/TN/FN counts.
- `prediction_raw` — the prediction's original JSON tree before rich value unwrapping. `None` for predictions built without `from_json()` (no rich value data was ever supplied). This is where you reach for custom metadata like `_bbox` or `_source_span`.

Use `RichValueHelper.process_rich_values(prediction_raw)` to walk `prediction_raw` and pick out your metadata key the same way `ConfidenceAccumulator` picks out `_confidence`.

### Errors are isolated

A bug in one accumulator can't corrupt another's state or the bulk confusion matrix. Each accumulator's `accumulate()` runs inside its own `try`/`except`; a failure is recorded as a separate error tagged with the accumulator's name and surfaced through `compute().errors`.

## Distributed Evaluation

Confidence pairs are included in state serialization and merging:

```python
wa = BulkStructuredModelEvaluator(target_schema=Product)
wb = BulkStructuredModelEvaluator(target_schema=Product)

# Process shards separately, then merge
wa.merge_state(wb.get_state())
results = wa.compute()
```

## See Also

- [Classification Logic](classification-logic.md): how match/no-match is determined for each field
- [Aggregate Metrics](aggregate-metrics.md): hierarchical metric rollup
