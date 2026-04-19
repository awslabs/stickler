---
title: Confidence Metrics
---

# Confidence Metrics

Stickler's confidence module measures how well a model's self-reported confidence scores correlate with actual prediction correctness. It consumes confidence data from the **Rich Value Pattern** and supports pluggable metrics (AUROC, Brier Score, ECE), per-field breakdowns, and coverage tracking.

## Rich Value Pattern

A rich value is any JSON dict with a `"value"` key. Everything else is metadata. Confidence is one type of metadata, but it's optional. This pattern also supports bounding boxes, source spans, and other future metadata types.

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
  "name": {"value": "Widget", "confidence": 0.95},
  "price": {"value": 29.99, "confidence": 0.8}
}
```

### Rich Value without Confidence

```json
{
  "name": {"value": "Widget", "bbox": [0.1, 0.2, 0.3, 0.4]},
  "price": {"value": 29.99}
}
```

### Mixed Format

Fields with and without rich values can coexist:

```json
{
  "name": {"value": "Widget", "confidence": 0.95},
  "price": 29.99,
  "sku": {"value": "ABC123", "bbox": [0.1, 0.2, 0.3, 0.4]}
}
```

### Nested Structures

Rich values work with nested objects and arrays:

```json
{
  "customer": {
    "name": {"value": "John Doe", "confidence": 0.92},
    "address": {
      "street": {"value": "123 Main St", "confidence": 0.85},
      "city": "New York"
    }
  },
  "items": [
    {
      "product": {"value": "Laptop", "confidence": 0.89},
      "price": {"value": 1299.99, "confidence": 0.76}
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
    "name": {"value": "Widget Pro", "confidence": 0.95},
    "price": {"value": 29.99, "confidence": 0.8},
    "sku": {"value": "XYZ789", "confidence": 0.3}
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
