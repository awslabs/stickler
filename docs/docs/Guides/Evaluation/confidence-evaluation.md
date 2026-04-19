---
title: Confidence Evaluation
---

# Confidence Evaluation Guide

How to evaluate whether your model's confidence scores are trustworthy, using Stickler's pluggable confidence metrics.

## Quick Start

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import BulkStructuredModelEvaluator
from stickler.structured_object_evaluator.models.confidence import (
    AUROCMetric, BrierScoreMetric, ECEMetric, ErrorCaptureAtBudgetMetric,
)

evaluator = BulkStructuredModelEvaluator(
    target_schema=Invoice,
    confidence_metrics=[
        AUROCMetric(),
        BrierScoreMetric(),
        ErrorCaptureAtBudgetMetric(budgets=[0.10, 0.30, 0.50]),
    ]
)

for gt, pred in dataset:
    evaluator.update(gt, pred)

results = evaluator.compute()

# Statistical: is confidence useful?
print(results.confidence_metrics["overall"]["auroc"]["value"])

# Practical: how useful?
ecab = results.confidence_metrics["overall"]["error_capture_at_budget"]
for budget, data in ecab["budgets"].items():
    print(f"Review {budget:.0%} of data: catch {data['pct_errors_caught']:.0%} of errors "
          f"({data['gain']:.1f}x vs random)")

# How much data has confidence?
print(results.confidence_metrics["coverage"])
```

## Rich Value Pattern

Predictions carry confidence via the Rich Value Pattern. Any JSON dict with a `"value"` key is a rich value. Confidence is optional metadata:

```json
{
  "invoice_id": {"value": "INV-001", "confidence": 0.97},
  "vendor": {"value": "Acme Corp"},
  "total": 1247.50
}
```

`from_json()` unwraps rich values automatically. Fields without confidence are skipped by the confidence module but still compared normally.

## Available Metrics

| Metric | What it measures | Key output |
|--------|-----------------|------------|
| `AUROCMetric` | Can confidence separate correct from incorrect? | `{"value": 0.85}` |
| `BrierScoreMetric` | Mean squared calibration error | `{"value": 0.12}` |
| `ECEMetric` | Expected calibration error with bin data | `{"value": 0.08, "bins": [...]}` |
| `ErrorCaptureAtBudgetMetric` | Errors caught at X% review effort | `{"value": 3.2, "budgets": {...}}` |

## Error Capture at Review Budget

The business metric. Sort fields by confidence (lowest first), review the bottom X%, count errors found:

```
Review 10% of data: catch 55% of errors (5.5x vs random)
Review 30% of data: catch 89% of errors (3.0x vs random)
Review 50% of data: catch 97% of errors (1.9x vs random)
```

A gain of 5.5x at 10% means confidence-guided review finds 5.5 times more errors than random sampling at the same effort level.

## Per-Field Breakdown

All metrics are computed at both overall and per-field levels:

```python
for field, metrics in results.confidence_metrics["fields"].items():
    auroc = metrics["auroc"]["value"]
    print(f"{field}: AUROC={auroc}")
```

This tells you which field types benefit most from confidence-guided review.

## Coverage

Not every field has confidence. Coverage tells you how much of your data is being evaluated:

```python
cov = results.confidence_metrics["coverage"]
# {"fields_with_confidence": 200, "fields_total": 400, "ratio": 0.5}
```

## Adding Custom Metrics

Subclass `ConfidenceMetric`:

```python
from stickler.structured_object_evaluator.models.confidence import ConfidenceMetric, ConfidencePairs

class MyMetric(ConfidenceMetric):
    @property
    def name(self) -> str:
        return "my_metric"

    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        # pairs: list of ConfidencePair(is_match, confidence, similarity)
        return {"value": ...}
```

## Further Reading

- [Confidence Metrics (Advanced)](../../Advanced/confidence-metrics.md): full technical reference
- [Bulk Evaluation](bulk-evaluation.md): the bulk evaluator that powers dataset-level confidence metrics
