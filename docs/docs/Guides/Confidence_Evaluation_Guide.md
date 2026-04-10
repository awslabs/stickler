---
title: Confidence Evaluation Guide
---

# Confidence Evaluation Guide

Stickler's confidence module measures how well a model's self-reported confidence scores correlate with actual prediction correctness. It consumes confidence data from the **Rich Value Pattern** (any JSON dict with a `"value"` key plus optional metadata like `"confidence"`). The module supports pluggable metrics (AUROC, Brier Score, ECE), works at both single-document and dataset levels, and reports per-field breakdowns with coverage tracking.

## Rich Values in JSON

Stickler uses the **Rich Value Pattern**: any JSON dict with a `"value"` key is treated as a rich value. The value is unwrapped into the model field, and metadata keys (like `"confidence"`) are stored separately. Confidence is optional. Plain values still work, and you can mix freely.

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, LevenshteinComparator

class Invoice(StructuredModel):
    invoice_number: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    vendor: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)
    total: float = ComparableField(threshold=0.95)

# Rich values with confidence
prediction = Invoice.from_json({
    "invoice_number": {"value": "INV-2024-001", "confidence": 0.97},
    "vendor": {"value": "Acme Corp", "confidence": 0.72},
    "total": 1247.50  # plain value, no metadata
})

# Rich values without confidence (e.g., for future bbox support)
prediction_no_conf = Invoice.from_json({
    "invoice_number": {"value": "INV-2024-001"},
    "vendor": {"value": "Acme Corp", "bbox": [0.1, 0.2, 0.3, 0.4]},
    "total": 1247.50
})

# Values are unwrapped; confidence is accessible when present
print(prediction.invoice_number)                          # "INV-2024-001"
print(prediction.get_field_confidence("invoice_number"))  # 0.97
print(prediction.get_field_confidence("total"))           # None
print(prediction_no_conf.get_field_confidence("vendor"))  # None (no confidence in that rich value)
```

## Single-Document Evaluation

Pass `add_confidence_metrics=True` and `document_field_comparisons=True` to `compare_with()`. The result includes a `confidence_metrics` dict.

```python
ground_truth = Invoice(invoice_number="INV-2024-001", vendor="Acme Corporation", total=1247.50)

result = ground_truth.compare_with(
    prediction,
    add_confidence_metrics=True,
    document_field_comparisons=True
)

# Result structure
print(result["confidence_metrics"]["overall"])
# {"auroc": {"value": 1.0}}

print(result["confidence_metrics"]["fields"])
# {"invoice_number": {"auroc": {"value": None}}, "vendor": {"auroc": {"value": None}}}

print(result["confidence_metrics"]["coverage"])
# {"fields_with_confidence": 2, "fields_total": 3, "ratio": 0.667}
```

Per-document AUROC with 3-5 fields is often `None` (single class, where all match or all fail). The real value comes from bulk evaluation.

## Bulk Evaluation

`BulkStructuredModelEvaluator` automatically accumulates confidence pairs across documents and computes dataset-level metrics at `compute()` time.

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

evaluator = BulkStructuredModelEvaluator(target_schema=Invoice)

for gt_json, pred_json in dataset:
    gt = Invoice(**gt_json)
    pred = Invoice.from_json(pred_json)
    evaluator.update(gt, pred)

results = evaluator.compute()

# Dataset-level confidence metrics
print(results.confidence_metrics["overall"]["auroc"]["value"])  # e.g., 0.85
print(results.confidence_metrics["coverage"]["ratio"])          # e.g., 0.92
```

### Per-Field Breakdown

```python
for field, metrics in results.confidence_metrics["fields"].items():
    auroc = metrics["auroc"]["value"]
    print(f"{field}: AUROC={auroc}")
```

Fields where all predictions match (or all fail) will have `auroc=None` since AUROC requires both classes.

## Multiple Metrics

By default, only AUROC is computed. Pass additional metrics to get Brier Score, ECE, or custom metrics.

```python
from stickler.structured_object_evaluator.models.confidence import (
    AUROCMetric, BrierScoreMetric, ECEMetric,
)

evaluator = BulkStructuredModelEvaluator(
    target_schema=Invoice,
    confidence_metrics=[AUROCMetric(), BrierScoreMetric(), ECEMetric(n_bins=10)]
)
```

### ECE Bin Data

ECE returns bin data for reliability diagrams:

```python
ece = results.confidence_metrics["overall"]["ece"]
print(f"ECE: {ece['value']:.4f}")

for b in ece["bins"]:
    lo, hi = b["range"]
    print(f"[{lo:.1f}, {hi:.1f}): count={b['count']}, accuracy={b['accuracy']:.2f}, mean_conf={b['mean_confidence']:.2f}")
```

## Coverage

Not every field has a confidence score. Coverage tells you how much of your data is actually being evaluated.

```python
cov = results.confidence_metrics["coverage"]
print(f"Fields with confidence: {cov['fields_with_confidence']}/{cov['fields_total']} ({cov['ratio']:.0%})")
```

A low ratio means the metrics only reflect a subset of your fields. Fields without confidence are silently skipped.

## Interpreting Results

| Metric | Good | Bad | Meaning |
|--------|------|-----|---------|
| AUROC | 0.7–1.0 | <0.5 | Confidence discriminates correct from incorrect |
| Brier Score | <0.1 | >0.25 | Mean squared calibration error (lower is better) |
| ECE | <0.05 | >0.15 | Weighted gap between confidence and accuracy per bin |

AUROC = `None` means the metric couldn't be computed (single class or no data).

## Distributed Evaluation

Confidence pairs are included in state serialization and merging:

```python
# Worker A
wa = BulkStructuredModelEvaluator(target_schema=Invoice)
for gt, pred in shard_a:
    wa.update(gt, pred)

# Worker B
wb = BulkStructuredModelEvaluator(target_schema=Invoice)
for gt, pred in shard_b:
    wb.update(gt, pred)

# Merge
wa.merge_state(wb.get_state())
results = wa.compute()  # Metrics from both shards
```

## Adding Custom Metrics

Subclass `ConfidenceMetric` and implement `name` and `compute()`:

```python
from stickler.structured_object_evaluator.models.confidence import (
    ConfidenceMetric, ConfidencePairs,
)

class ConfidenceSimilarityCorrelation(ConfidenceMetric):
    """Pearson correlation between confidence and similarity scores."""

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

Existing metrics use `is_match` and `confidence`. The `similarity` field enables future metrics that correlate confidence with degree of correctness rather than binary match.

## Nested and List Fields

Confidence works with any nesting depth. Field paths use dot notation for nested objects and array notation for list items:

- `address.street`, `address.city`
- `contact.address.zip_code` (double-nested)
- `items[0].product`, `items[1].price` (list items)

List indices are prediction-side indices. Hungarian matching may reorder items, but confidence paths always use the original prediction JSON indices.

## Notebooks

- [`Confidence_Estimation.ipynb`](../../examples/notebooks/Confidence_Estimation.ipynb): single-document API walkthrough
- [`Bulk_Confidence_AUROC.ipynb`](../../examples/notebooks/Bulk_Confidence_AUROC.ipynb): dataset-level evaluation with multiple metrics
