---
title: Bounding Box mAP Metrics
---

# Bounding Box mAP Metrics

Stickler supports bounding box evaluation for document processing use cases where you need to measure how accurately a model locates information on a page. The primary metric is **mean Average Precision (mAP)**, computed using Intersection over Union (IoU) at a configurable threshold.

## When to Use This

Bounding box evaluation is useful when your extraction pipeline returns spatial coordinates alongside field values. Common use cases include:

- Signature detection and localization
- Logo identification on documents
- Key-value pair localization on invoices and forms
- Table cell boundary detection

## JSON Structure

Bounding boxes are provided through the [Rich Value Pattern](rich-value-pattern.md), using the `_bbox` key alongside `_value` and optionally `_confidence`. Like all rich-value metadata, the key is underscore-prefixed.

### Bbox with value only

```json
{
  "vendor_name": {
    "_value": "Acme Corp",
    "_bbox": [[10, 20], [200, 50]]
  }
}
```

### Bbox with confidence

```json
{
  "vendor_name": {
    "_value": "Acme Corp",
    "_confidence": 0.95,
    "_bbox": [[10, 20], [200, 50]]
  }
}
```

### Supported formats

Two bounding box formats are accepted:

- **Two-point**: `[[x1, y1], [x2, y2]]` — top-left and bottom-right corners
- **Flat**: `[x1, y1, x2, y2]` — four coordinates in a single list

Coordinates can be in any unit system (pixels, normalized 0-1, etc.) as long as ground truth and predictions use the same system.

## Usage

The recommended path is bulk evaluation with `BBoxMAPAccumulator`, which aggregates bounding-box pairs across an entire test set and computes mAP once at the end. A single-document sanity-check path is also available via `compare_with(add_bbox_metrics=True)`.

### Defining models

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import LevenshteinComparator, NumericComparator

class Invoice(StructuredModel):
    vendor_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8
    )
    invoice_number: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )
    total_amount: float = ComparableField(
        comparator=NumericComparator(), threshold=0.95
    )
```

### Providing bbox data via rich values

```python
ground_truth = Invoice.from_json({
    "vendor_name": {"_value": "Acme Corp", "_bbox": [[10, 20], [200, 50]]},
    "invoice_number": {"_value": "INV-2024-001", "_bbox": [[10, 60], [200, 90]]},
    "total_amount": {"_value": 1500.00, "_bbox": [[10, 100], [200, 130]]},
})

prediction = Invoice.from_json({
    "vendor_name": {
        "_value": "Acme Corp", "_confidence": 0.95, "_bbox": [[12, 18], [198, 52]]
    },
    "invoice_number": {
        "_value": "INV-2024-001", "_confidence": 0.9, "_bbox": [[10, 60], [200, 90]]
    },
    "total_amount": {
        "_value": 1500.00, "_confidence": 0.85, "_bbox": [[50, 200], [150, 230]]
    },
})
```

### Bulk evaluation (recommended)

Pass a `BBoxMAPAccumulator` to `BulkStructuredModelEvaluator`. The accumulator extracts bounding-box pairs from each comparison, accumulates them across documents, and exposes the result under its name, `bbox_map_metrics`.

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.bbox_map_accumulator import (
    BBoxMAPAccumulator,
)

evaluator = BulkStructuredModelEvaluator(
    accumulators=[BBoxMAPAccumulator(iou_threshold=0.5)],
)

for gt, pred in dataset:
    evaluator.update(gt, pred)

result = evaluator.compute()
bbox_metrics = result.accumulator_metrics["bbox_map_metrics"]
print(f"Mean AP: {bbox_metrics['mean_ap']:.3f}")
print(f"Coverage: {bbox_metrics['coverage']}")
```

`BBoxMAPAccumulator` implements the same `PostComparisonAccumulator` interface as `ConfidenceAccumulator`, so it supports `get_state()` / `load_state()` / `merge_state()` for checkpointing and distributed (sharded) evaluation.

### Combining with confidence metrics

Because each accumulator is independent, you can evaluate localization and confidence calibration in the same pass:

```python
from stickler.structured_object_evaluator.models.confidence.accumulator import (
    ConfidenceAccumulator,
)

evaluator = BulkStructuredModelEvaluator(
    accumulators=[
        ConfidenceAccumulator(),
        BBoxMAPAccumulator(iou_threshold=0.5),
    ],
)

for gt, pred in dataset:
    evaluator.update(gt, pred)

metrics = evaluator.compute().accumulator_metrics
print(f"mAP:   {metrics['bbox_map_metrics']['mean_ap']:.3f}")
print(f"AUROC: {metrics['confidence_metrics']['overall']['auroc']['value']:.3f}")
```

### Single-document sanity check

For a quick check on one document pair, `compare_with(add_bbox_metrics=True)` computes the same metrics inline. This emits a `UserWarning` recommending bulk evaluation — per-document mAP over a handful of fields is noisy and not statistically meaningful.

```python
result = ground_truth.compare_with(
    prediction,
    add_bbox_metrics=True,
    document_field_comparisons=True,
)

bbox_metrics = result["bbox_metrics"]
print(f"Mean AP: {bbox_metrics['mean_ap']:.3f}")
for field, m in bbox_metrics["fields"].items():
    print(f"  {field}: IoU={m['iou']:.3f}, AP={m['ap']:.1f}")
```

### Custom IoU threshold

```python
# Stricter (mAP@0.75)
BBoxMAPAccumulator(iou_threshold=0.75)

# More lenient (mAP@0.3)
BBoxMAPAccumulator(iou_threshold=0.3)
```

The single-document path takes the same threshold via `compare_with(..., bbox_iou_threshold=0.75)`.

## How It Works

### IoU calculation

For each field that has bounding box data on both ground truth and prediction, the IoU (Intersection over Union) is computed:

```
IoU = Area of Intersection / Area of Union
```

A predicted box is a **true positive** when its IoU with the matched ground-truth box meets or exceeds the threshold. A below-threshold detection counts as **both a false positive and a false negative** (the prediction is in the wrong place, and the ground-truth box is left unmatched). A ground-truth box with no prediction is a **false negative** (a localization miss).

### Average Precision

AP is computed as the area under the confidence-ranked precision-recall curve, using the Pascal VOC 2010+ all-points interpolation:

1. Rank a field's predicted boxes by `_confidence` (descending).
2. Walk the ranking, labelling each detection TP or FP at the IoU threshold and tracking cumulative precision and recall (recall denominator = number of ground-truth boxes).
3. Make precision monotonically non-increasing, then integrate over recall.

Predicted boxes are ranked by `_confidence`, which rides the same rich-value pattern (`{"_value": ..., "_confidence": ..., "_bbox": ...}`). When a prediction has no `_confidence` it defaults to `1.0`; without real confidence scores the ranking is uninformative and AP collapses to a single operating point, so providing `_confidence` is recommended for meaningful AP.

Threshold classification is deferred to compute time, so the same accumulated observations can be re-scored at a different IoU threshold.

The per-field entry also reports `precision`, `recall`, `mean_iou`, `num_gt`, `num_detections`, and `num_true_positives` at the all-detections operating point for inspection.

### Mean AP

The mean Average Precision macro-averages per-field AP across field-type classes that carry at least one ground-truth box:

```
mAP = mean(AP for each field-type class with at least one GT bbox)
```

Two things worth knowing about the denominator:

- **Macro-average over classes.** Each field-type class weighs equally regardless of how many observations it has — a class seen once counts the same as one seen a thousand times. A class enters the mean if *any* document carried a ground-truth box for it.
- **List indices are normalized into one class.** `LineItems[0].StartDate` and `LineItems[2].StartDate` are grouped under `LineItems[].StartDate`, so AP is measured per field-type rather than per list slot.

This is a different universe from `coverage`, which counts per-(document, field) occurrences. The two will not line up by construction; `mean_ap` answers "how well is each field-type localized on average" while `coverage` answers "what fraction of compared fields carried a bounding box". When no field carries a ground-truth box, `mean_ap` is `None`.

## Result Structure

```python
{
    "mean_ap": 0.667,
    "iou_threshold": 0.5,
    "fields": {
        "vendor_name": {
            "ap": 1.0, "precision": 1.0, "recall": 1.0, "mean_iou": 0.92,
            "num_gt": 1, "num_detections": 1, "num_true_positives": 1
        },
        "invoice_number": {
            "ap": 1.0, "precision": 1.0, "recall": 1.0, "mean_iou": 1.0,
            "num_gt": 1, "num_detections": 1, "num_true_positives": 1
        },
        "total_amount": {
            "ap": 0.0, "precision": 0.0, "recall": 0.0, "mean_iou": 0.05,
            "num_gt": 1, "num_detections": 1, "num_true_positives": 0
        }
    },
    "coverage": {
        "fields_with_bbox": 3,
        "fields_total": 3,
        "ratio": 1.0
    }
}
```

The single-document path nests this same structure under the `bbox_metrics` key of the `compare_with()` result.

## BBoxIoUComparator

For direct bounding box comparison without the full evaluation pipeline, use the `BBoxIoUComparator`:

```python
from stickler.comparators import BBoxIoUComparator

cmp = BBoxIoUComparator(threshold=0.5)

# Returns IoU as a similarity score in [0.0, 1.0]
iou = cmp.compare([[0, 0], [100, 50]], [[10, 5], [110, 55]])
print(f"IoU: {iou:.3f}")
```

## See Also

- [Rich Value Pattern](rich-value-pattern.md) — how fields carry `_value`, `_confidence`, and `_bbox` metadata
- [Confidence Metrics](confidence-metrics.md) — evaluating prediction confidence calibration
- [Classification Logic](classification-logic.md) — how match/no-match is determined
- [Comparators](../Guides/Comparators/README.md) — all available comparators
