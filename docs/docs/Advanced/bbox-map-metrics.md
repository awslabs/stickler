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
from stickler.structured_object_evaluator.models.bbox import (
    BBoxMAPAccumulator,
)

evaluator = BulkStructuredModelEvaluator(
    accumulators=[BBoxMAPAccumulator()],
)

for gt, pred in dataset:
    evaluator.update(gt, pred)

result = evaluator.compute()
bbox_metrics = result.accumulator_metrics["bbox_map_metrics"]
print(f"mAP@[.50:.95]: {bbox_metrics['mean_ap']:.3f}")
print(f"mAP@.50:       {bbox_metrics['map_50']:.3f}")
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
        BBoxMAPAccumulator(),
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
    print(f"  {field}: mean_iou={m['mean_iou']:.3f}, AP={m['ap']:.3f}")
```

### IoU thresholds

By default the metric uses the COCO IoU range `[0.50, 0.55, ..., 0.95]` and reports `mean_ap` as the average over that range (COCO mAP@[.50:.95]), plus `map_50` and `map_75` for the individual thresholds. Pass a single value or a custom list to change this:

```python
# Single threshold: mean_ap == map_50 (no averaging over a range).
BBoxMAPAccumulator(iou_thresholds=0.5)

# A custom range.
BBoxMAPAccumulator(iou_thresholds=[0.5, 0.75, 0.9])
```

The single-document path takes the same argument via `compare_with(..., bbox_iou_thresholds=0.5)`.

## How It Works

### IoU calculation

For each field that has bounding box data on both ground truth and prediction, the IoU (Intersection over Union) is computed:

```
IoU = Area of Intersection / Area of Union
```

A predicted box is a **true positive** when its IoU with the matched ground-truth box meets or exceeds the threshold. A below-threshold detection counts as **both a false positive and a false negative** (the prediction is in the wrong place, and the ground-truth box is left unmatched). A ground-truth box with no prediction is a **false negative** (a localization miss).

### Average Precision

AP is computed COCO-style, matching the [pycocotools](https://github.com/cocodataset/cocoapi) / [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) implementation:

1. Rank a field's predicted boxes by `_confidence` (descending).
2. Walk the ranking, labelling each detection TP or FP at the IoU threshold and tracking cumulative precision and recall (recall denominator = number of ground-truth boxes).
3. Apply the precision envelope (make precision monotonically non-increasing from the right — "remove zig-zags").
4. Sample precision at 101 fixed recall points (0.00, 0.01, ..., 1.00) and average them.

Predicted boxes are ranked by `_confidence`, which rides the same rich-value pattern (`{"_value": ..., "_confidence": ..., "_bbox": ...}`). When a prediction has no `_confidence` it defaults to `1.0`; without real confidence scores the ranking is uninformative, so providing `_confidence` is recommended for meaningful AP.

The raw IoU and confidence are kept per observation, so the same accumulated data is re-scored at every IoU threshold in the configured range.

The per-field entry reports `ap` (averaged over the IoU range), `ap_50`, `ap_75`, `mean_iou`, `num_gt`, and `num_detections`.

### Mean AP

AP is computed per (field-type class, IoU threshold), then averaged over the configured IoU thresholds and over field-type classes that carry at least one ground-truth box to give `mean_ap` (COCO mAP). `map_50` and `map_75` are the same class-average at a single IoU threshold:

```
mean_ap = mean over IoU thresholds of [ mean over classes of AP ]
```

Two things worth knowing about the class denominator:

- **Macro-average over classes.** Each field-type class weighs equally regardless of how many observations it has — a class seen once counts the same as one seen a thousand times. A class enters the mean if *any* document carried a ground-truth box for it.
- **List indices are normalized into one class.** `LineItems[0].StartDate` and `LineItems[2].StartDate` are grouped under `LineItems[].StartDate`, so AP is measured per field-type rather than per list slot.

This is a different universe from `coverage`, which counts per-(document, field) occurrences. The two will not line up by construction; `mean_ap` answers "how well is each field-type localized on average" while `coverage` answers "what fraction of compared fields carried a bounding box". When no field carries a ground-truth box, `mean_ap` is `None`.

## Result Structure

```python
{
    "mean_ap": 0.667,       # mAP averaged over the IoU-threshold range
    "map_50": 0.667,
    "map_75": 0.667,
    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "fields": {
        "vendor_name": {
            "ap": 1.0, "ap_50": 1.0, "ap_75": 1.0, "mean_iou": 0.92,
            "num_gt": 1, "num_detections": 1
        },
        "invoice_number": {
            "ap": 1.0, "ap_50": 1.0, "ap_75": 1.0, "mean_iou": 1.0,
            "num_gt": 1, "num_detections": 1
        },
        "total_amount": {
            "ap": 0.0, "ap_50": 0.0, "ap_75": 0.0, "mean_iou": 0.05,
            "num_gt": 1, "num_detections": 1
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
