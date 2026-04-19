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

Bounding boxes are provided as part of the [Rich Value Pattern](confidence-metrics.md), using the `"bbox"` key alongside `"value"` and optionally `"confidence"`:

### Bbox with value only

```json
{
  "vendor_name": {
    "value": "Acme Corp",
    "bbox": [[10, 20], [200, 50]]
  }
}
```

### Bbox with confidence

```json
{
  "vendor_name": {
    "value": "Acme Corp",
    "confidence": 0.95,
    "bbox": [[10, 20], [200, 50]]
  }
}
```

### Supported formats

Two bounding box formats are accepted:

- **Two-point**: `[[x1, y1], [x2, y2]]` — top-left and bottom-right corners
- **Flat**: `[x1, y1, x2, y2]` — four coordinates in a single list

Coordinates can be in any unit system (pixels, normalized 0-1, etc.) as long as ground truth and predictions use the same system.

## Usage

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
    "vendor_name": {
        "value": "Acme Corp",
        "bbox": [[10, 20], [200, 50]]
    },
    "invoice_number": {
        "value": "INV-2024-001",
        "bbox": [[10, 60], [200, 90]]
    },
    "total_amount": {
        "value": 1500.00,
        "bbox": [[10, 100], [200, 130]]
    }
})

prediction = Invoice.from_json({
    "vendor_name": {
        "value": "Acme Corp",
        "confidence": 0.95,
        "bbox": [[12, 18], [198, 52]]
    },
    "invoice_number": {
        "value": "INV-2024-001",
        "confidence": 0.9,
        "bbox": [[10, 60], [200, 90]]
    },
    "total_amount": {
        "value": 1500.00,
        "confidence": 0.85,
        "bbox": [[50, 200], [150, 230]]
    }
})
```

### Running evaluation with bbox metrics

```python
result = ground_truth.compare_with(
    prediction,
    add_bbox_metrics=True,
    document_field_comparisons=True
)

bbox_metrics = result["bbox_metrics"]
print(f"Mean AP: {bbox_metrics['mean_ap']:.3f}")
print(f"IoU threshold: {bbox_metrics['iou_threshold']}")

# Per-field breakdown
for field, metrics in bbox_metrics["field_results"].items():
    print(f"  {field}: IoU={metrics['iou']:.3f}, AP={metrics['ap']:.1f}")
```

### Custom IoU threshold

```python
# Stricter threshold (mAP@0.75)
result = ground_truth.compare_with(
    prediction,
    add_bbox_metrics=True,
    bbox_iou_threshold=0.75
)

# More lenient threshold (mAP@0.3)
result = ground_truth.compare_with(
    prediction,
    add_bbox_metrics=True,
    bbox_iou_threshold=0.3
)
```

## How It Works

### IoU calculation

For each field that has bounding box data on both ground truth and prediction, the IoU (Intersection over Union) is computed:

```
IoU = Area of Intersection / Area of Union
```

A field is classified as a **true positive** if its IoU meets or exceeds the threshold, and a **false positive** otherwise.

### Per-field metrics

For each field with a ground truth bounding box:

| Metric | Formula |
|---|---|
| **IoU** | Intersection area / Union area |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 × Precision × Recall / (Precision + Recall) |
| **AP** | Precision × Recall |

Since each field has exactly one ground truth and one prediction box, TP is either 0 or 1 per field.

### Mean AP

The mean Average Precision is the average of per-field AP values across all fields that have ground truth bounding boxes:

```
mAP = mean(AP for each field with GT bbox)
```

Fields without ground truth bounding boxes are excluded from the calculation. Fields with ground truth but no prediction bounding box receive AP = 0.

## Result Structure

```python
{
    "bbox_metrics": {
        "mean_ap": 0.667,
        "iou_threshold": 0.5,
        "field_results": {
            "vendor_name": {
                "iou": 0.92,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "ap": 1.0
            },
            "invoice_number": {
                "iou": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "ap": 1.0
            },
            "total_amount": {
                "iou": 0.05,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0
            }
        },
        "coverage": {
            "fields_with_bbox": 3,
            "fields_total": 3,
            "ratio": 1.0
        }
    }
}
```

## Combining with Confidence Metrics

Bbox metrics and confidence metrics can be computed in the same call:

```python
result = ground_truth.compare_with(
    prediction,
    add_bbox_metrics=True,
    add_confidence_metrics=True,
    document_field_comparisons=True
)

print(f"mAP: {result['bbox_metrics']['mean_ap']:.3f}")
print(f"AUROC: {result['confidence_metrics']['overall']['auroc']['value']:.3f}")
```

## BBoxIoUComparator

For direct bounding box comparison without the full evaluation pipeline, you can use the `BBoxIoUComparator`:

```python
from stickler.comparators import BBoxIoUComparator

cmp = BBoxIoUComparator(threshold=0.5)

# Returns IoU as similarity score
iou = cmp.compare([[0, 0], [100, 50]], [[10, 5], [110, 55]])
print(f"IoU: {iou:.3f}")

# Binary classification
tp, fp = cmp.binary_compare([[0, 0], [100, 50]], [[10, 5], [110, 55]])
```

## See Also

- [Confidence Metrics](confidence-metrics.md) — evaluating prediction confidence calibration
- [Classification Logic](classification-logic.md) — how match/no-match is determined
- [Comparators](../Guides/Comparators/README.md) — all available comparators
