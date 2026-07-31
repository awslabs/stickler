# Bounding Box mAP Module

Measures how accurately a model **localizes** document fields, via mean Average
Precision (mAP) over bounding boxes.

This module consumes bounding-box data carried by the **Rich Value Pattern**. A
rich value is any JSON dict with a `"_value"` key; a bounding box rides alongside
under `"_bbox"` (e.g., `{"_value": "Acme", "_bbox": [[10, 20], [200, 50]], "_confidence": 0.9}`).
`process_rich_values()` (in `models/rich_value.py`) unwraps these during `from_json()`, and the box lands in the
instance extras (`get_all_extras()`); this module evaluates the localization
slice. Boxes are optional; fields without a `_bbox` are skipped for mAP but still
counted toward coverage.

## Architecture

```
bbox/
├── __init__.py     # Public API re-exports
├── calculator.py   # MAPCalculator: join, real AP, mean AP, coverage
├── accumulator.py  # BBoxMAPAccumulator: PostComparisonAccumulator for bulk eval
└── README.md
```

Mirrors the sibling `confidence/` package: a calculator (the math) plus an
accumulator (bulk/streaming aggregation through `PostComparisonAccumulator`).

## How the join works

`MAPCalculator.extract_from_dicts` joins the `field_comparisons` rows from
`compare_with` with two bounding-box maps:

- **Ground-truth boxes** are looked up by the GT-side `expected_key`.
- **Prediction boxes** are looked up by the prediction-side `actual_key`.

These two keys diverge once Hungarian matching reorders list items
(`LineItems[0].StartDate` vs `LineItems[2].StartDate`), so joining the wrong side
silently mis-scores reordered lists. FN rows (`actual_key is None`, often reported
at the object level like `items[1]`) record a localization miss for every GT box at
or under that prefix.

Observations are grouped by a **list-index-normalized class key**
(`LineItems[2].StartDate` -> `LineItems[].StartDate`) so AP is measured per
field-type, not per list slot.

## Average Precision

`compute_metrics` computes a true AP per field-type, COCO-style (matching
pycocotools / torchmetrics): predicted boxes are ranked by `_confidence`, each
labelled TP/FP at an IoU threshold (recall denominator = number of GT boxes),
the precision envelope is applied ("zig-zags removed"), and precision is sampled
at 101 fixed recall points. AP is computed at each IoU threshold in the
configured range (COCO `[0.50:0.95]` by default); `mean_ap` averages over those
thresholds and over field-type classes (with `map_50` / `map_75` exposed
separately). Missing `_confidence` defaults to 1.0, so real confidence scores
are recommended.

A below-threshold matched box counts as **both** a false positive and a false
negative (wrong location + unmatched ground truth).

**Scope:** this matches COCO's AP *definition* (envelope + 101-point + IoU
range), not a full detection evaluator. Each field has at most one GT box and
one predicted box, paired by field path — there is no many-to-many box
assignment per image. That suits document field localization but is not
interchangeable with COCO mAP on multi-instance detection datasets.

## BBoxObservation

```python
class BBoxObservation(BaseModel):
    has_gt: bool                  # a ground-truth box was present
    has_pred: bool                # a prediction box was present
    iou: float                    # IoU(pred, gt); 0.0 when either is missing
    confidence: Optional[float]   # prediction _confidence (None when absent)
```

Threshold classification is deferred to `compute_metrics`, so the same accumulated
observations can be re-scored at a different IoU threshold.

## Usage

Bulk (recommended):

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.bbox import BBoxMAPAccumulator

evaluator = BulkStructuredModelEvaluator(accumulators=[BBoxMAPAccumulator()])
for gt, pred in dataset:
    evaluator.update(gt, pred)
metrics = evaluator.compute().accumulator_metrics["bbox_map_metrics"]
```

Single-document sanity check: `gt.compare_with(pred, add_bbox_metrics=True)` nests
the same structure under the `bbox_metrics` result key.

See the [Bounding Box mAP Metrics](../../../../../../docs/docs/Advanced/bbox-map-metrics.md)
docs page for the full guide.
