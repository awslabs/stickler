---
title: Object Detections from Robots and Cameras
---

# Evaluating Object Detections from Robots and Cameras

A robot or camera system reports, per frame, a list of detected objects: a class and a position. Stickler scores that list against the ground truth for the frame and counts each object as a true positive (TP), false discovery (FD), false negative (FN) or false alarm (FA). The examples run in order and print the output shown.

## Model a frame

One `StructuredModel` per detection, and a frame that holds a `List` of them:

```python
from typing import List

from stickler import ComparableField, ExactComparator, NumericComparator, StructuredModel


class Det(StructuredModel):
    cls: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    x: float = ComparableField(comparator=NumericComparator(absolute_tolerance=0.5), threshold=1.0)
    y: float = ComparableField(comparator=NumericComparator(absolute_tolerance=0.5), threshold=1.0)
    z: float = ComparableField(comparator=NumericComparator(absolute_tolerance=0.2), threshold=1.0)


class Frame(StructuredModel):
    dets: List[Det] = ComparableField()


def show(frame_class, truth, pred, **compare_kwargs):
    result = frame_class(dets=truth).compare_with(
        frame_class(dets=pred), include_confusion_matrix=True, **compare_kwargs
    )
    dets = result["confusion_matrix"]["fields"]["dets"]
    for node in ("overall", "aggregate"):
        print(f"{node:9s}", {k: dets[node][k] for k in ("tp", "fd", "fn", "fa")})
    return dets


truth = [
    {"cls": "Forklift", "x": 2.0, "y": 5.0, "z": 0.0},
    {"cls": "Person", "x": 8.0, "y": 1.0, "z": 0.0},
]
pred = [
    {"cls": "Person", "x": 8.3, "y": 0.8, "z": 0.1},
    {"cls": "Forklift", "x": 2.1, "y": 5.0, "z": 0.0},
]
show(Frame, truth, pred)
```

```text
overall   {'tp': 2, 'fd': 0, 'fn': 0, 'fa': 0}
aggregate {'tp': 8, 'fd': 0, 'fn': 0, 'fa': 0}
```

- **Class:** `ExactComparator` scores `1.0` for the same string, else `0.0`.
- **Position:** `NumericComparator(absolute_tolerance=...)` scores `1.0` when the two values differ by at most the tolerance, else `0.0`. Set one per axis: here `z` is tighter than `x` and `y`.
- **Order:** the predictions arrive in a different order and both objects still pair.

## How detections pair

Stickler pairs the two lists one to one with the [Hungarian algorithm](../../Advanced/hungarian-matching.md). The rules for pairing, counts and scores are stated once in the [list contract](../../Advanced/classification-logic.md#list-contract). For a detection list they come to this:

- A pair's similarity is the weighted mean of its field scores. A pair at or above the element class's `match_threshold` (default `0.7`) is a TP, below it an FD.
- Hungarian always pairs `min(len(truth), len(pred))` objects. Only the rest are FN (truth) or FA (prediction).

## A wrong class can still pair

With four fields at equal weight, one wrong field scores `0.75`, which meets `0.7`. A Pallet reported where a Forklift stands is a TP:

```python
wrong_class = [pred[0], {**pred[1], "cls": "Pallet"}]
show(Frame, truth, wrong_class)
```

```text
overall   {'tp': 2, 'fd': 0, 'fn': 0, 'fa': 0}
aggregate {'tp': 7, 'fd': 1, 'fn': 0, 'fa': 0}
```

An `x` off by 1.0 m, twice its tolerance, gives the same counts. The error shows only in `aggregate`, as one wrong field.

To make every field decide the pair, set `match_threshold = 1.0` on the detection class. Raising the weight of one field protects that field only, because the other fields can still carry the mean over the bar.

```python
class StrictDet(Det):
    match_threshold = 1.0


class StrictFrame(StructuredModel):
    dets: List[StrictDet] = ComparableField()


strict = show(StrictFrame, truth, wrong_class)
print("recall", strict["overall"]["derived"]["cm_recall"])
strict = show(StrictFrame, truth, wrong_class, recall_with_fd=True)
print("recall", strict["overall"]["derived"]["cm_recall"])
```

```text
overall   {'tp': 1, 'fd': 1, 'fn': 0, 'fa': 0}
aggregate {'tp': 4, 'fd': 0, 'fn': 0, 'fa': 0}
recall 1.0
overall   {'tp': 1, 'fd': 1, 'fn': 0, 'fa': 0}
aggregate {'tp': 4, 'fd': 0, 'fn': 0, 'fa': 0}
recall 0.5
```

The wrong pair is one FD, not one FN plus one FA as a detection metric counts it. Recall leaves FDs out by default. Pass `recall_with_fd=True` to count them against recall ([why](../../Advanced/hungarian-matching.md#fd-and-recall)).

A missed object and an unrelated extra detection in the same frame also pair, as one FD:

```python
missed_and_extra = [pred[0], {"cls": "Box", "x": 20.0, "y": 20.0, "z": 3.0}]
show(StrictFrame, truth, missed_and_extra)
```

```text
overall   {'tp': 1, 'fd': 1, 'fn': 0, 'fa': 0}
aggregate {'tp': 4, 'fd': 0, 'fn': 0, 'fa': 0}
```

## 2D boxes

For camera boxes, score the box with `BBoxIoUComparator`. Its score is the intersection over union (IoU), and the field `threshold` is the IoU cut. Give the box as a tuple of corners, `(x1, y1, x2, y2)`. A `List` field is paired element by element and does not reach the comparator. Convert `(x, y, width, height)` boxes, as COCO and YOLO store them, to corners first: read as corners they give a wrong IoU and no error.

```python
from typing import Tuple

from stickler import BBoxIoUComparator


class Box(StructuredModel):
    cls: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    box: Tuple[float, float, float, float] = ComparableField(
        comparator=BBoxIoUComparator(), threshold=0.5
    )


class BoxFrame(StructuredModel):
    dets: List[Box] = ComparableField()


box_truth = [{"cls": "Person", "box": (0, 0, 100, 50)}]
for box in [(10, 5, 110, 55), (40, 20, 140, 70)]:
    iou = BBoxIoUComparator().compare(box_truth[0]["box"], box)
    print(f"IoU {iou:.2f}")
    show(BoxFrame, box_truth, [{"cls": "Person", "box": box}])
```

```text
IoU 0.68
overall   {'tp': 1, 'fd': 0, 'fn': 0, 'fa': 0}
aggregate {'tp': 2, 'fd': 0, 'fn': 0, 'fa': 0}
IoU 0.22
overall   {'tp': 0, 'fd': 1, 'fn': 0, 'fa': 0}
aggregate {'tp': 0, 'fd': 1, 'fn': 0, 'fa': 0}
```

An IoU below the field threshold scores `0.0`, so the pair falls to `0.5` and is an FD. [Bounding Box mAP Metrics](../../Advanced/bbox-map-metrics.md#scope) covers one box per document field, not many boxes of one class per frame.

## Read the results

On the list field, `overall` counts objects and `aggregate` counts field values inside the paired objects. Read `overall` for detection precision and recall. `aggregate` covers only the fields of TP pairs, so with `match_threshold = 1.0` a failed pair adds nothing to it. See [`overall` vs `aggregate`](../Evaluation/understanding-results.md#overall-vs-aggregate). To see what failed, read `non_matches`: each FD row holds both objects.

```python
x_off = [pred[0], {**pred[1], "x": 3.1}]
result = StrictFrame(dets=truth).compare_with(
    StrictFrame(dets=x_off), include_confusion_matrix=True, document_non_matches=True
)
print("aggregate fd", result["confusion_matrix"]["fields"]["dets"]["aggregate"]["fd"])
for row in result["non_matches"]:
    print(row["field_path"], row["non_match_type"].value)
    print(" truth", row["ground_truth_value"])
    print(" pred ", row["prediction_value"])
```

```text
aggregate fd 0
dets[0] false_discovery
 truth {'cls': 'Forklift', 'x': 2.0, 'y': 5.0, 'z': 0.0}
 pred  {'cls': 'Forklift', 'x': 3.1, 'y': 5.0, 'z': 0.0}
```

## Many frames

Feed one frame pair at a time to `BulkStructuredModelEvaluator` ([Bulk Evaluation](../Evaluation/bulk-evaluation.md)). `field_metrics["dets"]` holds the object counts summed over all frames:

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

frames = [
    (truth, pred),
    (truth, wrong_class),
    ([truth[1]], []),
    ([], [{"cls": "Pallet", "x": 4.0, "y": 4.0, "z": 0.0}]),
]
evaluator = BulkStructuredModelEvaluator(target_schema=StrictFrame)
for i, (frame_truth, frame_pred) in enumerate(frames):
    evaluator.update(
        StrictFrame(dets=frame_truth), StrictFrame(dets=frame_pred), doc_id=f"frame_{i}"
    )
dets = evaluator.compute().field_metrics["dets"]
print({k: dets[k] for k in ("tp", "fd", "fn", "fa")})
print({k: round(dets[k], 2) for k in ("cm_precision", "cm_recall", "cm_f1")})
print("recall with fd", dets["tp"] / (dets["tp"] + dets["fn"] + dets["fd"]))
```

```text
{'tp': 3, 'fd': 1, 'fn': 1, 'fa': 1}
{'cm_precision': 0.6, 'cm_recall': 0.75, 'cm_f1': 0.67}
recall with fd 0.6
```

The bulk evaluator has no `recall_with_fd`, so `cm_recall` leaves FDs out. Compute `tp / (tp + fn + fd)` from the counts to count them.
