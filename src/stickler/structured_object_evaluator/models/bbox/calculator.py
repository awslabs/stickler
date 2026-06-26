"""
Mean Average Precision (mAP) calculator for bounding box evaluation.

This is the orchestrator for bounding-box localization scoring. It mirrors
``ConfidenceCalculator``'s calculator/accumulator split:

1. ``extract_from_dicts`` joins ``field_comparisons`` with ground-truth and
   prediction bounding boxes (and prediction confidences) to produce keyed
   ``BBoxObservation`` rows.
2. ``compute_metrics`` ranks each field's predicted boxes by confidence,
   labels them TP/FP at the IoU threshold, and integrates the precision-recall
   curve into a true Average Precision per field, then averages across fields.

Average Precision
-----------------
AP is the area under the confidence-ranked precision-recall curve, computed
COCO-style (matching pycocotools / torchmetrics): the precision envelope is
applied ("zig-zags removed") and precision is sampled at 101 fixed recall points.
``mean_ap`` averages AP over the configured IoU thresholds (COCO mAP@[0.50:0.95]
by default) and over field-type classes; ``map_50`` / ``map_75`` expose the
single-threshold values.

Predicted boxes are ranked by their ``_confidence`` (which rides the same
rich-value pattern). When a prediction has no confidence, it defaults to 1.0;
without real confidence scores the ranking is uninformative, so providing
``_confidence`` is recommended for meaningful AP.

List fields
-----------
Boxes are joined per row using the GT-side ``expected_key`` and the
prediction-side ``actual_key`` (which diverge once Hungarian matching reorders
list items). Observations are then grouped by a *class key* that normalizes list
indices (``LineItems[2].StartDate`` -> ``LineItems[].StartDate``) so AP is
measured per field-type rather than per list slot.

Bounding boxes ride on the rich value pattern under the ``_bbox`` key, e.g.::

    {"_value": "John Doe", "_bbox": [[x1, y1], [x2, y2]], "_confidence": 0.9}
"""

import re
from bisect import bisect_left
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel

from stickler.comparators.bbox import BBoxIoUComparator

if TYPE_CHECKING:
    from stickler.structured_object_evaluator.models.structured_model import (
        StructuredModel,
    )

# The rich-value metadata key bounding boxes ride on.
BBOX_KEY = "_bbox"

# Confidence assigned to a predicted box that carries no _confidence.
_DEFAULT_CONFIDENCE = 1.0

# COCO default IoU thresholds: 0.50, 0.55, ..., 0.95 (averaged for the headline
# mAP). Matches torchmetrics' linspace(0.5, 0.95, 10).
DEFAULT_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

# COCO recall sampling points for 101-point interpolation: 0.00, 0.01, ..., 1.00.
_RECALL_THRESHOLDS = tuple(round(0.01 * i, 2) for i in range(101))

_LIST_INDEX_RE = re.compile(r"\[\d+\]")


def class_key(field_path: str) -> str:
    """Normalize list indices in a field path to group AP per field-type.

    ``LineItems[2].StartDate`` -> ``LineItems[].StartDate``. Flat field paths
    are returned unchanged.
    """
    return _LIST_INDEX_RE.sub("[]", field_path)


class BBoxObservation(BaseModel):
    """A single ground-truth/prediction bounding-box row for one field.

    - ``has_gt`` / ``has_pred``: whether a box was present on each side.
    - ``iou``: IoU between prediction and ground truth (0.0 when either box is
      missing).
    - ``confidence``: the prediction's ``_confidence`` (None when absent).

    Threshold classification is deferred to ``compute_metrics`` so the same
    accumulated observations can be scored at different IoU thresholds.
    """

    has_gt: bool
    has_pred: bool
    iou: float
    confidence: Optional[float] = None


# Backwards-compatible alias; observations are the unit accumulated per field.
BBoxPair = BBoxObservation
BBoxObservations = List[BBoxObservation]
KeyedBBoxPairs = Dict[str, BBoxObservations]


class BBoxExtractionResult(BaseModel):
    """Result of extracting bbox observations from a single comparison."""

    keyed_pairs: KeyedBBoxPairs
    fields_with_bbox: int
    fields_total: int


class MAPCalculator:
    """Extracts bounding-box observations and computes Mean Average Precision.

    AP is computed COCO-style: per field-type class, predicted boxes are ranked
    by confidence, classified TP/FP at an IoU threshold, and the precision-recall
    curve is interpolated with 101-point sampling (matching pycocotools /
    torchmetrics). The reported ``mean_ap`` averages AP over the configured IoU
    thresholds (COCO mAP@[0.50:0.95] by default) and over classes; ``map_50`` /
    ``map_75`` expose the single-threshold values.

    Args:
        iou_thresholds: A single IoU threshold or an iterable of them. Defaults
            to the COCO range ``(0.50, 0.55, ..., 0.95)``. Passing a single float
            (e.g. ``0.5``) yields a single-threshold ``mean_ap`` equal to
            ``map_50``.
    """

    def __init__(
        self, iou_thresholds: Union[float, Iterable[float]] = DEFAULT_IOU_THRESHOLDS
    ):
        if isinstance(iou_thresholds, (int, float)):
            iou_thresholds = (float(iou_thresholds),)
        self.iou_thresholds = tuple(sorted(float(t) for t in iou_thresholds))
        # Only compare() is used; classification happens in this calculator.
        self._comparator = BBoxIoUComparator()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def bboxes_from_extras(
        extras: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Project a field-extras mapping down to ``{field_path: bbox}``.

        Args:
            extras: Mapping of field path -> extras dict, as returned by
                ``StructuredModel.get_all_extras()``. Each extras dict may
                carry a ``_bbox`` entry.

        Returns:
            Mapping of field path -> bbox for fields that had a ``_bbox``.
        """
        bboxes: Dict[str, Any] = {}
        for field_path, field_extras in extras.items():
            if isinstance(field_extras, dict) and BBOX_KEY in field_extras:
                bboxes[field_path] = field_extras[BBOX_KEY]
        return bboxes

    def extract(
        self,
        comparison_result: Dict,
        gt_instance: "StructuredModel",
        pred_instance: "StructuredModel",
    ) -> BBoxExtractionResult:
        """Extract bbox observations from a comparison plus two model instances.

        Convenience wrapper for the single-document path. Pulls bounding boxes
        and confidences from the instances and delegates to
        ``extract_from_dicts``.

        Raises:
            ValueError: If no ``field_comparisons`` in ``comparison_result``.
        """
        field_comparisons = comparison_result.get("field_comparisons", [])
        if not field_comparisons:
            raise ValueError("No field comparisons found in comparison result.")

        gt_bboxes = self.bboxes_from_extras(gt_instance.get_all_extras())
        pred_bboxes = self.bboxes_from_extras(pred_instance.get_all_extras())
        confidences = (
            pred_instance.get_all_confidences()
            if hasattr(pred_instance, "get_all_confidences")
            else {}
        )
        return self.extract_from_dicts(
            field_comparisons, gt_bboxes, pred_bboxes, confidences
        )

    def extract_from_dicts(
        self,
        field_comparisons: List[Dict],
        gt_bboxes: Dict[str, Any],
        pred_bboxes: Dict[str, Any],
        confidences: Optional[Dict[str, float]] = None,
    ) -> BBoxExtractionResult:
        """Join raw ``field_comparisons`` with gt/pred bbox and confidence dicts.

        Ground-truth boxes are looked up by the GT-side ``expected_key`` and
        prediction boxes by the prediction-side ``actual_key`` (these diverge
        for reordered list items). A row contributes to the per-field metrics
        when it carries a ground-truth box, a prediction box, or both:

        - GT box + matching prediction -> a detection (TP if IoU >= threshold).
        - GT box, no prediction (FN row, ``actual_key`` is None) -> an unmatched
          ground truth (recall denominator, no detection).
        - Prediction, no GT box (spurious detection) -> a false positive.

        Observations are keyed by a list-index-normalized class key.

        Args:
            field_comparisons: The ``field_comparisons`` list from a
                ``compare_with`` result.
            gt_bboxes: Ground-truth ``{expected_key: bbox}``.
            pred_bboxes: Prediction ``{actual_key: bbox}``.
            confidences: Prediction ``{actual_key: confidence}`` (optional).

        Returns:
            BBoxExtractionResult with keyed observations and coverage counts.
        """
        confidences = confidences or {}
        keyed: KeyedBBoxPairs = {}
        fields_with_bbox = 0
        fields_total = 0

        for fc in field_comparisons:
            expected_key = fc.get("expected_key")
            actual_key = fc.get("actual_key")

            # FN row: an unmatched ground-truth entry (no prediction). The row
            # may be reported at the object level (e.g. "items[1]") while the
            # GT boxes live on nested fields ("items[1].description"), so record
            # a localization miss for every GT bbox at or under expected_key.
            if actual_key is None and isinstance(expected_key, str):
                gt_keys = self._gt_keys_under(expected_key, gt_bboxes)
                if gt_keys:
                    # Count each missed GT box as one (field) occurrence that
                    # carried a bbox, so coverage stays consistent with the
                    # matched path (and the ratio cannot exceed 1.0 when one
                    # object-level FN row hides several nested boxes).
                    for gt_key in gt_keys:
                        fields_total += 1
                        fields_with_bbox += 1
                        keyed.setdefault(class_key(gt_key), []).append(
                            BBoxObservation(
                                has_gt=True, has_pred=False, iou=0.0, confidence=None
                            )
                        )
                else:
                    # No GT bbox under this row, but it is still a real
                    # comparison row, so it counts toward the total.
                    fields_total += 1
                continue

            gt_bbox = (
                gt_bboxes.get(expected_key) if isinstance(expected_key, str) else None
            )
            pred_bbox = (
                pred_bboxes.get(actual_key) if isinstance(actual_key, str) else None
            )

            has_gt = gt_bbox is not None
            has_pred = pred_bbox is not None
            if not has_gt and not has_pred:
                # Field carried no bounding box on either side. Count toward
                # the total when it is a real comparison row.
                if isinstance(expected_key, str) or isinstance(actual_key, str):
                    fields_total += 1
                continue

            fields_total += 1
            if has_gt:
                fields_with_bbox += 1

            iou = self._comparator.compare(pred_bbox, gt_bbox) if has_pred else 0.0
            confidence = (
                confidences.get(actual_key) if isinstance(actual_key, str) else None
            )

            # Group by field-type (list indices normalized away). Prefer the GT
            # path so matched/FN rows for the same field-type coalesce.
            group_path = expected_key if isinstance(expected_key, str) else actual_key
            key = class_key(group_path)

            keyed.setdefault(key, []).append(
                BBoxObservation(
                    has_gt=has_gt,
                    has_pred=has_pred,
                    iou=iou,
                    confidence=confidence,
                )
            )

        return BBoxExtractionResult(
            keyed_pairs=keyed,
            fields_with_bbox=fields_with_bbox,
            fields_total=fields_total,
        )

    @staticmethod
    def _gt_keys_under(prefix: str, gt_bboxes: Dict[str, Any]) -> List[str]:
        """GT bbox keys equal to ``prefix`` or nested under it.

        Handles object-level FN rows (``items[1]``) whose ground-truth boxes
        live on nested fields (``items[1].description``).
        """
        keys = []
        for k in gt_bboxes:
            if k == prefix or k.startswith(prefix + ".") or k.startswith(prefix + "["):
                keys.append(k)
        return keys

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _average_precision(
        detections: List[tuple], num_gt: int, threshold: float
    ) -> Optional[float]:
        """COCO 101-point interpolated AP from confidence-ranked detections.

        Mirrors pycocotools / torchmetrics: rank detections by confidence,
        classify TP/FP at ``threshold``, build the precision-recall curve,
        remove zig-zags (precision envelope), then average precision sampled at
        101 fixed recall points (0.00, 0.01, ..., 1.00).

        Args:
            detections: List of ``(confidence, iou)`` tuples for predicted boxes.
            num_gt: Number of ground-truth boxes (recall denominator).
            threshold: IoU threshold for the TP/FP decision.

        Returns:
            AP in [0, 1], or None when ``num_gt`` is 0 (undefined).
        """
        if num_gt == 0:
            return None
        if not detections:
            return 0.0

        ranked = sorted(detections, key=lambda d: d[0], reverse=True)
        tp = 0
        fp = 0
        recalls: List[float] = []
        precisions: List[float] = []
        for _conf, iou in ranked:
            if iou >= threshold:
                tp += 1
            else:
                fp += 1
            recalls.append(tp / num_gt)
            precisions.append(tp / (tp + fp))

        # Precision envelope: make precision monotonically non-increasing from
        # the right ("remove zig-zags"), matching COCO/torchmetrics.
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        # 101-point sampling: precision at the first recall >= each sample point
        # (0 beyond the maximum achieved recall).
        n = len(recalls)
        total = 0.0
        for rt in _RECALL_THRESHOLDS:
            idx = bisect_left(recalls, rt)
            total += precisions[idx] if idx < n else 0.0
        return total / len(_RECALL_THRESHOLDS)

    def _score_field(self, observations: BBoxObservations) -> Dict[str, Any]:
        """Compute per-threshold AP and summary stats for one field's observations."""
        num_gt = sum(1 for o in observations if o.has_gt)

        detections: List[tuple] = []
        iou_sum = 0.0
        for o in observations:
            if not o.has_pred:
                continue
            conf = o.confidence if o.confidence is not None else _DEFAULT_CONFIDENCE
            detections.append((conf, o.iou))
            iou_sum += o.iou

        num_det = len(detections)
        mean_iou = iou_sum / num_det if num_det > 0 else 0.0

        # AP at each configured IoU threshold; per-field "ap" averages them.
        ap_by_threshold: Dict[float, Optional[float]] = {
            t: self._average_precision(detections, num_gt, t)
            for t in self.iou_thresholds
        }
        valid = [ap for ap in ap_by_threshold.values() if ap is not None]
        ap = sum(valid) / len(valid) if valid else None

        return {
            "ap": ap,
            "ap_by_threshold": ap_by_threshold,
            "ap_50": ap_by_threshold.get(0.5),
            "ap_75": ap_by_threshold.get(0.75),
            "mean_iou": mean_iou,
            "num_gt": num_gt,
            "num_detections": num_det,
        }

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def compute_metrics(
        self,
        keyed_pairs: KeyedBBoxPairs,
        fields_with_bbox: int = 0,
        fields_total: int = 0,
    ) -> Dict[str, Any]:
        """Compute per-field AP and overall mAP from accumulated observations.

        ``mean_ap`` is the COCO-style mAP: AP is computed per (field-type class,
        IoU threshold), then averaged over the configured thresholds and over
        classes that carry at least one ground-truth box. Each class weighs
        equally regardless of how many observations it has. ``map_50`` /
        ``map_75`` are the single-threshold means (None when that threshold is
        not configured). This class-averaging is a different universe from
        ``coverage``, which counts per-(document, field) occurrences; see the
        bbox mAP metrics doc.

        Args:
            keyed_pairs: Class key -> list of BBoxObservation.
            fields_with_bbox: Count of (doc, field) rows that carried a GT bbox.
            fields_total: Total count of compared (doc, field) rows.

        Returns:
            {
                "mean_ap": float | None,        # mAP over the IoU-threshold range
                "map_50": float | None,
                "map_75": float | None,
                "iou_thresholds": [float, ...],
                "fields": {class_key: {ap, ap_50, ap_75, mean_iou,
                                       num_gt, num_detections}},
                "coverage": {fields_with_bbox, fields_total, ratio},
            }
            ``mean_ap`` is None when no field carried a ground-truth box.
        """
        per_field: Dict[str, Dict[str, Any]] = {}
        # Collect per-threshold AP across classes so the headline averages over
        # both thresholds and classes (COCO map).
        ap_by_threshold: Dict[float, List[float]] = {t: [] for t in self.iou_thresholds}
        for key, observations in keyed_pairs.items():
            scored = self._score_field(observations)
            for t, ap in scored.pop("ap_by_threshold").items():
                if ap is not None:
                    ap_by_threshold[t].append(ap)
            per_field[key] = scored

        per_threshold_map = {t: self._mean(aps) for t, aps in ap_by_threshold.items()}
        valid_maps = [m for m in per_threshold_map.values() if m is not None]
        mean_ap = self._mean(valid_maps)

        return {
            "mean_ap": mean_ap,
            "map_50": per_threshold_map.get(0.5),
            "map_75": per_threshold_map.get(0.75),
            "iou_thresholds": list(self.iou_thresholds),
            "fields": per_field,
            "coverage": {
                "fields_with_bbox": fields_with_bbox,
                "fields_total": fields_total,
                "ratio": (fields_with_bbox / fields_total if fields_total > 0 else 0.0),
            },
        }
