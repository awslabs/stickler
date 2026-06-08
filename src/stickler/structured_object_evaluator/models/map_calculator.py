"""
Mean Average Precision (mAP) calculator for bounding box evaluation.

This is the orchestrator for bounding-box localization scoring. It mirrors
``ConfidenceCalculator``:

1. Joins ``field_comparisons`` with ground-truth and prediction bounding
   boxes to produce keyed ``BBoxPair`` observations.
2. Tracks coverage (how many compared fields carried a ground-truth bbox
   vs. the total number of compared fields).
3. Computes per-field and overall IoU/precision/recall/F1/AP and mean AP
   from accumulated pairs.

Bounding boxes ride on the rich value pattern under the ``_bbox`` key, e.g.::

    {"_value": "John Doe", "_bbox": [[x1, y1], [x2, y2]], "_confidence": 0.9}

During ``from_json()`` the ``_bbox`` lands in the instance extras, reachable
via ``StructuredModel.get_all_extras()``. ``MAPCalculator.bboxes_from_extras``
projects that extras mapping down to ``{field_path: bbox}``.

Usage (single document, convenience):

    >>> calculator = MAPCalculator(iou_threshold=0.5)
    >>> extraction = calculator.extract(comparison_result, gt_instance, pred_instance)
    >>> metrics = calculator.compute_metrics(
    ...     extraction.keyed_pairs,
    ...     fields_with_bbox=extraction.fields_with_bbox,
    ...     fields_total=extraction.fields_total,
    ... )
    >>> print(metrics["mean_ap"])

Usage (bulk): drive it through ``BBoxMAPAccumulator`` instead, which calls
``extract_from_dicts`` per document and ``compute_metrics`` once at the end.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from stickler.comparators.bbox import BBoxIoUComparator

# The rich-value metadata key bounding boxes ride on.
BBOX_KEY = "_bbox"


class BBoxPair(BaseModel):
    """A single ground-truth/prediction bounding-box observation for one field.

    A pair is created only when the ground truth carries a bounding box (the
    prediction side may be missing, which is a localization miss). ``iou`` is
    the Intersection over Union between the prediction and ground-truth boxes,
    or 0.0 when the prediction box is absent. Threshold classification is
    deferred to ``compute`` time so the same accumulated pairs can be scored
    at different IoU thresholds.
    """

    iou: float
    has_pred: bool


BBoxPairs = List[BBoxPair]
KeyedBBoxPairs = Dict[str, BBoxPairs]


class BBoxExtractionResult(BaseModel):
    """Result of extracting bbox pairs from a single comparison."""

    keyed_pairs: KeyedBBoxPairs
    fields_with_bbox: int
    fields_total: int


class MAPCalculator:
    """Extracts bounding-box pairs and computes Mean Average Precision.

    Args:
        iou_threshold: IoU threshold for considering a detection correct
            (default: 0.5, corresponding to mAP@0.5).
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._comparator = BBoxIoUComparator(threshold=iou_threshold)

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
        gt_instance: "StructuredModel",  # noqa: F821
        pred_instance: "StructuredModel",  # noqa: F821
    ) -> BBoxExtractionResult:
        """Extract bbox pairs from a comparison plus two model instances.

        Convenience wrapper for the single-document path. Pulls bounding
        boxes from each instance's extras and delegates to
        ``extract_from_dicts``.

        Raises:
            ValueError: If no ``field_comparisons`` in ``comparison_result``.
        """
        field_comparisons = comparison_result.get("field_comparisons", [])
        if not field_comparisons:
            raise ValueError("No field comparisons found in comparison result.")

        gt_bboxes = self.bboxes_from_extras(gt_instance.get_all_extras())
        pred_bboxes = self.bboxes_from_extras(pred_instance.get_all_extras())
        return self.extract_from_dicts(field_comparisons, gt_bboxes, pred_bboxes)

    def extract_from_dicts(
        self,
        field_comparisons: List[Dict],
        gt_bboxes: Dict[str, Any],
        pred_bboxes: Dict[str, Any],
    ) -> BBoxExtractionResult:
        """Join raw ``field_comparisons`` with gt/pred bbox dicts.

        A field contributes a ``BBoxPair`` only when the ground truth carries
        a bounding box for that field; fields without a ground-truth bbox are
        counted toward the total but skipped for mAP (there is nothing to
        localize against). Rows without a string ``actual_key`` (e.g. list FN
        placeholders) are skipped entirely, matching ``ConfidenceCalculator``.

        Args:
            field_comparisons: The ``field_comparisons`` list from a
                ``compare_with`` result.
            gt_bboxes: Ground-truth ``{field_path: bbox}``.
            pred_bboxes: Prediction ``{field_path: bbox}``.

        Returns:
            BBoxExtractionResult with keyed pairs and coverage counts.
        """
        keyed: KeyedBBoxPairs = {}
        fields_with_bbox = 0
        fields_total = 0

        for fc in field_comparisons:
            field_path = fc.get("actual_key")
            if not isinstance(field_path, str):
                continue
            fields_total += 1

            gt_bbox = gt_bboxes.get(field_path)
            if gt_bbox is None:
                # No ground-truth bbox -> nothing to localize against.
                continue

            fields_with_bbox += 1

            pred_bbox = pred_bboxes.get(field_path)
            if pred_bbox is None:
                # Ground truth exists but prediction is missing -> miss.
                keyed.setdefault(field_path, []).append(
                    BBoxPair(iou=0.0, has_pred=False)
                )
                continue

            iou = self._comparator.compare(pred_bbox, gt_bbox)
            keyed.setdefault(field_path, []).append(BBoxPair(iou=iou, has_pred=True))

        return BBoxExtractionResult(
            keyed_pairs=keyed,
            fields_with_bbox=fields_with_bbox,
            fields_total=fields_total,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _score_pairs(self, pairs: BBoxPairs) -> Dict[str, Any]:
        """Compute IoU/precision/recall/F1/AP for one field's pairs.

        Classification per pair at ``iou_threshold``:
            - prediction present and IoU >= threshold -> true positive
            - prediction present and IoU <  threshold -> false positive + false negative
            - prediction missing                      -> false negative
        """
        tp = fp = fn = 0
        iou_sum = 0.0
        for p in pairs:
            iou_sum += p.iou
            if not p.has_pred:
                fn += 1
            elif p.iou >= self.iou_threshold:
                tp += 1
            else:
                fp += 1
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        ap = precision * recall
        mean_iou = iou_sum / len(pairs) if pairs else 0.0

        return {
            "iou": mean_iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "support": len(pairs),
        }

    def compute_metrics(
        self,
        keyed_pairs: KeyedBBoxPairs,
        fields_with_bbox: int = 0,
        fields_total: int = 0,
    ) -> Dict[str, Any]:
        """Compute per-field and overall mAP from accumulated pairs.

        Args:
            keyed_pairs: Field path -> list of BBoxPair.
            fields_with_bbox: Count of fields that carried a ground-truth bbox.
            fields_total: Total count of compared fields.

        Returns:
            {
                "mean_ap": float | None,
                "iou_threshold": float,
                "fields": {field_path: {iou, precision, recall, f1, ap, support}},
                "coverage": {fields_with_bbox, fields_total, ratio},
            }
            ``mean_ap`` is None when no field carried a bounding box.
        """
        per_field: Dict[str, Dict[str, Any]] = {}
        for field_path, pairs in keyed_pairs.items():
            per_field[field_path] = self._score_pairs(pairs)

        mean_ap: Optional[float]
        if per_field:
            mean_ap = sum(f["ap"] for f in per_field.values()) / len(per_field)
        else:
            mean_ap = None

        return {
            "mean_ap": mean_ap,
            "iou_threshold": self.iou_threshold,
            "fields": per_field,
            "coverage": {
                "fields_with_bbox": fields_with_bbox,
                "fields_total": fields_total,
                "ratio": (fields_with_bbox / fields_total if fields_total > 0 else 0.0),
            },
        }
