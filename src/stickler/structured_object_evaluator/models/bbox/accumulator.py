"""Bounding-box mAP accumulator for bulk evaluation.

Mirrors ``ConfidenceAccumulator``: extracts per-field bounding-box observations
from each comparison result, accumulates them across documents, and computes
mean Average Precision once at the end.

Unlike confidence (which lives only on the prediction side), mAP needs *both*
the ground-truth and prediction bounding boxes plus the prediction confidences.
The accumulator never sees the model instances — it only receives the
``comparison_result`` — so ``compare_with`` pre-extracts everything it needs into
the result under ``ground_truth_bboxes``, ``prediction_bboxes`` and
``prediction_confidences``. Older serialized results lacking ``ground_truth_bboxes``
cannot have their localization recovered (the GT boxes aren't in
``prediction_raw``), so such rows contribute coverage only.
"""

from typing import Any, Dict, Iterable, Optional, Union

from stickler.structured_object_evaluator.models._keyed_pairs_accumulator import (
    _KeyedPairsAccumulator,
)
from stickler.structured_object_evaluator.models.bbox.calculator import (
    DEFAULT_IOU_THRESHOLDS,
    BBoxObservation,
    MAPCalculator,
)


class BBoxMAPAccumulator(_KeyedPairsAccumulator):
    """Accumulates bbox observations and computes aggregate mAP metrics."""

    _pair_type = BBoxObservation
    _state_keys = ("keyed_bbox_pairs", "bbox_fields_with", "bbox_fields_total")
    _warn_label = "BBoxMAPAccumulator"

    def __init__(
        self, iou_thresholds: Union[float, Iterable[float]] = DEFAULT_IOU_THRESHOLDS
    ):
        self._calculator = MAPCalculator(iou_thresholds=iou_thresholds)
        self.reset()

    @property
    def name(self) -> str:
        return "bbox_map_metrics"

    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        if not self._has_field_comparisons(comparison_result, prediction_raw):
            return

        field_comparisons = comparison_result.get("field_comparisons", [])

        # mAP needs ground-truth boxes, which are not recoverable from
        # prediction_raw. compare_with stashes all three maps; when they are
        # absent (older serialized results) the document contributes coverage
        # only and the calculator simply finds nothing to score.
        gt_bboxes = comparison_result.get("ground_truth_bboxes") or {}
        pred_bboxes = comparison_result.get("prediction_bboxes") or {}
        confidences = comparison_result.get("prediction_confidences") or {}

        extraction = self._calculator.extract_from_dicts(
            field_comparisons, gt_bboxes, pred_bboxes, confidences
        )

        self._extend(
            extraction.keyed_pairs,
            extraction.fields_with_bbox,
            extraction.fields_total,
        )

    def compute(self) -> Optional[Dict[str, Any]]:
        # Coverage-only (fields seen but none carried a bbox) still surfaces a
        # 0/N signal rather than None, matching ConfidenceAccumulator.
        if self._fields_total == 0:
            return None
        return self._calculator.compute_metrics(
            self._keyed_pairs,
            fields_with_bbox=self._fields_with,
            fields_total=self._fields_total,
        )
