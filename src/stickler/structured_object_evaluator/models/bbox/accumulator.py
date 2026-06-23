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

import warnings
from typing import Any, Dict, Optional

from stickler.structured_object_evaluator.models.bbox.calculator import (
    BBoxObservation,
    KeyedBBoxPairs,
    MAPCalculator,
)
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)


class BBoxMAPAccumulator(PostComparisonAccumulator):
    """Accumulates bbox observations and computes aggregate mAP metrics."""

    def __init__(self, iou_threshold: float = 0.5):
        self._calculator = MAPCalculator(iou_threshold=iou_threshold)
        self.reset()

    @property
    def name(self) -> str:
        return "bbox_map_metrics"

    def reset(self) -> None:
        self._keyed_pairs: KeyedBBoxPairs = {}
        self._fields_with: int = 0
        self._fields_total: int = 0

    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        field_comparisons = comparison_result.get("field_comparisons", [])
        if not field_comparisons:
            # prediction_raw present but no field_comparisons signals the
            # caller wanted bbox metrics but forgot document_field_comparisons.
            if prediction_raw is not None:
                warnings.warn(
                    "BBoxMAPAccumulator got prediction_raw but no "
                    "field_comparisons; re-run compare_with("
                    "document_field_comparisons=True).",
                    UserWarning,
                    stacklevel=2,
                )
            return

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

        for field_path, observations in extraction.keyed_pairs.items():
            self._keyed_pairs.setdefault(field_path, []).extend(observations)

        self._fields_with += extraction.fields_with_bbox
        self._fields_total += extraction.fields_total

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

    def get_state(self) -> Dict[str, Any]:
        return {
            "keyed_bbox_pairs": {
                field_path: [o.model_dump() for o in observations]
                for field_path, observations in self._keyed_pairs.items()
            },
            "bbox_fields_with": self._fields_with,
            "bbox_fields_total": self._fields_total,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self._keyed_pairs = {
            field_path: [BBoxObservation(**o) for o in observations]
            for field_path, observations in state.get("keyed_bbox_pairs", {}).items()
        }
        self._fields_with = state.get("bbox_fields_with", 0)
        self._fields_total = state.get("bbox_fields_total", 0)

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        for field_path, observations in other_state.get("keyed_bbox_pairs", {}).items():
            self._keyed_pairs.setdefault(field_path, []).extend(
                [BBoxObservation(**o) for o in observations]
            )
        self._fields_with += other_state.get("bbox_fields_with", 0)
        self._fields_total += other_state.get("bbox_fields_total", 0)
