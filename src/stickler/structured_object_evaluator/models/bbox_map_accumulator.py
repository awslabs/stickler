"""Bounding-box mAP accumulator for bulk evaluation.

Mirrors ``ConfidenceAccumulator``: extracts per-field bounding-box pairs from
each comparison result, accumulates them across documents, and computes mean
Average Precision once at the end.

Unlike confidence (which lives only on the prediction side), mAP needs *both*
the ground-truth and prediction bounding boxes. The accumulator never sees the
GT model instance — it only receives the ``comparison_result`` and the raw
prediction JSON — so ``compare_with`` pre-extracts both bbox maps into the
result under ``ground_truth_bboxes`` and ``prediction_bboxes``. When those keys
are absent (older serialized results), the prediction side is recovered from
``prediction_raw`` via ``RichValueHelper``; ground-truth boxes cannot be
recovered that way, so such rows contribute coverage only.
"""

import warnings
from typing import Any, Dict, List, Optional

from stickler.structured_object_evaluator.models.map_calculator import (
    BBoxPair,
    KeyedBBoxPairs,
    MAPCalculator,
)
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)
from stickler.structured_object_evaluator.models.rich_value_helper import (
    RichValueHelper,
)


class BBoxMAPAccumulator(PostComparisonAccumulator):
    """Accumulates bbox pairs and computes aggregate mAP metrics."""

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

        # Prefer the pre-extracted maps compare_with provides; fall back to
        # walking prediction_raw for older serialized results.
        gt_bboxes = comparison_result.get("ground_truth_bboxes")
        pred_bboxes = comparison_result.get("prediction_bboxes")

        if pred_bboxes is None:
            if prediction_raw is not None:
                _unwrapped, _confidences, extras = RichValueHelper.process_rich_values(
                    prediction_raw
                )
                pred_bboxes = self._calculator.bboxes_from_extras(extras)
            else:
                pred_bboxes = {}

        if gt_bboxes is None:
            # Ground-truth boxes aren't recoverable from prediction_raw.
            # Without them there is nothing to localize against, so this
            # document contributes coverage (fields_total) only.
            gt_bboxes = {}

        extraction = self._calculator.extract_from_dicts(
            field_comparisons, gt_bboxes, pred_bboxes
        )

        for field_path, pairs in extraction.keyed_pairs.items():
            self._keyed_pairs.setdefault(field_path, []).extend(pairs)

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
                field_path: [p.model_dump() for p in pairs]
                for field_path, pairs in self._keyed_pairs.items()
            },
            "bbox_fields_with": self._fields_with,
            "bbox_fields_total": self._fields_total,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self._keyed_pairs = {
            field_path: [BBoxPair(**p) for p in pairs]
            for field_path, pairs in state.get("keyed_bbox_pairs", {}).items()
        }
        self._fields_with = state.get("bbox_fields_with", 0)
        self._fields_total = state.get("bbox_fields_total", 0)

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        for field_path, pairs in other_state.get("keyed_bbox_pairs", {}).items():
            self._keyed_pairs.setdefault(field_path, []).extend(
                [BBoxPair(**p) for p in pairs]
            )
        self._fields_with += other_state.get("bbox_fields_with", 0)
        self._fields_total += other_state.get("bbox_fields_total", 0)


# Type alias for symmetry with KeyedConfidencePairs imports elsewhere.
KeyedBBoxPairsState = List[Dict[str, Any]]
