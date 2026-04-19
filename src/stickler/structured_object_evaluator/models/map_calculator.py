"""
Mean Average Precision (mAP) calculator for bounding box evaluation.

Computes per-field IoU, precision, recall, F1, AP, and overall mAP by
joining bounding box metadata from rich values with field comparison
results. This follows the same extraction pattern as ConfidenceCalculator.

Usage:
    >>> calculator = MAPCalculator(iou_threshold=0.5)
    >>> extraction = calculator.extract(comparison_result, gt_instance, pred_instance)
    >>> metrics = calculator.compute_metrics(extraction)
    >>> print(metrics["mean_ap"])

The calculator uses bounding boxes stored on StructuredModel instances
via the rich value pattern:
    {"value": "John Doe", "bbox": [[x1, y1], [x2, y2]], "confidence": 0.9}
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from stickler.comparators.bbox import BBoxIoUComparator


class BBoxFieldResult(BaseModel):
    """Result of evaluating a single field's bounding box."""

    field_path: str
    iou: float
    precision: float
    recall: float
    f1: float
    ap: float
    gt_bbox: Optional[Any] = None
    pred_bbox: Optional[Any] = None


class MAPExtractionResult(BaseModel):
    """Result of extracting bbox pairs from a comparison."""

    field_results: List[BBoxFieldResult]
    fields_with_bbox: int
    fields_total: int


class MAPCalculator:
    """Computes Mean Average Precision for bounding box evaluation.

    Joins field_comparisons with bounding box metadata from ground truth
    and prediction instances to compute IoU-based metrics per field.

    Args:
        iou_threshold: IoU threshold for considering a detection as correct
            (default: 0.5, corresponding to mAP@0.5).
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._comparator = BBoxIoUComparator(threshold=iou_threshold)

    def extract(
        self,
        comparison_result: Dict,
        gt_instance: "StructuredModel",  # noqa: F821
        pred_instance: "StructuredModel",  # noqa: F821
    ) -> MAPExtractionResult:
        """Extract bbox pairs and compute per-field IoU metrics.

        Joins field_comparisons with bbox data from both ground truth and
        prediction instances. Fields without bbox data on both sides are
        skipped but counted toward the total.

        Args:
            comparison_result: Must contain "field_comparisons".
            gt_instance: Ground truth model with bbox data.
            pred_instance: Prediction model with bbox data.

        Returns:
            MAPExtractionResult with per-field metrics and coverage.

        Raises:
            ValueError: If no field_comparisons in comparison_result.
        """
        field_comparisons = comparison_result.get("field_comparisons", [])
        if not field_comparisons:
            raise ValueError("No field comparisons found in comparison result.")

        gt_bboxes = gt_instance.get_all_bboxes()
        pred_bboxes = pred_instance.get_all_bboxes()

        field_results: List[BBoxFieldResult] = []
        fields_with_bbox = 0
        fields_total = 0

        for fc in field_comparisons:
            fields_total += 1
            field_path = fc["actual_key"]

            gt_bbox = gt_bboxes.get(field_path)
            pred_bbox = pred_bboxes.get(field_path)

            if gt_bbox is None:
                # No ground truth bbox — skip this field for mAP
                continue

            fields_with_bbox += 1

            if pred_bbox is None:
                # Ground truth exists but no prediction — miss
                field_results.append(
                    BBoxFieldResult(
                        field_path=field_path,
                        iou=0.0,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        ap=0.0,
                        gt_bbox=gt_bbox,
                        pred_bbox=None,
                    )
                )
                continue

            # Compute IoU
            iou = self._comparator.compare(pred_bbox, gt_bbox)

            # Binary classification at the IoU threshold
            tp = 1 if iou >= self.iou_threshold else 0
            fp = 1 - tp
            fn = 0 if tp == 1 else 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            ap = precision * recall

            field_results.append(
                BBoxFieldResult(
                    field_path=field_path,
                    iou=iou,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    ap=ap,
                    gt_bbox=gt_bbox,
                    pred_bbox=pred_bbox,
                )
            )

        return MAPExtractionResult(
            field_results=field_results,
            fields_with_bbox=fields_with_bbox,
            fields_total=fields_total,
        )

    def compute_metrics(
        self, extraction: MAPExtractionResult
    ) -> Dict[str, Any]:
        """Compute aggregate mAP metrics from extraction results.

        Args:
            extraction: Result from extract().

        Returns:
            Dictionary with:
            - mean_ap: Mean Average Precision across all bbox fields.
            - iou_threshold: The IoU threshold used.
            - field_results: Per-field breakdown with IoU, precision, recall, F1, AP.
            - coverage: How many fields had bbox data vs total.
        """
        field_results = extraction.field_results

        if not field_results:
            return {
                "mean_ap": None,
                "iou_threshold": self.iou_threshold,
                "field_results": {},
                "coverage": {
                    "fields_with_bbox": extraction.fields_with_bbox,
                    "fields_total": extraction.fields_total,
                    "ratio": 0.0,
                },
            }

        mean_ap = sum(r.ap for r in field_results) / len(field_results)

        per_field: Dict[str, Dict[str, Any]] = {}
        for r in field_results:
            per_field[r.field_path] = {
                "iou": r.iou,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "ap": r.ap,
            }

        return {
            "mean_ap": mean_ap,
            "iou_threshold": self.iou_threshold,
            "field_results": per_field,
            "coverage": {
                "fields_with_bbox": extraction.fields_with_bbox,
                "fields_total": extraction.fields_total,
                "ratio": (
                    extraction.fields_with_bbox / extraction.fields_total
                    if extraction.fields_total > 0
                    else 0.0
                ),
            },
        }
