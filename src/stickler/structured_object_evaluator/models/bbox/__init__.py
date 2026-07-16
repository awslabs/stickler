"""
Bounding-box mAP evaluation module for structured model comparisons.

Provides mean Average Precision (mAP) scoring for document field localization,
built on the rich-value ``_bbox`` pattern and the ``PostComparisonAccumulator``
interface. AP is a true confidence-ranked precision-recall curve area
(Pascal VOC 2010+ all-points).

Public API:
    MAPCalculator       - extracts keyed observations, computes AP / mean AP
    BBoxExtractionResult - pydantic model returned by MAPCalculator.extract
    BBoxObservation     - pydantic model: has_gt, has_pred, iou, confidence
    BBoxObservations    - type alias: List[BBoxObservation]
    KeyedBBoxPairs      - type alias: Dict[str, BBoxObservations]
    class_key           - normalizes list indices for per-field-type grouping
    BBoxMAPAccumulator  - PostComparisonAccumulator producing bbox_map_metrics
"""

from stickler.structured_object_evaluator.models.bbox.accumulator import (
    BBoxMAPAccumulator,
)
from stickler.structured_object_evaluator.models.bbox.calculator import (
    BBoxExtractionResult,
    BBoxObservation,
    BBoxObservations,
    KeyedBBoxPairs,
    MAPCalculator,
    class_key,
)

__all__ = [
    "MAPCalculator",
    "BBoxExtractionResult",
    "BBoxObservation",
    "BBoxObservations",
    "KeyedBBoxPairs",
    "class_key",
    "BBoxMAPAccumulator",
]
