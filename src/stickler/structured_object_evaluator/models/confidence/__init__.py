"""
Confidence evaluation module for structured model comparisons.

Provides pluggable metrics (AUROC, Brier, ECE, etc.) that measure how well
prediction confidence scores correlate with actual correctness.

Public API:
    ConfidenceCalculator  — extracts keyed pairs, runs metrics at all levels
    ConfidenceMetric      — base class for implementing new metrics
    AUROCMetric           — area under the ROC curve
    BrierScoreMetric      — mean squared calibration error
    ECEMetric             — expected calibration error with bin data
    ConfidencePairs       — type alias: List[Tuple[bool, float]]
    KeyedConfidencePairs  — type alias: Dict[str, ConfidencePairs]
"""

from stickler.structured_object_evaluator.models.confidence.calculator import (
    ConfidenceCalculator,
    ExtractionResult,
    KeyedConfidencePairs,
)
from stickler.structured_object_evaluator.models.confidence.metrics import (
    DEFAULT_METRICS,
    AUROCMetric,
    BrierScoreMetric,
    ConfidenceMetric,
    ConfidencePair,
    ConfidencePairs,
    ECEMetric,
)

__all__ = [
    "ConfidenceCalculator",
    "ExtractionResult",
    "ConfidenceMetric",
    "ConfidencePair",
    "AUROCMetric",
    "BrierScoreMetric",
    "ECEMetric",
    "ConfidencePairs",
    "KeyedConfidencePairs",
    "DEFAULT_METRICS",
]
