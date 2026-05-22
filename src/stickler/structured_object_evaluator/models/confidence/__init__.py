"""
Confidence evaluation module for structured model comparisons.

Provides pluggable metrics (AUROC, Brier, ECE, etc.) that measure how well
prediction confidence scores correlate with actual correctness.

Public API:
    ConfidenceCalculator       - extracts keyed pairs, runs metrics at all levels
    ExtractionResult           - pydantic model returned by ConfidenceCalculator.extract
    ConfidenceMetric           - base class for implementing new metrics
    AUROCMetric                - area under the ROC curve
    BrierScoreMetric           - mean squared calibration error
    ECEMetric                  - expected calibration error with bin data
    ErrorCaptureAtBudgetMetric - errors caught at X% review effort
    ConfidencePair             - pydantic model: is_match, confidence, similarity
    ConfidencePairs            - type alias: List[ConfidencePair]
    KeyedConfidencePairs       - type alias: Dict[str, ConfidencePairs]
    default_metrics            - factory returning the default metric list
"""

from stickler.structured_object_evaluator.models.confidence.calculator import (
    ConfidenceCalculator,
    ExtractionResult,
    KeyedConfidencePairs,
)
from stickler.structured_object_evaluator.models.confidence.metrics import (
    AUROCMetric,
    BrierScoreMetric,
    ConfidenceMetric,
    ConfidencePair,
    ConfidencePairs,
    ECEMetric,
    ErrorCaptureAtBudgetMetric,
    default_metrics,
)

__all__ = [
    "ConfidenceCalculator",
    "ExtractionResult",
    "ConfidenceMetric",
    "ConfidencePair",
    "AUROCMetric",
    "BrierScoreMetric",
    "ECEMetric",
    "ErrorCaptureAtBudgetMetric",
    "ConfidencePairs",
    "KeyedConfidencePairs",
    "default_metrics",
]
