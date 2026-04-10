"""
Pluggable confidence metrics.

Each metric operates on a list of ConfidencePair objects and returns a
result dict with at least {"value": float | None}. Metrics may include
additional structured data (e.g., bins for ECE).

ConfidencePair fields:
    is_match:   bool  — whether the field crossed its ComparableField threshold
    confidence: float — the model's self-reported confidence (from JSON)
    similarity: float — the raw comparator similarity score (0.0–1.0)

Existing metrics use is_match and confidence. The similarity score is
available for future metrics that correlate confidence with *how right*
the prediction is, not just whether it crossed a threshold.

To add a new metric:
    1. Subclass ConfidenceMetric
    2. Implement name (property) and compute(pairs)
    3. Pass it to ConfidenceCalculator(metrics=[...])
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel
from sklearn.metrics import roc_auc_score


class ConfidencePair(BaseModel):
    """A single observation pairing a match result with confidence and similarity."""

    is_match: bool
    confidence: float
    similarity: float


ConfidencePairs = List[ConfidencePair]


class ConfidenceMetric(ABC):
    """Base class for confidence calibration metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Key used in result dicts (e.g., 'auroc')."""
        ...

    @abstractmethod
    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        """Compute the metric.

        Args:
            pairs: List of ConfidencePair objects.

        Returns:
            Dict with at least {"value": float | None}.
        """
        ...


class AUROCMetric(ConfidenceMetric):
    """Area Under the ROC Curve.

    Measures how well confidence discriminates correct from incorrect.
    Returns None when AUROC is undefined (no pairs or single class).
    """

    @property
    def name(self) -> str:
        return "auroc"

    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        if not pairs or len(set(p.is_match for p in pairs)) < 2:
            return {"value": None}
        y_true = [1 if p.is_match else 0 for p in pairs]
        y_scores = [p.confidence for p in pairs]
        return {"value": roc_auc_score(y_true, y_scores)}


class BrierScoreMetric(ConfidenceMetric):
    """Brier Score — mean squared error between confidence and outcome.

    Lower is better. 0.0 = perfect, 0.25 = random on balanced classes.
    """

    @property
    def name(self) -> str:
        return "brier_score"

    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        if not pairs:
            return {"value": None}
        brier = sum(
            (p.confidence - (1.0 if p.is_match else 0.0)) ** 2 for p in pairs
        ) / len(pairs)
        return {"value": brier}


class ECEMetric(ConfidenceMetric):
    """Expected Calibration Error with bin data for reliability diagrams.

    Returns {"value": float, "bins": [...]} where each bin has
    range, count, accuracy, and mean_confidence.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "ece"

    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        if not pairs:
            return {"value": None, "bins": []}

        bins = []
        for i in range(self.n_bins):
            lo = i / self.n_bins
            hi = (i + 1) / self.n_bins
            bp = [
                p for p in pairs
                if (lo <= p.confidence < hi) or (i == self.n_bins - 1 and p.confidence == hi)
            ]
            if bp:
                acc = sum(1 for p in bp if p.is_match) / len(bp)
                mc = sum(p.confidence for p in bp) / len(bp)
            else:
                acc, mc = 0.0, 0.0
            bins.append({
                "range": [lo, hi],
                "count": len(bp),
                "accuracy": acc,
                "mean_confidence": mc,
            })

        total = len(pairs)
        ece = sum(
            (b["count"] / total) * abs(b["accuracy"] - b["mean_confidence"])
            for b in bins if b["count"] > 0
        )
        return {"value": ece, "bins": bins}


DEFAULT_METRICS: List[ConfidenceMetric] = [AUROCMetric()]
