"""
Pluggable confidence metrics.

Each metric operates on a list of ConfidencePair objects and returns a
result dict with at least {"value": float | None}. Metrics may include
additional structured data (e.g., bins for ECE).

ConfidencePair fields:
    is_match:   bool  - whether the field crossed its ComparableField threshold
    confidence: float - the model's self-reported confidence (from JSON)
    similarity: float - the raw comparator similarity score (0.0 to 1.0)

Existing metrics use is_match and confidence. The similarity score is
available for future metrics that correlate confidence with *how right*
the prediction is, not just whether it crossed a threshold.

TODO: The similarity field enables several future metric directions:
  - Parameterized AUROC: re-threshold using similarity >= custom_threshold
    instead of the pre-baked is_match, allowing AUROC computation at
    different correctness standards without re-running comparisons.
  - Confidence-similarity correlation (Spearman/Pearson): does higher
    confidence correspond to higher similarity? Pure continuous metric,
    no binary label needed.
  - Review Efficiency Metric: sort by confidence ascending, measure how
    quickly you discover errors. Could use similarity to define error
    severity instead of binary match. See the Review_Efficiency_Exploration
    notebook for a worked example.

To add a new metric:
    1. Subclass ConfidenceMetric
    2. Implement name (property) and compute(pairs)
    3. Pass it to ConfidenceCalculator(metrics=[...])
"""

from abc import ABC, abstractmethod
from math import isfinite
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator
from sklearn.metrics import roc_auc_score


class ConfidencePair(BaseModel):
    """A single observation pairing a match result with confidence and similarity.

    Confidence and similarity are both expected to be finite floats in
    [0.0, 1.0]. Out-of-range or non-finite inputs (NaN, inf) are rejected
    at construction time so a single bad row can't silently corrupt
    downstream metrics (Brier tolerates out-of-range values without
    complaint; AUROC crashes on NaN mid-run).
    """

    is_match: bool
    confidence: float
    similarity: float

    @field_validator("confidence", "similarity")
    @classmethod
    def _finite_in_unit_interval(cls, v: float, info) -> float:
        if not isfinite(v):
            raise ValueError(
                f"{info.field_name} must be a finite float in [0.0, 1.0], got {v!r}"
            )
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"{info.field_name} must be in [0.0, 1.0], got {v!r}"
            )
        return v


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

    Args:
        n_bins: Number of confidence bins to compute (default: 10).
            Must be a positive integer.

    Raises:
        ValueError: If n_bins is less than 1.
    """

    def __init__(self, n_bins: int = 10):
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
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


class ErrorCaptureAtBudgetMetric(ConfidenceMetric):
    """Error Capture at Review Budget.

    Answers: "If I review X% of my data (lowest confidence first),
    what percentage of errors do I catch?"

    Sort fields by confidence ascending. At each budget level, count
    what fraction of total errors fall in the bottom X% of fields.
    Random sampling would catch ``k/n`` of errors in expectation when
    ``k`` fields are reviewed, so gain is computed against the actual
    reviewed fraction rather than the requested budget — otherwise
    tight budgets and small datasets inflate the reported gain (e.g.
    ``n=1, budget=0.1`` reviews 100% of the data but would report a
    10x gain against a 10% baseline).

    Args:
        budgets: List of review budget levels as fractions (default: [0.1, 0.3, 0.5]).
            Each budget must be in the range (0.0, 1.0]. Budgets are
            sorted internally so the middle-budget headline and iteration
            order are deterministic regardless of input order.

    Raises:
        ValueError: If any budget is outside (0.0, 1.0].

    Result shape::

        {
            "value": float,              # headline gain at the middle budget
            "budgets": {
                <budget>: {
                    "fields_reviewed": int,
                    "errors_found": int,
                    "pct_errors_caught": float,
                    "pct_errors_random": float,   # actual reviewed fraction (k/n)
                    "gain": float                 # pct_errors_caught / (k/n)
                },
                ...
            }
        }
    """

    def __init__(self, budgets: Optional[List[float]] = None):
        budgets = budgets if budgets is not None else [0.10, 0.30, 0.50]
        for b in budgets:
            if not (0.0 < b <= 1.0):
                raise ValueError(
                    f"All budgets must be in the range (0.0, 1.0], got {b}"
                )
        # Sort so the "middle" budget and iteration order are deterministic
        # regardless of input order.
        self.budgets = sorted(budgets)

    @property
    def name(self) -> str:
        return "error_capture_at_budget"

    def compute(self, pairs: ConfidencePairs) -> Dict[str, Any]:
        if not pairs:
            return {"value": None, "budgets": {}}

        total_errors = sum(1 for p in pairs if not p.is_match)
        if total_errors == 0:
            return {"value": None, "budgets": {}}

        # Sort by confidence ascending (lowest first)
        sorted_pairs = sorted(pairs, key=lambda p: p.confidence)
        n = len(sorted_pairs)

        budgets_result = {}
        for budget in self.budgets:
            # Cap k at n so fields_reviewed never exceeds total fields.
            # max(1, ...) guarantees we review at least one row; this means
            # the actual reviewed fraction (k/n) can exceed the requested
            # budget at small n, so we compare gain against k/n rather
            # than the requested budget to avoid spurious "10x gain"
            # numbers when n=1 and budget=0.1.
            k = min(n, max(1, int(n * budget)))
            reviewed_fraction = k / n
            errors_found = sum(1 for p in sorted_pairs[:k] if not p.is_match)
            pct_errors_caught = errors_found / total_errors
            gain = (
                pct_errors_caught / reviewed_fraction
                if reviewed_fraction > 0
                else 0.0
            )
            budgets_result[budget] = {
                "fields_reviewed": k,
                "errors_found": errors_found,
                "pct_errors_caught": pct_errors_caught,
                # Report the actual reviewed fraction used as the random
                # baseline so the gain calculation is transparent.
                "pct_errors_random": reviewed_fraction,
                "gain": gain,
            }

        # Headline value: gain at the middle budget level
        middle = self.budgets[len(self.budgets) // 2]
        headline = budgets_result[middle]["gain"]

        return {"value": headline, "budgets": budgets_result}


def default_metrics() -> List[ConfidenceMetric]:
    """Return a fresh list of the default metrics.

    Using a factory ensures each calculator gets its own metric instances,
    which matters if a metric holds state (cached thresholds, precomputed
    values, etc.). Today AUROCMetric is stateless, but this keeps the
    contract safe for future metrics.
    """
    return [AUROCMetric()]
