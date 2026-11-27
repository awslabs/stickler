"""Bootstrap confidence intervals for one run's decision metrics.

Decisions in one document are correlated (one unreadable ticket gets several
questions wrong together), so documents, not decisions, are resampled.
"""

from typing import Any, Dict, Optional, Sequence

from .escalate import _decision_metrics, arrays_from_result

DEFAULT_N_BINS = 10


def confidence_intervals(
    res: Any,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    fields: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    n_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Bootstrap a CI for each decision metric, resampling documents.

    Args:
        res: A ``ProcessEvaluation`` from ``BulkStructuredModelEvaluator.compute()``,
            or its ``decision_metrics`` dict.
        n_boot: Number of bootstrap resamples.
        seed: Random seed; the same seed gives the same intervals.
        alpha: 0.05 gives 95% intervals.
        fields: Decision fields to include (default: all).
        metrics: Metric names to bound, such as ``["accuracy", "ece_mass"]``.
            The default is ``RATE_METRICS``: every rate and score, and no
            counts, confusion cells or bin edges.
        n_bins: Bins for ``ece_mass``. The default is the value the run's
            accumulator used, so the interval is for the same statistic as
            the point estimate.

    Returns:
        ``{field: {"correctness": {...}, "calibration": {...}}}``, where each
        bounded metric becomes ``{"low": float, "high": float}``. Also records
        ``n_boot``, ``seed``, ``alpha`` and ``n_bins`` under ``"settings"``.
        The intervals are percentile intervals. ``ece_mass`` is biased upward
        under resampling, so on small runs its interval can sit above the
        estimate.
    """
    from .metrics import RATE_METRICS, bootstrap_ci, calibration, correctness

    if n_bins is None:
        settings = _decision_metrics(res).get("settings") or {}
        n_bins = int(settings.get("n_bins", DEFAULT_N_BINS))
    keep = RATE_METRICS if metrics is None else frozenset(metrics)
    arrays = arrays_from_result(res, fields)
    out: Dict[str, Any] = {
        "settings": {"n_boot": n_boot, "seed": seed, "alpha": alpha, "n_bins": n_bins, "unit": "document"}
    }

    def calibration_at_bins(a):
        return calibration(a, n_bins=n_bins)

    for name, a in arrays.items():
        out[name] = {
            "correctness": bootstrap_ci(a, correctness, n_boot=n_boot, seed=seed, alpha=alpha, keep=keep),
            "calibration": bootstrap_ci(a, calibration_at_bins, n_boot=n_boot, seed=seed, alpha=alpha, keep=keep),
        }
    return out
