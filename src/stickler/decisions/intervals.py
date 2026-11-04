"""Bootstrap confidence intervals for one run's decision metrics.

Decisions in one document are correlated (one unreadable ticket gets several
questions wrong together), so documents, not decisions, are resampled.
"""

from typing import Any, Dict, Optional, Sequence

from .escalate import arrays_from_result


def confidence_intervals(
    res: Any,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Bootstrap a CI for every decision metric, resampling documents.

    Args:
        res: A ``ProcessEvaluation`` from ``BulkStructuredModelEvaluator.compute()``,
            or its ``decision_metrics`` dict.
        n_boot: Number of bootstrap resamples.
        seed: Random seed; the same seed gives the same intervals.
        alpha: 0.05 gives 95% intervals.
        fields: Decision fields to include (default: all).

    Returns:
        ``{field: {"correctness": {...}, "calibration": {...}}}``, where every
        numeric metric becomes ``{"low": float, "high": float}``. Also records
        ``n_boot``, ``seed`` and ``alpha`` under ``"settings"``.
    """
    from .metrics import bootstrap_ci, calibration, correctness

    arrays = arrays_from_result(res, fields)
    out: Dict[str, Any] = {"settings": {"n_boot": n_boot, "seed": seed, "alpha": alpha, "unit": "document"}}
    for name, a in arrays.items():
        out[name] = {
            "correctness": bootstrap_ci(a, correctness, n_boot=n_boot, seed=seed, alpha=alpha),
            "calibration": bootstrap_ci(a, calibration, n_boot=n_boot, seed=seed, alpha=alpha),
        }
    return out
