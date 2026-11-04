"""Correctness and calibration metrics for decision fields.

Every function reads a ``DecisionArrays`` (or plain arrays, for the small
helpers) and returns plain Python floats, ints and dicts, so results are
JSON-serialisable. A metric that is undefined on its input (no rows, a single
class) is ``None``.

Rules shared by every kind:

- ``accuracy`` counts a missing answer as wrong; ``answered_accuracy`` and
  every calibration metric use answered rows only.
- Confidence is ``a.conf``, the probability of the predicted (argmax) answer.
- ``coverage`` is ``{"ratio", "ok", "missing"}`` over asked decisions.

numpy only: sklearn and scipy are not runtime dependencies.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from stickler.utils.statistics import binary_roc_auc

from .arrays import DecisionArrays

NO_ANSWER = "(no answer)"
NOUL_THRESHOLD = 0.5
_EPS = 1e-15


def _f(x: Any) -> Optional[float]:
    """A plain float, or None for None or a non-finite value."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _ratio(num: float, den: float) -> float:
    """``num / den``, or 0.0 when ``den`` is 0 (sklearn ``zero_division=0``)."""
    return float(num) / float(den) if den else 0.0


def _prf(tp: int, n_pred: int, n_gold: int) -> Dict[str, float]:
    precision = _ratio(tp, n_pred)
    recall = _ratio(tp, n_gold)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _coverage(a: DecisionArrays) -> Dict[str, Any]:
    n_ok, n_total = a.n_ok, a.n_total
    return {"ratio": _ratio(n_ok, n_total) if n_total else None, "ok": n_ok, "missing": a.n_missing}


# Helpers


def auroc(labels: Sequence[Any], scores: Sequence[float]) -> Optional[float]:
    """Binary ROC AUC of ``scores`` against boolean ``labels``; None with a single class."""
    labels = np.asarray(labels, dtype=bool)
    if labels.size == 0 or labels.all() or not labels.any():
        return None
    return float(binary_roc_auc(labels, np.asarray(scores, dtype=float)))


def brier_multiclass(P: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Mean over rows of ``sum_c (P[c] - onehot(y)[c])^2``."""
    P = np.asarray(P, dtype=float)
    if P.shape[0] == 0:
        return None
    onehot = np.eye(P.shape[1])[np.asarray(y, dtype=int)]
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def nll(P: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Mean ``-log P[gold]``, with the probability clipped at 1e-15."""
    P = np.asarray(P, dtype=float)
    if P.shape[0] == 0:
        return None
    p = P[np.arange(P.shape[0]), np.asarray(y, dtype=int)]
    return float(-np.log(np.clip(p, _EPS, 1.0)).mean())


def binary_brier(p: Sequence[float], y: Sequence[Any]) -> Optional[float]:
    """Mean ``(p - y)^2`` for probabilities ``p`` of the positive class."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return None
    return float(((p - np.asarray(y, dtype=float)) ** 2).mean())


def log_loss(p: Sequence[float], y: Sequence[Any]) -> Optional[float]:
    """Binary cross-entropy for probabilities ``p`` of the positive class, clipped at 1e-15."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    if p.size == 0:
        return None
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> Optional[float]:
    """Cohen's kappa with quadratic weights over levels ``0..k-1``.

    None on empty input, or when the expected disagreement is zero (gold and
    predictions all on one level).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = y_true.size
    if n == 0:
        return None
    observed = np.zeros((k, k))
    np.add.at(observed, (y_true, y_pred), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    levels = np.arange(k)
    weights = (levels[:, None] - levels[None, :]) ** 2
    denom = float((weights * expected).sum())
    if denom == 0.0:
        return None
    return float(1.0 - (weights * observed).sum() / denom)


def ranked_probability_score(P: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Ranked probability score for ordered levels, normalised to ``[0, 1]``.

    ``mean_rows(sum_{j < k-1} (CDF_pred_j - CDF_obs_j)^2) / (k - 1)``.
    """
    P = np.asarray(P, dtype=float)
    if P.shape[0] == 0:
        return None
    k = P.shape[1]
    cdf_pred = np.cumsum(P, axis=1)[:, : k - 1]
    cdf_obs = (np.asarray(y, dtype=int)[:, None] <= np.arange(k - 1)[None, :]).astype(float)
    return float(((cdf_pred - cdf_obs) ** 2).sum(axis=1).mean() / (k - 1))


def ece_mass(conf: Sequence[float], correct: Sequence[Any], n_bins: int = 10) -> Dict[str, Any]:
    """Expected calibration error over equal-mass bins.

    Rows are sorted by confidence and split into ``n_bins`` bins of about
    ``n / n_bins`` rows. A bin edge that would split a run of equal
    confidences moves forward to the end of the run, so tied rows always
    share a bin and the result never depends on row order. Bins can
    therefore be fewer than ``n_bins``.

    Returns:
        ``{"value": float | None, "bins": [{"range", "count",
        "mean_confidence", "accuracy"}, ...]}``. ``value`` is None on empty input.

    Raises:
        ValueError: If ``n_bins`` is less than 1.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = conf.size
    if n == 0:
        return {"value": None, "bins": []}

    order = np.argsort(conf, kind="mergesort")
    conf, correct = conf[order], correct[order]
    _, run_counts = np.unique(conf, return_counts=True)
    run_ends = np.cumsum(run_counts)
    # Integer ceil of i * n / n_bins, then snap forward to the end of a tie run.
    targets = [-(-i * n // n_bins) for i in range(1, n_bins)]
    edges = sorted({int(run_ends[np.searchsorted(run_ends, t)]) for t in targets} | {n})

    bins: List[Dict[str, Any]] = []
    value = 0.0
    start = 0
    for end in edges:
        if end <= start:
            continue
        c, ok = conf[start:end], correct[start:end]
        mean_conf, acc = float(c.mean()), float(ok.mean())
        value += (end - start) / n * abs(acc - mean_conf)
        bins.append({
            "range": [float(c[0]), float(c[-1])],
            "count": int(end - start),
            "mean_confidence": mean_conf,
            "accuracy": acc,
        })
        start = end
    return {"value": float(value), "bins": bins}


# Field metrics


def correctness(a: DecisionArrays) -> Dict[str, Any]:
    """Correctness of one decision field.

    Every kind: ``accuracy`` (missing counts as wrong), ``answered_accuracy``
    and ``coverage``. Choice adds ``per_class``, ``macro_f1`` and
    ``confusion``; Noul adds yes-class ``precision``, ``recall``, ``f1`` at
    ``P(yes) >= threshold``; Score adds ``qwk``, ``mae_argmax`` and
    ``mae_expected`` on answered rows.

    A missing answer is a false negative for its gold class (Choice, Noul).
    """
    n_total, ok = a.n_total, a.answered
    correct = a.correct
    out: Dict[str, Any] = {
        "accuracy": _ratio(correct.sum(), n_total) if n_total else None,
        "answered_accuracy": _ratio(correct[ok].sum(), ok.sum()) if ok.any() else None,
        "coverage": _coverage(a),
    }
    spec, y, pred = a.spec, a.y, a.pred

    if spec.kind == "choice":
        per_class: Dict[str, Dict[str, Any]] = {}
        for c, label in enumerate(spec.labels):
            support = int((y == c).sum())
            stats = _prf(int(((pred == c) & (y == c)).sum()), int((pred == c).sum()), support)
            per_class[label] = {**stats, "support": support}
        out["per_class"] = per_class
        # Average over the options seen in gold or predictions, as sklearn's
        # f1_score(average="macro") does by default: an option nobody chose
        # and no gold used says nothing about the model.
        seen = [m["f1"] for c, m in enumerate(per_class.values()) if (y == c).any() or (pred == c).any()]
        out["macro_f1"] = float(np.mean(seen)) if seen else None
        confusion: Dict[str, Dict[str, int]] = {}
        pred_names = list(spec.labels) + [NO_ANSWER]
        for g, gold_label in enumerate(spec.labels):
            in_gold = y == g
            row = {}
            for p, pred_label in enumerate(pred_names):
                count = int((in_gold & (pred == (p if p < spec.k else -1))).sum())
                if count:
                    row[pred_label] = count
            if row:
                confusion[gold_label] = row
        out["confusion"] = confusion

    elif spec.kind == "noul":
        says_yes = np.zeros(n_total, dtype=bool)
        says_yes[ok] = a.P[ok, 1] >= NOUL_THRESHOLD
        gold_yes = y == 1
        out.update(_prf(int((says_yes & gold_yes).sum()), int(says_yes.sum()), int(gold_yes.sum())))
        out["threshold"] = NOUL_THRESHOLD

    elif spec.kind == "score":
        y_ok, P_ok = y[ok], a.P[ok]
        out["qwk"] = quadratic_weighted_kappa(y_ok, pred[ok], spec.k)
        if ok.any():
            out["mae_argmax"] = float(np.abs(pred[ok] - y_ok).mean())
            out["mae_expected"] = float(np.abs(P_ok @ np.arange(spec.k) - y_ok).mean())
        else:
            out["mae_argmax"] = out["mae_expected"] = None

    return out


def calibration(a: DecisionArrays, n_bins: int = 10) -> Dict[str, Any]:
    """Calibration of one decision field, on answered rows only.

    Every kind: ``auroc`` (confidence vs correct), ``ece_mass`` (its value;
    the bins are under ``ece_mass_bins``) and ``coverage``. Choice adds
    ``brier_multiclass`` and ``nll``; Noul adds ``auroc_label`` (P(yes) vs
    gold), ``brier`` and ``log_loss`` on P(yes); Score adds ``rps``.
    """
    ok = a.answered
    P, y = a.P[ok], a.y[ok]
    conf, correct = a.conf[ok], a.correct[ok]
    ece = ece_mass(conf, correct, n_bins)
    out: Dict[str, Any] = {
        "auroc": auroc(correct, conf),
        "ece_mass": ece["value"],
        "ece_mass_bins": ece["bins"],
        "coverage": _coverage(a),
    }
    kind = a.spec.kind
    if kind == "choice":
        out["brier_multiclass"] = brier_multiclass(P, y)
        out["nll"] = nll(P, y)
    elif kind == "noul":
        p_yes = P[:, 1]
        out["auroc_label"] = auroc(y == 1, p_yes)
        out["brier"] = binary_brier(p_yes, y)
        out["log_loss"] = log_loss(p_yes, y)
    elif kind == "score":
        out["rps"] = ranked_probability_score(P, y)
    return out


# Error bars


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _collect(point: Dict[str, Any], sample: Any, acc: Dict[str, Any]) -> None:
    """Append each numeric leaf of ``sample`` found at a numeric leaf of ``point``."""
    for key, value in point.items():
        sub = sample.get(key) if isinstance(sample, dict) else None
        if isinstance(value, dict):
            _collect(value, sub, acc.setdefault(key, {}))
        elif _is_number(value) and _is_number(sub) and np.isfinite(sub):
            acc.setdefault(key, []).append(float(sub))


def _bounds(point: Dict[str, Any], acc: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in point.items():
        if isinstance(value, dict):
            nested = _bounds(value, acc.get(key, {}), alpha)
            if nested:
                out[key] = nested
        elif _is_number(value):
            values = acc.get(key)
            if values:
                low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                out[key] = {"low": float(low), "high": float(high)}
            else:
                out[key] = {"low": None, "high": None}
    return out


def bootstrap_ci(
    a: DecisionArrays,
    metric_fn: Callable[[DecisionArrays], Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Percentile bootstrap bounds for every numeric leaf of ``metric_fn(a)``.

    Resamples documents (``a.groups``) with replacement, not rows, because
    decisions within a document are correlated. Each resample keeps every
    row of each drawn document, once per draw.

    Returns:
        The numeric leaves of ``metric_fn(a)`` in the same nested shape, each
        as ``{"low", "high"}``. Non-numeric and None leaves are left out.
        A leaf that is None or absent in a resample is skipped for that
        resample; a leaf with no values in any resample has None bounds.
        Deterministic for a given ``seed``.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    point = metric_fn(a)
    docs, inverse = np.unique(a.groups, return_inverse=True)
    acc: Dict[str, Any] = {}
    if docs.size:
        rows_of = [np.flatnonzero(inverse == d) for d in range(docs.size)]
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            drawn = rng.integers(0, docs.size, size=docs.size)
            rows = np.concatenate([rows_of[d] for d in drawn])
            _collect(point, metric_fn(a.subset(rows)), acc)
    return _bounds(point, acc, alpha)


__all__: List[str] = [
    "auroc",
    "binary_brier",
    "bootstrap_ci",
    "brier_multiclass",
    "calibration",
    "correctness",
    "ece_mass",
    "log_loss",
    "nll",
    "quadratic_weighted_kappa",
    "ranked_probability_score",
]
