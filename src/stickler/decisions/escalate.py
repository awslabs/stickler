"""Accept or escalate: what a confidence threshold buys.

For each threshold tau, a decision is accepted when its confidence (the
probability of its predicted answer) is at least tau, and escalated otherwise.
A missing answer is always escalated. Stickler measures at the thresholds it
is given; it does not choose one. Choose tau on a dev run, then report it on
a test run.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .arrays import DecisionArrays
from .primitives import spec_from_question

DEFAULT_TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _decision_metrics(res: Any) -> Mapping[str, Any]:
    dm = res if isinstance(res, Mapping) else getattr(res, "decision_metrics", None)
    if not dm or "fields" not in dm:
        raise ValueError(
            "No decision metrics found. Evaluate with a target_schema that declares "
            "Choice, Score or Noul fields, then pass the result of compute()."
        )
    return dm


def arrays_from_result(res: Any, fields: Optional[Sequence[str]] = None) -> Dict[str, DecisionArrays]:
    """Rebuild each field's DecisionArrays from ``res.decision_metrics``.

    Args:
        res: A ProcessEvaluation or its ``decision_metrics`` dict.
        fields: Decision field paths to keep (default: all), as they appear in
            ``decision_metrics["fields"]``, e.g. ``msgs[].intent``.

    Raises:
        ValueError: If there are no decision metrics, or a field is unknown.
    """
    stored = _decision_metrics(res)["fields"]
    if fields is not None:
        unknown = sorted(set(fields) - set(stored))
        if unknown:
            raise ValueError(f"Unknown decision fields: {unknown}; known: {sorted(stored)}")
        stored = {f: stored[f] for f in fields}
    out: Dict[str, DecisionArrays] = {}
    for name, f in stored.items():
        spec = spec_from_question(f["kind"], None if f["kind"] == "noul" else list(f["labels"]))
        out[name] = DecisionArrays.from_records(spec, f["records"])
    return out


def risk_coverage(conf: np.ndarray, correct: np.ndarray):
    """The points a ``confidence >= tau`` rule can reach, highest tau first.

    One point per distinct confidence: accept every decision at or above it.
    Tied confidences are accepted together, as they would be at runtime.

    Returns:
        ``(taus, accepted, accuracy)`` arrays, one entry per point: the
        threshold, how many decisions it accepts, and their accuracy.
    """
    conf = np.asarray(conf, dtype=float)
    if conf.size == 0:
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0)
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], np.asarray(correct)[order].astype(float)
    ends = np.flatnonzero(np.r_[c[1:] != c[:-1], True]) + 1
    return c[ends - 1], ends, np.cumsum(ok)[ends - 1] / ends


def _aurc(accepted: np.ndarray, accuracy: np.ndarray, n: int) -> Optional[float]:
    """Area under the risk-coverage curve: the mean selective risk over coverage.

    Each decision contributes the risk (1 - accuracy) of the smallest accepted
    set that contains it, so ``AURC = sum(block_size * risk_at_block) / n``
    over the blocks of tied confidences. Without ties this is the usual
    mean-of-risks definition (Geifman et al., 2019). Lower is better.
    """
    if n == 0 or accepted.size == 0:
        return None
    sizes = np.diff(np.r_[0, accepted])
    return float(np.sum(sizes * (1.0 - accuracy)) / n)


def _curve(conf: np.ndarray, correct: np.ndarray, answered: np.ndarray, target: Optional[float]):
    """AURC both ways, and the coverage at a target accuracy.

    ``aurc`` runs over every asked decision, with missing answers accepted last
    as errors, so a model cannot improve it by not answering. ``aurc_answered``
    is the standard selective-prediction AURC over answered decisions only,
    with coverage relative to them. The two agree when nothing is missing.
    ``coverage_at_target`` divides by every asked decision and never accepts a
    missing answer.
    """
    n_total = int(answered.size)
    if n_total == 0:
        return {"aurc": None, "aurc_answered": None, "coverage_at_target": None, "tau_at_target": None}
    filled = np.where(answered, np.nan_to_num(conf, nan=-np.inf), -np.inf)
    _, acc_all, accuracy_all = risk_coverage(filled, correct & answered)
    taus, accepted, accuracy = risk_coverage(conf[answered], correct[answered])
    out = {
        "aurc": _aurc(acc_all, accuracy_all, n_total),
        "aurc_answered": _aurc(accepted, accuracy, int(answered.sum())),
        "coverage_at_target": None,
        "tau_at_target": None,
    }
    if target is not None:
        hits = np.flatnonzero(accuracy >= target)
        if hits.size:
            i = hits[-1]
            out["coverage_at_target"] = float(accepted[i] / n_total)
            out["tau_at_target"] = float(taus[i])
        else:
            out["coverage_at_target"] = 0.0
    return out


def _rows(conf: np.ndarray, correct: np.ndarray, answered: np.ndarray, taus: Sequence[float]) -> List[Dict[str, Any]]:
    n_total = int(answered.size)
    errors_total = int((~correct).sum())
    rows = []
    for tau in taus:
        accept = answered & (np.nan_to_num(conf, nan=-1.0) >= tau)
        n_acc = int(accept.sum())
        rows.append({
            "tau": float(tau),
            "coverage": n_acc / n_total if n_total else None,
            "accuracy_accepted": float(correct[accept].mean()) if n_acc else None,
            "accepted": n_acc,
            "escalated": n_total - n_acc,
            "errors_escalated": int((~correct & ~accept).sum()),
            "errors_total": errors_total,
        })
    return rows


@dataclass
class EscalationReport:
    """Accept-or-escalate results, per field (or one pooled entry)."""

    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_accuracy: Optional[float] = None

    def table(self) -> str:
        """A plain-text table, one block per field."""
        lines: List[str] = []
        for name, f in self.fields.items():
            aurc = f"AURC={_fmt(f['aurc'], 3)}"
            if f["aurc_answered"] != f["aurc"]:
                aurc += f", answered only {_fmt(f['aurc_answered'], 3)}"
            lines.append(f"{name}  (n={f['n_total']}, {aurc})")
            lines.append("  tau   coverage  accuracy_accepted  errors_escalated")
            for r in f["rows"]:
                lines.append(
                    f"  {r['tau']:<5.2f} {_fmt(r['coverage']):<9} {_fmt(r['accuracy_accepted']):<18} "
                    f"{r['errors_escalated']} / {r['errors_total']}"
                )
            if self.target_accuracy is not None:
                lines.append(
                    f"  coverage at accuracy >= {self.target_accuracy:g}: {_fmt(f['coverage_at_target'])}"
                    + (f" (tau = {f['tau_at_target']:.3f})" if f["tau_at_target"] is not None else "")
                )
            lines.append("")
        return "\n".join(lines).rstrip()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {"target_accuracy": self.target_accuracy, "fields": self.fields}


def _fmt(x: Optional[float], digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def accept_or_escalate(
    res: Any,
    taus: Union[Sequence[float], Mapping[str, Sequence[float]]] = DEFAULT_TAUS,
    target_accuracy: Optional[float] = None,
    *,
    fields: Optional[Sequence[str]] = None,
    pool: bool = False,
) -> EscalationReport:
    """Measure accepting decisions at or above each confidence threshold.

    Args:
        res: A ``ProcessEvaluation`` from ``BulkStructuredModelEvaluator.compute()``,
            or its ``decision_metrics`` dict.
        taus: Confidence thresholds to report: one list for every field, or a
            dict of field to list. A field missing from the dict gets
            ``DEFAULT_TAUS``.
        target_accuracy: If given, also report the largest coverage whose
            accepted decisions reach this accuracy, and the threshold that gives it.
        fields: Decision fields to include (default: all).
        pool: Pool all fields into one entry. Confidence scales differ by
            question type (a Noul's never goes below 0.5), so per-field is the default.

    Returns:
        An EscalationReport. ``report.table()`` prints it. Each field has
        ``rows`` (one per tau), ``aurc``, ``aurc_answered``,
        ``coverage_at_target`` and ``tau_at_target``; see ``_curve`` for how
        missing answers enter each.

    Raises:
        ValueError: If ``taus`` is a dict with ``pool=True``, or names a field
            that is not in the result.
    """
    arrays = arrays_from_result(res, fields)
    if isinstance(taus, Mapping):
        if pool:
            raise ValueError("Per-field taus cannot be pooled; pass one list of taus with pool=True")
        unknown = sorted(set(taus) - set(arrays))
        if unknown:
            raise ValueError(f"taus names unknown decision fields: {unknown}; known: {sorted(arrays)}")
    groups = {"all": list(arrays.values())} if pool else {f: [a] for f, a in arrays.items()}
    report = EscalationReport(target_accuracy=target_accuracy)
    for name, parts in groups.items():
        conf = np.concatenate([a.conf for a in parts]) if parts else np.zeros(0)
        correct = np.concatenate([a.correct for a in parts]) if parts else np.zeros(0, dtype=bool)
        answered = np.concatenate([a.answered for a in parts]) if parts else np.zeros(0, dtype=bool)
        field_taus = taus.get(name, DEFAULT_TAUS) if isinstance(taus, Mapping) else taus
        report.fields[name] = {
            "n_total": int(answered.size),
            "rows": _rows(conf, correct, answered, field_taus),
            **_curve(conf, correct, answered, target_accuracy),
        }
    return report
