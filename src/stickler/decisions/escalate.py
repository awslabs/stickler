"""Accept or escalate: what a confidence threshold buys.

For each threshold tau, a decision is accepted when its confidence (the
probability of its predicted answer) is at least tau, and escalated otherwise.
A missing answer is always escalated. Stickler measures at the thresholds it
is given; it does not choose one. Choose tau on a dev run, then report it on
a test run.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

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


def _curve(conf: np.ndarray, correct: np.ndarray, n_total: int, target: Optional[float]):
    """AURC and coverage at a target accuracy, over every possible threshold."""
    if n_total == 0 or conf.size == 0:
        return None, None, None
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order].astype(float)
    # Cut only between distinct confidences, so tied decisions move together.
    ends = np.flatnonzero(np.r_[c[1:] != c[:-1], True]) + 1
    accepted = ends.astype(float)
    acc = np.cumsum(ok)[ends - 1] / accepted
    coverage = accepted / n_total
    risk = 1.0 - acc
    # Area under risk against coverage, from coverage 0 (risk taken as the first point's).
    xs = np.r_[0.0, coverage]
    ys = np.r_[risk[0], risk]
    aurc = float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))
    cov_at, tau_at = None, None
    if target is not None:
        hits = np.flatnonzero(acc >= target)
        if hits.size:
            i = hits[-1]
            cov_at, tau_at = float(coverage[i]), float(c[ends[i] - 1])
        else:
            cov_at = 0.0
    return aurc, cov_at, tau_at


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
            lines.append(f"{name}  (n={f['n_total']}, AURC={_fmt(f['aurc'])})")
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


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.2f}"


def accept_or_escalate(
    res: Any,
    taus: Sequence[float] = DEFAULT_TAUS,
    target_accuracy: Optional[float] = None,
    *,
    fields: Optional[Sequence[str]] = None,
    pool: bool = False,
) -> EscalationReport:
    """Measure accepting decisions at or above each confidence threshold.

    Args:
        res: A ``ProcessEvaluation`` from ``BulkStructuredModelEvaluator.compute()``,
            or its ``decision_metrics`` dict.
        taus: Confidence thresholds to report.
        target_accuracy: If given, also report the largest coverage whose
            accepted decisions reach this accuracy, and the threshold that gives it.
        fields: Decision fields to include (default: all).
        pool: Pool all fields into one entry. Confidence scales differ by
            question type (a Noul's never goes below 0.5), so per-field is the default.

    Returns:
        An EscalationReport. ``report.table()`` prints it.
    """
    arrays = arrays_from_result(res, fields)
    groups = {"all": list(arrays.values())} if pool else {f: [a] for f, a in arrays.items()}
    report = EscalationReport(target_accuracy=target_accuracy)
    for name, parts in groups.items():
        conf = np.concatenate([a.conf for a in parts]) if parts else np.zeros(0)
        correct = np.concatenate([a.correct for a in parts]) if parts else np.zeros(0, dtype=bool)
        answered = np.concatenate([a.answered for a in parts]) if parts else np.zeros(0, dtype=bool)
        aurc, cov_at, tau_at = _curve(conf[answered], correct[answered], int(answered.size), target_accuracy)
        report.fields[name] = {
            "n_total": int(answered.size),
            "rows": _rows(conf, correct, answered, taus),
            "aurc": aurc,
            "coverage_at_target": cov_at,
            "tau_at_target": tau_at,
        }
    return report
