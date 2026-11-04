"""The array form every decision metric reads.

One ``DecisionArrays`` holds every *asked* decision of one field across a run:
a question with no gold answer was not asked and is never recorded. Rows whose
answer is missing (refused, errored, timed out) stay in the arrays with NaN
probabilities, so accuracy can count them as wrong while calibration skips them.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .primitives import DecisionSpec


@dataclass(frozen=True)
class DecisionArrays:
    """Asked decisions for one field.

    Attributes:
        spec: The field's question.
        P: ``n x k`` probabilities in label order. A missing answer's row is NaN.
        y: ``n`` gold label indexes.
        groups: ``n`` document indexes, the bootstrap's resampling unit.
    """

    spec: DecisionSpec
    P: np.ndarray
    y: np.ndarray
    groups: np.ndarray

    @property
    def n_total(self) -> int:
        """Number of asked decisions."""
        return int(self.y.shape[0])

    @property
    def answered(self) -> np.ndarray:
        """Boolean mask of rows with an answer."""
        if self.P.size == 0:
            return np.zeros(self.n_total, dtype=bool)
        return ~np.isnan(self.P).any(axis=1)

    @property
    def n_ok(self) -> int:
        """Number of answered decisions."""
        return int(self.answered.sum())

    @property
    def n_missing(self) -> int:
        """Number of asked decisions with no answer."""
        return self.n_total - self.n_ok

    @property
    def pred(self) -> np.ndarray:
        """Predicted label index per row; -1 where the answer is missing."""
        out = np.full(self.n_total, -1, dtype=int)
        ok = self.answered
        if ok.any():
            if self.spec.kind == "noul":
                # Yes at P(yes) >= 0.5, matching normalize_answer and the Noul P/R/F1.
                out[ok] = (self.P[ok, 1] >= 0.5).astype(int)
            else:
                out[ok] = np.argmax(self.P[ok], axis=1)
        return out

    @property
    def conf(self) -> np.ndarray:
        """Probability of the predicted answer per row; NaN where missing."""
        out = np.full(self.n_total, np.nan)
        ok = self.answered
        if ok.any():
            out[ok] = self.P[ok].max(axis=1)
        return out

    @property
    def correct(self) -> np.ndarray:
        """Whether each row's predicted answer equals gold; False where missing."""
        return self.pred == self.y

    def subset(self, rows: np.ndarray) -> "DecisionArrays":
        """A copy with only the given row indexes (used by the bootstrap)."""
        return DecisionArrays(self.spec, self.P[rows], self.y[rows], self.groups[rows])

    @classmethod
    def from_records(cls, spec: DecisionSpec, records: Sequence[Dict[str, Any]]) -> "DecisionArrays":
        """Build from accumulator records ``{"doc": int, "gold": int, "probs": list | None}``."""
        n, k = len(records), spec.k
        P = np.full((n, k), np.nan)
        y = np.zeros(n, dtype=int)
        groups = np.zeros(n, dtype=int)
        for i, rec in enumerate(records):
            y[i] = rec["gold"]
            groups[i] = rec["doc"]
            if rec.get("probs") is not None:
                P[i] = rec["probs"]
        return cls(spec, P, y, groups)


def empty_arrays(spec: DecisionSpec) -> DecisionArrays:
    """Arrays with no rows."""
    return DecisionArrays(spec, np.zeros((0, spec.k)), np.zeros(0, dtype=int), np.zeros(0, dtype=int))


__all__: List[str] = ["DecisionArrays", "empty_arrays"]
