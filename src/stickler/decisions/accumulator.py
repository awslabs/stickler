"""Accumulate decision fields across a bulk run.

``BulkStructuredModelEvaluator`` installs a ``DecisionAccumulator`` on its own
when its ``target_schema`` declares a Choice, Score or Noul field, at any depth.
Each document adds one record per *asked* decision:

    {"doc": <document index>, "gold": <label index>, "probs": [...] | None}

Records are grouped by path with list indexes removed (``messages[].intent``),
so every item of a list contributes to one set of metrics. A decision whose gold
is ``None`` was not asked and is not recorded. ``probs`` is ``None`` when the
answer is missing. A prediction given as a bare value rather than a System One
answer counts as a certain answer (probability 1.0).
"""

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)

from .arrays import DecisionArrays
from .normalize import is_systemone_answer, normalize_answer
from .primitives import DecisionSpec, decision_paths

ACCUMULATOR_NAME = "decision_metrics"

_INDEX = re.compile(r"\[\d+\]")
_PART = re.compile(r"([^.\[\]]+)(?:\[(\d+)\])?")


def _pattern(key: str) -> str:
    """``messages[2].intent`` -> ``messages[].intent``."""
    return _INDEX.sub("[]", key)


def _plain(value: Any) -> Any:
    """Model instances (and lists of them) as plain dicts, so their fields can be read."""
    if isinstance(value, list):
        return [_plain(v) for v in value]
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


def _lookup(data: Any, key: Optional[str]) -> Any:
    """Read ``a.b[1].c`` out of nested dicts and lists; None if absent."""
    if not key:
        return None
    for name, index in _PART.findall(key):
        if not isinstance(data, dict):
            return None
        data = data.get(name)
        if index:
            if not isinstance(data, list) or int(index) >= len(data):
                return None
            data = data[int(index)]
    return data


class DecisionAccumulator(PostComparisonAccumulator):
    """Collect Choice, Score and Noul decisions and compute their metrics.

    Args:
        target_schema: The StructuredModel class whose decision fields to read.
        n_bins: Bins for ``ece_mass``.
    """

    def __init__(self, target_schema: Type[Any], n_bins: int = 10):
        self.specs: Dict[str, DecisionSpec] = decision_paths(target_schema)
        self.n_bins = n_bins
        self.reset()

    @property
    def name(self) -> str:
        return ACCUMULATOR_NAME

    def reset(self) -> None:
        self._records: Dict[str, List[Dict[str, Any]]] = {p: [] for p in self.specs}
        self._docs = 0

    def _decisions(self, rows: List[Dict[str, Any]]) -> Iterator[Tuple[str, Any, Optional[str], Any]]:
        """Yield ``(path, gold, prediction_key, predicted_value)`` for every decision in the rows.

        A row is either one decision leaf, or a whole object (a list item whose
        pair fell below its match threshold, or an unmatched item), whose
        decision fields are read from its values.
        """
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = row.get("expected_key") or row.get("actual_key")
            if not key:
                continue
            pattern = _pattern(key)
            if pattern in self.specs:
                yield pattern, row.get("expected_value"), row.get("actual_key"), row.get("actual_value")
                continue
            gold_obj, pred_obj = _plain(row.get("expected_value")), _plain(row.get("actual_value"))
            if isinstance(gold_obj, list) and not isinstance(pred_obj, list):
                # A gold list with no predicted list: every item's decisions are missing.
                items = [
                    {"expected_key": f"{key}[{i}]", "expected_value": item, "actual_key": None, "actual_value": None}
                    for i, item in enumerate(gold_obj)
                ]
                yield from self._decisions(items)
                continue
            if not isinstance(gold_obj, dict) and not isinstance(pred_obj, dict):
                continue
            prefix = pattern + "."
            for path in self.specs:
                if not path.startswith(prefix):
                    continue
                rest = path[len(prefix):]
                actual_key = row.get("actual_key")
                if "[]" in rest:
                    # A list inside an object that was not expanded. With no
                    # predicted object, every gold item's decision is missing.
                    # With one, its items were never paired, so they are not
                    # reported per item.
                    head, tail = rest.split("[]", 1)
                    gold_items = _lookup(gold_obj, head) if isinstance(gold_obj, dict) else None
                    if pred_obj is None and isinstance(gold_items, list):
                        if not tail.startswith("."):
                            continue
                        missing_rows = [
                            {"expected_key": f"{key}.{head}[{i}]", "expected_value": item,
                             "actual_key": None, "actual_value": None}
                            for i, item in enumerate(gold_items)
                        ]
                        for p, gold, _, _ in self._decisions(missing_rows):
                            if p == path:
                                yield p, gold, None, None
                    continue
                yield (
                    path,
                    _lookup(gold_obj, rest) if isinstance(gold_obj, dict) else None,
                    f"{actual_key}.{rest}" if actual_key else None,
                    _lookup(pred_obj, rest) if isinstance(pred_obj, dict) else None,
                )

    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        if prediction_raw is None:
            prediction_raw = comparison_result.get("prediction_raw") or {}
        doc = self._docs
        self._docs += 1
        for path, gold, pred_key, pred_value in self._decisions(comparison_result.get("field_comparisons", [])):
            if gold is None:
                continue  # not asked for this document
            spec = self.specs[path]
            raw = _lookup(prediction_raw, pred_key)
            self._records[path].append(
                {"doc": doc, "gold": spec.index_of_value(gold), "probs": self._probs(spec, path, raw, pred_value)}
            )

    @staticmethod
    def _probs(spec: DecisionSpec, path: str, raw: Any, value: Any) -> Optional[List[float]]:
        if is_systemone_answer(raw):
            return [float(p) for p in normalize_answer(spec, raw, path).probs]
        if value is None:
            return None
        probs = [0.0] * spec.k
        probs[spec.index_of_value(value)] = 1.0
        return probs

    def arrays(self) -> Dict[str, DecisionArrays]:
        """The accumulated decisions of each path, as DecisionArrays."""
        return {p: DecisionArrays.from_records(self.specs[p], recs) for p, recs in self._records.items()}

    def compute(self) -> Optional[Dict[str, Any]]:
        if not any(self._records.values()):
            return None
        from .metrics import calibration, correctness

        fields: Dict[str, Any] = {}
        for path, a in self.arrays().items():
            if a.n_total == 0:
                continue
            spec = self.specs[path]
            fields[path] = {
                "kind": spec.kind,
                "labels": list(spec.labels),
                "correctness": correctness(a),
                "calibration": calibration(a, n_bins=self.n_bins),
                "records": [dict(r) for r in self._records[path]],
            }
        return {"fields": fields, "document_count": self._docs}

    def get_state(self) -> Dict[str, Any]:
        return {"records": {p: [dict(r) for r in recs] for p, recs in self._records.items()}, "docs": self._docs}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.reset()
        self._docs = int(state.get("docs", 0))
        for p, recs in state.get("records", {}).items():
            if p in self._records:
                self._records[p] = [dict(r) for r in recs]

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        offset = self._docs
        for p, recs in other_state.get("records", {}).items():
            if p in self._records:
                self._records[p].extend({**r, "doc": r["doc"] + offset} for r in recs)
        self._docs += int(other_state.get("docs", 0))
