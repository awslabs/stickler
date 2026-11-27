"""Accumulate decision fields across a bulk run.

``BulkStructuredModelEvaluator`` installs a ``DecisionAccumulator`` on its own
when its ``target_schema`` declares a Choice, Score or Noul field, at any depth.
Each document adds one record per *asked* decision:

    {"doc": <document index>, "gold": <label index>, "probs": [...] | None}

Records are grouped by path with list indexes removed (``messages[].intent``),
so every item of a list contributes to one set of metrics. A decision whose gold
is ``None`` was not asked and is not recorded. ``probs`` is ``None`` when the
answer is missing. A prediction given as a bare value rather than a System One
answer has no probabilities to score, so it raises ``AnswerError`` instead of
entering calibration as a certain answer. On a field with
``on_invalid="missing"``, a bare value or a malformed answer is recorded as
``{"probs": None, "rejected": True}`` instead.

Documents are identified by the evaluator's ``doc_id``. A ``doc_id`` seen twice
raises, because the bootstrap resamples documents and a repeated document would
be counted as an independent one. ``repeated_doc_ids="group"`` instead keeps
every row of a repeated ``doc_id`` in one document, which is how repeated runs
of the same items are pooled.
"""

import math
import re
import warnings
from typing import Any, Dict, Hashable, Iterator, List, Literal, Optional, Tuple, Type

from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)

from .arrays import DecisionArrays
from .normalize import AnswerError, is_systemone_answer, normalize_answer
from .primitives import DecisionSpec, decision_paths

ACCUMULATOR_NAME = "decision_metrics"
# Below this many answered decisions a field's calibration metrics are mostly
# noise: a perfectly calibrated model shows an ece_mass near 0.2 at n = 20.
MIN_CALIBRATION_N = 30
# warn -> compute -> BulkStructuredModelEvaluator._build_process_evaluation ->
# compute() / get_current_metrics() / save_metrics() -> the caller's line.
_SMALL_SAMPLE_STACKLEVEL = 4

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


def _doc_key(doc_id: Any) -> Optional[Hashable]:
    """The identity a ``doc_id`` stands for; None for no id (None or NaN)."""
    if doc_id is None or (isinstance(doc_id, float) and math.isnan(doc_id)):
        return None
    return doc_id


class DecisionAccumulator(PostComparisonAccumulator):
    """Collect Choice, Score and Noul decisions and compute their metrics.

    Args:
        target_schema: The StructuredModel class whose decision fields to read.
        n_bins: Bins for ``ece_mass``.
        repeated_doc_ids: ``"raise"`` (the default) rejects a ``doc_id`` seen
            before; ``"group"`` adds its rows to the earlier document, so
            repeated runs share one bootstrap group.
    """

    def __init__(
        self,
        target_schema: Type[Any],
        n_bins: int = 10,
        repeated_doc_ids: Literal["raise", "group"] = "raise",
    ):
        if repeated_doc_ids not in ("raise", "group"):
            raise ValueError(f"repeated_doc_ids must be 'raise' or 'group', got {repeated_doc_ids!r}")
        self.specs: Dict[str, DecisionSpec] = decision_paths(target_schema)
        self.n_bins = n_bins
        self.repeated_doc_ids = repeated_doc_ids
        self.reset()

    @property
    def name(self) -> str:
        return ACCUMULATOR_NAME

    def reset(self) -> None:
        self._records: Dict[str, List[Dict[str, Any]]] = {p: [] for p in self.specs}
        self._docs = 0
        self._doc_ids: List[Any] = []  # the doc_id of each document index
        self._index_of: Dict[Hashable, int] = {}
        self._current: Optional[int] = None

    def start_document(self, doc_id: Any) -> None:
        """Set the document the next ``accumulate`` call belongs to.

        Raises:
            ValueError: If ``doc_id`` was seen before and ``repeated_doc_ids``
                is ``"raise"``.
        """
        key = _doc_key(doc_id)
        if key is not None and key in self._index_of:
            if self.repeated_doc_ids == "raise":
                raise ValueError(
                    f"doc_id {doc_id!r} was already evaluated. Documents are the bootstrap's "
                    "resampling unit, so a repeat would count as an independent document. Use "
                    "unique doc_ids, or DecisionAccumulator(..., repeated_doc_ids='group') to "
                    "pool repeated runs of the same items."
                )
            self._current = self._index_of[key]
            return
        self._current = self._new_doc(doc_id)

    def _new_doc(self, doc_id: Any) -> int:
        index = self._docs
        self._docs += 1
        self._doc_ids.append(doc_id if _doc_key(doc_id) is not None else None)
        if _doc_key(doc_id) is not None:
            self._index_of[doc_id] = index
        return index

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
        doc, self._current = self._current, None
        # Read every decision before recording any, so a document that raises
        # adds nothing rather than some of its fields.
        pending = []
        for path, gold, pred_key, pred_value in self._decisions(comparison_result.get("field_comparisons", [])):
            if gold is None:
                continue  # not asked for this document
            spec = self.specs[path]
            raw = _lookup(prediction_raw, pred_key)
            record = {"gold": spec.index_of_value(gold)}
            try:
                record["probs"] = self._probs(spec, path, raw, pred_value)
            except AnswerError:
                if spec.on_invalid != "missing":
                    raise
                record.update(probs=None, rejected=True)
            pending.append((path, record))
        if doc is None:  # called without start_document (not through the evaluator)
            doc = self._new_doc(None)
        for path, record in pending:
            self._records[path].append({"doc": doc, **record})

    @staticmethod
    def _probs(spec: DecisionSpec, path: str, raw: Any, value: Any) -> Optional[List[float]]:
        if is_systemone_answer(raw):
            return [float(p) for p in normalize_answer(spec, raw, path).probs]
        if value is None:
            return None
        raise AnswerError(
            f"field {path!r}: the prediction {value!r} is a bare value, not a System One answer, "
            "so it has no probabilities to score. Build the prediction with from_json() from the "
            "model's answer, or set the field to None if the model did not answer."
        )

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
            if a.n_ok < MIN_CALIBRATION_N:
                warnings.warn(
                    f"Decision field {path!r} has {a.n_ok} answered decisions. Calibration "
                    f"metrics below {MIN_CALIBRATION_N} are mostly noise; read them with "
                    "confidence_intervals().",
                    UserWarning,
                    stacklevel=_SMALL_SAMPLE_STACKLEVEL,
                )
            spec = self.specs[path]
            fields[path] = {
                "kind": spec.kind,
                "labels": list(spec.labels),
                "correctness": correctness(a),
                "calibration": calibration(a, n_bins=self.n_bins),
                "records": [dict(r) for r in self._records[path]],
            }
        return {"fields": fields, "document_count": self._docs, "settings": {"n_bins": self.n_bins}}

    def get_state(self) -> Dict[str, Any]:
        return {
            "records": {p: [dict(r) for r in recs] for p, recs in self._records.items()},
            "docs": self._docs,
            "doc_ids": list(self._doc_ids),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.reset()
        docs = int(state.get("docs", 0))
        doc_ids = list(state.get("doc_ids") or [None] * docs)
        for doc_id in doc_ids:
            self._new_doc(doc_id)
        for p, recs in state.get("records", {}).items():
            if p in self._records:
                self._records[p] = [dict(r) for r in recs]

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        """Add another shard's state. Its documents follow this one's.

        Raises:
            ValueError: If both shards saw a ``doc_id`` and ``repeated_doc_ids``
                is ``"raise"``. Nothing is merged in that case.
        """
        docs = int(other_state.get("docs", 0))
        doc_ids = list(other_state.get("doc_ids") or [None] * docs)
        if self.repeated_doc_ids == "raise":
            clash = [d for d in doc_ids if _doc_key(d) is not None and d in self._index_of]
            if clash:
                raise ValueError(f"doc_ids {clash[:5]} appear in both states; use repeated_doc_ids='group' to pool them")
        mapping = {}
        for j, doc_id in enumerate(doc_ids):
            key = _doc_key(doc_id)
            mapping[j] = self._index_of[key] if key is not None and key in self._index_of else self._new_doc(doc_id)
        for p, recs in other_state.get("records", {}).items():
            if p in self._records:
                self._records[p].extend({**r, "doc": mapping[r["doc"]]} for r in recs)
