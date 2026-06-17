"""Aggregate confusion-matrix accumulator for bulk evaluation.

Rolls up per-document confusion matrix aggregates (the ``aggregate`` field
emitted by :class:`ConfusionMatrixBuilder` when ``include_confusion_matrix=True``
is passed to :meth:`StructuredModel.compare_with`) into corpus-level totals.

Usage::

    accumulator = AggregateConfusionMatrixAccumulator()
    evaluator = BulkStructuredModelEvaluator(
        target_schema=MyModel,
        accumulators=[accumulator],  # included in defaults too
    )
    # ... drive evaluator with include_confusion_matrix=True ...
    process_eval = evaluator.compute()
    corpus_agg = process_eval.accumulator_metrics["aggregate_metrics"]
    # corpus_agg["overall"]["tp"]
    # corpus_agg["overall"]["derived"]["cm_precision"]
    # corpus_agg["fields"]["contact.email"]["tp"]
"""

from collections import defaultdict
from typing import Any, Dict, Optional

from stickler.structured_object_evaluator.models.metrics_helper import MetricsHelper
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)


_METRIC_KEYS = ("tp", "fp", "fn", "fa", "fd", "tn")


def _join_path(prefix: str, name: str) -> str:
    """Join a dotted field path component (mirrors bulk evaluator helper)."""
    return f"{prefix}.{name}" if prefix else name


def _empty_counts() -> Dict[str, int]:
    return {k: 0 for k in _METRIC_KEYS}


class AggregateConfusionMatrixAccumulator(PostComparisonAccumulator):
    """Accumulates confusion matrix aggregates across documents.

    Reads per-document confusion matrix results produced when
    ``include_confusion_matrix=True`` is passed to ``compare_with``,
    extracts the top-level ``aggregate`` block plus all nested per-field
    aggregates, and rolls them up into corpus-level totals exposed via
    ``ProcessEvaluation.accumulator_metrics["aggregate_metrics"]``.

    State shape::

        {
            "overall": {tp, fp, fn, fa, fd, tn},
            "fields": {
                "dotted.field.path": {tp, fp, fn, fa, fd, tn},
                ...
            },
        }

    Documents that did not include a confusion matrix are silently
    skipped — there is no warning, since mixing old (pre-feature) and
    new results in a single corpus is an explicit goal.

    Args:
        recall_with_fd: When ``True``, derived ``cm_recall`` (and therefore
            ``cm_f1``) at every level uses the include-FD formula
            ``TP / (TP + FN + FD)``, penalizing partial matches that
            didn't clear ``match_threshold``. When ``False`` (default),
            uses the textbook ``TP / (TP + FN)``. Mirrors the
            ``recall_with_fd`` knob on
            :meth:`StructuredModel.compare_with`, kept off by default
            so corpus-level numbers match the per-document defaults.
    """

    def __init__(self, recall_with_fd: bool = False) -> None:
        """Initialize the accumulator with empty state."""
        self._metrics_helper = MetricsHelper()
        self._recall_with_fd = bool(recall_with_fd)
        self.reset()

    @property
    def name(self) -> str:
        return "aggregate_metrics"

    # --- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        """Clear accumulated counts."""
        self._overall: Dict[str, int] = defaultdict(int)
        # Lazy-create per-field counts as they appear.
        self._fields: Dict[str, Dict[str, int]] = defaultdict(_empty_counts)
        # Track whether any document contributed; compute() returns None
        # when nothing was accumulated, matching ConfidenceAccumulator.
        self._docs_with_data: int = 0

    # --- accumulation ----------------------------------------------------

    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        """Extract aggregate metrics from a single comparison result.

        Silently returns without modifying state when the document did
        not opt into ``include_confusion_matrix``.
        """
        confusion_matrix = comparison_result.get("confusion_matrix")
        if not isinstance(confusion_matrix, dict):
            # No confusion matrix recorded for this doc — skip silently.
            return

        contributed = False

        top_aggregate = confusion_matrix.get("aggregate")
        if self._add_metrics(self._overall, top_aggregate):
            contributed = True

        fields = confusion_matrix.get("fields")
        if isinstance(fields, dict):
            if self._accumulate_field_aggregates(fields, ""):
                contributed = True

        if contributed:
            self._docs_with_data += 1

    def _accumulate_field_aggregates(
        self, fields: Dict[str, Any], prefix: str
    ) -> bool:
        """Walk a fields dict, summing each node's ``aggregate`` block.

        Recurses into both ``fields`` and ``nested_fields`` children.
        Returns True iff any metric was added at this level or below.
        """
        contributed = False
        for field_name, field_data in fields.items():
            if not isinstance(field_data, dict):
                continue

            current_path = _join_path(prefix, field_name)

            if self._add_metrics(
                self._fields[current_path], field_data.get("aggregate")
            ):
                contributed = True

            child_fields = field_data.get("fields")
            if isinstance(child_fields, dict):
                if self._accumulate_field_aggregates(child_fields, current_path):
                    contributed = True

            nested_fields = field_data.get("nested_fields")
            if isinstance(nested_fields, dict):
                if self._accumulate_field_aggregates(nested_fields, current_path):
                    contributed = True

        return contributed

    @staticmethod
    def _add_metrics(target: Dict[str, int], source: Any) -> bool:
        """Add tp/fp/fn/fa/fd/tn from ``source`` into ``target``.

        ``source`` may be missing, ``None``, or a non-dict; in those
        cases nothing is added. Non-numeric or negative values are
        silently skipped to keep accumulator state consistent.

        Returns True iff at least one metric was added.
        """
        if not isinstance(source, dict):
            return False

        added = False
        for key in _METRIC_KEYS:
            value = source.get(key)
            if isinstance(value, bool):
                # bool is a subclass of int; reject to avoid silent
                # corruption from accidental True/False values.
                continue
            if not isinstance(value, (int, float)):
                continue
            if value < 0:
                continue
            target[key] += int(value)
            added = True
        return added

    # --- output ----------------------------------------------------------

    def compute(self) -> Optional[Dict[str, Any]]:
        """Compute corpus-level aggregate metrics.

        Returns ``None`` when no document contributed aggregate data
        (matching the ABC contract used by other accumulators).
        """
        if self._docs_with_data == 0:
            return None

        overall_counts = self._counts_dict(self._overall)
        overall_node = dict(overall_counts)
        overall_node["derived"] = self._metrics_helper.calculate_derived_metrics(
            overall_counts, recall_with_fd=self._recall_with_fd
        )

        field_nodes: Dict[str, Dict[str, Any]] = {}
        for path in sorted(self._fields):
            counts = self._counts_dict(self._fields[path])
            node = dict(counts)
            node["derived"] = self._metrics_helper.calculate_derived_metrics(
                counts, recall_with_fd=self._recall_with_fd
            )
            field_nodes[path] = node

        return {"overall": overall_node, "fields": field_nodes}

    @staticmethod
    def _counts_dict(source: Dict[str, int]) -> Dict[str, int]:
        """Return a fresh dict with all metric keys filled (0 when absent)."""
        return {key: int(source.get(key, 0)) for key in _METRIC_KEYS}

    # --- serialization / merge ------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "overall": self._counts_dict(self._overall),
            "fields": {
                path: self._counts_dict(counts)
                for path, counts in self._fields.items()
            },
            "docs_with_data": self._docs_with_data,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.reset()
        overall = state.get("overall") or {}
        if isinstance(overall, dict):
            for key in _METRIC_KEYS:
                value = overall.get(key, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._overall[key] = int(value)

        fields = state.get("fields") or {}
        if isinstance(fields, dict):
            for path, counts in fields.items():
                if not isinstance(counts, dict):
                    continue
                for key in _METRIC_KEYS:
                    value = counts.get(key, 0)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self._fields[path][key] += int(value)

        # Backward compat: older states may not record docs_with_data.
        # Treat any non-zero counts as evidence that data was accumulated.
        # Note: if all counts are zero, docs_with_data defaults to 0, which
        # causes compute() to return None for that accumulator instance.
        docs_with_data = state.get("docs_with_data")
        if isinstance(docs_with_data, int) and docs_with_data >= 0:
            self._docs_with_data = docs_with_data
        else:
            self._docs_with_data = (
                1
                if any(self._overall.values())
                or any(any(counts.values()) for counts in self._fields.values())
                else 0
            )

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        """Additive merge of a peer accumulator's state."""
        other_overall = other_state.get("overall") or {}
        if isinstance(other_overall, dict):
            for key in _METRIC_KEYS:
                value = other_overall.get(key, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._overall[key] += int(value)

        other_fields = other_state.get("fields") or {}
        if isinstance(other_fields, dict):
            for path, counts in other_fields.items():
                if not isinstance(counts, dict):
                    continue
                for key in _METRIC_KEYS:
                    value = counts.get(key, 0)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self._fields[path][key] += int(value)

        other_docs = other_state.get("docs_with_data", 0)
        if isinstance(other_docs, int) and other_docs > 0:
            self._docs_with_data += other_docs
        elif (
            any((other_overall or {}).get(k, 0) for k in _METRIC_KEYS)
            or any(
                any(counts.get(k, 0) for k in _METRIC_KEYS)
                for counts in (other_fields or {}).values()
                if isinstance(counts, dict)
            )
        ):
            # Older peers without docs_with_data: count their state as
            # at least one contributing document so compute() emits.
            self._docs_with_data += 1
