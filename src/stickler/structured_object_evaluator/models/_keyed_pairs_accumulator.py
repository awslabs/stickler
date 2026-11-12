"""Shared state handling for accumulators that key observations by field path.

``ConfidenceAccumulator`` and ``BBoxMAPAccumulator`` store the same shape:
a ``{field_path: [observation, ...]}`` mapping plus a ``with``/``total``
counter pair. Only the observation type and the names used in the serialized
state differ, so the four state methods used to be copied line for line.

Keeping them here means a change to the merge semantics cannot silently apply
to one accumulator and not the other. This class is private: it is an
implementation detail of the two built-ins, not part of the public
``PostComparisonAccumulator`` contract.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Type

from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)


class _KeyedPairsAccumulator(PostComparisonAccumulator):
    """Intermediate base for accumulators keyed by field path."""

    #: Observation model reconstructed when loading or merging state.
    _pair_type: Type
    #: ``(pairs_key, fields_with_key, fields_total_key)`` as they appear on the wire.
    _state_keys: Tuple[str, str, str]
    #: Class name used in the "no field_comparisons" warning.
    _warn_label: str

    def reset(self) -> None:
        self._keyed_pairs: Dict[str, list] = {}
        self._fields_with: int = 0
        self._fields_total: int = 0

    def _has_field_comparisons(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> bool:
        """Whether the result carries field comparisons, warning if it should have.

        ``prediction_raw`` present without ``field_comparisons`` means the caller
        wanted this accumulator's metrics but forgot
        ``document_field_comparisons=True``. Warn instead of raising: a bulk run
        over thousands of documents should not die on one malformed row.

        ``stacklevel=3``: level 1 is the ``warnings.warn`` line in this helper,
        level 2 is the subclass's ``accumulate``, and level 3 is the evaluator's
        call to it — which is where the warning should point.
        """
        if comparison_result.get("field_comparisons"):
            return True

        if prediction_raw is not None:
            warnings.warn(
                f"{self._warn_label} got prediction_raw but no "
                "field_comparisons; re-run compare_with("
                "document_field_comparisons=True).",
                UserWarning,
                stacklevel=3,
            )
        return False

    def _extend(
        self, keyed_pairs: Dict[str, list], fields_with: int, fields_total: int
    ) -> None:
        """Append observations per field path and bump both counters."""
        for field_path, pairs in keyed_pairs.items():
            self._keyed_pairs.setdefault(field_path, []).extend(pairs)
        self._fields_with += fields_with
        self._fields_total += fields_total

    def get_state(self) -> Dict[str, Any]:
        pairs_key, with_key, total_key = self._state_keys
        return {
            pairs_key: {
                field_path: [p.model_dump() for p in pairs]
                for field_path, pairs in self._keyed_pairs.items()
            },
            with_key: self._fields_with,
            total_key: self._fields_total,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        pairs_key, with_key, total_key = self._state_keys
        self._keyed_pairs = {
            field_path: [self._pair_type(**p) for p in pairs]
            for field_path, pairs in state.get(pairs_key, {}).items()
        }
        self._fields_with = state.get(with_key, 0)
        self._fields_total = state.get(total_key, 0)

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        pairs_key, with_key, total_key = self._state_keys
        for field_path, pairs in other_state.get(pairs_key, {}).items():
            self._keyed_pairs.setdefault(field_path, []).extend(
                [self._pair_type(**p) for p in pairs]
            )
        self._fields_with += other_state.get(with_key, 0)
        self._fields_total += other_state.get(total_key, 0)
