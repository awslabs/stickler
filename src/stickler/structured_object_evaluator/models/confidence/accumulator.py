"""
Confidence accumulator for bulk evaluation.

Extracts confidence pairs from comparison results, accumulates them
across documents, and computes aggregate confidence metrics (AUROC,
Brier, ECE, ECARB, etc.).

This is a PostComparisonAccumulator that the BulkStructuredModelEvaluator
delegates to for all confidence-related state management.
"""

from typing import Any, Dict, List, Optional

from stickler.structured_object_evaluator.models.confidence.calculator import (
    ConfidenceCalculator,
    KeyedConfidencePairs,
)
from stickler.structured_object_evaluator.models.confidence.metrics import (
    ConfidenceMetric,
    ConfidencePair,
)
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)
from stickler.structured_object_evaluator.models.rich_value_helper import (
    RichValueHelper,
)


class ConfidenceAccumulator(PostComparisonAccumulator):
    """Accumulates confidence pairs and computes aggregate confidence metrics."""

    def __init__(self, metrics: Optional[List[ConfidenceMetric]] = None):
        self._calculator = ConfidenceCalculator(metrics=metrics)
        self.reset()

    @property
    def name(self) -> str:
        return "confidence_metrics"

    def reset(self) -> None:
        self._keyed_pairs: KeyedConfidencePairs = {}
        self._fields_with: int = 0
        self._fields_total: int = 0

    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        field_comparisons = comparison_result.get("field_comparisons", [])
        if not field_comparisons:
            # Align behavior with ConfidenceCalculator.extract: both paths
            # used to diverge on this input (extract raised, accumulator
            # silently returned), which meant a caller who forgot
            # ``document_field_comparisons=True`` got zero confidence with
            # no signal. When we have prediction_raw we clearly *expected*
            # confidence, so a missing field_comparisons list is likely a
            # misuse; surface it with a warning so it shows up in logs
            # without killing the bulk run.
            if prediction_raw is not None:
                import warnings as _warnings

                _warnings.warn(
                    "ConfidenceAccumulator got a comparison_result with "
                    "prediction_raw but no field_comparisons. Confidence "
                    "cannot be accumulated for this document. Re-run "
                    "compare_with(..., document_field_comparisons=True) "
                    "so the field-level join data is available.",
                    UserWarning,
                    stacklevel=2,
                )
            return

        # Extract confidences from prediction_raw if available
        if prediction_raw is not None:
            _unwrapped, confidences, _extras = RichValueHelper.process_rich_values(
                prediction_raw
            )
        else:
            confidences = {}

        extraction = self._calculator.extract_from_dicts(
            field_comparisons, confidences
        )

        if confidences:
            for field_path, pairs in extraction.keyed_pairs.items():
                self._keyed_pairs.setdefault(field_path, []).extend(pairs)

        self._fields_with += extraction.fields_with_confidence
        self._fields_total += extraction.fields_total

    def compute(self) -> Optional[Dict[str, Any]]:
        # Report coverage-only when we've seen fields but none had confidence.
        # Previously the accumulator returned None when keyed_pairs was empty,
        # which dropped the coverage signal entirely. After the coverage
        # accounting fix, an evaluator that sees 400 fields without
        # confidence should still publish ``{fields_with: 0, fields_total:
        # 400}`` rather than ``confidence_metrics=None``.
        if self._fields_total == 0:
            return None
        return self._calculator.compute_metrics(
            self._keyed_pairs,
            fields_with_confidence=self._fields_with,
            fields_total=self._fields_total,
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "keyed_confidence_pairs": {
                field_path: [p.model_dump() for p in pairs]
                for field_path, pairs in self._keyed_pairs.items()
            },
            "confidence_fields_with": self._fields_with,
            "confidence_fields_total": self._fields_total,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self._keyed_pairs = {
            field_path: [ConfidencePair(**p) for p in pairs]
            for field_path, pairs in state.get("keyed_confidence_pairs", {}).items()
        }
        self._fields_with = state.get("confidence_fields_with", 0)
        self._fields_total = state.get("confidence_fields_total", 0)

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        for field_path, pairs in other_state.get(
            "keyed_confidence_pairs", {}
        ).items():
            self._keyed_pairs.setdefault(field_path, []).extend(
                [ConfidencePair(**p) for p in pairs]
            )
        self._fields_with += other_state.get("confidence_fields_with", 0)
        self._fields_total += other_state.get("confidence_fields_total", 0)
