"""Confidence accumulator for bulk evaluation."""

from typing import Any, Dict, List, Optional

from stickler.structured_object_evaluator.models._keyed_pairs_accumulator import (
    _KeyedPairsAccumulator,
)
from stickler.structured_object_evaluator.models.confidence.calculator import (
    ConfidenceCalculator,
)
from stickler.structured_object_evaluator.models.confidence.metrics import (
    ConfidenceMetric,
    ConfidencePair,
)
from stickler.structured_object_evaluator.models.rich_value_helper import (
    RichValueHelper,
)


class ConfidenceAccumulator(_KeyedPairsAccumulator):
    """Accumulates confidence pairs and computes aggregate confidence metrics."""

    _pair_type = ConfidencePair
    _state_keys = (
        "keyed_confidence_pairs",
        "confidence_fields_with",
        "confidence_fields_total",
    )
    _warn_label = "ConfidenceAccumulator"

    def __init__(self, metrics: Optional[List[ConfidenceMetric]] = None):
        self._calculator = ConfidenceCalculator(metrics=metrics)
        self.reset()

    @property
    def name(self) -> str:
        return "confidence_metrics"

    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        if not self._has_field_comparisons(comparison_result, prediction_raw):
            return

        field_comparisons = comparison_result.get("field_comparisons", [])

        # Prefer the pre-extracted dict when compare_with provided it;
        # fall back to walking prediction_raw for older serialized results.
        confidences = comparison_result.get("prediction_confidences")
        if confidences is None:
            if prediction_raw is not None:
                _unwrapped, confidences, _extras = (
                    RichValueHelper.process_rich_values(prediction_raw)
                )
            else:
                confidences = {}

        extraction = self._calculator.extract_from_dicts(
            field_comparisons, confidences
        )

        # Pairs are extended only when confidence values were present; the
        # counters include every field seen, so a 0/N coverage signal survives.
        # BBoxMAPAccumulator extends unconditionally.
        self._extend(
            extraction.keyed_pairs if confidences else {},
            extraction.fields_with_confidence,
            extraction.fields_total,
        )

    def compute(self) -> Optional[Dict[str, Any]]:
        # Report coverage-only when fields were seen but none had confidence,
        # so a 0/N coverage signal still surfaces (vs. None).
        if self._fields_total == 0:
            return None
        return self._calculator.compute_metrics(
            self._keyed_pairs,
            fields_with_confidence=self._fields_with,
            fields_total=self._fields_total,
        )
