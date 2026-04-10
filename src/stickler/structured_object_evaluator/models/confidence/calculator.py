"""
Confidence calculator — extracts keyed pairs and runs metrics.

This is the single orchestrator for confidence evaluation. It:
1. Joins field_comparisons with confidence data to produce keyed pairs
2. Tracks coverage (how many fields had confidence vs. total)
3. Runs configured metrics at overall and per-field levels
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from stickler.structured_object_evaluator.models.confidence.metrics import (
    DEFAULT_METRICS,
    ConfidenceMetric,
    ConfidencePair,
    ConfidencePairs,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

KeyedConfidencePairs = Dict[str, ConfidencePairs]


class ExtractionResult(BaseModel):
    """Result of extracting confidence pairs from a comparison."""

    keyed_pairs: Dict[str, List[ConfidencePair]]
    fields_with_confidence: int
    fields_total: int


class ConfidenceCalculator:
    """Extracts confidence pairs and computes metrics.

    Args:
        metrics: List of ConfidenceMetric instances. Defaults to [AUROCMetric()].
    """

    def __init__(self, metrics: Optional[List[ConfidenceMetric]] = None):
        self.metrics = metrics if metrics is not None else list(DEFAULT_METRICS)

    def extract(
        self, comparison_result: Dict, pred_instance: StructuredModel
    ) -> ExtractionResult:
        """Extract ConfidencePair objects keyed by field path, with coverage stats.

        Joins field_comparisons (from compare_with) with confidence data
        (from from_json). Fields without confidence are skipped but counted.

        Args:
            comparison_result: Must contain "field_comparisons".
            pred_instance: Prediction with confidence data.

        Returns:
            ExtractionResult with keyed_pairs and coverage counts.

        Raises:
            ValueError: If no field_comparisons in comparison_result.
        """
        field_comparisons = comparison_result.get("field_comparisons", [])
        if not field_comparisons:
            raise ValueError("No field comparisons found in comparison result.")

        pred_confidences = pred_instance.get_all_confidences()
        keyed: KeyedConfidencePairs = {}
        fields_with = 0
        fields_total = 0

        for fc in field_comparisons:
            fields_total += 1
            field_path = fc["actual_key"]
            confidence = pred_confidences.get(field_path)
            if confidence is not None:
                fields_with += 1
                pair = ConfidencePair(
                    is_match=bool(fc["match"]),
                    confidence=confidence,
                    similarity=fc.get("score", 0.0),
                )
                keyed.setdefault(field_path, []).append(pair)

        return ExtractionResult(
            keyed_pairs=keyed,
            fields_with_confidence=fields_with,
            fields_total=fields_total,
        )

    def extract_keyed_pairs(
        self, comparison_result: Dict, pred_instance: StructuredModel
    ) -> KeyedConfidencePairs:
        """Extract keyed pairs only (convenience wrapper around extract).

        Use extract() when you also need coverage stats.
        """
        return self.extract(comparison_result, pred_instance).keyed_pairs

    def compute_metrics(
        self,
        keyed_pairs: KeyedConfidencePairs,
        fields_with_confidence: int = 0,
        fields_total: int = 0,
    ) -> Dict[str, Any]:
        """Run all metrics at overall and per-field levels.

        Args:
            keyed_pairs: Field path -> list of ConfidencePair.
            fields_with_confidence: Count of fields that had confidence data.
            fields_total: Total count of compared fields.

        Returns:
            {
                "overall": {"auroc": {"value": ...}, ...},
                "fields": {"vendor": {"auroc": {"value": ...}}, ...},
                "coverage": {
                    "fields_with_confidence": int,
                    "fields_total": int,
                    "ratio": float
                }
            }
        """
        all_pairs: ConfidencePairs = []
        for pairs in keyed_pairs.values():
            all_pairs.extend(pairs)

        result: Dict[str, Any] = {"overall": {}, "fields": {}}

        for metric in self.metrics:
            result["overall"][metric.name] = metric.compute(all_pairs)

        for field_path, pairs in keyed_pairs.items():
            result["fields"][field_path] = {
                metric.name: metric.compute(pairs) for metric in self.metrics
            }

        result["coverage"] = {
            "fields_with_confidence": fields_with_confidence,
            "fields_total": fields_total,
            "ratio": (
                fields_with_confidence / fields_total if fields_total > 0 else 0.0
            ),
        }

        return result
