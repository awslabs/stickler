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
    ConfidenceMetric,
    ConfidencePair,
    ConfidencePairs,
    default_metrics,
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
        self.metrics = metrics if metrics is not None else default_metrics()

    def extract(
        self, comparison_result: Dict, pred_instance: StructuredModel
    ) -> ExtractionResult:
        """Extract ConfidencePair objects keyed by field path, with coverage stats.

        Raises:
            ValueError: If no field_comparisons in comparison_result.
        """
        field_comparisons = comparison_result.get("field_comparisons", [])
        if not field_comparisons:
            raise ValueError("No field comparisons found in comparison result.")
        return self.extract_from_dicts(
            field_comparisons, pred_instance.get_all_confidences()
        )

    def extract_keyed_pairs(
        self, comparison_result: Dict, pred_instance: StructuredModel
    ) -> KeyedConfidencePairs:
        """Extract keyed pairs only (convenience wrapper around extract)."""
        return self.extract(comparison_result, pred_instance).keyed_pairs

    def extract_from_dicts(
        self,
        field_comparisons: List[Dict],
        confidences: Dict[str, float],
    ) -> ExtractionResult:
        """Extract ConfidencePair objects from raw field_comparisons + confidences dicts.

        Rows without a string ``actual_key`` (e.g., list FN entries where
        the prediction has fewer items than ground truth) are skipped — they
        can't be joined to a confidence score and would inflate fields_total.
        """
        keyed: KeyedConfidencePairs = {}
        fields_with = 0
        fields_total = 0

        for fc in field_comparisons:
            field_path = fc.get("actual_key")
            if not isinstance(field_path, str):
                continue
            fields_total += 1
            confidence = confidences.get(field_path)
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
