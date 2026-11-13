"""Field comparator for StructuredModel comparisons.

This module provides the FieldComparator class that handles comparison of
primitive and structured fields during structured object comparison.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from .reporting_context import ReportingContext
from .threshold_helper import ThresholdHelper

if TYPE_CHECKING:
    from .structured_model import StructuredModel


def _classify(raw: float, threshold: float, clip: bool) -> tuple[Dict[str, int], float]:
    """Classify a compared field and apply its score-clipping policy."""
    if ThresholdHelper.is_above_threshold(raw, threshold):
        return {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}, raw
    return (
        {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0},
        0.0 if clip else raw,
    )


class FieldComparator:
    """Compares primitive and structured fields.
    
    This class is responsible for comparing individual fields between
    StructuredModel instances. It handles:
    - Primitive field comparison (strings, integers, floats)
    - Nested StructuredModel field comparison
    - Threshold-based binary classification
    - Score calculation and metrics generation
    """

    def __init__(
        self, model: "StructuredModel", report_context: Optional[ReportingContext] = None
    ):
        """Initialize comparator with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model
        self.report_context = report_context

    def compare_primitive_with_scores(
        self, 
        gt_val: Any, 
        pred_val: Any, 
        field_name: str
    ) -> Dict[str, Any]:
        """Compare primitive fields and return metrics + scores.
        
        This method compares primitive values (strings, integers, floats) using
        the configured comparator for the field. It applies threshold-based
        binary classification and returns both raw and threshold-applied scores.
        
        Args:
            gt_val: Ground truth value (primitive type)
            pred_val: Predicted value (primitive type)
            field_name: Name of the field being compared
            
        Returns:
            Dictionary with structure:
            {
                "overall": {
                    "tp": int, "fa": int, "fd": int, 
                    "fp": int, "tn": int, "fn": int
                },
                "raw_similarity_score": float,
                "similarity_score": float,
                "threshold_applied_score": float,
                "weight": float
            }
        """
        info = self.model.__class__._get_comparison_info(field_name)
        raw_similarity = info.comparator.compare(gt_val, pred_val)
        weight = info.weight
        threshold = info.threshold

        metrics, threshold_applied_score = _classify(
            raw_similarity, threshold, info.clip_under_threshold
        )

        # UNIFIED STRUCTURE: Always use 'overall' for metrics
        # 'fields' key omitted for primitive leaf nodes (semantic meaning: not a parent container)
        return {
            "overall": metrics,
            "raw_similarity_score": raw_similarity,
            "similarity_score": raw_similarity,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
        }

    def compare_structured_field(
        self,
        gt_val: "StructuredModel",
        pred_val: "StructuredModel",
        field_name: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Compare nested StructuredModel fields.
        
        This method compares nested StructuredModel instances, applying
        object-level threshold-based classification while preserving nested
        field details for debugging purposes.
        
        Args:
            gt_val: Ground truth StructuredModel instance
            pred_val: Predicted StructuredModel instance
            field_name: Name of the field being compared
            threshold: Matching threshold for object-level classification
            
        Returns:
            Dictionary with structure:
            {
                "overall": {
                    "tp": int, "fa": int, "fd": int, 
                    "fp": int, "tn": int, "fn": int,
                    "similarity_score": float,
                },
                "fields": dict,  # Nested field comparison details
                "raw_similarity_score": float,
                "similarity_score": float,
                "threshold_applied_score": float,
                "weight": float,
                "non_matches": list
            }
        """
        # Get field configuration
        info = self.model._get_comparison_info(field_name)
        weight = info.weight
        
        # Classify the object's similarity, separately from its nested field metrics.

        # Get object-level similarity score
        raw_score = gt_val.compare(pred_val)  # Overall object similarity

        object_metrics, threshold_applied_score = _classify(
            raw_score, threshold, info.clip_under_threshold
        )

        # Still generate nested field details for debugging, but don't roll them up
        # 
        # TODO (#392): compare() walks the subtree for its score, then
        # compare_recursive() walks it again for the nested details.
        from .comparison_engine import ComparisonEngine
        from .structured_model import StructuredModel

        if type(gt_val).compare_recursive is StructuredModel.compare_recursive:
            child_context = None
            if self.report_context is not None:
                child_context = ReportingContext()
                self.report_context.children[field_name] = child_context
            nested_result = ComparisonEngine(gt_val).compare_recursive(
                pred_val, _report_context=child_context
            )
        else:
            # Existing overrides need not accept internal metadata options.
            nested_result = gt_val.compare_recursive(pred_val)
        nested_details = nested_result["fields"]

        # Return structure with object-level metrics and nested field details kept separate
        return {
            "overall": {
                **object_metrics,
                "similarity_score": raw_score,
            },
            "fields": nested_details,  # Nested details available for debugging
            "raw_similarity_score": raw_score,
            "similarity_score": raw_score,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
            "non_matches": [],  # Add empty non_matches for consistency
        }
