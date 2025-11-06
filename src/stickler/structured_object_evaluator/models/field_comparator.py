"""Field comparator for StructuredModel comparisons.

This module provides the FieldComparator class that handles comparison of
primitive and structured fields during structured object comparison.
"""

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class FieldComparator:
    """Compares primitive and structured fields.
    
    This class is responsible for comparing individual fields between
    StructuredModel instances. It handles:
    - Primitive field comparison (strings, integers, floats)
    - Nested StructuredModel field comparison
    - Threshold-based binary classification
    - Score calculation and metrics generation
    """

    def __init__(self, model: "StructuredModel"):
        """Initialize comparator with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model

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

        # For binary classification metrics, always use threshold
        if raw_similarity >= threshold:
            metrics = {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
            threshold_applied_score = raw_similarity
        else:
            metrics = {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0}
            # For score calculation, respect clip_under_threshold setting
            threshold_applied_score = (
                0.0 if info.clip_under_threshold else raw_similarity
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
                    "all_fields_matched": bool
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
        
        # CRITICAL FIX: For StructuredModel fields, object-level metrics should be based on
        # object similarity, not rollup of nested field metrics

        # Get object-level similarity score
        raw_score = gt_val.compare(pred_val)  # Overall object similarity

        # Apply object-level binary classification based on threshold
        if raw_score >= threshold:
            # Object matches threshold -> True Positive
            object_metrics = {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
            threshold_applied_score = raw_score
        else:
            # Object below threshold -> False Discovery
            object_metrics = {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0}
            threshold_applied_score = (
                0.0 if info.clip_under_threshold else raw_score
            )

        # Still generate nested field details for debugging, but don't roll them up
        # 
        # TODO: PERFORMANCE ISSUE - Redundant traversal of nested object tree
        #       This call to compare_recursive() creates a new ComparisonEngine and
        #       re-traverses all nested fields, even though we're already in the middle
        #       of a recursive traversal from the parent ComparisonEngine. This causes
        #       O(n²) behavior for deeply nested structures.
        #
        #       Example: Invoice -> Contact -> Address
        #       - Parent engine traverses Invoice fields
        #       - Hits Contact field, calls this method
        #       - This creates NEW engine to traverse Contact fields
        #       - Hits Address field, creates ANOTHER new engine
        #       - Each level re-traverses its subtree unnecessarily
        #
        #       Better approach: Pass dispatcher context down through recursion to avoid
        #       creating new engines. This would require refactoring the recursion model
        #       to be more explicit about context passing.
        #
        #       Impact: Moderate - primarily affects deeply nested structures (3+ levels)
        #       Estimated overhead: 2-3x for structures with 3 levels of nesting
        nested_details = gt_val.compare_recursive(pred_val)["fields"]

        # Return structure with object-level metrics and nested field details kept separate
        return {
            "overall": {
                **object_metrics,
                "similarity_score": raw_score,
                "all_fields_matched": raw_score >= threshold,
            },
            "fields": nested_details,  # Nested details available for debugging
            "raw_similarity_score": raw_score,
            "similarity_score": raw_score,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
            "non_matches": [],  # Add empty non_matches for consistency
        }
