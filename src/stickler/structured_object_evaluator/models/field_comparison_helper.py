"""Field comparison helper for StructuredModel comparisons."""

from typing import List, Dict, Any, Optional
from .comparison_helper_base import ComparisonHelperBase

class FieldComparisonHelper(ComparisonHelperBase):
    """Helper class for collecting and formatting field comparisons in StructuredModel comparisons."""

    def create_entry(
        self,
        expected_key: str,
        expected_value: Any,
        actual_key: str,
        actual_value: Any,
        match: bool,
        score: float,
        weighted_score: float,
        reason: str,
    ) -> Dict[str, Any]:
        """Create a field comparison entry for detailed analysis.

        Args:
            expected_key: Ground truth field key/path
            expected_value: Ground truth field value
            actual_key: Predicted field key/path (can be None for FN)
            actual_value: Predicted field value (can be None for FN)
            match: Whether this comparison is considered a match
            score: Raw similarity score
            weighted_score: Score multiplied by field weight
            reason: Descriptive reason for the match/no-match result

        Returns:
            Dictionary with field comparison information
        """
        return self.create_field_comparison_entry(
            expected_key, expected_value, actual_key, actual_value,
            match, score, weighted_score, reason
        )

    def create_field_comparison_entry(
        self,
        expected_key: str,
        expected_value: Any,
        actual_key: str,
        actual_value: Any,
        match: bool,
        score: float,
        weighted_score: float,
        reason: str,
    ) -> Dict[str, Any]:
        """Create a field comparison entry for detailed analysis.

        Args:
            expected_key: Ground truth field key/path
            expected_value: Ground truth field value
            actual_key: Predicted field key/path (can be None for FN)
            actual_value: Predicted field value (can be None for FN)
            match: Whether this comparison is considered a match
            score: Raw similarity score
            weighted_score: Score multiplied by field weight
            reason: Descriptive reason for the match/no-match result

        Returns:
            Dictionary with field comparison information
        """
        # Handle StructuredModel serialization
        expected_value = self.serialize_object(expected_value)
        actual_value = self.serialize_object(actual_value)

        entry = {
            "expected_key": expected_key,
            "expected_value": expected_value,
            "actual_key": actual_key,
            "actual_value": actual_value,
            "match": match,
            "score": score,
            "weighted_score": weighted_score,
            "reason": reason,
        }

        return entry

    def _extract_entries_from_objects(
        self, 
        field_name: str, 
        gt_object: Any, 
        pred_object: Any, 
        gt_index: Optional[int],
        pred_index: Optional[int],
        is_match: bool,
        similarity_score: float,
        reason: str
    ) -> List[Dict[str, Any]]:
        """Extract field-level comparisons from structured model objects.
        
        Args:
            field_name: Name of the parent list field
            gt_object: Ground truth structured model
            pred_object: Prediction structured model  
            gt_index: Index in the GT list (None for FA cases)
            pred_index: Index in the pred list (None for FN cases)
            is_match: Whether the overall objects match
            similarity_score: Overall similarity score
            reason: Overall comparison reason
            
        Returns:
            List of field-level comparison entries
        """
        from .structured_model import StructuredModel
        
        # Check if both objects are structured models
        if (isinstance(gt_object, StructuredModel) and isinstance(pred_object, StructuredModel)):
            # Perform field-by-field comparison to get detailed field comparisons
            comparison_result = gt_object.compare_with(
                pred_object, 
                document_non_matches=False,
                include_confusion_matrix=False
            )
            
            field_comparisons = []
            
            # Extract field scores and create comparison entries
            for nested_field_name, field_score in comparison_result.get("field_scores", {}).items():
                gt_nested_val = getattr(gt_object, nested_field_name, None)
                pred_nested_val = getattr(pred_object, nested_field_name, None)
                
                # Get field configuration for threshold
                info = gt_object._get_comparison_info(nested_field_name)
                field_is_match = field_score >= info.threshold
                
                # Create field paths with indices
                expected_key = f"{field_name}[{gt_index}].{nested_field_name}" if gt_index is not None else f"{field_name}[].{nested_field_name}"
                actual_key = f"{field_name}[{pred_index}].{nested_field_name}" if pred_index is not None else None
                
                # Create reason for this specific field
                field_reason = self.generate_comparison_reason(field_is_match, field_score, info.threshold)
                
                # Handle missing fields
                if pred_nested_val is None and gt_nested_val is not None:
                    field_reason = "false negative (unmatched ground truth)"
                    field_is_match = False
                    field_score = 0.0
                elif gt_nested_val is None and pred_nested_val is not None:
                    field_reason = "false alarm (unmatched prediction)"
                    field_is_match = False
                    field_score = 0.0
                
                weighted_score = field_score * info.weight
                
                field_entry = self.create_field_comparison_entry(
                    expected_key=expected_key,
                    expected_value=gt_nested_val,
                    actual_key=actual_key,
                    actual_value=pred_nested_val,
                    match=field_is_match,
                    score=field_score,
                    weighted_score=weighted_score,
                    reason=field_reason
                )
                
                field_comparisons.append(field_entry)
                
            return field_comparisons
        
        else:
            # For primitive objects or single object comparisons, create a single entry
            expected_key = f"{field_name}[{gt_index}]" if gt_index is not None else f"{field_name}[]"
            actual_key = f"{field_name}[{pred_index}]" if pred_index is not None else None
            
            
            return [self.create_field_comparison_entry(
                expected_key=expected_key,
                expected_value=gt_object,
                actual_key=actual_key,
                actual_value=pred_object,
                match=is_match,
                score=similarity_score,
                weighted_score=similarity_score, # No weight score for primitive lists
                reason=reason
            )]
