"""Field comparison helper for StructuredModel comparisons."""

from typing import List, Dict, Any
from .hungarian_helper import HungarianHelper


class FieldComparisonHelper:
    """Helper class for collecting and formatting field comparisons in StructuredModel comparisons."""

    def __init__(self):
        self.hungarian_helper = HungarianHelper()

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
        if expected_value and hasattr(expected_value, "model_dump"):
            expected_value = expected_value.model_dump()
        if actual_value and hasattr(actual_value, "model_dump"):
            actual_value = actual_value.model_dump()

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

    def collect_list_field_comparisons(
        self, field_name: str, gt_list: List[Any], pred_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Collect individual field-level comparisons from a list field.

        Args:
            field_name: Name of the list field
            gt_list: Ground truth list
            pred_list: Prediction list

        Returns:
            List of field comparison dictionaries with individual field information
        """
        field_comparisons = []

        if not gt_list and not pred_list:
            return field_comparisons

        # Get optimal assignments with scores
        assignments = []
        matched_pairs_with_scores = []
        if gt_list and pred_list:
            hungarian_info = self.hungarian_helper.get_complete_matching_info(
                gt_list, pred_list
            )
            matched_pairs_with_scores = hungarian_info["matched_pairs"]
            assignments = [(i, j) for i, j, _ in matched_pairs_with_scores]

        # Get the match threshold from the model class
        if (
            gt_list
            and hasattr(gt_list[0], "__class__")
            and hasattr(gt_list[0].__class__, "match_threshold")
        ):
            match_threshold = gt_list[0].__class__.match_threshold
        else:
            match_threshold = 0.7

        # Process matched pairs for all comparisons (both matches and non-matches)
        for gt_idx, pred_idx, similarity_score in matched_pairs_with_scores:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]
                
                # Determine if this is a match
                is_match = similarity_score >= match_threshold
                
                # Create reason
                reason = self._generate_comparison_reason(is_match, similarity_score, match_threshold)
                
                # Extract field-level comparisons from structured model objects
                field_level_comparisons = self._extract_field_level_comparisons(
                    field_name,
                    gt_item,
                    pred_item,
                    gt_idx,
                    pred_idx,
                    is_match,
                    similarity_score,
                    reason,
                )
                field_comparisons.extend(field_level_comparisons)

        # Process unmatched ground truth items (FN)
        matched_gt_indices = set(idx for idx, _ in assignments)
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                field_level_comparisons = self._extract_field_level_comparisons(
                    field_name,
                    gt_item,
                    None,
                    gt_idx,
                    None,
                    False,  # is_match
                    0.0,    # score
                    "false negative (unmatched ground truth)",
                )
                field_comparisons.extend(field_level_comparisons)

        # Process unmatched prediction items (FA)
        matched_pred_indices = set(idx for _, idx in assignments)
        for pred_idx, pred_item in enumerate(pred_list):
            if pred_idx not in matched_pred_indices:
                field_level_comparisons = self._extract_field_level_comparisons(
                    field_name,
                    None,
                    pred_item,
                    None,
                    pred_idx,
                    False,  # is_match
                    0.0,    # score
                    "false alarm (unmatched prediction)",
                )
                field_comparisons.extend(field_level_comparisons)

        return field_comparisons

    def add_field_comparisons_for_null_cases(
        self, field_name: str, gt_list: List[Any], pred_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Add field comparisons for null cases (empty lists).

        Args:
            field_name: Name of the field
            gt_list: Ground truth list (may be empty/None)
            pred_list: Prediction list (may be empty/None)

        Returns:
            List of field comparison entries for null cases
        """
        field_comparisons = []

        # Check if both lists are empty
        if not gt_list and not pred_list:
            return field_comparisons

        # Handle gt null case
        if not gt_list and pred_list:
            # Add field comparisons for each FA item when GT is empty
            for pred_idx, pred_item in enumerate(pred_list):
                field_level_comparisons = self._extract_field_level_comparisons(
                    field_name,
                    None,
                    pred_item,
                    None,
                    pred_idx,
                    False,  # is_match
                    0.0,    # score
                    "false alarm (unmatched prediction)",
                )
                field_comparisons.extend(field_level_comparisons)

        # Handle pred null case
        elif gt_list and not pred_list:
            # Add field comparisons for each FN item when prediction is empty
            for gt_idx, gt_item in enumerate(gt_list):
                field_level_comparisons = self._extract_field_level_comparisons(
                    field_name,
                    gt_item,
                    None,
                    gt_idx,
                    None,
                    False,  # is_match
                    0.0,    # score
                    "false negative (unmatched ground truth)",
                )
                field_comparisons.extend(field_level_comparisons)

        return field_comparisons
    
    def _generate_comparison_reason(self, is_match: bool, score: float, threshold: float) -> str:
        """Generate a descriptive reason for a field comparison result."""
        if is_match:
            if score == 1.0:
                return "exact match"
            else:
                return f"above threshold ({score:.3f} >= {threshold})"
        else:
            return f"below threshold ({score:.3f} < {threshold})"

    
    def _extract_field_level_comparisons(
        self, 
        field_name: str, 
        gt_object: Any, 
        pred_object: Any, 
        gt_index: int,
        pred_index: int,
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
                field_reason = self._generate_comparison_reason(field_is_match, field_score, info.threshold)
                
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
