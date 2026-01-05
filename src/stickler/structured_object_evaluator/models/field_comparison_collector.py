"""Field comparison collector for StructuredModel comparisons.

This module provides the FieldComparisonCollector class that handles the collection
and documentation of ALL field comparisons (both matches and non-matches) during 
structured object comparison.
"""
import math
from typing import List, Dict, Any, TYPE_CHECKING
from .field_comparison_helper import FieldComparisonHelper

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class FieldComparisonCollector:
    """Collects all field-level comparison data for detailed analysis.
    
    This class is responsible for collecting and documenting ALL field comparisons
    between compared StructuredModel instances, including both matches and non-matches.
    It provides comprehensive field-level granularity for analysis purposes.
    """

    def __init__(self, model: "StructuredModel"):
        """Initialize collector with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model
        self.helper = FieldComparisonHelper()

    def collect_field_comparisons(
        self, 
        recursive_result: dict, 
        other: "StructuredModel"
    ) -> List[Dict[str, Any]]:
        """Collect all field comparisons with detailed metadata.
        
        This method walks through the recursive comparison result and collects
        ALL field comparisons (matches and non-matches) at the field level,
        providing detailed information about each comparison.
        
        Args:
            recursive_result: Result from compare_recursive containing field comparison details
            other: The predicted StructuredModel instance
            
        Returns:
            List of field comparison dictionaries with detailed information:
            [
                {
                    'expected_key': 'Agency',
                    'expected_value': 'BUYING TIME, LLC',
                    'actual_key': 'Agency',
                    'actual_value': 'BUYING TIME, LLC',
                    'match': True,
                    'score': 1.0,
                    'weighted_score': 2.0,
                    'reason': 'exact match'
                },
                {
                    'expected_key': 'LineItems[0].StartDate',
                    'expected_value': '10/22',
                    'actual_key': 'LineItems[2].StartDate',
                    'actual_value': '10/22/2016',
                    'match': False,
                    'score': 0.5,
                    'weighted_score': 0.5,
                    'reason': 'below threshold (0.5 < 1.0)'
                }
            ]
        """
        all_field_comparisons = []

        # Walk through the recursive result and collect all field comparisons
        for field_name, field_result in recursive_result.get("fields", {}).items():
            gt_val = getattr(self.model, field_name)
            pred_val = getattr(other, field_name, None)

            gt_is_empty = bool(isinstance(gt_val, list) and len(gt_val) == 0)
            pred_is_empty = bool(isinstance(pred_val, list) and len(pred_val) == 0)

            # Handle null list cases
            if gt_is_empty or pred_is_empty:
                # GT empty, pred has items → use helper for FA entries
                null_comparisons = self.helper.process_null_cases(
                    field_name, gt_val, pred_val
                )
                all_field_comparisons.extend(null_comparisons)

            elif (
                isinstance(gt_val, list)
                and isinstance(pred_val, list)
            ):
                # Use FieldComparisonHelper for primitive list collection
                list_comparisons = self.helper.collect_list_entries(
                    field_name, gt_val, pred_val
                )
                all_field_comparisons.extend(list_comparisons)

            else:
                from .structured_model import StructuredModel
                # Handle nested StructuredModel objects for detailed field comparison collection
                if (
                    isinstance(gt_val, StructuredModel)
                    and isinstance(pred_val, StructuredModel)
                    and "fields" in field_result
                ):
                    # Recursively collect field comparisons from nested objects
                    nested_collector = FieldComparisonCollector(gt_val)
                    nested_comparisons = nested_collector.collect_field_comparisons(
                        field_result, pred_val
                    )
                    # Prefix nested field paths with the parent field name
                    for nested_comp in nested_comparisons:
                        nested_comp["expected_key"] = (
                            f"{field_name}.{nested_comp['expected_key']}"
                        )
                        nested_comp["actual_key"] = (
                            f"{field_name}.{nested_comp['actual_key']}"
                        )
                        all_field_comparisons.append(nested_comp)

                # Create field-level comparison entries for primitive fields
                else:
                    # Extract comparison details from field_result
                    raw_score = field_result.get("raw_similarity_score", 0.0)
                    threshold_score = field_result.get("threshold_applied_score", raw_score)
                    weight = field_result.get("weight", 1.0)
                    weighted_score = threshold_score * weight
                    
                    # Determine if this is a match based on threshold
                    info = self.model._get_comparison_info(field_name)
                    is_match = bool(raw_score >= info.threshold)
                    
                    # Determine reason
                    if is_match:
                        if math.isclose(raw_score, 1.0):
                            reason = "exact match"
                        else:
                            reason = f"above threshold ({raw_score:.3f} >= {info.threshold})"
                    else:
                        reason = f"below threshold ({raw_score:.3f} < {info.threshold})"
                    
                    # Handle missing fields
                    if pred_val is None and gt_val is not None:
                        reason = "false negative (unmatched ground truth)"
                        is_match = False
                        raw_score = 0.0
                        weighted_score = 0.0
                    elif gt_val is None and pred_val is not None:
                        reason = "false alarm (unmatched prediction)"
                        is_match = False
                        raw_score = 0.0
                        weighted_score = 0.0

                    entry = {
                        "expected_key": field_name,
                        "expected_value": gt_val,
                        "actual_key": field_name,
                        "actual_value": pred_val,
                        "match": is_match,
                        "score": raw_score,
                        "weighted_score": weighted_score,
                        "reason": reason
                    }
                    all_field_comparisons.append(entry)

        return all_field_comparisons
