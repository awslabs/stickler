"""
Dedicated class for handling Hungarian matching of List[StructuredModel] fields.

This class extracts the Hungarian matching logic from StructuredModel to improve 
code organization and maintainability. The extraction preserves existing behavior
exactly, including current bugs that will be fixed in subsequent phases.

Current Behavior Preserved (including bugs):
- Uses parent field threshold instead of object match_threshold (bug)  
- Generates nested metrics for all matched pairs regardless of threshold (bug)
- Object-level counting discrepancies in some scenarios (bug)
"""

from typing import List, Dict, Any, TYPE_CHECKING
from .hungarian_helper import HungarianHelper
from .metrics_helper import MetricsHelper
from .field_helper import FieldHelper
from .comparable_field import ComparableField

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class StructuredListComparator:
    """Handles comparison of List[StructuredModel] fields using Hungarian matching."""
    
    def __init__(self, parent_model: "StructuredModel"):
        """Initialize the comparator with reference to parent model.
        
        Args:
            parent_model: The StructuredModel instance that owns the list field
        """
        self.parent_model = parent_model
    
    def compare_struct_list_with_scores(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        field_name: str,
    ) -> dict:
        """Enhanced structural list comparison that returns both metrics AND scores.
        
        CRITICAL: This is the main entry point extracted from StructuredModel.
        Maintains identical behavior including current bugs for Phase 2 compatibility.
        
        Args:
            gt_list: Ground truth list of StructuredModel objects
            pred_list: Predicted list of StructuredModel objects  
            field_name: Name of the list field being compared
            
        Returns:
            Dictionary with overall metrics, nested field details, and scores
        """
        # Get field configuration - same as original
        info = self.parent_model.__class__._get_comparison_info(field_name)
        weight = info.weight
        threshold = info.threshold
        
        # PHASE 3 FIX: Use correct threshold source for Hungarian matching decisions
        # Should use the list element model's match_threshold, not the parent field's threshold
        if gt_list and hasattr(gt_list[0].__class__, "match_threshold"):
            match_threshold = gt_list[0].__class__.match_threshold
        else:
            # Fallback to default if no match_threshold defined
            match_threshold = getattr(
                self.parent_model.__class__, "match_threshold", 0.7
            )
        
        # Handle empty list cases with beautiful match statements
        early_exit_result = self._handle_struct_list_empty_cases(
            gt_list, pred_list, weight
        )
        if early_exit_result is not None:
            return early_exit_result
        
        # Normalize None to empty lists for consistent processing below
        gt_list = gt_list or []
        pred_list = pred_list or []
        
        # Calculate object-level metrics using extracted method
        (
            object_level_metrics,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
        ) = self._calculate_object_level_metrics(gt_list, pred_list, match_threshold)
        
        # Calculate raw similarity score using extracted method
        raw_similarity = self._calculate_struct_list_similarity(
            gt_list, pred_list, info
        )
        
        # CRITICAL FIX: For structured lists, we NEVER clip under threshold - partial matches are important
        threshold_applied_score = raw_similarity  # Always use raw score for lists
        
        # Get field-level details for nested structure (but DON'T aggregate to list level)
        # THRESHOLD-GATED RECURSION: Only generate field details for good matches
        field_details = self._calculate_nested_field_metrics(
            field_name,
            gt_list,
            pred_list,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
            match_threshold,
        )
        
        # Build final result structure
        final_result = {
            "overall": object_level_metrics,  # Count OBJECTS, not fields
            "fields": field_details,  # Field-level details kept separate
            "raw_similarity_score": raw_similarity,
            "similarity_score": raw_similarity,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
        }
        
        return final_result
    
    def _handle_struct_list_empty_cases(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        weight: float,
    ) -> dict:
        """Handle empty list cases with beautiful match statements.
        
        Args:
            gt_list: Ground truth list (may be None)
            pred_list: Predicted list (may be None) 
            weight: Field weight for scoring
            
        Returns:
            Result dictionary if early exit needed, None if should continue processing
        """
        # Normalize None to empty lists for consistent handling
        gt_len = len(gt_list or [])
        pred_len = len(pred_list or [])
        
        match (gt_len, pred_len):
            case (0, 0):
                # Both empty lists → True Negative
                return {
                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                    "fields": {},
                    "raw_similarity_score": 1.0,
                    "similarity_score": 1.0,
                    "threshold_applied_score": 1.0,
                    "weight": weight,
                }
            case (0, pred_len):
                # GT empty, pred has items → False Alarms
                return {
                    "overall": {
                        "tp": 0,
                        "fa": pred_len,
                        "fd": 0,
                        "fp": pred_len,
                        "tn": 0,
                        "fn": 0,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case (gt_len, 0):
                # GT has items, pred empty → False Negatives
                return {
                    "overall": {
                        "tp": 0,
                        "fa": 0,
                        "fd": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": gt_len,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case _:
                # Both non-empty, continue processing
                return None

    def _calculate_object_level_metrics(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        match_threshold: float,
    ) -> tuple:
        """Calculate object-level metrics using Hungarian matching.
        
        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            match_threshold: Threshold for considering objects as matches
            
        Returns:
            Tuple of (object_metrics_dict, matched_pairs, matched_gt_indices, matched_pred_indices)
        """
        # Use Hungarian matching for OBJECT-LEVEL counts
        hungarian_helper = HungarianHelper()
        hungarian_info = hungarian_helper.get_complete_matching_info(gt_list, pred_list)
        matched_pairs = hungarian_info["matched_pairs"]
        
        # Count OBJECTS, not individual fields
        tp_objects = 0  # Objects with similarity >= match_threshold
        fd_objects = 0  # Objects with similarity < match_threshold
        for gt_idx, pred_idx, similarity in matched_pairs:
            if similarity >= match_threshold:
                tp_objects += 1
            else:
                fd_objects += 1
        
        # Count unmatched objects
        matched_gt_indices = {idx for idx, _, _ in matched_pairs}
        matched_pred_indices = {idx for _, idx, _ in matched_pairs}
        fn_objects = len(gt_list) - len(matched_gt_indices)  # Unmatched GT objects
        fa_objects = len(pred_list) - len(
            matched_pred_indices
        )  # Unmatched pred objects
        
        # Build list-level metrics counting OBJECTS (not fields)
        object_level_metrics = {
            "tp": tp_objects,
            "fa": fa_objects,  
            "fd": fd_objects,
            "fp": fa_objects + fd_objects,  # Total false positives
            "tn": 0,  # No true negatives at object level for non-empty lists
            "fn": fn_objects,
        }
        
        return (
            object_level_metrics,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
        )

    def _calculate_struct_list_similarity(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        info: "ComparableField",
    ) -> float:
        """Calculate raw similarity score for structured list.
        
        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            info: Field comparison info
            
        Returns:
            Raw similarity score between 0.0 and 1.0
        """
        if len(pred_list) > 0:
            # Use parent model's comparison method
            match_result = self.parent_model._compare_unordered_lists(
                gt_list, pred_list, info.comparator, info.threshold
            )
            return match_result.get("overall_score", 0.0)
        else:
            return 0.0

    def _calculate_nested_field_metrics(
        self,
        list_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
                                       matched_pairs: List,
                                       matched_gt_indices: set,
                                       matched_pred_indices: set,
        match_threshold: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate field-level details with threshold-aware metric population.
        
        Process ALL matched pairs for field-level analysis, but populate metrics sections
        based on object-level similarity threshold:
        - Above threshold: populate both "overall" and "aggregate" sections
        - Below threshold: populate only "aggregate" section
        
        Args:
            list_field_name: Name of the parent list field
            gt_list: Ground truth list
            pred_list: Predicted list  
            matched_pairs: List of (gt_idx, pred_idx, similarity) tuples
            matched_gt_indices: Set of matched GT indices
            matched_pred_indices: Set of matched pred indices
            match_threshold: Match threshold for determining metric section population
            
        Returns:
            Dictionary mapping field names to their metrics
        """
        field_details = {}
        
        if gt_list and isinstance(gt_list[0], StructuredModel):
            model_class = gt_list[0].__class__
            
            # Process ALL matched pairs for field-level analysis
            all_matched_pairs = matched_pairs
            
            # Generate field details if we have any matched pairs OR unmatched objects
            has_matched_pairs = len(all_matched_pairs) > 0
            has_unmatched = (len(matched_gt_indices) < len(gt_list)) or (len(matched_pred_indices) < len(pred_list))
            
            if has_matched_pairs or has_unmatched:
                for sub_field_name in model_class.model_fields:
                    if sub_field_name == "extra_fields":
                        continue
                    
                    # Check if this field is a List[StructuredModel] that needs hierarchical treatment
                    field_info = model_class.model_fields.get(sub_field_name)
                    is_hierarchical_field = (
                        field_info and model_class._is_structured_field_type(field_info)
                    )
                    
                    if is_hierarchical_field:
                        # Handle hierarchical fields with threshold-aware metric population
                        field_details[sub_field_name] = self._handle_hierarchical_field_threshold_aware(
                            sub_field_name, gt_list, pred_list, all_matched_pairs, 
                            matched_gt_indices, matched_pred_indices, match_threshold
                        )
                    else:
                        # Handle primitive fields with threshold-aware metric population
                        field_details[sub_field_name] = self._handle_primitive_field_threshold_aware(
                            sub_field_name, gt_list, pred_list, all_matched_pairs,
                            matched_gt_indices, matched_pred_indices, match_threshold
                        )
        
        return field_details
    
    def _handle_hierarchical_field_threshold_aware(
        self,
        sub_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        matched_pairs: List,
        matched_gt_indices: set,
        matched_pred_indices: set,
        match_threshold: float) -> Dict[str, Any]:
        """Handle hierarchical List[StructuredModel] fields with threshold-aware metric population.
        
        This method processes ALL matched pairs for field-level analysis but populates
        metrics sections based on object-level similarity threshold.
        """
        
        # Collect pair results for recursive aggregation
        above_threshold_results = []  # For overall metrics
        all_pair_results = []  # For aggregate metrics
        
        # Process matched pairs with threshold-aware field-level comparison
        for gt_idx, pred_idx, similarity in matched_pairs:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]
                gt_sub_value = getattr(gt_item, sub_field_name, None)
                pred_sub_value = getattr(pred_item, sub_field_name, None)
                
                # Always perform field-level comparison for aggregate metrics
                pair_result = gt_item._dispatch_field_comparison(
                    sub_field_name, gt_sub_value, pred_sub_value
                )
                
                # Create aggregate section by summing nested field contributions
                aggregate_metrics = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
                
                # Check if both GT and Pred are null
                gt_is_null = (gt_sub_value is None or gt_sub_value == [] or gt_sub_value == "")
                pred_is_null = (pred_sub_value is None or pred_sub_value == [] or pred_sub_value == "")
                
                if gt_is_null and pred_is_null:
                    # Both GT and Pred are null - create nested field entries with TN contributions
                    nested_field_count = gt_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0:
                        self._create_nested_field_entries_for_null_case(pair_result, gt_item, sub_field_name)
                        aggregate_metrics["tn"] = nested_field_count
                else:
                    # Sum up contributions from nested fields (for non-null cases)
                    if "fields" in pair_result:
                        for nested_field_name, nested_field_result in pair_result["fields"].items():
                            if "aggregate" in nested_field_result:
                                # Use aggregate metrics from nested fields
                                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                                    aggregate_metrics[metric] += nested_field_result["aggregate"].get(metric, 0)
                            elif "overall" in nested_field_result:
                                # Fallback to overall metrics if no aggregate
                                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                                    aggregate_metrics[metric] += nested_field_result["overall"].get(metric, 0)
                
                # Set the aggregate section
                pair_result["aggregate"] = aggregate_metrics
                
                # Add to all results for aggregate calculation
                all_pair_results.append(pair_result)
                
                # Only add to above-threshold results if similarity meets threshold
                if self._should_populate_overall_metrics(similarity, match_threshold):
                    above_threshold_results.append(pair_result)
        
        # Handle unmatched objects (contribute to aggregate only)
        unmatched_results = self._create_unmatched_results_for_hierarchical_field(
            sub_field_name, gt_list, pred_list, matched_gt_indices, matched_pred_indices
        )
        all_pair_results.extend(unmatched_results)
        
        # Calculate overall metrics from above-threshold results only
        overall_aggregated = self._recursive_aggregate_metrics(above_threshold_results)
        
        # Calculate aggregate metrics from all results
        aggregate_aggregated = self._recursive_aggregate_metrics(all_pair_results)
        
        # Preserve hierarchical fields structure from the first valid result
        hierarchical_fields = {}
        for pair_result in all_pair_results:
            if "fields" in pair_result and pair_result["fields"]:
                hierarchical_fields = pair_result["fields"]
                break
        
        # Combine results with proper structure
        final_result = {
            "overall": overall_aggregated.get("overall", {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}),
            "aggregate": aggregate_aggregated.get("aggregate", {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}),
            "fields": hierarchical_fields
        }
        
        # Merge field-level results with threshold-aware logic
        if above_threshold_results:
            overall_fields = overall_aggregated.get("fields", {})
            for field_name, field_data in overall_fields.items():
                if field_name not in final_result["fields"]:
                    final_result["fields"][field_name] = {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
                        "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
                    }
                final_result["fields"][field_name]["overall"] = field_data.get("overall", {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0})
        
        if all_pair_results:
            aggregate_fields = aggregate_aggregated.get("fields", {})
            for field_name, field_data in aggregate_fields.items():
                if field_name not in final_result["fields"]:
                    final_result["fields"][field_name] = {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
                        "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
                    }
                final_result["fields"][field_name]["aggregate"] = field_data.get("aggregate", {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0})
        
        # Add derived metrics recursively
        self._add_derived_metrics_recursively(final_result)
        
        # Add metadata from first pair if available
        if all_pair_results:
            for key in [
                "raw_similarity_score",
                "similarity_score", 
                "threshold_applied_score",
                "weight",
            ]:
                if key in all_pair_results[0]:
                    final_result[key] = all_pair_results[0][key]
        
        return final_result

    def _create_nested_field_entries_for_null_case(self, pair_result: Dict[str, Any], 
                                                  gt_item: 'StructuredModel', 
                                                  sub_field_name: str) -> None:
        """Create nested field entries for null cases to support metric aggregation."""
        if "fields" not in pair_result:
            pair_result["fields"] = {}
        
        # Get the nested model class to find field names
        field_info = gt_item.__class__.model_fields.get(sub_field_name)
        if field_info:
            from typing import get_origin, get_args, Union
            field_type = field_info.annotation if hasattr(field_info, 'annotation') else None
            if field_type:
                # Handle Union types (Optional[List[StructuredModel]])
                if get_origin(field_type) is Union:
                    # Find the List type in the Union
                    for arg in get_args(field_type):
                        if get_origin(arg) is list:
                            field_type = arg
                            break
                
                # Check if it's a List[StructuredModel]
                if get_origin(field_type) is list:
                    args = get_args(field_type)
                    if args and hasattr(args[0], 'model_fields'):
                        nested_model_class = args[0]
                        # Create TN entries for each nested field
                        for nested_field_name in nested_model_class.model_fields:
                            if nested_field_name != "extra_fields":
                                pair_result["fields"][nested_field_name] = {
                                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                                    "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0}  # Add to aggregate only
                                }

    def _create_unmatched_results_for_hierarchical_field(self, sub_field_name: str,
                                                        gt_list: List["StructuredModel"],
                                                        pred_list: List["StructuredModel"],
                                                        matched_gt_indices: set,
                                                        matched_pred_indices: set) -> List[Dict[str, Any]]:
        """Create results for unmatched objects in hierarchical fields."""
        unmatched_results = []
        
        # Handle unmatched GT objects (contribute FN for non-null fields, TN for null fields)
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                gt_sub_value = getattr(gt_item, sub_field_name, None)
                
                if gt_sub_value is not None and gt_sub_value != [] and gt_sub_value != "":
                    # Non-null field - count as FN with nested field contributions
                    nested_field_count = gt_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0 and isinstance(gt_sub_value, list):
                        # Count the list items and their nested fields
                        list_length = len(gt_sub_value)
                        fn_count = list_length * nested_field_count
                    else:
                        # Not a hierarchical field or empty list, count as 1
                        fn_count = 1
                    
                    unmatched_result = {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                        "fields": {},
                        "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": fn_count}  # Add to aggregate only
                    }
                    unmatched_results.append(unmatched_result)
                else:
                    # Null/empty field - count as TN with nested field contributions
                    nested_field_count = gt_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0:
                        # For hierarchical fields, count nested fields as TN
                        tn_count = nested_field_count
                        
                        # Create nested field entries so that _collect_all_primitive_metrics can find them
                        unmatched_result = {
                            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                            "fields": {},
                            "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": tn_count, "fn": 0}  # Add to aggregate only
                        }
                        
                        self._create_nested_field_entries_for_null_case(unmatched_result, gt_item, sub_field_name)
                    else:
                        # Not a hierarchical field, count as 1
                        tn_count = 1
                        unmatched_result = {
                            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                            "fields": {},
                            "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": tn_count, "fn": 0}  # Add to aggregate only
                        }
                    
                    unmatched_results.append(unmatched_result)
        
        # Handle unmatched pred objects (contribute FA for hierarchical fields)
        for pred_idx, pred_item in enumerate(pred_list):
            if pred_idx not in matched_pred_indices:
                pred_sub_value = getattr(pred_item, sub_field_name, None)
                
                if pred_sub_value is not None and pred_sub_value != [] and pred_sub_value != "":
                    # Non-null field - count as FA with nested field contributions
                    nested_field_count = pred_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0 and isinstance(pred_sub_value, list):
                        # Count the list items and their nested fields
                        list_length = len(pred_sub_value)
                        fa_count = list_length * nested_field_count
                    else:
                        # Not a hierarchical field or empty list, count as 1
                        fa_count = 1
                    
                    unmatched_result = {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                        "fields": {},
                        "aggregate": {"tp": 0, "fa": fa_count, "fd": 0, "fp": fa_count, "tn": 0, "fn": 0}  # Add to aggregate only
                    }
                    unmatched_results.append(unmatched_result)
        
        return unmatched_results

    def _handle_hierarchical_field(
        self,
        sub_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
                                  matched_pairs: List,
                                  matched_gt_indices: set,
                                  matched_pred_indices: set,
                                  match_threshold: float) -> Dict[str, Any]:
        """Handle hierarchical List[StructuredModel] fields with proper nested field counting.
        
        This method properly counts nested field contributions for unmatched objects
        in hierarchical fields (List[StructuredModel] types).
        """
        
        # Collect all pair results for recursive aggregation
        pair_results = []
        
        # Process matched pairs with field-level comparison (regardless of object-level similarity)
        for gt_idx, pred_idx, similarity in matched_pairs:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]
                gt_sub_value = getattr(gt_item, sub_field_name, None)
                pred_sub_value = getattr(pred_item, sub_field_name, None)
                
                # Always use field-level comparison for matched pairs
                # The object-level similarity doesn't affect field-level metrics
                pair_result = gt_item._dispatch_field_comparison(
                    sub_field_name, gt_sub_value, pred_sub_value
                )
                
                # For hierarchical fields, create aggregate section by summing nested field contributions
                # The aggregate should include contributions from nested fields
                aggregate_metrics = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
                
                # Check if both GT and Pred are null
                gt_is_null = (gt_sub_value is None or gt_sub_value == [] or gt_sub_value == "")
                pred_is_null = (pred_sub_value is None or pred_sub_value == [] or pred_sub_value == "")
                
                if gt_is_null and pred_is_null:
                    # Both GT and Pred are null - create nested field entries with TN contributions
                    nested_field_count = gt_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0:
                        # Create nested field entries so that _collect_all_primitive_metrics can find them
                        if "fields" not in pair_result:
                            pair_result["fields"] = {}
                        
                        # Get the nested model class to find field names
                        field_info = gt_item.__class__.model_fields.get(sub_field_name)
                        if field_info:
                            from typing import get_origin, get_args, Union
                            field_type = field_info.annotation if hasattr(field_info, 'annotation') else None
                            if field_type:
                                # Handle Union types (Optional[List[StructuredModel]])
                                if get_origin(field_type) is Union:
                                    # Find the List type in the Union
                                    for arg in get_args(field_type):
                                        if get_origin(arg) is list:
                                            field_type = arg
                                            break
                                
                                # Check if it's a List[StructuredModel]
                                if get_origin(field_type) is list:
                                    args = get_args(field_type)
                                    if args and hasattr(args[0], 'model_fields'):
                                        nested_model_class = args[0]
                                        # Create TN entries for each nested field
                                        for nested_field_name in nested_model_class.model_fields:
                                            if nested_field_name != "extra_fields":
                                                pair_result["fields"][nested_field_name] = {
                                                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                                                    "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0}
                                                }
                        
                        # Set aggregate to sum of nested fields
                        aggregate_metrics["tn"] = nested_field_count
                else:
                    # Sum up contributions from nested fields (for non-null cases)
                    if "fields" in pair_result:
                        for nested_field_name, nested_field_result in pair_result["fields"].items():
                            if "aggregate" in nested_field_result:
                                # Use aggregate metrics from nested fields
                                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                                    aggregate_metrics[metric] += nested_field_result["aggregate"].get(metric, 0)
                            elif "overall" in nested_field_result:
                                # Fallback to overall metrics if no aggregate
                                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                                    aggregate_metrics[metric] += nested_field_result["overall"].get(metric, 0)
                
                # Set the aggregate section
                pair_result["aggregate"] = aggregate_metrics
                
                pair_results.append(pair_result)
        
        # Handle unmatched GT objects (contribute FN for non-null fields, TN for null fields)
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                gt_sub_value = getattr(gt_item, sub_field_name, None)
                
                if gt_sub_value is not None and gt_sub_value != [] and gt_sub_value != "":
                    # Non-null field - count as FN with nested field contributions
                    nested_field_count = gt_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0 and isinstance(gt_sub_value, list):
                        # Count the list items and their nested fields
                        list_length = len(gt_sub_value)
                        fn_count = list_length * nested_field_count
                    else:
                        # Not a hierarchical field or empty list, count as 1
                        fn_count = 1
                    
                    unmatched_result = {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                        "fields": {},
                        "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": fn_count}  # Add to aggregate only
                    }
                    pair_results.append(unmatched_result)
                else:
                    # Null/empty field - count as TN with nested field contributions
                    nested_field_count = gt_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0:
                        # For hierarchical fields, count nested fields as TN
                        tn_count = nested_field_count
                        
                        # Create nested field entries so that _collect_all_primitive_metrics can find them
                        unmatched_result = {
                            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                            "fields": {},
                            "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": tn_count, "fn": 0}  # Add to aggregate only
                        }
                        
                        # Get the nested model class to create field entries
                        field_info = gt_item.__class__.model_fields.get(sub_field_name)
                        if field_info:
                            from typing import get_origin, get_args, Union
                            field_type = field_info.annotation if hasattr(field_info, 'annotation') else None
                            if field_type:
                                # Handle Union types (Optional[List[StructuredModel]])
                                if get_origin(field_type) is Union:
                                    # Find the List type in the Union
                                    for arg in get_args(field_type):
                                        if get_origin(arg) is list:
                                            field_type = arg
                                            break
                                
                                # Check if it's a List[StructuredModel]
                                if get_origin(field_type) is list:
                                    args = get_args(field_type)
                                    if args and hasattr(args[0], 'model_fields'):
                                        nested_model_class = args[0]
                                        # Create TN entries for each nested field
                                        for nested_field_name in nested_model_class.model_fields:
                                            if nested_field_name != "extra_fields":
                                                unmatched_result["fields"][nested_field_name] = {
                                                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                                                    "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0}  # Add to aggregate only
                                                }
                    else:
                        # Not a hierarchical field, count as 1
                        tn_count = 1
                        unmatched_result = {
                            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                            "fields": {},
                            "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": tn_count, "fn": 0}  # Add to aggregate only
                        }
                    
                    pair_results.append(unmatched_result)
        
        # Handle unmatched pred objects (contribute FA for hierarchical fields)
        for pred_idx, pred_item in enumerate(pred_list):
            if pred_idx not in matched_pred_indices:
                pred_sub_value = getattr(pred_item, sub_field_name, None)
                
                if pred_sub_value is not None and pred_sub_value != [] and pred_sub_value != "":
                    # Non-null field - count as FA with nested field contributions
                    nested_field_count = pred_item._get_nested_field_count(sub_field_name)
                    if nested_field_count > 0 and isinstance(pred_sub_value, list):
                        # Count the list items and their nested fields
                        list_length = len(pred_sub_value)
                        fa_count = list_length * nested_field_count
                    else:
                        # Not a hierarchical field or empty list, count as 1
                        fa_count = 1
                    
                    unmatched_result = {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},  # Empty for unmatched objects
                        "fields": {},
                        "aggregate": {"tp": 0, "fa": fa_count, "fd": 0, "fp": fa_count, "tn": 0, "fn": 0}  # Add to aggregate only
                    }
                    pair_results.append(unmatched_result)
        
        # Use recursive aggregation function
        aggregated_result = self._recursive_aggregate_metrics(pair_results)
        
        # Add derived metrics recursively
        self._add_derived_metrics_recursively(aggregated_result)
        
        # Add metadata from first pair if available
        if pair_results:
            for key in [
                "raw_similarity_score",
                "similarity_score",
                "threshold_applied_score",
                "weight",
            ]:
                if key in pair_results[0]:
                    aggregated_result[key] = pair_results[0][key]
        
        return (
            aggregated_result
            if pair_results
            else {"overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}}
        )

    
    def _expand_hierarchical_field_metrics(self, pair_result: Dict[str, Any], gt_value: Any, pred_value: Any, field_name: str, gt_item: 'StructuredModel') -> Dict[str, Any]:
        """Expand hierarchical field metrics to include nested field contributions.
        
        When a hierarchical field (like List[StructuredModel]) is classified as FA or TN,
        we need to add contributions for its nested fields to match the expected behavior.
        
        Args:
            pair_result: The current pair result from field comparison
            gt_value: Ground truth value for the field
            pred_value: Predicted value for the field  
            field_name: Name of the field
            gt_item: The ground truth item containing the field
            
        Returns:
            Updated pair result with expanded nested field metrics
        """
        # Use the helper method to get nested field count
        nested_field_count = gt_item._get_nested_field_count(field_name)
        
        if nested_field_count > 0:
            # If this is a FA case (GT=None, Pred=non-None), add FA for each nested field
            if pair_result["overall"].get("fa", 0) > 0:
                pair_result["overall"]["fa"] += nested_field_count
                pair_result["overall"]["fp"] += nested_field_count
                # Also update aggregate section if present
                if "aggregate" in pair_result:
                    pair_result["aggregate"]["fa"] += nested_field_count
                    pair_result["aggregate"]["fp"] += nested_field_count
            
            # If this is a TN case (GT=None, Pred=None), add TN for each nested field  
            elif pair_result["overall"].get("tn", 0) > 0:
                pair_result["overall"]["tn"] += nested_field_count
                # Also update aggregate section if present
                if "aggregate" in pair_result:
                    pair_result["aggregate"]["tn"] += nested_field_count
            
            # If this is a FN case (GT=non-None, Pred=None), add FN for each nested field
            elif pair_result["overall"].get("fn", 0) > 0:
                pair_result["overall"]["fn"] += nested_field_count
                # Also update aggregate section if present
                if "aggregate" in pair_result:
                    pair_result["aggregate"]["fn"] += nested_field_count
        
        return pair_result
    
    def _recursive_aggregate_metrics(self, pair_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recursively aggregate metrics from multiple pair results - handles arbitrary depth."""
        if not pair_results:
            return {
                "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
                "fields": {},
            }
        
        # Initialize the aggregated result
        aggregated = {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
            "fields": {},
            "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
        }
        
        for pair_result in pair_results:
            # Aggregate overall metrics
            if "overall" in pair_result:
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    aggregated["overall"][metric] += pair_result["overall"].get(
                        metric, 0
                    )
            
            # Aggregate aggregate metrics
            if "aggregate" in pair_result:
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    aggregated["aggregate"][metric] += pair_result["aggregate"].get(
                        metric, 0
                    )
            
            # Recursively aggregate fields
            if "fields" in pair_result:
                aggregated["fields"] = self._recursive_merge_fields(
                    aggregated["fields"], pair_result["fields"]
                )
        
        return aggregated
    
    def _recursive_merge_fields(
        self, target_fields: Dict[str, Any], source_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge field metrics - TRUE recursion for arbitrary depth."""
        for field_name, field_metrics in source_fields.items():
            if field_name not in target_fields:
                # Initialize field in target with same structure as source
                if "overall" in field_metrics:
                    # Hierarchical structure
                    target_fields[field_name] = {
                        "overall": {
                            "tp": 0,
                            "fa": 0,
                            "fd": 0,
                            "fp": 0,
                            "tn": 0,
                            "fn": 0,
                        },
                        "fields": {},
                        "aggregate": {
                            "tp": 0,
                            "fa": 0,
                            "fd": 0,
                            "fp": 0,
                            "tn": 0,
                            "fn": 0,
                        },
                    }
                else:
                    # Flat structure
                    target_fields[field_name] = {
                        "tp": 0,
                        "fa": 0,
                        "fd": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": 0,
                    }
            
            # Aggregate metrics based on structure type
            if "overall" in field_metrics:
                # Hierarchical structure - aggregate overall and aggregate sections, and recurse into fields
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    target_fields[field_name]["overall"][metric] += field_metrics[
                        "overall"
                    ].get(metric, 0)
                
                # Aggregate the aggregate section if present
                if "aggregate" in field_metrics:
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        target_fields[field_name]["aggregate"][metric] += field_metrics[
                            "aggregate"
                        ].get(metric, 0)
                
                # RECURSIVE CALL: Handle nested fields at arbitrary depth
                if "fields" in field_metrics:
                    if "fields" not in target_fields[field_name]:
                        target_fields[field_name]["fields"] = {}
                    target_fields[field_name]["fields"] = self._recursive_merge_fields(
                        target_fields[field_name]["fields"], field_metrics["fields"]
                    )
            else:
                # Flat structure - aggregate directly
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    target_fields[field_name][metric] += field_metrics.get(metric, 0)
        
        return target_fields
    
    def _add_derived_metrics_recursively(self, metrics_dict: Dict[str, Any]) -> None:
        """Recursively add derived metrics to all levels of the structure."""
        metrics_helper = MetricsHelper()
        
        # Add derived metrics to overall if present
        if "overall" in metrics_dict:
            metrics_dict["overall"]["derived"] = (
                metrics_helper.calculate_derived_metrics(metrics_dict["overall"])
            )
        
        # Add derived metrics to aggregate if present
        if "aggregate" in metrics_dict:
            metrics_dict["aggregate"]["derived"] = (
                metrics_helper.calculate_derived_metrics(metrics_dict["aggregate"])
            )
        
        # Recursively process fields
        if "fields" in metrics_dict:
            for field_name, field_data in metrics_dict["fields"].items():
                if "overall" in field_data:
                    # Hierarchical structure - add derived and recurse
                    field_data["overall"]["derived"] = (
                        metrics_helper.calculate_derived_metrics(field_data["overall"])
                    )
                    if "aggregate" in field_data:
                        field_data["aggregate"]["derived"] = (
                            metrics_helper.calculate_derived_metrics(field_data["aggregate"])
                        )
                    self._add_derived_metrics_recursively(field_data)  # RECURSIVE CALL
                elif "tp" in field_data:
                    # Flat structure with metrics - add derived metrics directly
                    field_data["derived"] = metrics_helper.calculate_derived_metrics(
                        field_data
                    )
                # If neither "overall" nor "tp" is present, it might be an empty structure - skip

    def _handle_primitive_field_threshold_aware(
        self,
        sub_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        matched_pairs: List,
        matched_gt_indices: set,
        matched_pred_indices: set,
        match_threshold: float) -> Dict[str, Any]:
        """Handle primitive fields with threshold-aware metric population.
        
        Process ALL matched pairs for field-level analysis, but populate metrics sections
        based on object-level similarity threshold.
        """
        
        # Initialize metrics sections
        overall_metrics = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
        aggregate_metrics = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
        
        # Process matched pairs with threshold-aware metric population
        for gt_idx, pred_idx, similarity in matched_pairs:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]
                gt_sub_value = getattr(gt_item, sub_field_name, None)
                pred_sub_value = getattr(pred_item, sub_field_name, None)
                
                # Get field-level classification for this pair
                field_classification = gt_item._classify_field_for_confusion_matrix(sub_field_name, pred_sub_value)
                
                # Add to appropriate sections based on threshold
                self._add_to_metric_sections(field_classification, similarity, match_threshold,
                                           overall_metrics, aggregate_metrics)
        
        # Handle unmatched GT objects (contribute to aggregate only)
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                gt_sub_value = getattr(gt_item, sub_field_name, None)
                if gt_sub_value is not None and gt_sub_value != "" and gt_sub_value != []:
                    aggregate_metrics["fn"] += 1
                else:
                    # GT field is None/empty - this is a TN (correctly predicted as absent)
                    aggregate_metrics["tn"] += 1
        
        # Handle unmatched pred objects (contribute to aggregate only)
        for pred_idx, pred_item in enumerate(pred_list):
            if pred_idx not in matched_pred_indices:
                pred_sub_value = getattr(pred_item, sub_field_name, None)
                if pred_sub_value is not None and pred_sub_value != "" and pred_sub_value != []:
                    aggregate_metrics["fa"] += 1
                    aggregate_metrics["fp"] += 1
        
        # Return both overall and aggregate sections
        return {
            "overall": overall_metrics,
            "aggregate": aggregate_metrics
        }

    def _should_populate_overall_metrics(self, similarity: float, match_threshold: float) -> bool:
        """Determine if metrics should populate the "overall" section based on threshold.
        
        This method implements threshold validation logic to determine whether field-level
        metrics from a matched pair should be included in the "overall" section.
        
        Args:
            similarity: The similarity score between the matched objects
            match_threshold: The threshold for considering objects as matches
            
        Returns:
            True if metrics should populate "overall" section, False otherwise
        """
        # Handle None or invalid threshold cases - populate both sections for backward compatibility
        if match_threshold is None:
            return True
            
        # Validate threshold is in valid range
        if not isinstance(match_threshold, (int, float)) or not (0.0 <= match_threshold <= 1.0):
            # Invalid threshold - default to populating both sections
            return True
            
        # Use ThresholdHelper for consistent threshold checking with floating point precision
        from .threshold_helper import ThresholdHelper
        return ThresholdHelper.is_above_threshold(similarity, match_threshold)

    def _should_populate_aggregate_metrics(self, similarity: float, match_threshold: float) -> bool:
        """Determine if metrics should populate the "aggregate" section.
        
        The aggregate section is always populated for universal aggregation functionality,
        regardless of threshold or similarity score.
        
        Args:
            similarity: The similarity score between the matched objects (unused)
            match_threshold: The threshold for considering objects as matches (unused)
            
        Returns:
            Always True - aggregate metrics are always populated
        """
        return True

    def _add_to_metric_sections(self, field_classification: dict, similarity: float, 
                               match_threshold: float, overall_metrics: dict, 
                               aggregate_metrics: dict) -> None:
        """Add field classification results to appropriate metric sections based on threshold.
        
        This method implements the core threshold-aware metric population logic:
        - Above threshold: populate both "overall" and "aggregate" sections
        - Below threshold: populate only "aggregate" section
        
        Args:
            field_classification: Dictionary with metric counts (tp, fa, fd, fp, tn, fn)
            similarity: The similarity score between the matched objects
            match_threshold: The threshold for considering objects as matches
            overall_metrics: Dictionary to accumulate overall metrics
            aggregate_metrics: Dictionary to accumulate aggregate metrics
        """
        # Always populate aggregate section for universal aggregation
        if self._should_populate_aggregate_metrics(similarity, match_threshold):
            for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                aggregate_metrics[metric] += field_classification.get(metric, 0)
        
        # Only populate overall section if above threshold
        if self._should_populate_overall_metrics(similarity, match_threshold):
            for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                overall_metrics[metric] += field_classification.get(metric, 0)

# Import needed at bottom to avoid circular imports
from .structured_model import StructuredModel
