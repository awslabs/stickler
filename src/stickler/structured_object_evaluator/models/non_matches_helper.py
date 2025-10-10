"""Non-matches helper for StructuredModel comparisons."""

from typing import List, Dict, Any, Optional
from .hungarian_helper import HungarianHelper
from .non_match_field import NonMatchType


class NonMatchesHelper:
    """Helper class for collecting and formatting non-matches in StructuredModel comparisons."""
    
    def __init__(self):
        self.hungarian_helper = HungarianHelper()
    
    def create_non_match_entry(
        self,
        field_name: str,
        gt_object: Any,
        pred_object: Any,
        non_match_type: str,
        object_index: int = None,
        similarity_score: float = None,
    ) -> Dict[str, Any]:
        """Create a non-match entry for detailed analysis.
        
        Args:
            field_name: Name of the field
            gt_object: Ground truth object (can be None for FA)
            pred_object: Prediction object (can be None for FN)
            non_match_type: Type of non-match ("FD", "FN", "FA")
            object_index: Optional index of the object in the list for indexed field paths
            similarity_score: Similarity score for FD entries
            
        Returns:
            Dictionary with non-match information
        """
        # Generate indexed field path if object_index provided
        indexed_field_path = (
            f"{field_name}[{object_index}]" if object_index is not None else field_name
        )
        
        # Map short codes to actual NonMatchType enum values
        type_mapping = {
            "FD": NonMatchType.FALSE_DISCOVERY,
            "FN": NonMatchType.FALSE_NEGATIVE, 
            "FA": NonMatchType.FALSE_ALARM,
        }
        
        entry = {
            "field_path": indexed_field_path,
            "non_match_type": type_mapping.get(non_match_type, non_match_type),
            "ground_truth_value": gt_object.model_dump()
            if gt_object and hasattr(gt_object, "model_dump")
            else gt_object,
            "prediction_value": pred_object.model_dump()
            if pred_object and hasattr(pred_object, "model_dump")
            else pred_object,
        }
        
        # Add descriptive reason based on non-match type
        if non_match_type == "FD":
            # False Discovery: matched but below threshold
            if similarity_score is not None:
                # Get the match threshold from the object
                if (
                    gt_object
                    and hasattr(gt_object, "__class__")
                    and hasattr(gt_object.__class__, "match_threshold")
                ):
                    threshold = gt_object.__class__.match_threshold
                else:
                    threshold = 0.7  # Default threshold
                entry["reason"] = (
                    f"below threshold ({similarity_score:.3f} < {threshold})"
                )
                entry["similarity"] = similarity_score
                entry["similarity_score"] = similarity_score
            else:
                entry["reason"] = "below threshold"
        elif non_match_type == "FN":
            # False Negative: unmatched ground truth
            entry["reason"] = "unmatched ground truth"
        elif non_match_type == "FA":
            # False Alarm: unmatched prediction
            entry["reason"] = "unmatched prediction"
        else:
            entry["reason"] = "unknown non-match type"
        
        return entry
    
    def collect_list_non_matches(
        self, field_name: str, gt_list: List[Any], pred_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Collect individual object-level non-matches from a list field.
        
        Args:
            field_name: Name of the list field
            gt_list: Ground truth list
            pred_list: Prediction list
            
        Returns:
            List of non-match dictionaries with individual object information
        """
        non_matches = []
        
        if not gt_list and not pred_list:
            return non_matches
        
        # Get optimal assignments with scores
        assignments = []
        matched_pairs_with_scores = []
        if gt_list and pred_list:
            hungarian_info = self.hungarian_helper.get_complete_matching_info(
                gt_list, pred_list
            )
            matched_pairs_with_scores = hungarian_info["matched_pairs"]
            assignments = [(i, j) for i, j, score in matched_pairs_with_scores]
        
        # Get the match threshold from the model class
        if (
            gt_list
            and hasattr(gt_list[0], "__class__")
            and hasattr(gt_list[0].__class__, "match_threshold")
        ):
            match_threshold = gt_list[0].__class__.match_threshold
        else:
            match_threshold = 0.7
        
        # Process matched pairs for FD entries
        for gt_idx, pred_idx, similarity_score in matched_pairs_with_scores:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]
                
                # Check if this is a False Discovery (below threshold)
                is_below_threshold = (
                    similarity_score < match_threshold
                    and abs(similarity_score - match_threshold) >= 1e-10
                )
                if is_below_threshold:
                    non_matches.append(
                        self.create_non_match_entry(
                            field_name,
                            gt_item,
                            pred_item,
                            "FD",
                            gt_idx,
                            similarity_score,
                        )
                    )
        
        # Process unmatched ground truth items (FN)
        matched_gt_indices = set(idx for idx, _ in assignments)
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                non_matches.append(
                    self.create_non_match_entry(field_name, gt_item, None, "FN", gt_idx)
                )
        
        # Process unmatched prediction items (FA)
        matched_pred_indices = set(idx for _, idx in assignments)
        for pred_idx, pred_item in enumerate(pred_list):
            if pred_idx not in matched_pred_indices:
                non_matches.append(
                    self.create_non_match_entry(
                    field_name, None, pred_item, "FA", pred_idx
                    )
                )
        
        return non_matches
    
    def add_non_matches_for_null_cases(
        self, field_name: str, gt_list: List[Any], pred_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Add non-matches for null cases (empty lists).
        
        Args:
            field_name: Name of the field
            gt_list: Ground truth list (may be empty/None)
            pred_list: Prediction list (may be empty/None)
            
        Returns:
            List of non-match entries for null cases
        """
        non_matches = []
        
        # Handle null cases
        if not gt_list and pred_list:
            # Add non-matches for each FA item when GT is empty
            for pred_idx, pred_item in enumerate(pred_list):
                non_matches.append(
                    self.create_non_match_entry(
                    field_name, None, pred_item, "FA", pred_idx
                    )
                )
        elif gt_list and not pred_list:
            # Add non-matches for each FN item when prediction is empty
            for gt_idx, gt_item in enumerate(gt_list):
                non_matches.append(self.create_non_match_entry(
                    field_name, gt_item, None, "FN", gt_idx
                ))
        
        return non_matches
    def create_leaf_non_match_entry(self, field_path: str, gt_value: Any, pred_value: Any, 
                                   comparison_type: str, similarity_score: Optional[float] = None,
                                   threshold: Optional[float] = None) -> Dict[str, Any]:
        """Create a leaf-level non-match entry for primitive field comparisons.
        
        Args:
            field_path: Full dot-notation path to the leaf field (e.g., 'people[0].name')
            gt_value: Ground truth value
            pred_value: Prediction value
            comparison_type: Type of comparison result ("TP", "FA", "FD", "TN", "FN")
            similarity_score: Similarity score from comparator
            threshold: Threshold used for classification
            
        Returns:
            Dictionary with leaf-level non-match information
        """
        # Map comparison types to non-match types for failed comparisons
        type_mapping = {
            "FD": NonMatchType.FALSE_DISCOVERY,
            "FN": NonMatchType.FALSE_NEGATIVE, 
            "FA": NonMatchType.FALSE_ALARM
        }
        
        # Only create non-match entries for failed comparisons
        if comparison_type not in type_mapping:
            return None
            
        entry = {
            "field_path": field_path,
            "non_match_type": type_mapping[comparison_type],
            "ground_truth_value": gt_value,
            "prediction_value": pred_value,
            "comparison_type": comparison_type,
            "is_leaf_level": True  # Flag to distinguish from object-level non_matches
        }
        
        # Add similarity and threshold information for debugging
        if similarity_score is not None:
            entry["similarity_score"] = similarity_score
            entry["similarity"] = similarity_score
            
        if threshold is not None:
            entry["threshold"] = threshold
            
        # Add descriptive reason based on comparison type
        if comparison_type == "FD":
            if similarity_score is not None and threshold is not None:
                entry["reason"] = f"below threshold ({similarity_score:.3f} < {threshold})"
            else:
                entry["reason"] = "below threshold"
        elif comparison_type == "FN":
            entry["reason"] = "ground truth present, prediction missing/null"
        elif comparison_type == "FA":
            entry["reason"] = "prediction present, ground truth missing/null"
        else:
            entry["reason"] = f"comparison failed ({comparison_type})"
        
        return entry
    
    def collect_primitive_field_non_matches(self, field_path: str, gt_value: Any, pred_value: Any,
                                          comparison_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect leaf-level non_matches from primitive field comparison results.
        
        Args:
            field_path: Full path to the field being compared
            gt_value: Ground truth value
            pred_value: Prediction value  
            comparison_result: Result from primitive field comparison
            
        Returns:
            List of leaf-level non-match entries
        """
        non_matches = []
        
        # Extract metrics from comparison result
        overall = comparison_result.get("overall", {})
        similarity_score = comparison_result.get("similarity_score")
        threshold_score = comparison_result.get("threshold_applied_score")
        
        # Determine threshold from comparison info if available
        threshold = None
        if hasattr(comparison_result, 'threshold'):
            threshold = comparison_result.threshold
        elif similarity_score is not None and threshold_score is not None:
            # If threshold was applied and score changed, we can infer threshold was used
            if abs(similarity_score - threshold_score) > 1e-10:
                # This suggests threshold clipping occurred
                threshold = 0.7  # Default threshold assumption
        
        # Check each metric type and create non-match entries for failures
        if overall.get("fd", 0) > 0:
            entry = self.create_leaf_non_match_entry(
                field_path, gt_value, pred_value, "FD", similarity_score, threshold
            )
            if entry:
                non_matches.append(entry)
                
        if overall.get("fn", 0) > 0:
            entry = self.create_leaf_non_match_entry(
                field_path, gt_value, pred_value, "FN", similarity_score, threshold
            )
            if entry:
                non_matches.append(entry)
                
        if overall.get("fa", 0) > 0:
            entry = self.create_leaf_non_match_entry(
                field_path, gt_value, pred_value, "FA", similarity_score, threshold
            )
            if entry:
                non_matches.append(entry)
        
        return non_matches
    
    def collect_primitive_list_non_matches(self, field_path: str, gt_list: List[Any], pred_list: List[Any],
                                         comparison_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect leaf-level non_matches from primitive list comparison results.
        
        Args:
            field_path: Full path to the list field being compared
            gt_list: Ground truth list
            pred_list: Prediction list
            comparison_result: Result from primitive list comparison
            
        Returns:
            List of leaf-level non-match entries for individual list items
        """
        non_matches = []
        
        # For primitive lists, we need to analyze the Hungarian matching results
        # to determine which specific items contributed to FA/FD/FN counts
        
        # Extract metrics from comparison result
        overall = comparison_result.get("overall", {})
        similarity_score = comparison_result.get("similarity_score", 0.0)
        
        # Handle null/empty cases
        if not gt_list and not pred_list:
            return non_matches
        elif not gt_list and pred_list:
            # All prediction items are FA
            for i, pred_item in enumerate(pred_list):
                entry = self.create_leaf_non_match_entry(
                    f"{field_path}[{i}]", None, pred_item, "FA", 0.0
                )
                if entry:
                    non_matches.append(entry)
        elif gt_list and not pred_list:
            # All ground truth items are FN
            for i, gt_item in enumerate(gt_list):
                entry = self.create_leaf_non_match_entry(
                    f"{field_path}[{i}]", gt_item, None, "FN", 0.0
                )
                if entry:
                    non_matches.append(entry)
        else:
            # Both lists have items - need to analyze matching results
            # For now, create aggregate entries based on overall counts
            # TODO: In future, could integrate with Hungarian matching to get specific item pairs
            
            fd_count = overall.get("fd", 0)
            fa_count = overall.get("fa", 0) 
            fn_count = overall.get("fn", 0)
            
            # Create representative entries for each type of failure
            if fd_count > 0:
                entry = self.create_leaf_non_match_entry(
                    f"{field_path}[*]", f"{len(gt_list)} items", f"{len(pred_list)} items", 
                    "FD", similarity_score
                )
                if entry:
                    entry["count"] = fd_count
                    entry["reason"] = f"{fd_count} items below threshold"
                    non_matches.append(entry)
                    
            if fa_count > 0:
                entry = self.create_leaf_non_match_entry(
                    f"{field_path}[*]", f"{len(gt_list)} items", f"{len(pred_list)} items",
                    "FA", similarity_score
                )
                if entry:
                    entry["count"] = fa_count
                    entry["reason"] = f"{fa_count} unmatched prediction items"
                    non_matches.append(entry)
                    
            if fn_count > 0:
                entry = self.create_leaf_non_match_entry(
                    f"{field_path}[*]", f"{len(gt_list)} items", f"{len(pred_list)} items",
                    "FN", similarity_score
                )
                if entry:
                    entry["count"] = fn_count
                    entry["reason"] = f"{fn_count} unmatched ground truth items"
                    non_matches.append(entry)
        
        return non_matches
    
    def organize_non_matches_hierarchically(self, non_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Organize non_matches into a hierarchical structure for better debugging.
        
        Args:
            non_matches: List of non-match entries
            
        Returns:
            Hierarchical structure organizing non_matches by field path and type
        """
        hierarchical = {
            "summary": {
                "total_count": len(non_matches),
                "leaf_level_count": 0,
                "object_level_count": 0,
                "by_type": {"FA": 0, "FD": 0, "FN": 0}
            },
            "by_field_path": {},
            "by_type": {"FA": [], "FD": [], "FN": []},
            "leaf_level": [],
            "object_level": []
        }
        
        for nm in non_matches:
            # Count by type
            comparison_type = nm.get("comparison_type", "")
            if comparison_type in hierarchical["summary"]["by_type"]:
                hierarchical["summary"]["by_type"][comparison_type] += 1
            
            # Separate leaf-level from object-level
            if nm.get("is_leaf_level", False):
                hierarchical["leaf_level"].append(nm)
                hierarchical["summary"]["leaf_level_count"] += 1
            else:
                hierarchical["object_level"].append(nm)
                hierarchical["summary"]["object_level_count"] += 1
            
            # Organize by field path
            field_path = nm.get("field_path", "unknown")
            if field_path not in hierarchical["by_field_path"]:
                hierarchical["by_field_path"][field_path] = []
            hierarchical["by_field_path"][field_path].append(nm)
            
            # Organize by type
            if comparison_type in hierarchical["by_type"]:
                hierarchical["by_type"][comparison_type].append(nm)
        
        return hierarchical
    
    def add_aggregate_contribution_tracing(self, non_matches: List[Dict[str, Any]], 
                                         aggregate_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add traceability from leaf non_matches to aggregate metric contributions.
        
        Args:
            non_matches: List of non-match entries
            aggregate_metrics: Aggregate metrics from comparison result
            
        Returns:
            Enhanced non_matches with aggregate contribution information
        """
        enhanced_non_matches = []
        
        for nm in non_matches:
            enhanced_nm = nm.copy()
            
            # Add aggregate contribution information for leaf-level non_matches
            if nm.get("is_leaf_level", False):
                comparison_type = nm.get("comparison_type", "")
                field_path = nm.get("field_path", "")
                
                # Determine which aggregate this contributes to
                path_parts = field_path.split(".")
                parent_field = path_parts[0] if path_parts else "root"
                
                enhanced_nm["aggregate_contribution"] = {
                    "contributes_to": parent_field,
                    "metric_type": comparison_type.lower(),
                    "field_hierarchy": path_parts
                }
                
                # Add context about how this affects aggregate counts
                if comparison_type == "FD":
                    enhanced_nm["aggregate_contribution"]["impact"] = "Increases FD count in parent aggregate"
                elif comparison_type == "FA":
                    enhanced_nm["aggregate_contribution"]["impact"] = "Increases FA count in parent aggregate"
                elif comparison_type == "FN":
                    enhanced_nm["aggregate_contribution"]["impact"] = "Increases FN count in parent aggregate"
            
            enhanced_non_matches.append(enhanced_nm)
        
        return enhanced_non_matches
    
    def create_debugging_report(self, non_matches: List[Dict[str, Any]], 
                              aggregate_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a comprehensive debugging report from non_matches.
        
        Args:
            non_matches: List of non-match entries
            aggregate_metrics: Optional aggregate metrics for correlation
            
        Returns:
            Comprehensive debugging report
        """
        # Organize hierarchically
        hierarchical = self.organize_non_matches_hierarchically(non_matches)
        
        # Add aggregate contribution tracing if metrics provided
        if aggregate_metrics:
            enhanced_non_matches = self.add_aggregate_contribution_tracing(non_matches, aggregate_metrics)
            hierarchical["enhanced_entries"] = enhanced_non_matches
        
        # Create field-level analysis
        field_analysis = {}
        for field_path, field_non_matches in hierarchical["by_field_path"].items():
            field_analysis[field_path] = {
                "count": len(field_non_matches),
                "types": {},
                "has_leaf_level": any(nm.get("is_leaf_level", False) for nm in field_non_matches),
                "has_object_level": any(not nm.get("is_leaf_level", False) for nm in field_non_matches)
            }
            
            # Count by type for this field
            for nm in field_non_matches:
                comp_type = nm.get("comparison_type", "unknown")
                field_analysis[field_path]["types"][comp_type] = field_analysis[field_path]["types"].get(comp_type, 0) + 1
        
        # Create debugging insights
        insights = []
        
        if hierarchical["summary"]["leaf_level_count"] > 0:
            insights.append(f"Found {hierarchical['summary']['leaf_level_count']} leaf-level non_matches that explain primitive field failures")
        
        if hierarchical["summary"]["object_level_count"] > 0:
            insights.append(f"Found {hierarchical['summary']['object_level_count']} object-level non_matches for structured comparisons")
        
        # Identify fields with high failure rates
        high_failure_fields = [
            field for field, analysis in field_analysis.items() 
            if analysis["count"] > 1
        ]
        if high_failure_fields:
            insights.append(f"Fields with multiple failures: {', '.join(high_failure_fields)}")
        
        return {
            "hierarchical_structure": hierarchical,
            "field_analysis": field_analysis,
            "debugging_insights": insights,
            "total_non_matches": len(non_matches)
        }