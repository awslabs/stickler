"""Non-matches helper for StructuredModel comparisons."""

from typing import List, Dict, Any, Optional
from .comparison_helper_base import ComparisonHelperBase
from .non_match_field import NonMatchType

class NonMatchesHelper(ComparisonHelperBase):
    """Helper class for collecting and formatting non-matches in StructuredModel comparisons."""

    def create_entry(
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
        return self.create_non_match_entry(
            field_name, gt_object, pred_object, non_match_type, object_index, similarity_score
        )

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
            "ground_truth_value": self.serialize_object(gt_object),
            "prediction_value": self.serialize_object(pred_object),
        }

        # Add descriptive reason based on non-match type
        if non_match_type == "FD":
            # False Discovery: matched but below threshold
            if similarity_score is not None:
                # Get the match threshold from the object
                threshold = self.get_match_threshold([gt_object] if gt_object else [pred_object])
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
        # Use base class method but filter for non-matches only
        all_entries = self.collect_list_entries(field_name, gt_list, pred_list)
        
        # Filter for non-matches only (entries where match is False or non_match_type exists)
        non_matches = []
        for entry in all_entries:
            # Check if this is a non-match entry
            if "non_match_type" in entry or (entry.get("match") is False):
                non_matches.append(entry)
                
        return non_matches

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
        """Extract non-match entries from structured model objects.
        
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
            List of non-match entries
        """
        # Only return entries for non-matches
        if is_match:
            return []
            
        # Determine non-match type based on available objects
        if gt_object is None:
            non_match_type = "FA"
            object_index = pred_index
        elif pred_object is None:
            non_match_type = "FN"
            object_index = gt_index
        else:
            # Both objects exist but similarity is below threshold
            non_match_type = "FD"
            object_index = gt_index
            
        return self._extract_field_level_non_matches(
            field_name, gt_object, pred_object, object_index, non_match_type, similarity_score
        )
    
    def _extract_field_level_non_matches(
        self, 
        field_name: str, 
        gt_object: Any, 
        pred_object: Any, 
        object_index: int,
        non_match_type: str,
        similarity_score: float = None
    ) -> List[Dict[str, Any]]:
        """Extract field-level non-matches from structured model objects.
        
        Args:
            field_name: Name of the parent list field
            gt_object: Ground truth structured model
            pred_object: Prediction structured model  
            object_index: Index in the list
            non_match_type: Type of non-match ("FD", "FN", "FA")
            similarity_score: Overall similarity score
            
        Returns:
            List of field-level non-match entries
        """
        from .structured_model import StructuredModel
        # Check if both objects are structured models
        if (isinstance(gt_object, StructuredModel) and isinstance(pred_object, StructuredModel)):
            # Perform field-by-field comparison
            comparison_result = gt_object.compare_with(
                pred_object, 
                document_non_matches=True,
                include_confusion_matrix=False
            )
            
            field_non_matches = []
            
            for non_match in comparison_result.get("non_matches", []):
                indexed_field_path = f"{field_name}[{object_index}].{non_match['field_path']}"
                
                field_entry = {
                    "field_path": indexed_field_path,
                    "non_match_type": non_match["non_match_type"],
                    "ground_truth_value": non_match["ground_truth_value"],
                    "prediction_value": non_match["prediction_value"],
                    "reason": non_match.get("reason", "field mismatch"),
                }
                
                if "similarity_score" in non_match:
                    field_entry["similarity_score"] = non_match["similarity_score"]
                    
                field_non_matches.append(field_entry)
                
            return field_non_matches
        
        else:
            return [self.create_non_match_entry(
                field_name, gt_object, pred_object, non_match_type, object_index, similarity_score
            )]
