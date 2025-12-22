"""Base class for comparison helpers in StructuredModel comparisons."""

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .hungarian_helper import HungarianHelper

DEFAULT_MATCH_THRESHOLD = 0.7

class ComparisonHelperBase(ABC):
    """Base class for collecting and formatting comparison data in StructuredModel comparisons."""

    def __init__(self):
        self.hungarian_helper = HungarianHelper()

    @abstractmethod
    def create_entry(self, *args, **kwargs) -> Dict[str, Any]:
        """Create an entry for the specific helper type. Must be implemented by subclasses."""
        pass

    def get_match_threshold(self, obj_list: List[Any]) -> float:
        """Get the match threshold from the model class or return default.
        
        Args:
            obj_list: List of objects to extract threshold from
            
        Returns:
            Match threshold value
        """
        if (
            obj_list
            and hasattr(obj_list[0], "__class__")
            and hasattr(obj_list[0].__class__, "match_threshold")
        ):
            return obj_list[0].__class__.match_threshold
        return DEFAULT_MATCH_THRESHOLD

    def get_optimal_assignments(self, gt_list: List[Any], pred_list: List[Any]) -> tuple:
        """Get optimal assignments and matched pairs with scores.
        
        Args:
            gt_list: Ground truth list
            pred_list: Prediction list
            
        Returns:
            Tuple of (assignments, matched_pairs_with_scores)
        """
        assignments = []
        matched_pairs_with_scores = []
        
        if gt_list and pred_list:
            hungarian_info = self.hungarian_helper.get_complete_matching_info(
                gt_list, pred_list
            )
            matched_pairs_with_scores = hungarian_info["matched_pairs"]
            assignments = [(i, j) for i, j, _ in matched_pairs_with_scores]
            
        return assignments, matched_pairs_with_scores

    def generate_comparison_reason(self, is_match: bool, score: float, threshold: float) -> str:
        """Generate a descriptive reason for a comparison result.
        
        Args:
            is_match: Whether this is considered a match
            score: Similarity score
            threshold: Match threshold
            
        Returns:
            Descriptive reason string
        """
        if is_match:
            if math.isclose(score,1.0):
                return "exact match"
            else:
                return f"above threshold ({score:.3f} >= {threshold})"
        else:
            return f"below threshold ({score:.3f} < {threshold})"

    def serialize_object(self, obj: Any) -> Any:
        """Serialize an object, handling StructuredModel serialization.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized object
        """
        if obj and hasattr(obj, "model_dump"):
            return obj.model_dump()
        return obj

    def process_matched_pairs(
        self, 
        field_name: str, 
        gt_list: List[Any], 
        pred_list: List[Any],
        matched_pairs_with_scores: List[tuple],
        match_threshold: float
    ) -> List[Dict[str, Any]]:
        """Process matched pairs and create entries.
        
        Args:
            field_name: Name of the list field
            gt_list: Ground truth list
            pred_list: Prediction list
            matched_pairs_with_scores: List of (gt_idx, pred_idx, score) tuples
            match_threshold: Threshold for determining matches
            
        Returns:
            List of entries for matched pairs
        """
        entries = []
        
        for gt_idx, pred_idx, similarity_score in matched_pairs_with_scores:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]
                
                # Determine if this is a match
                is_match = bool(similarity_score >= match_threshold)
                
                # Create reason
                reason = self.generate_comparison_reason(is_match, similarity_score, match_threshold)
                
                # Extract entries from structured model objects
                extracted_entries = self._extract_entries_from_objects(
                    field_name,
                    gt_item,
                    pred_item,
                    gt_idx,
                    pred_idx,
                    is_match,
                    similarity_score,
                    reason,
                )
                entries.extend(extracted_entries)
                
        return entries

    def process_unmatched_gt_items(
        self, 
        field_name: str, 
        gt_list: List[Any], 
        assignments: List[tuple]
    ) -> List[Dict[str, Any]]:
        """Process unmatched ground truth items (False Negatives).
        
        Args:
            field_name: Name of the list field
            gt_list: Ground truth list
            assignments: List of (gt_idx, pred_idx) assignment pairs
            
        Returns:
            List of entries for unmatched GT items
        """
        entries = []
        matched_gt_indices = set(idx for idx, _ in assignments)
        
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                extracted_entries = self._extract_entries_from_objects(
                    field_name=field_name,
                    gt_object=gt_item,
                    pred_object=None,
                    gt_index=gt_idx,
                    pred_index=None,
                    is_match=False,
                    similarity_score=0.0, 
                    reason="false negative (unmatched ground truth)",
                )
                entries.extend(extracted_entries)
                
        return entries

    def process_unmatched_pred_items(
        self, 
        field_name: str, 
        pred_list: List[Any], 
        assignments: List[tuple]
    ) -> List[Dict[str, Any]]:
        """Process unmatched prediction items (False Alarms).
        
        Args:
            field_name: Name of the list field
            pred_list: Prediction list
            assignments: List of (gt_idx, pred_idx) assignment pairs
            
        Returns:
            List of entries for unmatched prediction items
        """
        entries = []
        matched_pred_indices = set(idx for _, idx in assignments)
        
        for pred_idx, pred_item in enumerate(pred_list):
            if pred_idx not in matched_pred_indices:
                extracted_entries = self._extract_entries_from_objects(
                    field_name=field_name,
                    gt_object=None,
                    pred_object=pred_item,
                    gt_index=None,
                    pred_index=pred_idx,
                    is_match=False,  
                    similarity_score=0.0,   
                    reason="false alarm (unmatched prediction)",
                )
                entries.extend(extracted_entries)
                
        return entries

    def process_null_cases(
        self, 
        field_name: str, 
        gt_list: List[Any], 
        pred_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Process null cases (empty lists).
        
        Args:
            field_name: Name of the field
            gt_list: Ground truth list (may be empty/None)
            pred_list: Prediction list (may be empty/None)
            
        Returns:
            List of entries for null cases
        """
        entries = []

        # Check if both lists are empty
        if not gt_list and not pred_list:
            return entries

        # Handle gt null case (FA items when GT is empty)
        if not gt_list and pred_list:
            for pred_idx, pred_item in enumerate(pred_list):
                extracted_entries = self._extract_entries_from_objects(
                    field_name=field_name,
                    gt_object=None,
                    pred_object=pred_item,
                    gt_index=None,
                    pred_index=pred_idx,
                    is_match=False,  
                    similarity_score=0.0,    
                    reason="false alarm (unmatched prediction)",
                )
                entries.extend(extracted_entries)

        # Handle pred null case (FN items when prediction is empty)
        elif gt_list and not pred_list:
            for gt_idx, gt_item in enumerate(gt_list):
                extracted_entries = self._extract_entries_from_objects(
                    field_name=field_name,
                    gt_object=gt_item,
                    pred_object=None,
                    gt_index=gt_idx,
                    pred_index=None,
                    is_match=False,  
                    similarity_score=0.0, 
                    reason="false negative (unmatched ground truth)",
                )
                entries.extend(extracted_entries)

        return entries

    @abstractmethod
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
        """Extract entries from structured model objects. Must be implemented by subclasses.
        
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
            List of entries specific to the helper type
        """
        pass

    def collect_list_entries(
        self, field_name: str, gt_list: List[Any], pred_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """Collect entries from a list field using the template method pattern.

        Args:
            field_name: Name of the list field
            gt_list: Ground truth list
            pred_list: Prediction list

        Returns:
            List of entries with individual information
        """
        entries = []

        if not gt_list and not pred_list:
            return entries

        # Get optimal assignments with scores
        assignments, matched_pairs_with_scores = self.get_optimal_assignments(gt_list, pred_list)
        
        match_threshold = self.get_match_threshold(gt_list or pred_list)

        entries.extend(self.process_matched_pairs(
            field_name, gt_list, pred_list, matched_pairs_with_scores, match_threshold
        ))

        # Process unmatched ground truth items (FN)
        entries.extend(self.process_unmatched_gt_items(field_name, gt_list, assignments))

        # Process unmatched prediction items (FA)
        entries.extend(self.process_unmatched_pred_items(field_name, pred_list, assignments))

        return entries
