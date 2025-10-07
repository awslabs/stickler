"""Non-matches helper for StructuredModel comparisons."""

from typing import List, Dict, Any
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
                non_matches.append(
                    self.create_non_match_entry(field_name, gt_item, None, "FN", gt_idx)
                )

        return non_matches
