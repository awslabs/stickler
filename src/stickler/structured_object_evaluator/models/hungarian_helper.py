"""Hungarian matching helper for StructuredModel comparisons."""

from typing import List, Any, Dict
from stickler.algorithms.hungarian import HungarianMatcher
from stickler.comparators.structured import StructuredModelComparator


class HungarianHelper:
    """Helper class for Hungarian matching operations with StructuredModel objects."""

    def __init__(self):
        self.hungarian = HungarianMatcher(StructuredModelComparator())

    def get_complete_matching_info(
        self, gt_list: List[Any], pred_list: List[Any]
    ) -> Dict[str, Any]:
        """Get all Hungarian matching information in one call to eliminate redundant calculations.

        This method performs Hungarian matching once and returns all derived information,
        eliminating the need for multiple calls that recalculate the same matching.

        Args:
            gt_list: Ground truth list
            pred_list: Prediction list

        Returns:
            Dictionary containing:
            - matched_pairs: List of tuples (gt_idx, pred_idx, similarity_score)
            - assignments: List of tuples (gt_idx, pred_idx) without scores
            - unmatched_gt_indices: List of unmatched ground truth indices
            - unmatched_pred_indices: List of unmatched prediction indices
            - matched_gt_indices: Set of matched ground truth indices
            - matched_pred_indices: Set of matched prediction indices
        """
        if not gt_list or not pred_list:
            return {
                "matched_pairs": [],
                "assignments": [],
                "unmatched_gt_indices": list(range(len(gt_list or []))),
                "unmatched_pred_indices": list(range(len(pred_list or []))),
                "matched_gt_indices": set(),
                "matched_pred_indices": set(),
            }

        # Single Hungarian algorithm call - this is the only expensive operation
        metrics = self.hungarian.calculate_metrics(gt_list, pred_list)
        matched_pairs = metrics["matched_pairs"]

        # Derive all other information from this single result
        assignments = [(i, j) for i, j, _ in matched_pairs]
        matched_gt_indices = {i for i, _, _ in matched_pairs}
        matched_pred_indices = {j for _, j, _ in matched_pairs}

        unmatched_gt_indices = [
            i for i in range(len(gt_list)) if i not in matched_gt_indices
        ]
        unmatched_pred_indices = [
            j for j in range(len(pred_list)) if j not in matched_pred_indices
        ]

        return {
            "matched_pairs": matched_pairs,
            "assignments": assignments,
            "unmatched_gt_indices": unmatched_gt_indices,
            "unmatched_pred_indices": unmatched_pred_indices,
            "matched_gt_indices": matched_gt_indices,
            "matched_pred_indices": matched_pred_indices,
        }
