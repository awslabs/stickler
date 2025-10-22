"""Helper class for creating standard comparison result dictionaries.

This module provides utility methods for creating common result structures
used throughout the comparison process.
"""

from typing import Dict, Any


class ResultHelper:
    """Helper class for creating standard comparison result dictionaries."""

    @staticmethod
    def create_true_negative_result(weight: float) -> Dict[str, Any]:
        """Create a true negative result.

        Args:
            weight: Field weight for scoring

        Returns:
            True negative result dictionary
        """
        return {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
            "fields": {},
            "raw_similarity_score": 1.0,
            "similarity_score": 1.0,
            "threshold_applied_score": 1.0,
            "weight": weight,
        }

    @staticmethod
    def create_false_alarm_result(weight: float) -> Dict[str, Any]:
        """Create a false alarm result.

        Args:
            weight: Field weight for scoring

        Returns:
            False alarm result dictionary
        """
        return {
            "overall": {"tp": 0, "fa": 1, "fd": 0, "fp": 1, "tn": 0, "fn": 0},
            "fields": {},
            "raw_similarity_score": 0.0,
            "similarity_score": 0.0,
            "threshold_applied_score": 0.0,
            "weight": weight,
        }

    @staticmethod
    def create_false_negative_result(weight: float) -> Dict[str, Any]:
        """Create a false negative result.

        Args:
            weight: Field weight for scoring

        Returns:
            False negative result dictionary
        """
        return {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 1},
            "fields": {},
            "raw_similarity_score": 0.0,
            "similarity_score": 0.0,
            "threshold_applied_score": 0.0,
            "weight": weight,
        }

    @staticmethod
    def create_empty_list_result(
        gt_len: int, pred_len: int, weight: float
    ) -> Dict[str, Any]:
        """Create result for empty list cases using match statements.

        Args:
            gt_len: Length of ground truth list
            pred_len: Length of predicted list
            weight: Field weight for scoring

        Returns:
            Result dictionary if early exit needed, None if should continue processing
        """
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
            case (0, _):
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
            case (_, 0):
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
