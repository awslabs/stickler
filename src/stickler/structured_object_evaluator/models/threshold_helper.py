"""Threshold checking helper for StructuredModel comparisons."""

from typing import Any


class ThresholdHelper:
    """Helper class for consistent threshold checking with floating point precision handling."""

    @staticmethod
    def is_above_threshold(score: float, threshold: float) -> bool:
        """Check if a score is above threshold with floating point precision handling.

        Args:
            score: The similarity score to check
            threshold: The threshold value

        Returns:
            True if score is above or equal to threshold (considering floating point precision)
        """
        return score >= threshold or abs(score - threshold) < 1e-10

    @staticmethod
    def is_below_threshold(score: float, threshold: float) -> bool:
        """Check if a score is below threshold with floating point precision handling.

        Args:
            score: The similarity score to check
            threshold: The threshold value

        Returns:
            True if score is below threshold (considering floating point precision)
        """
        return score < threshold and abs(score - threshold) >= 1e-10

    @staticmethod
    def classify_match(score: float, threshold: float) -> str:
        """Classify a match based on threshold.

        Args:
            score: The similarity score
            threshold: The threshold value

        Returns:
            "TP" if above threshold, "FD" if below threshold
        """
        if ThresholdHelper.is_above_threshold(score, threshold):
            return "TP"
        else:
            return "FD"

    @staticmethod
    def get_match_threshold(obj: Any, default: float = 0.7) -> float:
        """Get the match threshold from an object or return default.

        Args:
            obj: Object to get threshold from (should have match_threshold attribute)
            default: Default threshold if object doesn't have match_threshold

        Returns:
            The match threshold value
        """
        if (
            obj
            and hasattr(obj, "__class__")
            and hasattr(obj.__class__, "match_threshold")
        ):
            return obj.__class__.match_threshold
        return default

    @staticmethod
    def apply_threshold_logic(matched_pairs, threshold: float) -> tuple[int, int]:
        """Apply threshold logic to matched pairs to get TP and FD counts.

        Args:
            matched_pairs: List of (gt_idx, pred_idx, similarity_score) tuples
            threshold: The threshold value

        Returns:
            Tuple of (tp_count, fd_count)
        """
        tp = 0
        fd = 0

        for i, j, score in matched_pairs:
            if ThresholdHelper.is_above_threshold(score, threshold):
                tp += 1
            else:
                fd += 1

        return tp, fd

    @staticmethod
    def format_threshold_reason(score: float, threshold: float) -> str:
        """Format a threshold-related reason string.

        Args:
            score: The similarity score
            threshold: The threshold value

        Returns:
            Formatted reason string
        """
        return f"below threshold ({score:.3f} < {threshold})"
