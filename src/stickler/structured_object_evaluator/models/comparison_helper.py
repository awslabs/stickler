"""Comparison helper for StructuredModel field comparison operations.

This module provides utilities for comparing fields, lists, and nested structures
within StructuredModel instances.
"""

from typing import Any, Dict, List
from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator

from .hungarian_helper import HungarianHelper
from .threshold_helper import ThresholdHelper


class ComparisonHelper:
    """Helper class for StructuredModel field comparison operations."""

    @staticmethod
    def compare_unordered_lists(
        list1: List[Any], list2: List[Any], comparator: BaseComparator, threshold: float
    ) -> Dict[str, Any]:
        """Compare two lists as unordered collections using Hungarian matching.

        Args:
            list1: First list
            list2: Second list
            comparator: Comparator to use for item comparison
            threshold: Minimum score to consider a match

        Returns:
            Dictionary with confusion matrix metrics including:
            - tp: True positives (matches >= threshold)
            - fd: False discoveries (matches < threshold)
            - fa: False alarms (unmatched prediction items)
            - fn: False negatives (unmatched ground truth items)
            - fp: Total false positives (fd + fa)
            - overall_score: Similarity score for backward compatibility
        """
        # Handle empty lists
        if not list1 and not list2:
            return {"tp": 0, "fd": 0, "fa": 0, "fn": 0, "fp": 0, "overall_score": 1.0}

        if not list1:
            return {
                "tp": 0,
                "fd": 0,
                "fa": len(list2),
                "fn": 0,
                "fp": len(list2),
                "overall_score": 0.0,
            }

        if not list2:
            return {
                "tp": 0,
                "fd": 0,
                "fa": 0,
                "fn": len(list1),
                "fp": 0,
                "overall_score": 0.0,
            }

        # Use HungarianHelper for Hungarian matching operations
        hungarian_helper = HungarianHelper()

        # Use the appropriate comparator based on item types
        # Import here to avoid circular import
        from .structured_model import StructuredModel

        if all(isinstance(item, StructuredModel) for item in list1[:1]) and all(
            isinstance(item, StructuredModel) for item in list2[:1]
        ):
            # For StructuredModel lists, we need to use individual comparison scoring for consistency
            # Use HungarianHelper to get optimal pairings - OPTIMIZED: Single call gets all info
            hungarian_info = hungarian_helper.get_complete_matching_info(list1, list2)
            matched_pairs = hungarian_info["matched_pairs"]

            # CRITICAL FIX: Replace raw scores with threshold-applied scores from individual comparison
            # This ensures consistency between individual and list comparison results
            threshold_corrected_pairs = []
            for gt_idx, pred_idx, raw_score in matched_pairs:
                if gt_idx < len(list1) and pred_idx < len(list2):
                    gt_item = list1[gt_idx]
                    pred_item = list2[pred_idx]

                    # Use individual comparison with threshold application (same as .compare_with())
                    individual_result = gt_item.compare_with(pred_item)
                    threshold_applied_score = individual_result["overall_score"]

                    threshold_corrected_pairs.append(
                        (gt_idx, pred_idx, threshold_applied_score)
                    )
                else:
                    threshold_corrected_pairs.append((gt_idx, pred_idx, raw_score))

            # Replace matched_pairs with threshold-corrected version
            matched_pairs = threshold_corrected_pairs

            # Use a very low threshold since we've already applied thresholds in individual comparison
            classification_threshold = (
                0.01  # Almost everything that's not 0.0 should be TP
            )
        else:
            # Use the provided comparator for other types
            from stickler.algorithms.hungarian import HungarianMatcher

            # CRITICAL FIX: Use match_threshold=0.0 to capture ALL matches, not just those above threshold
            # This allows us to keep track of partial matches for scoring
            hungarian = HungarianMatcher(comparator, match_threshold=0.0)
            classification_threshold = threshold

            # Get detailed metrics from HungarianMatcher
            metrics = hungarian.calculate_metrics(list1, list2)
            matched_pairs = metrics["matched_pairs"]

        # Apply threshold logic to classify matches
        tp = 0  # True positives (score >= threshold)
        fd = 0  # False discoveries (score < threshold, including 0)

        for i, j, score in matched_pairs:
            # Use ThresholdHelper for consistent threshold checking
            if ThresholdHelper.is_above_threshold(score, classification_threshold):
                tp += 1
            else:
                # All matches below threshold are False Discoveries, including 0.0 scores
                fd += 1

        # False alarms are unmatched prediction items
        fa = len(list2) - len(matched_pairs)

        # False negatives are unmatched ground truth items
        fn = len(list1) - len(matched_pairs)

        # Total false positives include both false discoveries and false alarms
        fp = fd + fa

        # CRITICAL FIX: Use threshold-applied scores for consistency with individual comparison
        # This ensures list comparison matches the same scoring logic as individual comparison
        if not matched_pairs:
            overall_score = 0.0
        else:
            # Apply threshold to each similarity score (same logic as individual comparison)
            threshold_applied_similarities = []
            for _, _, score in matched_pairs:
                # Use ThresholdHelper for consistent threshold checking
                if ThresholdHelper.is_above_threshold(score, classification_threshold):
                    threshold_applied_similarities.append(score)
                else:
                    # Below threshold gets 0.0 (same as individual comparison clipping)
                    threshold_applied_similarities.append(0.0)

            # Average the threshold-applied similarities
            avg_threshold_similarity = sum(threshold_applied_similarities) / len(
                threshold_applied_similarities
            )

            # Scale by coverage ratio (matched pairs / max list size)
            max_items = max(len(list1), len(list2))
            coverage_ratio = len(matched_pairs) / max_items if max_items > 0 else 1.0
            overall_score = avg_threshold_similarity * coverage_ratio

        return {
            "tp": tp,
            "fd": fd,
            "fa": fa,
            "fn": fn,
            "fp": fp,
            "overall_score": overall_score,
        }

    @staticmethod
    def compare_field_with_threshold(
        structured_model_instance, field_name: str, other_value: Any
    ) -> float:
        """Compare a single field with a value using the configured comparator.

        Args:
            structured_model_instance: StructuredModel instance
            field_name: Name of the field to compare
            other_value: Value to compare with

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Import here to avoid circular import
        from .configuration_helper import ConfigurationHelper

        info = ConfigurationHelper.get_comparison_info(
            structured_model_instance.__class__, field_name
        )

        # We should always get a ComparableField object now
        comparator = info.comparator
        threshold = info.threshold

        # Get field value from self
        self_value = getattr(structured_model_instance, field_name)

        # Handle None values
        if self_value is None or other_value is None:
            return 1.0 if self_value == other_value else 0.0

        # Handle lists with special processing
        if isinstance(self_value, list) and isinstance(other_value, list):
            # Use Hungarian matching for any type of list items, including primitive types
            # Pass the field comparator
            result = ComparisonHelper.compare_unordered_lists(
                self_value, other_value, comparator, threshold
            )
            # Return the overall score for backward compatibility
            return result["overall_score"]

        # Handle nested StructuredModel objects - use recursive comparison
        from .structured_model import StructuredModel

        if isinstance(self_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            return self_value.compare(other_value)

        # Handle dictionary objects using the field's configured comparator
        if isinstance(self_value, dict) and isinstance(other_value, dict):
            # TODO: Fix Dictionary Comparison Architecture Issue
            # Prevent Dict[str, Any] + LevenshteinComparator anti-pattern
            # This defeats the purpose of structured comparison by converting dictionaries to strings
            if isinstance(comparator, LevenshteinComparator):
                raise TypeError(
                    f"Dictionary comparison with LevenshteinComparator is not supported. "
                    f"Dictionary fields should either: "
                    f"1) Use ANLS_star comparison method, or "
                    f"2) Be decomposed into proper StructuredModel subclasses. "
                    f"Field '{field_name}' contains dict data that would be incorrectly compared as strings."
                )
            return comparator.compare(self_value, other_value)

        # Use the comparator to calculate similarity
        similarity = comparator.compare(self_value, other_value)

        # Apply threshold
        return 0.0 if similarity < threshold else similarity

    @staticmethod
    def compare_field_raw(
        structured_model_instance, field_name: str, other_value: Any
    ) -> float:
        """Compare a single field with a value WITHOUT applying thresholds.

        This version is used by the compare method to get raw similarity scores.

        Args:
            structured_model_instance: StructuredModel instance
            field_name: Name of the field to compare
            other_value: Value to compare with

        Returns:
            Raw similarity score between 0.0 and 1.0 without threshold filtering
        """
        # Import here to avoid circular import
        from .configuration_helper import ConfigurationHelper

        info = ConfigurationHelper.get_comparison_info(
            structured_model_instance.__class__, field_name
        )

        # We should always get a ComparableField object now
        comparator = info.comparator

        # Get field value from self
        self_value = getattr(structured_model_instance, field_name)

        # Handle None values
        if self_value is None or other_value is None:
            return 1.0 if self_value == other_value else 0.0

        # Handle lists with special processing
        if isinstance(self_value, list) and isinstance(other_value, list):
            threshold = 0.0  # Use zero threshold for raw comparisons
            result = ComparisonHelper.compare_unordered_lists(
                self_value, other_value, comparator, threshold
            )
            return result["overall_score"]

        # Handle nested StructuredModel objects
        from .structured_model import StructuredModel

        if isinstance(self_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            return self_value.compare(other_value)

        # Handle dictionary objects using the field's configured comparator
        if isinstance(self_value, dict) and isinstance(other_value, dict):
            return comparator.compare(self_value, other_value)

        # Use the comparator to calculate raw similarity (no threshold)
        return comparator.compare(self_value, other_value)
