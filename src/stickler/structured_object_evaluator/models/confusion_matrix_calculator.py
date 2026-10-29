"""Confusion matrix calculation for structured model comparisons.

This module provides the ConfusionMatrixCalculator class for calculating
confusion matrix metrics (TP, FP, TN, FN, FD, FA) for field comparisons.
"""

from typing import TYPE_CHECKING, Any, Dict

from .field_helper import FieldHelper
from .null_helper import NullHelper
from .threshold_helper import ThresholdHelper

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class ConfusionMatrixCalculator:
    """Calculates confusion matrix metrics for field comparisons.

    This class is responsible for computing confusion matrix statistics
    (True Positives, False Positives, True Negatives, False Negatives,
    False Discoveries, False Alarms) for individual fields.
    """

    @staticmethod
    def classify_field_for_confusion_matrix(
        model: "StructuredModel",
        field_name: str,
        other_value: Any,
        threshold: float = None,
    ) -> Dict[str, int]:
        """Classify a field comparison according to confusion matrix rules.

        This method determines the confusion matrix classification for a single
        field comparison based on null states and similarity scores.

        Classification rules:
        - Both null: TN (True Negative)
        - GT null, pred non-null: FA (False Alarm)
        - GT non-null, pred null: FN (False Negative)
        - Both non-null and match: TP (True Positive)
        - Both non-null but don't match: FD (False Discovery)

        Args:
            model: The ground truth StructuredModel instance that owns the field
            field_name: Name of the field being compared
            other_value: Value to compare with
            threshold: Threshold for matching (uses field's threshold if None)

        Returns:
            Dictionary with the six counts: tp, fa, fd, fp, tn, fn

        Example:
            >>> result = ConfusionMatrixCalculator.classify_field_for_confusion_matrix(
            ...     gt_model, "name", "John"
            ... )
            >>> print(result["tp"], result["fd"])
        """
        # Get field values
        gt_value = getattr(model, field_name)
        pred_value = other_value

        # Get field configuration
        info = model.__class__._get_comparison_info(field_name)
        if threshold is None:
            threshold = info.threshold
        comparator = info.comparator

        # Determine if values are null
        gt_is_null = FieldHelper.is_null_value(gt_value)
        pred_is_null = FieldHelper.is_null_value(pred_value)

        # Calculate similarity if both aren't null
        similarity = None
        if not gt_is_null and not pred_is_null:
            # Import here to avoid circular imports
            from .structured_model import StructuredModel

            if isinstance(gt_value, StructuredModel) and isinstance(
                pred_value, StructuredModel
            ):
                comparison = gt_value.compare_with(pred_value)
                similarity = comparison["overall_score"]
            else:
                # A None element is absent (#386): an `Any` sub-field holding a
                # list reaches the comparator here, so drop absent elements as
                # the list comparators do.
                if isinstance(gt_value, list) and isinstance(pred_value, list):
                    gt_value = NullHelper.present(gt_value)
                    pred_value = NullHelper.present(pred_value)
                # Use the field's configured comparator for primitive comparison
                similarity = comparator.compare(gt_value, pred_value)
            values_match = ThresholdHelper.is_above_threshold(similarity, threshold)
        else:
            values_match = False

        # Apply confusion matrix classification
        if gt_is_null and pred_is_null:
            # TN: Both null
            result = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0}
        elif gt_is_null and not pred_is_null:
            # FA: GT null, prediction non-null (False Alarm)
            result = {"tp": 0, "fa": 1, "fd": 0, "fp": 1, "tn": 0, "fn": 0}
        elif not gt_is_null and pred_is_null:
            # FN: GT non-null, prediction null
            result = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 1}
        elif values_match:
            # TP: Both non-null and match
            result = {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
        else:
            # FD: Both non-null but don't match (False Discovery)
            result = {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0}

        return result
