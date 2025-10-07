"""Evaluator format helper for StructuredModel output formatting.

This module provides utilities for formatting comparison results for evaluator
compatibility and calculating list item metrics.
"""

from typing import Any, Dict, List
from .hungarian_helper import HungarianHelper
from .metrics_helper import MetricsHelper


class EvaluatorFormatHelper:
    """Helper class for StructuredModel evaluator formatting operations."""

    @staticmethod
    def format_for_evaluator(
        structured_model_instance,
        result: Dict[str, Any],
        other: Any,
        recall_with_fd: bool = False,
    ) -> Dict[str, Any]:
        """Format comparison results for evaluator compatibility.

        Args:
            structured_model_instance: StructuredModel instance
            result: Standard comparison result from compare_with
            other: The other model being compared
            recall_with_fd: Whether to include FD in recall denominator

        Returns:
            Dictionary in evaluator format with overall, fields, confusion_matrix
        """
        field_scores = result["field_scores"]
        overall_score = result["overall_score"]
        confusion_matrix = result.get("confusion_matrix", {})
        non_matches = result.get("non_matches", [])

        # If we have confusion matrix data, use its derived metrics for overall
        if (
            confusion_matrix
            and "overall" in confusion_matrix
            and "derived" in confusion_matrix["overall"]
        ):
            # Use the derived metrics from confusion matrix for overall
            cm_derived = confusion_matrix["overall"]["derived"]
            overall_metrics = {
                "precision": cm_derived["cm_precision"],
                "recall": cm_derived["cm_recall"],
                "f1": cm_derived["cm_f1"],
                "accuracy": cm_derived["cm_accuracy"],
                "anls_score": overall_score,
            }
        else:
            # Fallback to binary conversion if no confusion matrix
            metrics_helper = MetricsHelper()
            overall_metrics = metrics_helper.convert_score_to_binary_metrics(
                overall_score
            )

        # Calculate field metrics with proper nested structure for list fields
        field_metrics = {}

        # Determine which fields are list fields by checking the actual field types
        list_fields = set()
        for field_name in field_scores.keys():
            field_value = getattr(structured_model_instance, field_name)
            if isinstance(field_value, list):
                list_fields.add(field_name)

        for field_name, score in field_scores.items():
            if field_name in list_fields:
                # This is a list field - create nested structure expected by tests
                metrics_helper = MetricsHelper()
                overall_metrics_for_list = (
                    metrics_helper.convert_score_to_binary_metrics(score)
                )

                # Get individual item metrics by comparing list items
                items_metrics = EvaluatorFormatHelper.calculate_list_item_metrics(
                    field_name,
                    getattr(structured_model_instance, field_name),
                    getattr(other, field_name, []),
                    recall_with_fd,
                )

                field_metrics[field_name] = {
                    "overall": overall_metrics_for_list,
                    "items": items_metrics,
                }
            else:
                # Regular field
                metrics_helper = MetricsHelper()
                field_metrics[field_name] = (
                    metrics_helper.convert_score_to_binary_metrics(score)
                )

        # Flatten confusion matrix fields for evaluator compatibility
        # The evaluator expects direct access like cm["fields"]["name"]["tp"]
        # but my hierarchical structure has cm["fields"]["name"]["overall"]["tp"]
        flattened_confusion_matrix = (
            EvaluatorFormatHelper._flatten_confusion_matrix_for_evaluator(
                confusion_matrix
            )
        )

        return {
            "overall": overall_metrics,
            "fields": field_metrics,
            "confusion_matrix": flattened_confusion_matrix,
            "non_matches": non_matches,
        }

    @staticmethod
    def _flatten_confusion_matrix_for_evaluator(
        confusion_matrix: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Flatten hierarchical confusion matrix structure for evaluator compatibility.

        Converts:
            cm["fields"]["name"]["overall"]["tp"] -> cm["fields"]["name"]["tp"]

        Args:
            confusion_matrix: Hierarchical confusion matrix structure

        Returns:
            Flattened confusion matrix structure for evaluator
        """
        if not confusion_matrix or "fields" not in confusion_matrix:
            return confusion_matrix

        flattened_cm = confusion_matrix.copy()
        flattened_fields = {}

        for field_name, field_data in confusion_matrix["fields"].items():
            if isinstance(field_data, dict):
                if "overall" in field_data:
                    # Hierarchical structure - flatten it
                    flattened_field = field_data["overall"].copy()

                    # If there are nested fields, keep them as-is for nested access
                    if "fields" in field_data:
                        flattened_field["fields"] = field_data["fields"]

                    # Add other keys from the hierarchical structure
                    for key, value in field_data.items():
                        if key not in ["overall", "fields"]:
                            flattened_field[key] = value

                    flattened_fields[field_name] = flattened_field
                else:
                    # Already flat structure - keep as-is
                    flattened_fields[field_name] = field_data
            else:
                # Non-dict field data - keep as-is
                flattened_fields[field_name] = field_data

        flattened_cm["fields"] = flattened_fields
        return flattened_cm

    @staticmethod
    def calculate_list_item_metrics(
        field_name: str,
        gt_list: List[Any],
        pred_list: List[Any],
        recall_with_fd: bool = False,
    ) -> List[Dict[str, Any]]:
        """Calculate metrics for individual items in a list field.

        Args:
            field_name: Name of the list field
            gt_list: Ground truth list
            pred_list: Prediction list
            recall_with_fd: Whether to include FD in recall denominator

        Returns:
            List of metrics dictionaries for each matched item pair
        """
        items_metrics = []

        if not gt_list or not pred_list:
            return items_metrics

        # For StructuredModel lists, compare items individually
        # Import here to avoid circular import
        from .structured_model import StructuredModel

        if gt_list and isinstance(gt_list[0], StructuredModel):
            # Use HungarianHelper for Hungarian matching operations
            hungarian_helper = HungarianHelper()

            # Use HungarianHelper to get optimal assignments - OPTIMIZED: Single call gets all info
            hungarian_info = hungarian_helper.get_complete_matching_info(
                gt_list, pred_list
            )
            assignments = hungarian_info["assignments"]

            # Create metrics for each matched pair
            for gt_idx, pred_idx in assignments:
                if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                    gt_item = gt_list[gt_idx]
                    pred_item = pred_list[pred_idx]

                    # Compare the items with evaluator format
                    item_comparison = gt_item.compare_with(
                        pred_item, evaluator_format=True, recall_with_fd=recall_with_fd
                    )
                    items_metrics.append(item_comparison)

        return items_metrics
