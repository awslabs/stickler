"""Evaluator format helper for StructuredModel output formatting.

This module provides utilities for formatting comparison results for evaluator
compatibility and calculating list item metrics.
"""

from typing import Any, Dict, List, Optional

from .hungarian_helper import HungarianHelper
from .metrics_helper import MetricsHelper
from .null_helper import NullHelper


class EvaluatorFormatHelper:
    """Helper class for StructuredModel evaluator formatting operations."""

    @staticmethod
    def format_for_evaluator(
        structured_model_instance,
        result: Dict[str, Any],
        other: Any,
        recall_with_fd: bool = False,
        recursive_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format comparison results for evaluator compatibility.

        Args:
            structured_model_instance: StructuredModel instance
            result: Standard comparison result from compare_with
            other: The other model being compared
            recall_with_fd: Whether to include FD in recall denominator
            recursive_result: The ``compare_recursive`` result ``result`` was
                built from. Its confusion counts give ``overall`` and each list
                field's ``overall`` their precision and recall. Without it they
                fall back to a conversion of the score, which cannot tell a
                missed value from a spurious one.

        Returns:
            Dictionary in evaluator format with overall, fields,
            confusion_matrix, and ``non_matches`` when ``result`` carries
            records (``document_non_matches=True``)
        """
        field_scores = result["field_scores"]
        overall_score = result["overall_score"]
        confusion_matrix = result.get("confusion_matrix", {})
        counts_by_field = (recursive_result or {}).get("fields", {})

        overall_counts = (recursive_result or {}).get("overall")

        # If we have confusion matrix data, use its derived metrics for overall
        if (
            confusion_matrix
            and "overall" in confusion_matrix
            and "derived" in confusion_matrix["overall"]
        ):
            overall_metrics = EvaluatorFormatHelper._metrics_from_derived(
                confusion_matrix["overall"]["derived"], overall_score
            )
        elif EvaluatorFormatHelper._has_decisions(overall_counts):
            # The same counts, read before the confusion matrix is built, so a
            # node with a TP, FP or FN gets the same precision and recall with
            # or without include_confusion_matrix. A TN-only node does not
            # (#434).
            overall_metrics = EvaluatorFormatHelper._metrics_from_counts(
                overall_counts, overall_score, recall_with_fd
            )
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
                list_counts = counts_by_field.get(field_name, {}).get("overall")
                if EvaluatorFormatHelper._has_decisions(list_counts):
                    # Count-based, from the list's own elements. A score
                    # conversion read [A, B] vs [A] (one TP, one FN) as
                    # precision 0.5 and recall 1.0, the wrong way round.
                    overall_metrics_for_list = (
                        EvaluatorFormatHelper._metrics_from_counts(
                            list_counts, score, recall_with_fd
                        )
                    )
                else:
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

        formatted = {
            "overall": overall_metrics,
            "fields": field_metrics,
            "confusion_matrix": flattened_confusion_matrix,
        }
        # Only when records were collected, as in the default compare_with()
        # dict. An always-present empty list read as "no non-matches" on a
        # document with a wrong field (#413).
        if "non_matches" in result:
            formatted["non_matches"] = result["non_matches"]
        return formatted

    @staticmethod
    def _has_decisions(counts: Any) -> bool:
        """Whether a node has a TP, FP or FN to compute precision and recall from.

        A node that is only true negatives (both sides empty) has neither, and
        keeps the score conversion it always had rather than reading 0.0.
        """
        return isinstance(counts, dict) and any(
            counts.get(k, 0) for k in ("tp", "fa", "fd", "fn")
        )

    @staticmethod
    def _metrics_from_counts(
        counts: Dict[str, Any], score: float, recall_with_fd: bool = False
    ) -> Dict[str, float]:
        """Evaluator-format metrics for one node, from its confusion counts.

        Precision is ``tp / (tp + fp)`` and recall ``tp / (tp + fn)``
        (``tp / (tp + fn + fd)`` with ``recall_with_fd``), as in the confusion
        matrix's own ``derived`` block. ``anls_score`` stays the node's score.
        """
        derived = MetricsHelper().calculate_derived_metrics(
            {k: counts.get(k, 0) for k in ("tp", "fa", "fd", "fp", "tn", "fn")},
            recall_with_fd,
        )
        return EvaluatorFormatHelper._metrics_from_derived(derived, score)

    @staticmethod
    def _metrics_from_derived(
        derived: Dict[str, Any], score: float
    ) -> Dict[str, float]:
        """Evaluator-format metrics from a ``derived`` block and the node's score."""
        return {
            "precision": derived["cm_precision"],
            "recall": derived["cm_recall"],
            "f1": derived["cm_f1"],
            "accuracy": derived["cm_accuracy"],
            "anls_score": score,
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

        if isinstance(NullHelper.first_present(gt_list, pred_list), StructuredModel):
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
