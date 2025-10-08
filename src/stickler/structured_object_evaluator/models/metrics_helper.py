"""Metrics calculation helper for StructuredModel comparisons."""

from typing import Dict, Any
from collections import OrderedDict


class MetricsHelper:
    """Helper class for calculating and aggregating confusion matrix metrics."""

    def calculate_derived_metrics(
        self, metrics: Dict[str, int], recall_with_fd: bool = False
    ) -> Dict[str, float]:
        """Calculate derived metrics from confusion matrix counts.

        Args:
            metrics: Dictionary with TP, FP, TN, FN, FD counts
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                            If False, use traditional recall (TP/(TP+FN))

        Returns:
            Dictionary with precision, recall, F1, and accuracy
        """
        tp = metrics["tp"]
        fp = metrics["fp"]
        tn = metrics["tn"]
        fn = metrics["fn"]
        fd = metrics["fd"]
        fa = metrics["fa"]

        # Calculate precision: TP / (TP + FP) where FP includes both FA and FD
        # Note: fp field should already equal fa + fd from individual classifications
        total_fp = fa + fd  # Total False Positives = False Alarms + False Discoveries
        precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0

        # Calculate recall based on the selected formula
        if recall_with_fd:
            # Alternative recall: TP / (TP + FN + FD)
            recall = tp / (tp + fn + fd) if (tp + fn + fd) > 0 else 0.0
        else:
            # Traditional recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "cm_precision": precision,
            "cm_recall": recall,
            "cm_f1": f1,
            "cm_accuracy": accuracy,
        }

    def aggregate_field_to_overall(
        self, field_result: Any, overall_metrics: Dict[str, int]
    ) -> None:
        """Aggregate field results to overall metrics.

        Args:
            field_result: Result from processing a field (could be dict or OrderedDict)
            overall_metrics: Overall metrics to update
        """
        if isinstance(field_result, dict):
            if "overall" in field_result:
                # This is a hierarchical result, use its overall metrics
                source = field_result["overall"]
            else:
                # This is a leaf result, use it directly
                source = field_result

            # Aggregate basic confusion matrix metrics
            for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                overall_metrics[metric] += source.get(metric, 0)

    def calculate_recursive_aggregates(
        self, fields_dict: OrderedDict
    ) -> Dict[str, int]:
        """Calculate aggregate metrics by recursively traversing the hierarchy.

        Args:
            fields_dict: Dictionary of field results

        Returns:
            Dictionary with aggregate metrics
        """
        aggregate_metrics = {
            "aggregate_tp": 0,
            "aggregate_fa": 0,
            "aggregate_fd": 0,
            "aggregate_fp": 0,
            "aggregate_tn": 0,
            "aggregate_fn": 0,
        }

        def collect_all_leaf_metrics(node):
            """Recursively collect metrics from all leaf nodes."""
            collected = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}

            if isinstance(node, dict):
                if "fields" in node:
                    # This has sub-structure, recurse into fields
                    for field_key, field_value in node["fields"].items():
                        child_metrics = collect_all_leaf_metrics(field_value)
                        for metric in collected:
                            collected[metric] += child_metrics[metric]
                elif "overall" in node:
                    # This is a structured node, use its overall metrics
                    overall = node["overall"]
                    for metric in collected:
                        collected[metric] += overall.get(metric, 0)
                else:
                    # This is a leaf node with direct metrics
                    for metric in collected:
                        collected[metric] += node.get(metric, 0)

            return collected

        # Collect from all fields
        for field_name, field_data in fields_dict.items():
            field_leaf_metrics = collect_all_leaf_metrics(field_data)
            for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                aggregate_metrics[f"aggregate_{metric}"] += field_leaf_metrics[metric]

        return aggregate_metrics

    def convert_score_to_binary_metrics(
        self, score: float, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Convert similarity score to binary classification metrics.

        Args:
            score: Similarity score [0-1]
            threshold: Threshold for considering a match

        Returns:
            Dictionary with TP, FP, FN, TN counts converted to metrics
        """
        # For single field comparison: if score >= threshold, it's TP, otherwise FP/FN
        if score >= threshold:
            tp = score  # Proportional TP credit
            fp = 1 - score if score < 1.0 else 0  # Small FP for imperfect matches
            fn = 0
            tn = 0
        else:
            tp = 0
            fp = score  # Partial FP credit for some similarity
            fn = 1 - score  # Higher FN for very different values
            tn = 0

        # Calculate derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "anls_score": score,
        }

    def initialize_metrics_dict(self) -> Dict[str, int]:
        """Initialize a standard metrics dictionary.

        Returns:
            Dictionary with all metrics initialized to 0
        """
        return {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}

    def add_aggregate_derived_metrics(
        self,
        metrics: Dict[str, Any],
        aggregate_metrics: Dict[str, int],
        recall_with_fd: bool = False,
    ) -> None:
        """Add aggregate-derived metrics to a metrics dictionary.

        Args:
            metrics: Metrics dictionary to update
            aggregate_metrics: Aggregate metrics to derive from
            recall_with_fd: Whether to include FD in recall denominator
        """
        if aggregate_metrics:
            metrics["aggregate_derived"] = self.calculate_derived_metrics(
                {
                    "tp": aggregate_metrics.get("aggregate_tp", 0),
                    "fp": aggregate_metrics.get("aggregate_fp", 0),
                    "tn": aggregate_metrics.get("aggregate_tn", 0),
                    "fn": aggregate_metrics.get("aggregate_fn", 0),
                    "fd": aggregate_metrics.get("aggregate_fd", 0),
                    "fa": aggregate_metrics.get("aggregate_fa", 0),
                },
                recall_with_fd,
            )
