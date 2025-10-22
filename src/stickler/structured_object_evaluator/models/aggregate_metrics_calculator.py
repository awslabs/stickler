"""Aggregate metrics calculator for confusion matrix rollup.

This module provides the AggregateMetricsCalculator class for calculating
aggregate confusion matrix metrics by rolling up child field metrics to parent nodes.
"""

from typing import Dict, Any


class AggregateMetricsCalculator:
    """Calculates aggregate metrics by rolling up child field metrics.
    
    The aggregate field contains the sum of all primitive field confusion matrices
    below that node in the tree. This provides universal field-level granularity
    for analyzing model comparison results.
    
    Architecture:
    -------------
    This class is part of the StructuredModel refactoring that extracts metrics
    calculation logic into dedicated helper classes. It works in conjunction with:
    
    - ConfusionMatrixCalculator: Calculates basic confusion matrix metrics
    - DerivedMetricsCalculator: Calculates derived metrics (F1, precision, recall)
    - ConfusionMatrixBuilder: Orchestrates all metrics calculation
    
    The calculator performs a recursive traversal of the comparison result tree,
    calculating aggregate metrics at each level by summing child field metrics.
    
    Features:
    ---------
    - Recursive traversal of comparison result tree
    - Handles arbitrary nesting depth
    - Sums child aggregate metrics to parent nodes
    - Handles both hierarchical and legacy flat structures
    - Preserves all existing result structure and metadata
    
    Example:
    --------
    >>> calculator = AggregateMetricsCalculator()
    >>> result = {
    ...     "overall": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
    ...     "fields": {
    ...         "name": {"overall": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "fa": 0, "fd": 0}}
    ...     }
    ... }
    >>> result_with_aggregate = calculator.calculate_aggregate_metrics(result)
    >>> print(result_with_aggregate["aggregate"])
    {'tp': 1, 'fa': 0, 'fd': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    """
    
    def calculate_aggregate_metrics(self, result: dict) -> dict:
        """Calculate aggregate metrics for all nodes in the result tree.
        
        This method performs a recursive traversal of the comparison result tree,
        calculating aggregate metrics at each level by summing child field metrics.
        
        The aggregate field is added as a sibling to 'overall' and 'fields' at each
        level, containing the sum of all primitive field confusion matrices below
        that node in the tree.
        
        Algorithm:
        ----------
        1. Recursively process all child fields first (depth-first traversal)
        2. Sum child aggregate metrics to calculate parent aggregate
        3. For leaf nodes, use overall metrics as aggregate
        4. Handle both hierarchical (with 'overall') and legacy flat structures
        5. Add aggregate field at each level
        
        Args:
            result: Result from compare_recursive with hierarchical structure.
                   Expected structure:
                   {
                       "overall": {"tp": int, "fp": int, ...},
                       "fields": {
                           "field_name": {
                               "overall": {...},
                               "fields": {...}
                           }
                       }
                   }
        
        Returns:
            Modified result with 'aggregate' fields added at each level.
            The aggregate field contains:
            {
                "tp": int,  # Sum of all child TP
                "fa": int,  # Sum of all child FA
                "fd": int,  # Sum of all child FD
                "fp": int,  # Sum of all child FP
                "tn": int,  # Sum of all child TN
                "fn": int   # Sum of all child FN
            }
        
        Notes:
        ------
        - The method does not modify the original result dictionary
        - Handles arbitrary nesting depth through recursion
        - Preserves all existing keys and structure
        - Works with both new hierarchical and legacy flat result formats
        
        Example:
        --------
        >>> calculator = AggregateMetricsCalculator()
        >>> result = {
        ...     "overall": {"tp": 2, "fp": 1, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
        ...     "fields": {
        ...         "name": {
        ...             "overall": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
        ...             "fields": {}
        ...         },
        ...         "age": {
        ...             "overall": {"tp": 1, "fp": 1, "fn": 0, "tn": 0, "fa": 0, "fd": 0},
        ...             "fields": {}
        ...         }
        ...     }
        ... }
        >>> result_with_aggregate = calculator.calculate_aggregate_metrics(result)
        >>> # Parent aggregate is sum of child aggregates
        >>> assert result_with_aggregate["aggregate"]["tp"] == 2
        >>> assert result_with_aggregate["aggregate"]["fp"] == 1
        """
        if not isinstance(result, dict):
            return result

        # Make a copy to avoid modifying the original
        result_copy = result.copy()

        # Calculate aggregate for this node
        aggregate_metrics = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}

        # Recursively process 'fields' first to get child aggregates
        if "fields" in result_copy and isinstance(result_copy["fields"], dict):
            fields_copy = {}
            for field_name, field_result in result_copy["fields"].items():
                if isinstance(field_result, dict):
                    # Recursively calculate aggregate for child field
                    processed_field = self.calculate_aggregate_metrics(field_result)
                    fields_copy[field_name] = processed_field

                    # CRITICAL FIX: Sum child's aggregate metrics to parent
                    if "aggregate" in processed_field and self._has_basic_metrics(
                        processed_field["aggregate"]
                    ):
                        child_aggregate = processed_field["aggregate"]
                        for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                            aggregate_metrics[metric] += child_aggregate.get(metric, 0)
                else:
                    # Non-dict field - keep as is
                    fields_copy[field_name] = field_result
            result_copy["fields"] = fields_copy

        # CRITICAL FIX: Enhanced leaf node detection for deep nesting
        # Handle both empty fields dict and missing fields key as leaf indicators
        is_leaf_node = (
            "fields" not in result_copy
            or not result_copy["fields"]
            or (
                isinstance(result_copy["fields"], dict)
                and len(result_copy["fields"]) == 0
            )
        )

        if is_leaf_node:
            # Check if this is a leaf node with basic metrics (either in "overall" or directly)
            if "overall" in result_copy and self._has_basic_metrics(
                result_copy["overall"]
            ):
                # Hierarchical leaf node: aggregate = overall metrics
                overall = result_copy["overall"]
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    aggregate_metrics[metric] = overall.get(metric, 0)
            elif self._has_basic_metrics(result_copy):
                # CRITICAL FIX: Legacy primitive leaf node - wrap in "overall" structure
                # This preserves Universal Aggregate Field structure compliance
                legacy_metrics = {}
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    legacy_metrics[metric] = result_copy.get(metric, 0)
                    aggregate_metrics[metric] = result_copy.get(metric, 0)

                # Wrap legacy structure in "overall" key to maintain consistency
                if not "overall" in result_copy:
                    # Move all basic metrics to "overall" key
                    result_copy["overall"] = legacy_metrics
                    # Remove basic metrics from top level to avoid duplication
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        if metric in result_copy:
                            del result_copy[metric]
                    # Preserve other keys like derived, raw_similarity_score, etc.

        # CRITICAL FIX: Always sum child field metrics if no child aggregates were found
        # This handles the deep nesting case where leaf nodes have overall metrics but empty fields
        if (
            aggregate_metrics["tp"] == 0
            and aggregate_metrics["fa"] == 0
            and aggregate_metrics["fd"] == 0
            and aggregate_metrics["fp"] == 0
            and aggregate_metrics["tn"] == 0
            and aggregate_metrics["fn"] == 0
        ):
            # Check if we have fields with overall metrics that we can sum
            if "fields" in result_copy and isinstance(result_copy["fields"], dict):
                for field_name, field_result in result_copy["fields"].items():
                    if isinstance(field_result, dict):
                        # ENHANCED: Check for both direct metrics and overall metrics
                        if "overall" in field_result and self._has_basic_metrics(
                            field_result["overall"]
                        ):
                            field_overall = field_result["overall"]
                            for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                                aggregate_metrics[metric] += field_overall.get(
                                    metric, 0
                                )
                        elif self._has_basic_metrics(field_result):
                            # Direct metrics (legacy format)
                            for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                                aggregate_metrics[metric] += field_result.get(metric, 0)

        # Add aggregate as a sibling of 'overall' and 'fields'
        result_copy["aggregate"] = aggregate_metrics

        return result_copy
    
    def _has_basic_metrics(self, metrics_dict: dict) -> bool:
        """Check if a dictionary has basic confusion matrix metrics.

        Args:
            metrics_dict: Dictionary to check

        Returns:
            True if it has the basic metrics (tp, fp, fn, etc.)
        """
        basic_metrics = ["tp", "fp", "fn", "tn", "fa", "fd"]
        return all(metric in metrics_dict for metric in basic_metrics)
