"""Derived metrics calculator for confusion matrix analysis.

This module provides the DerivedMetricsCalculator class for calculating
derived metrics (precision, recall, F1, accuracy) from basic confusion matrix counts.
"""

from typing import Dict, Any


class DerivedMetricsCalculator:
    """Calculates derived metrics from basic confusion matrix counts.
    
    The derived metrics include precision, recall, F1 score, and accuracy,
    calculated from the basic confusion matrix counts (TP, FP, FN, TN, FD, FA).
    
    Architecture:
    -------------
    This class is part of the StructuredModel refactoring that extracts metrics
    calculation logic into dedicated helper classes. It works in conjunction with:
    
    - ConfusionMatrixCalculator: Calculates basic confusion matrix metrics
    - AggregateMetricsCalculator: Calculates aggregate metrics by rolling up children
    - MetricsHelper: Provides the actual calculation logic for derived metrics
    - ConfusionMatrixBuilder: Orchestrates all metrics calculation
    
    The calculator performs a recursive traversal of the comparison result tree,
    adding derived metrics at each level based on the confusion matrix counts.
    It delegates the actual metric calculations to MetricsHelper to avoid code
    duplication.
    
    Features:
    ---------
    - Recursive traversal of comparison result tree
    - Calculates precision, recall, F1, and accuracy via MetricsHelper
    - Supports both traditional and FD-inclusive recall formulas
    - Handles both 'overall' and 'aggregate' metrics
    - Preserves all existing result structure and metadata
    
    Formulas (implemented in MetricsHelper):
    -----------------------------------------
    - Precision: TP / (TP + FP) where FP = FA + FD
    - Recall (traditional): TP / (TP + FN)
    - Recall (with FD): TP / (TP + FN + FD)
    - F1: 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    
    Example:
    --------
    >>> calculator = DerivedMetricsCalculator()
    >>> result = {
    ...     "overall": {"tp": 8, "fp": 2, "fn": 1, "tn": 0, "fa": 1, "fd": 1},
    ...     "fields": {}
    ... }
    >>> result_with_derived = calculator.add_derived_metrics_to_result(result)
    >>> print(result_with_derived["derived"])
    {'cm_precision': 0.8, 'cm_recall': 0.888..., 'cm_f1': 0.842..., 'cm_accuracy': 0.727...}
    """
    
    def add_derived_metrics_to_result(
        self, 
        result: Dict[str, Any],
        recall_with_fd: bool = False
    ) -> Dict[str, Any]:
        """Walk through result and add 'derived' fields at each level.
        
        This method performs a recursive traversal of the comparison result tree,
        adding derived metrics at each level based on the confusion matrix counts.
        
        Derived metrics are calculated for:
        - 'overall' metrics (if present)
        - 'aggregate' metrics (if present)
        
        The derived metrics are added as a 'derived' field at each level, containing:
        - cm_precision: Precision score
        - cm_recall: Recall score (formula depends on recall_with_fd)
        - cm_f1: F1 score
        - cm_accuracy: Accuracy score
        
        Algorithm:
        ----------
        1. Recursively process all child fields first (depth-first traversal)
        2. Calculate derived metrics for 'overall' if present
        3. Calculate derived metrics for 'aggregate' if present
        4. Add 'derived' and 'aggregate_derived' fields at each level
        5. Preserve all existing keys and structure
        
        Args:
            result: Result from compare_recursive with confusion matrix metrics.
                   Expected structure:
                   {
                       "overall": {"tp": int, "fp": int, "fn": int, ...},
                       "aggregate": {"tp": int, "fp": int, "fn": int, ...},
                       "fields": {
                           "field_name": {
                               "overall": {...},
                               "fields": {...}
                           }
                       }
                   }
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                           If False, use traditional recall (TP/(TP+FN))
        
        Returns:
            Modified result with 'derived' and 'aggregate_derived' fields added.
            The derived field contains:
            {
                "cm_precision": float,  # TP / (TP + FP)
                "cm_recall": float,     # TP / (TP + FN) or TP / (TP + FN + FD)
                "cm_f1": float,         # 2 * (P * R) / (P + R)
                "cm_accuracy": float    # (TP + TN) / (TP + TN + FP + FN)
            }
        
        Notes:
        ------
        - The method does not modify the original result dictionary
        - Handles arbitrary nesting depth through recursion
        - Preserves all existing keys and structure
        - Works with both new hierarchical and legacy flat result formats
        - Gracefully handles missing metrics (returns 0.0 for derived metrics)
        
        Example:
        --------
        >>> calculator = DerivedMetricsCalculator()
        >>> result = {
        ...     "overall": {"tp": 8, "fp": 2, "fn": 1, "tn": 0, "fa": 1, "fd": 1},
        ...     "aggregate": {"tp": 8, "fp": 2, "fn": 1, "tn": 0, "fa": 1, "fd": 1},
        ...     "fields": {
        ...         "name": {
        ...             "overall": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "fa": 0, "fd": 0}
        ...         }
        ...     }
        ... }
        >>> result_with_derived = calculator.add_derived_metrics_to_result(result)
        >>> assert "derived" in result_with_derived
        >>> assert "aggregate_derived" in result_with_derived
        >>> assert "cm_precision" in result_with_derived["derived"]
        """
        from .metrics_helper import MetricsHelper
        
        if not isinstance(result, dict):
            return result

        # Make a copy to avoid modifying the original
        result_copy = result.copy()

        # Add derived metrics to 'overall' if it exists and has basic metrics
        if "overall" in result_copy and isinstance(result_copy["overall"], dict):
            overall = result_copy["overall"]
            if self._has_basic_metrics(overall):
                metrics_helper = MetricsHelper()
                overall["derived"] = metrics_helper.calculate_derived_metrics(
                    overall, recall_with_fd
                )

                # Also add derived metrics to aggregate if it exists
                if "aggregate" in overall and self._has_basic_metrics(
                    overall["aggregate"]
                ):
                    overall["aggregate"]["derived"] = (
                        metrics_helper.calculate_derived_metrics(
                            overall["aggregate"], recall_with_fd
                        )
                    )

        # Add derived metrics to top-level aggregate if it exists
        if "aggregate" in result_copy and self._has_basic_metrics(
            result_copy["aggregate"]
        ):
            metrics_helper = MetricsHelper()
            result_copy["aggregate"]["derived"] = (
                metrics_helper.calculate_derived_metrics(
                    result_copy["aggregate"], recall_with_fd
                )
            )

        # Recursively process 'fields' if it exists
        if "fields" in result_copy and isinstance(result_copy["fields"], dict):
            fields_copy = {}
            for field_name, field_result in result_copy["fields"].items():
                if isinstance(field_result, dict):
                    # Check if this is a hierarchical field (has overall/fields) or a unified structure field
                    if "overall" in field_result and "fields" in field_result:
                        # Hierarchical field - process recursively
                        fields_copy[field_name] = self.add_derived_metrics_to_result(
                            field_result, recall_with_fd
                        )
                    elif "overall" in field_result and self._has_basic_metrics(
                        field_result["overall"]
                    ):
                        # Unified structure field - add derived metrics to overall
                        field_copy = field_result.copy()
                        metrics_helper = MetricsHelper()
                        field_copy["overall"]["derived"] = (
                            metrics_helper.calculate_derived_metrics(
                                field_result["overall"], recall_with_fd
                            )
                        )

                        # Also add derived metrics to aggregate if it exists
                        if "aggregate" in field_copy and self._has_basic_metrics(
                            field_copy["aggregate"]
                        ):
                            field_copy["aggregate"]["derived"] = (
                                metrics_helper.calculate_derived_metrics(
                                    field_copy["aggregate"], recall_with_fd
                                )
                            )

                        fields_copy[field_name] = field_copy
                    elif self._has_basic_metrics(field_result):
                        # CRITICAL FIX: Legacy leaf field with basic metrics - wrap in "overall" structure
                        field_copy = field_result.copy()
                        metrics_helper = MetricsHelper()

                        # Extract basic metrics and wrap in "overall" structure
                        legacy_metrics = {}
                        for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                            if metric in field_copy:
                                legacy_metrics[metric] = field_copy[metric]
                                del field_copy[metric]  # Remove from top level

                        # Add derived metrics to the legacy metrics
                        legacy_metrics["derived"] = (
                            metrics_helper.calculate_derived_metrics(
                                legacy_metrics, recall_with_fd
                            )
                        )

                        # Wrap in "overall" structure
                        field_copy["overall"] = legacy_metrics

                        fields_copy[field_name] = field_copy
                    else:
                        # Other structure - keep as is
                        fields_copy[field_name] = field_result
                else:
                    # Non-dict field - keep as is
                    fields_copy[field_name] = field_result
            result_copy["fields"] = fields_copy

        return result_copy
    
    def _has_basic_metrics(self, metrics_dict: Dict[str, Any]) -> bool:
        """Check if a dictionary has basic confusion matrix metrics.
        
        Args:
            metrics_dict: Dictionary to check
        
        Returns:
            True if it has the basic metrics (tp, fp, fn, etc.)
        """
        basic_metrics = ["tp", "fp", "fn", "tn", "fa", "fd"]
        return all(metric in metrics_dict for metric in basic_metrics)
