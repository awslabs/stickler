"""Confusion matrix builder for orchestrating metrics calculation.

This module provides the ConfusionMatrixBuilder class that orchestrates
the calculation of complete confusion matrices with aggregate and derived metrics.
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class ConfusionMatrixBuilder:
    """Builds complete confusion matrices with aggregate and derived metrics.
    
    This class orchestrates the calculation of complete confusion matrices by
    coordinating between:
    - ConfusionMatrixCalculator: Calculates basic confusion matrix metrics
    - AggregateMetricsCalculator: Rolls up child metrics to parent nodes
    - DerivedMetricsCalculator: Calculates precision, recall, F1, accuracy
    
    The builder ensures that all metrics are calculated in the correct order
    and that the final confusion matrix contains all necessary information for
    analysis.
    
    Architecture:
    -------------
    This class is part of the StructuredModel refactoring that extracts metrics
    calculation logic into dedicated helper classes. It serves as the main
    orchestrator for metrics calculation, coordinating between the three
    calculator classes.
    
    The builder is used by ComparisonEngine to add confusion matrix metrics
    to comparison results when requested by the user.
    
    Calculation Order:
    ------------------
    1. Basic confusion matrix metrics (already in recursive_result from compare_recursive)
    2. Aggregate metrics (roll up child metrics to parent nodes)
    3. Derived metrics (calculate precision, recall, F1, accuracy)
    
    This order is important because:
    - Aggregate metrics depend on child metrics being available
    - Derived metrics depend on basic metrics (TP, FP, FN, etc.) being available
    
    Attributes:
        model: The ground truth StructuredModel instance used for comparison
        calculator: ConfusionMatrixCalculator for basic metrics
        aggregate_calculator: AggregateMetricsCalculator for rollup
        derived_calculator: DerivedMetricsCalculator for derived metrics
    
    Example:
    --------
    >>> builder = ConfusionMatrixBuilder(gt_model)
    >>> recursive_result = engine.compare_recursive(pred_model)
    >>> confusion_matrix = builder.build_confusion_matrix(
    ...     recursive_result,
    ...     add_derived_metrics=True
    ... )
    >>> print(confusion_matrix["aggregate"]["derived"]["cm_f1"])
    """

    def __init__(self, model: "StructuredModel"):
        """Initialize builder with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model
        
        # Initialize calculators lazily to avoid circular imports
        self._calculator = None
        self._aggregate_calculator = None
        self._derived_calculator = None

    @property
    def calculator(self):
        """Lazy initialization of ConfusionMatrixCalculator."""
        if self._calculator is None:
            from .confusion_matrix_calculator import ConfusionMatrixCalculator
            self._calculator = ConfusionMatrixCalculator(self.model)
        return self._calculator

    @property
    def aggregate_calculator(self):
        """Lazy initialization of AggregateMetricsCalculator."""
        if self._aggregate_calculator is None:
            from .aggregate_metrics_calculator import AggregateMetricsCalculator
            self._aggregate_calculator = AggregateMetricsCalculator()
        return self._aggregate_calculator

    @property
    def derived_calculator(self):
        """Lazy initialization of DerivedMetricsCalculator."""
        if self._derived_calculator is None:
            from .derived_metrics_calculator import DerivedMetricsCalculator
            self._derived_calculator = DerivedMetricsCalculator()
        return self._derived_calculator

    def build_confusion_matrix(
        self,
        recursive_result: Dict[str, Any],
        add_derived_metrics: bool = True,
        recall_with_fd: bool = False
    ) -> Dict[str, Any]:
        """Build complete confusion matrix from recursive result.
        
        This method orchestrates the calculation of a complete confusion matrix
        by coordinating between the three calculator classes:
        
        1. Basic confusion matrix metrics are already in recursive_result
           (calculated during compare_recursive)
        2. Add aggregate metrics by rolling up child metrics to parent nodes
        3. Add derived metrics (precision, recall, F1, accuracy) if requested
        
        The method ensures that all metrics are calculated in the correct order
        and that the final confusion matrix contains all necessary information.
        
        Args:
            recursive_result: Result from compare_recursive with basic metrics.
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
            add_derived_metrics: Whether to add derived metrics (precision, recall, F1, accuracy)
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                           If False, use traditional recall (TP/(TP+FN))
        
        Returns:
            Complete confusion matrix with aggregate and derived metrics:
            {
                "overall": {
                    "tp": int,
                    "fa": int,
                    "fd": int,
                    "fp": int,
                    "tn": int,
                    "fn": int,
                    "similarity_score": float,
                    "all_fields_matched": bool,
                    "derived": {  # If add_derived_metrics=True
                        "cm_precision": float,
                        "cm_recall": float,
                        "cm_f1": float,
                        "cm_accuracy": float
                    }
                },
                "fields": {
                    "field_name": {
                        "overall": {...},
                        "fields": {...},
                        "aggregate": {...}  # Rolled up child metrics
                    }
                },
                "aggregate": {  # Top-level aggregate metrics
                    "tp": int,
                    "fa": int,
                    "fd": int,
                    "fp": int,
                    "tn": int,
                    "fn": int,
                    "derived": {  # If add_derived_metrics=True
                        "cm_precision": float,
                        "cm_recall": float,
                        "cm_f1": float,
                        "cm_accuracy": float
                    }
                }
            }
        
        Notes:
        ------
        - The method does not modify the original recursive_result
        - Handles arbitrary nesting depth through recursive calculators
        - Preserves all existing keys and structure from recursive_result
        - Works with both new hierarchical and legacy flat result formats
        
        Example:
        --------
        >>> builder = ConfusionMatrixBuilder(gt_model)
        >>> recursive_result = engine.compare_recursive(pred_model)
        >>> confusion_matrix = builder.build_confusion_matrix(
        ...     recursive_result,
        ...     add_derived_metrics=True,
        ...     recall_with_fd=False
        ... )
        >>> # Access top-level aggregate metrics
        >>> print(confusion_matrix["aggregate"]["tp"])
        >>> # Access derived metrics
        >>> print(confusion_matrix["aggregate"]["derived"]["cm_f1"])
        >>> # Access field-level metrics
        >>> print(confusion_matrix["fields"]["name"]["overall"]["tp"])
        """
        # Start with the recursive result (already has basic confusion matrix metrics)
        confusion_matrix = recursive_result

        # Step 1: Add universal aggregate metrics to all nodes
        # This rolls up child metrics to parent nodes
        confusion_matrix = self.aggregate_calculator.calculate_aggregate_metrics(
            confusion_matrix
        )

        # Step 2: Add derived metrics if requested
        # This calculates precision, recall, F1, accuracy from basic metrics
        if add_derived_metrics:
            confusion_matrix = self.derived_calculator.add_derived_metrics_to_result(
                confusion_matrix,
                recall_with_fd
            )

        return confusion_matrix
