"""Comparison engine for orchestrating StructuredModel comparisons.

This module provides the ComparisonEngine class that orchestrates the overall
comparison process for StructuredModel instances, coordinating between the
dispatcher, collectors, and calculators.
"""

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class ComparisonEngine:
    """Orchestrates the comparison process for StructuredModel instances.
    
    This class is the main orchestrator for comparing two StructuredModel instances.
    It coordinates between:
    - ComparisonDispatcher: Routes field comparisons to appropriate handlers
    - NonMatchCollector: Collects and documents non-matching fields
    - ConfusionMatrixCalculator: Calculates confusion matrix metrics
    
    The engine implements a single-traversal optimization where all comparison
    data (scores, metrics, non-matches) is collected in one pass through the
    model structure.
    
    Attributes:
        model: The ground truth StructuredModel instance used for comparison
        dispatcher: ComparisonDispatcher for routing field comparisons
        non_match_collector: NonMatchCollector for documenting non-matches
        confusion_matrix_calculator: ConfusionMatrixCalculator for metrics
    """

    def __init__(self, model: "StructuredModel"):
        """Initialize engine with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model
        
        # Initialize components lazily to avoid circular imports
        self._dispatcher = None
        self._non_match_collector = None
        self._confusion_matrix_calculator = None
        self._confusion_matrix_builder = None

    @property
    def dispatcher(self):
        """Lazy initialization of ComparisonDispatcher."""
        if self._dispatcher is None:
            from .comparison_dispatcher import ComparisonDispatcher
            self._dispatcher = ComparisonDispatcher(self.model)
        return self._dispatcher

    @property
    def non_match_collector(self):
        """Lazy initialization of NonMatchCollector."""
        if self._non_match_collector is None:
            from .non_match_collector import NonMatchCollector
            self._non_match_collector = NonMatchCollector(self.model)
        return self._non_match_collector

    @property
    def confusion_matrix_calculator(self):
        """Lazy initialization of ConfusionMatrixCalculator."""
        if self._confusion_matrix_calculator is None:
            from .confusion_matrix_calculator import ConfusionMatrixCalculator
            self._confusion_matrix_calculator = ConfusionMatrixCalculator(self.model)
        return self._confusion_matrix_calculator

    @property
    def confusion_matrix_builder(self):
        """Lazy initialization of ConfusionMatrixBuilder."""
        if self._confusion_matrix_builder is None:
            from .confusion_matrix_builder import ConfusionMatrixBuilder
            self._confusion_matrix_builder = ConfusionMatrixBuilder(self.model)
        return self._confusion_matrix_builder

    def compare_recursive(self, other: "StructuredModel") -> Dict[str, Any]:
        """The core recursive comparison function.
        
        This method performs a single-traversal comparison of two StructuredModel
        instances, collecting both confusion matrix metrics and similarity scores
        in one pass through the model structure.
        
        The comparison process:
        1. Iterates through all fields in the ground truth model
        2. Dispatches each field comparison to the appropriate handler
        3. Aggregates metrics and scores from field comparisons
        4. Handles hallucinated fields (extra fields) as False Alarms
        5. Calculates overall similarity score and match status
        
        Args:
            other: Another instance of the same model to compare with
            
        Returns:
            Dictionary with hierarchical comparison results:
            {
                "overall": {
                    "tp": int,
                    "fa": int,
                    "fd": int,
                    "fp": int,
                    "tn": int,
                    "fn": int,
                    "similarity_score": float,
                    "all_fields_matched": bool
                },
                "fields": {
                    "field_name": {
                        "overall": {...},
                        "fields": {...},  # For nested structures
                        "raw_similarity_score": float,
                        "similarity_score": float,
                        "threshold_applied_score": float,
                        "weight": float
                    }
                },
                "non_matches": []  # Populated by collectors if requested
            }
            
        Example:
            >>> engine = ComparisonEngine(gt_model)
            >>> result = engine.compare_recursive(pred_model)
            >>> print(result["overall"]["similarity_score"])
        """
        result = {
            "overall": {
                "tp": 0,
                "fa": 0,
                "fd": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "similarity_score": 0.0,
                "all_fields_matched": False,
            },
            "fields": {},
            "non_matches": [],
        }

        # Score percolation variables
        total_score = 0.0
        total_weight = 0.0
        threshold_matched_fields = set()

        for field_name in self.model.__class__.model_fields:
            if field_name == "extra_fields":
                continue

            gt_val = getattr(self.model, field_name)
            pred_val = getattr(other, field_name, None)

            # Enhanced dispatch returns both metrics AND scores
            field_result = self.dispatcher.dispatch_field_comparison(field_name, gt_val, pred_val)

            result["fields"][field_name] = field_result

            # Simple aggregation to overall metrics
            self._aggregate_to_overall(field_result, result["overall"])

            # Score percolation - aggregate scores upward
            if "similarity_score" in field_result and "weight" in field_result:
                weight = field_result["weight"]
                threshold_applied_score = field_result["threshold_applied_score"]
                total_score += threshold_applied_score * weight
                total_weight += weight

                # Track threshold-matched fields
                info = self.model._get_comparison_info(field_name)
                if field_result["raw_similarity_score"] >= info.threshold:
                    threshold_matched_fields.add(field_name)

        # CRITICAL FIX: Handle hallucinated fields (extra fields) as False Alarms
        extra_fields_fa = self._count_extra_fields_as_false_alarms(other)
        result["overall"]["fa"] += extra_fields_fa
        result["overall"]["fp"] += extra_fields_fa

        # Calculate overall similarity score from percolated scores
        if total_weight > 0:
            result["overall"]["similarity_score"] = total_score / total_weight

        # Determine all_fields_matched
        model_fields_for_comparison = set(self.model.__class__.model_fields.keys()) - {
            "extra_fields"
        }
        result["overall"]["all_fields_matched"] = len(threshold_matched_fields) == len(
            model_fields_for_comparison
        )

        return result

    def compare_with(
        self,
        other: "StructuredModel",
        include_confusion_matrix: bool = False,
        document_non_matches: bool = False,
        evaluator_format: bool = False,
        recall_with_fd: bool = False,
        add_derived_metrics: bool = True,
    ) -> Dict[str, Any]:
        """Compare with another instance using single traversal.
        
        This is the main public API for comparing two StructuredModel instances.
        It uses the single-traversal optimization to collect all comparison data
        in one pass, then optionally adds confusion matrix metrics and non-match
        documentation based on the provided flags.
        
        The comparison process:
        1. Calls compare_recursive to get base comparison results
        2. Extracts field scores and overall metrics
        3. Optionally adds confusion matrix with aggregate and derived metrics
        4. Optionally adds non-match documentation
        5. Optionally formats results for evaluator
        
        Args:
            other: Another instance of the same model to compare with
            include_confusion_matrix: Whether to include confusion matrix calculations
            document_non_matches: Whether to document non-matches for analysis
            evaluator_format: Whether to format results for the evaluator
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                            If False, use traditional recall (TP/(TP+FN))
            add_derived_metrics: Whether to add derived metrics to confusion matrix
            
        Returns:
            Dictionary with comparison results:
            {
                "field_scores": {"field_name": float, ...},
                "overall_score": float,
                "all_fields_matched": bool,
                "confusion_matrix": {...},  # If include_confusion_matrix=True
                "non_matches": [...]  # If document_non_matches=True
            }
            
        Example:
            >>> engine = ComparisonEngine(gt_model)
            >>> result = engine.compare_with(
            ...     pred_model,
            ...     include_confusion_matrix=True,
            ...     document_non_matches=True
            ... )
            >>> print(result["overall_score"])
            >>> print(result["confusion_matrix"]["overall"]["tp"])
        """
        # SINGLE TRAVERSAL: Get everything in one pass
        recursive_result = self.compare_recursive(other)

        # Extract scoring information from recursive result
        field_scores = {}
        for field_name, field_result in recursive_result["fields"].items():
            if isinstance(field_result, dict):
                # Use threshold_applied_score when available, which respects clip_under_threshold setting
                if "threshold_applied_score" in field_result:
                    field_scores[field_name] = field_result["threshold_applied_score"]
                # Fallback to raw_similarity_score if threshold_applied_score not available
                elif "raw_similarity_score" in field_result:
                    field_scores[field_name] = field_result["raw_similarity_score"]

        # Extract overall metrics
        overall_result = recursive_result["overall"]
        overall_score = overall_result.get("similarity_score", 0.0)
        all_fields_matched = overall_result.get("all_fields_matched", False)

        # Build basic result structure
        result = {
            "field_scores": field_scores,
            "overall_score": overall_score,
            "all_fields_matched": all_fields_matched,
        }

        # Add optional features using already-computed recursive result
        if include_confusion_matrix:
            # Use ConfusionMatrixBuilder to orchestrate metrics calculation
            confusion_matrix = self.confusion_matrix_builder.build_confusion_matrix(
                recursive_result,
                add_derived_metrics=add_derived_metrics,
                recall_with_fd=recall_with_fd
            )

            result["confusion_matrix"] = confusion_matrix

        # Add optional non-match documentation
        if document_non_matches:
            # Use NonMatchCollector for enhanced object-level non-matches
            non_matches = self.non_match_collector.collect_enhanced_non_matches(recursive_result, other)
            result["non_matches"] = non_matches

        # If evaluator_format is requested, transform the result
        if evaluator_format:
            return self.model._format_for_evaluator(result, other, recall_with_fd)

        return result

    def _aggregate_to_overall(self, field_result: dict, overall: dict) -> None:
        """Simple aggregation to overall metrics.
        
        Args:
            field_result: Result from a field comparison
            overall: Overall metrics dictionary to update
        """
        for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
            if isinstance(field_result, dict):
                if metric in field_result:
                    overall[metric] += field_result[metric]
                elif "overall" in field_result and metric in field_result["overall"]:
                    overall[metric] += field_result["overall"][metric]

    def _count_extra_fields_as_false_alarms(self, other: "StructuredModel") -> int:
        """Count hallucinated fields (extra fields) in the prediction as False Alarms.

        Args:
            other: The predicted StructuredModel instance to check for extra fields

        Returns:
            Number of hallucinated fields that should count as False Alarms
        """
        fa_count = 0

        # Check if the other model has extra fields (hallucinated content)
        if hasattr(other, "__pydantic_extra__"):
            # Count each extra field as one False Alarm
            fa_count += len(other.__pydantic_extra__)

        # Also recursively check nested StructuredModel objects for extra fields
        for field_name in self.model.__class__.model_fields:
            if field_name == "extra_fields":
                continue

            gt_val = getattr(self.model, field_name, None)
            pred_val = getattr(other, field_name, None)

            # Check nested StructuredModel objects
            if isinstance(gt_val, type(self.model)) and isinstance(pred_val, type(self.model)):
                # Create engine for nested model and count its extra fields
                nested_engine = ComparisonEngine(gt_val)
                fa_count += nested_engine._count_extra_fields_as_false_alarms(pred_val)

            # Check lists of StructuredModel objects
            elif (
                isinstance(gt_val, list)
                and isinstance(pred_val, list)
                and gt_val
                and pred_val
            ):
                # Check if list contains StructuredModel instances
                if gt_val and hasattr(gt_val[0], '__class__') and hasattr(gt_val[0].__class__, 'model_fields'):
                    # Import HungarianHelper for matching
                    from .hungarian_helper import HungarianHelper
                    
                    # For lists, we need to match them up properly using Hungarian matching
                    hungarian_helper = HungarianHelper()
                    hungarian_info = hungarian_helper.get_complete_matching_info(
                        gt_val, pred_val
                    )
                    matched_pairs = hungarian_info["matched_pairs"]

                    # Count extra fields in matched pairs
                    for gt_idx, pred_idx, similarity in matched_pairs:
                        if gt_idx < len(gt_val) and pred_idx < len(pred_val):
                            gt_item = gt_val[gt_idx]
                            pred_item = pred_val[pred_idx]
                            nested_engine = ComparisonEngine(gt_item)
                            fa_count += nested_engine._count_extra_fields_as_false_alarms(pred_item)

                    # For unmatched prediction items, count their extra fields too
                    matched_pred_indices = {pred_idx for _, pred_idx, _ in matched_pairs}
                    for pred_idx, pred_item in enumerate(pred_val):
                        if pred_idx not in matched_pred_indices:
                            # Check if it's a StructuredModel
                            if hasattr(pred_item, '__class__') and hasattr(pred_item.__class__, 'model_fields'):
                                # For unmatched items, we need a dummy GT to compare against
                                if gt_val:  # Use first GT item as template
                                    dummy_gt = gt_val[0]
                                    nested_engine = ComparisonEngine(dummy_gt)
                                    fa_count += nested_engine._count_extra_fields_as_false_alarms(pred_item)
                                else:
                                    # If no GT items, count all extra fields in this pred item
                                    if hasattr(pred_item, "__pydantic_extra__"):
                                        fa_count += len(pred_item.__pydantic_extra__)

        return fa_count
