"""
Evaluator for StructuredModel objects.

This module provides an evaluator class for computing metrics on StructuredModel objects,
leveraging their built-in comparison capabilities to generate comprehensive metrics.
It also supports documenting non-matches (false positives, false negatives) for detailed analysis.
"""

## The dictionary format of the confusion matrix needs to change to an instance of a data class
## Need to remove false positives and make it a calculation

import os
import psutil
from typing import List, Dict, Any, Optional, Union, Type
import warnings

from stickler.structured_object_evaluator.models.non_match_field import (
    NonMatchField,
    NonMatchType,
)

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.comparators.structured import StructuredModelComparator
from stickler.algorithms.hungarian import HungarianMatcher


def get_memory_usage():
    """
    Get current memory usage of the process in MB.

    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


class StructuredModelEvaluator:
    """
    Evaluator for StructuredModel objects.

    This evaluator computes comprehensive metrics for StructuredModel objects,
    leveraging their built-in comparison capabilities. It includes confusion matrix
    calculations, field-level metrics, non-match documentation, and memory optimization capabilities.
    """

    def __init__(
        self,
        model_class: Optional[Type[StructuredModel]] = None,
        threshold: float = 0.5,
        verbose: bool = False,
        document_non_matches: bool = True,
        recall_with_fd: bool = False
    ):
        """
        Initialize the evaluator.

        Args:
            model_class: Optional StructuredModel class for type checking
            threshold: Similarity threshold for considering a match
            verbose: Whether to print detailed progress information
            document_non_matches: Whether to document detailed non-match information
        """
        self.model_class = model_class
        self.threshold = threshold
        self.verbose = verbose
        self.peak_memory_usage = 0
        self.recall_with_fd = recall_with_fd
        self.start_memory = get_memory_usage()

        # New attributes for documenting non-matches
        self.document_non_matches = document_non_matches
        self.non_match_documents: List[NonMatchField] = []

        warnings.warn(
            "This module is going to be removed in future versions. Use the StructuredModel.compare_with() method.",
            DeprecationWarning,
            stacklevel=2,
        )



        if self.verbose:
            print(
                f"Initialized StructuredModelEvaluator. Starting memory: {self.start_memory:.2f} MB"
            )

    def _check_memory(self):
        """Check current memory usage and update peak memory."""
        current_memory = get_memory_usage()

        if current_memory > self.peak_memory_usage:
            self.peak_memory_usage = current_memory

        if self.verbose and current_memory > self.start_memory + 100:  # 100MB increase
            print(f"Memory usage increased: {current_memory:.2f} MB")

        return current_memory

    def _calculate_metrics_from_binary(
        self,
        tp: float,
        fp: float,
        fn: float,
        tn: float = 0.0,
        fd: float = 0.0,
        recall_with_fd: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate metrics from binary classification counts.

        Args:
            tp: True positive count
            fp: False positive count
            fn: False negative count
            tn: True negative count (default 0)
            fd: False discovery count (default 0) - used only when recall_with_fd=True
            recall_with_fd: Whether to use alternative recall formula including FD in denominator

        Returns:
            Dictionary with precision, recall, F1, and accuracy
        """
        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

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

        # Calculate accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def calculate_derived_confusion_matrix_metrics(
        self, cm_counts: Dict[str, Union[int, float]]
    ) -> Dict[str, float]:
        """
        Calculate derived metrics from confusion matrix counts.

        This method uses MetricsHelper to maintain consistency and avoid code duplication.

        Args:
            cm_counts: Dictionary with confusion matrix counts containing keys:
                      'tp', 'fp', 'tn', 'fn', and optionally 'fd', 'fa'

        Returns:
            Dictionary with derived metrics: cm_precision, cm_recall, cm_f1, cm_accuracy
        """
        # Use MetricsHelper for consistent metric calculation
        from stickler.structured_object_evaluator.models.metrics_helper import (
            MetricsHelper,
        )

        metrics_helper = MetricsHelper()

        # Convert counts to the format expected by MetricsHelper
        metrics_dict = {
            "tp": int(cm_counts.get("tp", 0)),
            "fp": int(cm_counts.get("fp", 0)),
            "tn": int(cm_counts.get("tn", 0)),
            "fn": int(cm_counts.get("fn", 0)),
            "fd": int(cm_counts.get("fd", 0)),
            "fa": int(cm_counts.get("fa", 0)),
        }

        # Use MetricsHelper to calculate derived metrics
        return metrics_helper.calculate_derived_metrics(metrics_dict)

    def _convert_score_to_binary(self, score: float) -> Dict[str, float]:
        """
        Convert an ANLS Star score to binary classification counts.

        Args:
            score: ANLS Star similarity score [0-1]

        Returns:
            Dictionary with TP, FP, FN, TN counts
        """
        # For a single field comparison, there are different approaches
        # to convert a similarity score to binary classification:

        # Approach used here: If score >= threshold, count as TP with
        # proportional value, otherwise count as partial FP and partial FN
        if score >= self.threshold:
            # Handle as true positive with proportional credit
            tp = score  # Proportional TP
            fp = (
                1 - score if score < 1.0 else 0
            )  # Proportional FP for imperfect matches
            fn = 0
            tn = 0
        else:
            # Handle as false classification
            tp = 0
            fp = score  # Give partial credit for similarity even if below threshold
            fn = 1 - score  # More different = higher FN
            tn = 0

        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def _is_null_value(self, value: Any) -> bool:
        """
        Determine if a value should be considered null or empty.

        Args:
            value: The value to check

        Returns:
            True if the value is null/empty, False otherwise
        """
        if value is None:
            return True
        elif hasattr(value, "__len__") and not isinstance(
            value, (str, bytes, bytearray)
        ):
            # Consider empty lists/collections as null values
            return len(value) == 0
        elif isinstance(value, (str, bytes, bytearray)):
            return len(value.strip()) == 0
        return False

    def combine_cm_dicts(
        self, cm1: Dict[str, int], cm2: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Combine two confusion matrix dictionaries by adding corresponding values.

        Args:
            cm1: First confusion matrix dictionary
            cm2: Second confusion matrix dictionary

        Returns:
            Combined confusion matrix dictionary
        """
        return {
            key: cm1.get(key, 0) + cm2.get(key, 0)
            for key in ["tp", "fa", "fd", "fp", "tn", "fn"]
        }

    def add_non_match(
        self,
        field_path: str,
        non_match_type: NonMatchType,
        gt_value: Any,
        pred_value: Any,
        similarity_score: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Document a non-match with detailed information.

        Args:
            field_path: Dot-notation path to the field (e.g., 'address.city')
            non_match_type: Type of non-match
            gt_value: Ground truth value
            pred_value: Predicted value
            similarity_score: Optional similarity score if available
            details: Optional additional context or details
            document_id: Optional ID of the document this non-match belongs to
        """
        if not self.document_non_matches:
            return

        self.non_match_documents.append(
            NonMatchField(
                field_path=field_path,
                non_match_type=non_match_type,
                ground_truth_value=gt_value,
                prediction_value=pred_value,
                similarity_score=similarity_score,
                details=details or {},
            )
        )

    def clear_non_match_documents(self):
        """Clear the stored non-match documents."""
        self.non_match_documents = []

    def _convert_enhanced_non_match_to_field(
        self, nm_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert enhanced non-match format to NonMatchField format.

        Args:
            nm_dict: Enhanced non-match dictionary from StructuredModel

        Returns:
            Dictionary in NonMatchField format
        """
        # Map enhanced format to NonMatchField format
        converted = {
            "field_path": nm_dict.get("field_path", ""),
            "ground_truth_value": nm_dict.get("ground_truth_value"),
            "prediction_value": nm_dict.get("prediction_value"),
            "similarity_score": nm_dict.get("similarity_score"),
            "details": nm_dict.get("details", {}),
        }

        # The non_match_type is already a NonMatchType enum from StructuredModel
        converted["non_match_type"] = nm_dict.get("non_match_type")

        return converted

    def _compare_models(
        self, gt_model: StructuredModel, pred_model: StructuredModel
    ) -> Dict[str, Any]:
        """
        Compare two StructuredModel instances and return metrics.

        Args:
            gt_model: Ground truth model
            pred_model: Predicted model

        Returns:
            Dict with comparison metrics including tp, fp, fn, tn, field_scores, overall_score
        """
        # Check if inputs are valid StructuredModel instances
        if not (
            isinstance(gt_model, StructuredModel)
            and isinstance(pred_model, StructuredModel)
        ):
            raise TypeError("Both models must be StructuredModel instances")

        # If model_class is specified, check type
        if self.model_class and not (
            isinstance(gt_model, self.model_class)
            and isinstance(pred_model, self.model_class)
        ):
            raise TypeError(
                f"Both models must be instances of {self.model_class.__name__}"
            )

        # Use the built-in compare_with method from StructuredModel
        comparison_result = gt_model.compare_with(pred_model)

        # Initialize metrics
        tp = fp = fn = tn = 0

        # Determine match status
        if comparison_result["overall_score"] >= self.threshold:
            # Good enough match
            tp = 1
        else:
            # Not a good enough match
            fp = 1

        # Prepare result
        result = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "field_scores": comparison_result["field_scores"],
            "overall_score": comparison_result["overall_score"],
            # match_status removed - now unnecessary
        }

        return result

    def evaluate(
        self,
        ground_truth: StructuredModel,
        predictions: StructuredModel,
        recall_with_fd: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth and return comprehensive metrics.

        Args:
            ground_truth: Ground truth data (StructuredModel instance)
            predictions: Predicted data (StructuredModel instance)
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                            If False, use traditional recall (TP/(TP+FN))

        Returns:
            Dictionary with the following structure:

            {
                "overall": {
                    "precision": float,     # Overall precision [0-1]
                    "recall": float,        # Overall recall [0-1]
                    "f1": float,           # Overall F1 score [0-1]
                    "accuracy": float,     # Overall accuracy [0-1]
                    "anls_score": float    # Overall ANLS similarity score [0-1]
                },

                "fields": {
                    "<field_name>": {
                        # For primitive fields (str, int, float, bool):
                        "precision": float,
                        "recall": float,
                        "f1": float,
                        "accuracy": float,
                        "anls_score": float
                    },

                    "<list_field_name>": {
                        # For list fields (e.g., products: List[Product]):
                        "overall": {
                            "precision": float,
                            "recall": float,
                            "f1": float,
                            "accuracy": float,
                            "anls_score": float
                        },
                        "items": [
                            # Individual metrics for each matched item pair
                            {
                                "overall": {...},  # Item-level overall metrics
                                "fields": {        # Field metrics within each item
                                    "<nested_field>": {...}
                                }
                            }
                        ]
                    }
                },

                "confusion_matrix": {
                    "fields": {
                        # AGGREGATED metrics for all field types
                        "<field_name>": {
                            "tp": int,          # True positives
                            "fp": int,          # False positives
                            "tn": int,          # True negatives
                            "fn": int,          # False negatives
                            "fd": int,          # False discoveries (non-null but don't match)
                            "fa": int,          # False alarms
                            "derived": {
                                "cm_precision": float,
                                "cm_recall": float,
                                "cm_f1": float,
                                "cm_accuracy": float
                            }
                        },

                        # For list fields with nested objects, aggregated field metrics:
                        "<list_field>.<nested_field>": {
                            # Aggregated counts across ALL instances in the list
                            "tp": int,    # Total true positives for this field across all items
                            "fp": int,    # Total false positives for this field across all items
                            "fn": int,    # Total false negatives for this field across all items
                            "fd": int,    # Total false discoveries for this field across all items
                            "fa": int,    # Total false alarms for this field across all items
                            "derived": {...}
                        }
                    },

                    "overall": {
                        # Overall confusion matrix aggregating all fields
                        "tp": int, "fp": int, "tn": int, "fn": int, "fd": int, "fa": int
                        "derived": {...}
                    }
                }
            }

        Key Usage Patterns:

        1. **Individual Item Metrics** (per-instance analysis):
           ```python
           # Access metrics for each individual item in a list
           for i, item_metrics in enumerate(results['fields']['products']['items']):
               print(f"Product {i}: {item_metrics['overall']['f1']}")
           ```

        2. **Aggregated Field Metrics** (recommended for field performance analysis):
           ```python
           # Access aggregated metrics across all instances of a field type
           cm_fields = results['confusion_matrix']['fields']
           product_id_performance = cm_fields['products.product_id']
           print(f"Product ID field: {product_id_performance['derived']['cm_precision']}")

           # Get all aggregated product field metrics
           product_fields = {k: v for k, v in cm_fields.items()
                           if k.startswith('products.')}
           ```

        3. **Helper Function for Aggregated Metrics**:
           ```python
           def get_aggregated_metrics(results, list_field_name):
               '''Extract aggregated field metrics for a list field.'''
               cm_fields = results['confusion_matrix']['fields']
               prefix = f"{list_field_name}."
               return {k.replace(prefix, ''): v for k, v in cm_fields.items()
                      if k.startswith(prefix)}

           # Usage:
           product_metrics = get_aggregated_metrics(results, 'products')
           print(f"Product name precision: {product_metrics['name']['derived']['cm_precision']}")
           ```

        Note:
            - Use `results['fields'][field]['items']` for per-instance analysis
            - Use `results['confusion_matrix']['fields'][field.subfield]` for aggregated field analysis
            - Aggregated metrics provide rolled-up performance across all instances of a field type
            - Confusion matrix metrics use standard TP/FP/TN/FN/FD classification with derived metrics
        """
        # Clear any existing non-match documents
        self.clear_non_match_documents()

        # Use StructuredModel's enhanced comparison with evaluator format
        # This pushes all the heavy lifting into the StructuredModel as requested
        result = ground_truth.compare_with(
            predictions,
            include_confusion_matrix=True,
            document_non_matches=self.document_non_matches,
            evaluator_format=True,  # This makes StructuredModel return evaluator-compatible format
            recall_with_fd=recall_with_fd,
        )

        # Add non-matches to evaluator's collection if they exist
        if result.get("non_matches"):
            for nm_dict in result["non_matches"]:
                # Convert enhanced non-match format to NonMatchField format
                converted_nm = self._convert_enhanced_non_match_to_field(nm_dict)
                self.non_match_documents.append(NonMatchField(**converted_nm))

        # Process derived metrics explicitly with recall_with_fd parameter
        if "confusion_matrix" in result and "overall" in result["confusion_matrix"]:
            overall_cm = result["confusion_matrix"]["overall"]

            # Update derived metrics directly in the result
            from stickler.structured_object_evaluator.models.metrics_helper import (
                MetricsHelper,
            )

            metrics_helper = MetricsHelper()

            # Apply correct recall_with_fd to overall metrics
            derived_metrics = metrics_helper.calculate_derived_metrics(
                overall_cm, recall_with_fd=recall_with_fd
            )
            result["confusion_matrix"]["overall"]["derived"] = derived_metrics

            # Copy these to the top-level metrics if needed
            if "overall" in result:
                result["overall"]["precision"] = derived_metrics["cm_precision"]
                result["overall"]["recall"] = derived_metrics["cm_recall"]
                result["overall"]["f1"] = derived_metrics["cm_f1"]

        return result

    def _format_evaluation_results(
        self, comparison_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format StructuredModel comparison results to match expected evaluator output format.

        Args:
            comparison_result: Result from StructuredModel.compare_with()

        Returns:
            Dictionary in the expected evaluator format
        """
        # Extract components from StructuredModel result
        field_scores = comparison_result["field_scores"]
        overall_score = comparison_result["overall_score"]
        confusion_matrix = comparison_result.get("confusion_matrix", {})
        non_matches = comparison_result.get("non_matches", [])

        # Calculate field metrics using existing logic for backward compatibility
        field_metrics = {}

        for field_name, score in field_scores.items():
            # Convert field score to binary metrics using existing method
            binary = self._convert_score_to_binary(score)
            # For field metrics, fd is often not available directly, so we ignore recall_with_fd
            metrics = self._calculate_metrics_from_binary(
                binary["tp"], binary["fp"], binary["fn"], binary["tn"]
            )
            metrics["anls_score"] = score
            field_metrics[field_name] = metrics

        # Calculate overall metrics
        binary = self._convert_score_to_binary(overall_score)
        # For overall metrics, use confusion_matrix data which should have fd
        overall_fd = confusion_matrix.get("overall", {}).get("fd", 0)
        overall_metrics = self._calculate_metrics_from_binary(
            binary["tp"],
            binary["fp"],
            binary["fn"],
            binary["tn"],
            fd=overall_fd,
            recall_with_fd=self.recall_with_fd,
        )
        overall_metrics["anls_score"] = overall_score

        # Add non-matches to evaluator's collection if they exist
        if non_matches:
            for nm_dict in non_matches:
                self.non_match_documents.append(NonMatchField(**nm_dict))

        # Prepare final result in expected format
        result = {
            "overall": overall_metrics,
            "fields": field_metrics,
            "confusion_matrix": confusion_matrix,
            "non_matches": non_matches,
        }

        return result

    def _compare_model_lists(
        self, gt_models: List[StructuredModel], pred_models: List[StructuredModel]
    ) -> Dict[str, Any]:
        """
        Compare two lists of StructuredModel instances using Hungarian matching.

        Args:
            gt_models: List of ground truth models
            pred_models: List of predicted models

        Returns:
            Dict with comparison metrics including tp, fp, fn, overall_score
        """
        # Handle empty lists
        if not gt_models and not pred_models:
            return {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "overall_score": 1.0,  # Empty lists are a perfect match
            }

        if not gt_models:
            return {
                "tp": 0,
                "fp": len(pred_models),
                "fn": 0,
                "tn": 0,
                "overall_score": 0.0,  # All predictions are false positives
            }

        if not pred_models:
            return {
                "tp": 0,
                "fp": 0,
                "fn": len(gt_models),
                "tn": 0,
                "overall_score": 0.0,  # All ground truths are false negatives
            }

        # Ensure all items are StructuredModel instances
        if not all(
            isinstance(model, StructuredModel) for model in gt_models + pred_models
        ):
            raise TypeError("All items in both lists must be StructuredModel instances")

        # If model_class is specified, check type for all models
        if self.model_class:
            if not all(
                isinstance(model, self.model_class) for model in gt_models + pred_models
            ):
                raise TypeError(
                    f"All models must be instances of {self.model_class.__name__}"
                )

        # Create a Hungarian matcher with StructuredModelComparator
        hungarian = HungarianMatcher(StructuredModelComparator())

        # Run Hungarian matching
        tp, fp = hungarian(gt_models, pred_models)

        # Calculate false negatives
        fn = len(gt_models) - tp

        # Calculate overall score (proportion of correct matches)
        max_items = max(len(gt_models), len(pred_models))
        overall_score = tp / max_items if max_items > 0 else 1.0

        return {"tp": tp, "fp": fp, "fn": fn, "tn": 0, "overall_score": overall_score}
