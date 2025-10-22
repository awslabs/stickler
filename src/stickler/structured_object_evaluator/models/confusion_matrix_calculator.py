"""Confusion matrix calculation for structured model comparisons.

This module provides the ConfusionMatrixCalculator class for calculating
confusion matrix metrics (TP, FP, TN, FN, FD, FA) for field comparisons.
"""

from typing import Any, Dict, List, TYPE_CHECKING

from .field_helper import FieldHelper
from .hungarian_helper import HungarianHelper
from .non_matches_helper import NonMatchesHelper
from .metrics_helper import MetricsHelper

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class ConfusionMatrixCalculator:
    """Calculates confusion matrix metrics for field comparisons.
    
    This class is responsible for computing confusion matrix statistics
    (True Positives, False Positives, True Negatives, False Negatives,
    False Discoveries, False Alarms) for individual fields and lists of
    structured models.
    
    Attributes:
        model: The ground truth StructuredModel instance used for comparison
    """

    def __init__(self, model: "StructuredModel"):
        """Initialize calculator with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model

    def calculate_list_confusion_matrix(
        self, field_name: str, other_list: List[Any]
    ) -> Dict[str, Any]:
        """Calculate confusion matrix for a list field.
        
        This method computes confusion matrix metrics for a list field,
        including nested field metrics for List[StructuredModel] fields.
        It uses Hungarian matching for optimal pairing of list elements.
        
        Args:
            field_name: Name of the list field being compared
            other_list: Predicted list to compare with
            
        Returns:
            Dictionary with:
            - Top-level TP, FP, TN, FN, FD, FA counts and derived metrics
            - nested_fields: Dict with metrics for individual fields within list items
            - non_matches: List of individual object-level non-matches
            
        Example:
            >>> calculator = ConfusionMatrixCalculator(gt_model)
            >>> result = calculator.calculate_list_confusion_matrix("items", pred_list)
            >>> print(result["tp"], result["fn"])
        """
        # Import here to avoid circular imports
        from .structured_model import StructuredModel
        
        gt_list = getattr(self.model, field_name)
        pred_list = other_list

        # Initialize result structure
        result = {
            "tp": 0,
            "fa": 0,
            "fd": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "nested_fields": {},  # Store nested field metrics here
            "non_matches": [],  # Store individual object-level non-matches here
        }

        # Handle null cases first
        if FieldHelper.is_null_value(gt_list) and FieldHelper.is_null_value(pred_list):
            result.update({"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0})
        elif FieldHelper.is_null_value(gt_list):
            result.update(
                {
                    "tp": 0,
                    "fa": len(pred_list),
                    "fd": 0,
                    "fp": len(pred_list),
                    "tn": 0,
                    "fn": 0,
                }
            )
            # Add non-matches for each FA item using NonMatchesHelper
            non_matches_helper = NonMatchesHelper()
            result["non_matches"] = non_matches_helper.add_non_matches_for_null_cases(
                field_name, gt_list, pred_list
            )
        elif FieldHelper.is_null_value(pred_list):
            result.update(
                {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": len(gt_list)}
            )
            # Add non-matches for each FN item using NonMatchesHelper
            non_matches_helper = NonMatchesHelper()
            result["non_matches"] = non_matches_helper.add_non_matches_for_null_cases(
                field_name, gt_list, pred_list
            )
        else:
            # Use existing comparison logic for list-level metrics
            info = self.model.__class__._get_comparison_info(field_name)
            comparator = info.comparator
            threshold = info.threshold

            # Reuse existing Hungarian matching logic
            match_result = self.model._compare_unordered_lists(
                gt_list, pred_list, comparator, threshold
            )

            # Use the detailed confusion matrix results directly from Hungarian matcher
            result.update(
                {
                    "tp": match_result["tp"],
                    "fa": match_result[
                        "fa"
                    ],  # False alarms (unmatched prediction items)
                    "fd": match_result[
                        "fd"
                    ],  # False discoveries (matches below threshold)
                    "fp": match_result["fp"],  # Total false positives (fa + fd)
                    "tn": 0,
                    "fn": match_result["fn"],  # False negatives (unmatched GT items)
                }
            )

            # Collect individual object-level non-matches using NonMatchesHelper
            if gt_list and isinstance(gt_list[0], StructuredModel):
                non_matches_helper = NonMatchesHelper()
                non_matches = non_matches_helper.collect_list_non_matches(
                    field_name, gt_list, pred_list
                )
                result["non_matches"] = non_matches

            # If list contains StructuredModel objects, calculate nested field metrics
            if gt_list and isinstance(gt_list[0], StructuredModel):
                nested_metrics = self.calculate_nested_field_metrics(
                    field_name, gt_list, pred_list, threshold
                )
                result["nested_fields"] = nested_metrics

        # For List[StructuredModel], we should NOT aggregate nested fields to list level
        # List level metrics represent object-level matches from Hungarian algorithm
        # Nested field metrics represent field-level matches within those objects
        # They are separate concerns and should not be aggregated

        # Only aggregate if this is explicitly marked as an aggregate field AND it's not a list
        is_aggregate = self.model.__class__._is_aggregate_field(field_name)
        if is_aggregate and not isinstance(gt_list, list):
            # Initialize top-level confusion matrix values to 0
            result["tp"] = 0
            result["fa"] = 0
            result["fd"] = 0
            result["fp"] = 0
            result["tn"] = 0
            result["fn"] = 0
            # Sum up the confusion matrix values from nested fields
            for field, field_metrics in result["nested_fields"].items():
                result["tp"] += field_metrics["tp"]
                result["fa"] += field_metrics["fa"]
                result["fd"] += field_metrics["fd"]
                result["fp"] += field_metrics["fp"]
                result["tn"] += field_metrics["tn"]
                result["fn"] += field_metrics["fn"]

        # Add derived metrics
        metrics_helper = MetricsHelper()
        result["derived"] = metrics_helper.calculate_derived_metrics(result)

        return result

    def classify_field_for_confusion_matrix(
        self, field_name: str, other_value: Any, threshold: float = None
    ) -> Dict[str, Any]:
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
            field_name: Name of the field being compared
            other_value: Value to compare with
            threshold: Threshold for matching (uses field's threshold if None)
            
        Returns:
            Dictionary with TP, FP, TN, FN, FD counts and derived metrics
            
        Example:
            >>> calculator = ConfusionMatrixCalculator(gt_model)
            >>> result = calculator.classify_field_for_confusion_matrix("name", "John")
            >>> print(result["tp"], result["derived"]["precision"])
        """
        # Get field values
        gt_value = getattr(self.model, field_name)
        pred_value = other_value

        # Get field configuration
        info = self.model.__class__._get_comparison_info(field_name)
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
                # Use the field's configured comparator for primitive comparison
                similarity = comparator.compare(gt_value, pred_value)
            values_match = similarity >= threshold
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

        # Add derived metrics
        metrics_helper = MetricsHelper()
        result["derived"] = metrics_helper.calculate_derived_metrics(result)
        # Don't include similarity_score in the result as tests don't expect it

        return result

    def calculate_nested_field_metrics(
        self,
        list_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        threshold: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for fields within list items.
        
        This method performs threshold-gated recursive analysis of fields within
        matched list items. Only pairs with similarity >= match_threshold undergo
        recursive field analysis. Poor matches and unmatched items are treated
        as atomic units.
        
        Args:
            list_field_name: Name of the parent list field (e.g., "transactions")
            gt_list: Ground truth list of StructuredModel objects
            pred_list: Predicted list of StructuredModel objects
            threshold: Matching threshold for recursive analysis
            
        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics.
            E.g., {"transactions.date": {...}, "transactions.description": {...}}
            
        Example:
            >>> calculator = ConfusionMatrixCalculator(gt_model)
            >>> metrics = calculator.calculate_nested_field_metrics(
            ...     "items", gt_items, pred_items, 0.7
            ... )
            >>> print(metrics["items.name"]["tp"])
        """
        # Import here to avoid circular imports
        from .structured_model import StructuredModel
        
        nested_metrics = {}

        if not gt_list or not isinstance(gt_list[0], StructuredModel):
            return nested_metrics

        # Get the model class from the first item
        model_class = gt_list[0].__class__

        # CRITICAL FIX: Use field's threshold, not class's match_threshold
        # Get the field info from the parent object to use the correct threshold
        parent_field_info = self.model.__class__._get_comparison_info(list_field_name)
        match_threshold = parent_field_info.threshold

        # For each field in the nested model
        for field_name in model_class.model_fields:
            if field_name == "extra_fields":
                continue

            nested_field_path = f"{list_field_name}.{field_name}"

            # Initialize aggregated counts for this nested field
            total_tp = total_fa = total_fd = total_fp = total_tn = total_fn = 0

            # Use HungarianHelper for Hungarian matching operations - OPTIMIZED: Single call gets all info
            hungarian_helper = HungarianHelper()

            # Use HungarianHelper to get optimal assignments with similarity scores
            assignments = []
            matched_pairs_with_scores = []
            if gt_list and pred_list:
                hungarian_info = hungarian_helper.get_complete_matching_info(
                    gt_list, pred_list
                )
                matched_pairs_with_scores = hungarian_info["matched_pairs"]
                # Extract (gt_idx, pred_idx) pairs from the matched_pairs
                assignments = [(i, j) for i, j, score in matched_pairs_with_scores]

            # THRESHOLD-GATED RECURSION: Only process pairs that meet the match_threshold
            for gt_idx, pred_idx, similarity_score in matched_pairs_with_scores:
                if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                    gt_item = gt_list[gt_idx]
                    pred_item = pred_list[pred_idx]

                    # Handle floating point precision issues
                    is_above_threshold = (
                        similarity_score >= match_threshold
                        or abs(similarity_score - match_threshold) < 1e-10
                    )

                    # Only perform recursive field analysis if similarity meets threshold
                    if is_above_threshold:
                        # Get field values
                        gt_value = getattr(gt_item, field_name, None)
                        pred_value = getattr(pred_item, field_name, None)

                        # Check if this field is a List[StructuredModel] that needs recursive processing
                        if (
                            isinstance(gt_value, list)
                            and isinstance(pred_value, list)
                            and gt_value
                            and isinstance(gt_value[0], StructuredModel)
                        ):
                            # Handle List[StructuredModel] recursively
                            # Create a calculator for the gt_item
                            item_calculator = ConfusionMatrixCalculator(gt_item)
                            list_classification = item_calculator.calculate_list_confusion_matrix(
                                field_name, pred_value
                            )

                            # Aggregate the list-level counts
                            total_tp += list_classification["tp"]
                            total_fa += list_classification["fa"]
                            total_fd += list_classification["fd"]
                            total_fp += list_classification["fp"]
                            total_tn += list_classification["tn"]
                            total_fn += list_classification["fn"]

                            # IMPORTANT: Also collect the deeper nested field metrics
                            if "nested_fields" in list_classification:
                                for (
                                    deeper_field_path,
                                    deeper_metrics,
                                ) in list_classification["nested_fields"].items():
                                    # Create the full path: e.g., "products.attributes.name"
                                    full_deeper_path = (
                                        f"{list_field_name}.{deeper_field_path}"
                                    )

                                    # Initialize or aggregate into the deeper nested metrics
                                    if full_deeper_path not in nested_metrics:
                                        nested_metrics[full_deeper_path] = {
                                            "tp": 0,
                                            "fa": 0,
                                            "fd": 0,
                                            "fp": 0,
                                            "tn": 0,
                                            "fn": 0,
                                        }

                                    nested_metrics[full_deeper_path]["tp"] += (
                                        deeper_metrics["tp"]
                                    )
                                    nested_metrics[full_deeper_path]["fa"] += (
                                        deeper_metrics["fa"]
                                    )
                                    nested_metrics[full_deeper_path]["fd"] += (
                                        deeper_metrics["fd"]
                                    )
                                    nested_metrics[full_deeper_path]["fp"] += (
                                        deeper_metrics["fp"]
                                    )
                                    nested_metrics[full_deeper_path]["tn"] += (
                                        deeper_metrics["tn"]
                                    )
                                    nested_metrics[full_deeper_path]["fn"] += (
                                        deeper_metrics["fn"]
                                    )
                        else:
                            # Handle primitive fields or single StructuredModel fields
                            # Create a calculator for the gt_item
                            item_calculator = ConfusionMatrixCalculator(gt_item)
                            field_classification = item_calculator.classify_field_for_confusion_matrix(
                                field_name,
                                pred_value,
                                None,  # Use field's own threshold
                            )

                            # Aggregate counts
                            total_tp += field_classification["tp"]
                            total_fa += field_classification["fa"]
                            total_fd += field_classification["fd"]
                            total_fp += field_classification["fp"]
                            total_tn += field_classification["tn"]
                            total_fn += field_classification["fn"]
                    else:
                        # Skip recursive analysis for pairs below threshold
                        # These will be handled as FD at the object level
                        pass

            # Handle unmatched ground truth items (false negatives)
            matched_gt_indices = set(idx for idx, _ in assignments)
            for gt_idx, gt_item in enumerate(gt_list):
                if gt_idx not in matched_gt_indices:
                    gt_value = getattr(gt_item, field_name, None)
                    if not FieldHelper.is_null_value(gt_value):
                        # Check if this is a List[StructuredModel] that needs deeper processing for FN
                        if (
                            isinstance(gt_value, list)
                            and gt_value
                            and isinstance(gt_value[0], StructuredModel)
                        ):
                            # For List[StructuredModel], count each item in the list as a separate FN
                            # and handle deeper nested fields
                            total_fn += len(gt_value)  # Each list item is a separate FN

                            # Also handle deeper nested fields for unmatched items
                            dummy_empty_list = []  # Empty list for comparison
                            item_calculator = ConfusionMatrixCalculator(gt_item)
                            list_classification = item_calculator.calculate_list_confusion_matrix(
                                field_name, dummy_empty_list
                            )
                            if "nested_fields" in list_classification:
                                for (
                                    deeper_field_path,
                                    deeper_metrics,
                                ) in list_classification["nested_fields"].items():
                                    full_deeper_path = (
                                        f"{list_field_name}.{deeper_field_path}"
                                    )
                                    if full_deeper_path not in nested_metrics:
                                        nested_metrics[full_deeper_path] = {
                                            "tp": 0,
                                            "fa": 0,
                                            "fd": 0,
                                            "fp": 0,
                                            "tn": 0,
                                            "fn": 0,
                                        }
                                    nested_metrics[full_deeper_path]["fn"] += (
                                        deeper_metrics["fn"]
                                    )
                        else:
                            # Handle primitive fields or single StructuredModel fields
                            total_fn += 1

            # Handle unmatched prediction items (false alarms)
            matched_pred_indices = set(idx for _, idx in assignments)
            for pred_idx, pred_item in enumerate(pred_list):
                if pred_idx not in matched_pred_indices:
                    pred_value = getattr(pred_item, field_name, None)
                    if not FieldHelper.is_null_value(pred_value):
                        # Check if this is a List[StructuredModel] that needs deeper processing for FA
                        if (
                            isinstance(pred_value, list)
                            and pred_value
                            and isinstance(pred_value[0], StructuredModel)
                        ):
                            # For List[StructuredModel], count each item in the list as a separate FA
                            # and handle deeper nested fields
                            total_fa += len(
                                pred_value
                            )  # Each list item is a separate FA
                            total_fp += len(
                                pred_value
                            )  # Each list item is also a separate FP

                            # Also handle deeper nested fields for unmatched items
                            dummy_empty_list = []  # Empty list for comparison
                            # We need to create a dummy GT item for comparison to get the structure
                            if gt_list:  # Use structure from an existing GT item
                                dummy_gt_item = gt_list[0]
                                dummy_calculator = ConfusionMatrixCalculator(dummy_gt_item)
                                list_classification = dummy_calculator.calculate_list_confusion_matrix(
                                    field_name, pred_value
                                )
                                if "nested_fields" in list_classification:
                                    for (
                                        deeper_field_path,
                                        deeper_metrics,
                                    ) in list_classification["nested_fields"].items():
                                        full_deeper_path = (
                                            f"{list_field_name}.{deeper_field_path}"
                                        )
                                        if full_deeper_path not in nested_metrics:
                                            nested_metrics[full_deeper_path] = {
                                                "tp": 0,
                                                "fa": 0,
                                                "fd": 0,
                                                "fp": 0,
                                                "tn": 0,
                                                "fn": 0,
                                            }
                                        nested_metrics[full_deeper_path]["fa"] += (
                                            deeper_metrics["fa"]
                                        )
                                        nested_metrics[full_deeper_path]["fp"] += (
                                            deeper_metrics["fp"]
                                        )
                        else:
                            # Handle primitive fields or single StructuredModel fields
                            total_fa += 1
                            total_fp += 1

            # Store the aggregated metrics for this nested field
            nested_metrics[nested_field_path] = {
                "tp": total_tp,
                "fa": total_fa,
                "fd": total_fd,
                "fp": total_fp,
                "tn": total_tn,
                "fn": total_fn,
                "derived": MetricsHelper().calculate_derived_metrics(
                    {
                        "tp": total_tp,
                        "fa": total_fa,
                        "fd": total_fd,
                        "fp": total_fp,
                        "tn": total_tn,
                        "fn": total_fn,
                    }
                ),
            }

        # Add derived metrics for all deeper nested fields that were collected
        for deeper_path, deeper_metrics in nested_metrics.items():
            if deeper_path != nested_field_path and "derived" not in deeper_metrics:
                deeper_metrics["derived"] = MetricsHelper().calculate_derived_metrics(
                    {
                        "tp": deeper_metrics["tp"],
                        "fa": deeper_metrics["fa"],
                        "fd": deeper_metrics["fd"],
                        "fp": deeper_metrics["fp"],
                        "tn": deeper_metrics["tn"],
                        "fn": deeper_metrics["fn"],
                    }
                )

        return nested_metrics

    def calculate_single_nested_field_metrics(
        self,
        parent_field_name: str,
        gt_nested: "StructuredModel",
        pred_nested: "StructuredModel",
        parent_is_aggregate: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for fields within a single nested StructuredModel.

        Args:
            parent_field_name: Name of the parent field (e.g., "address")
            gt_nested: Ground truth nested StructuredModel
            pred_nested: Predicted nested StructuredModel
            parent_is_aggregate: Whether the parent field should aggregate child metrics

        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics
            E.g., {"address.street": {...}, "address.city": {...}}
        """
        # Import here to avoid circular imports
        from .structured_model import StructuredModel
        
        nested_metrics = {}

        if not isinstance(gt_nested, StructuredModel) or not isinstance(
            pred_nested, StructuredModel
        ):
            # Handle case where one of the fields is a list of StructuredModel objects
            if (
                not isinstance(gt_nested, list)
                or not gt_nested
                or not isinstance(gt_nested[0], StructuredModel)
            ):
                return nested_metrics
            return nested_metrics

        # Initialize aggregation metrics for parent field if it's an aggregated field
        parent_metrics = (
            {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
            if parent_is_aggregate
            else None
        )

        # Track which fields are aggregate fields themselves to avoid double counting
        child_aggregate_fields = set()

        # For each field in the nested model
        for field_name in gt_nested.__class__.model_fields:
            if field_name == "extra_fields":
                continue

            nested_field_path = f"{parent_field_name}.{field_name}"

            # Check if this nested field is itself an aggregate field
            is_child_aggregate = False
            if hasattr(gt_nested.__class__, "_is_aggregate_field"):
                is_child_aggregate = gt_nested.__class__._is_aggregate_field(field_name)
                if is_child_aggregate:
                    child_aggregate_fields.add(field_name)

            # Get the field value from the prediction
            pred_value = getattr(pred_nested, field_name, None)
            gt_value = getattr(gt_nested, field_name)

            # Handle lists of StructuredModel objects
            if (
                isinstance(gt_value, list)
                and isinstance(pred_value, list)
                and gt_value
                and isinstance(gt_value[0], StructuredModel)
            ):
                # Use the list comparison logic for lists of StructuredModel objects
                nested_calculator = ConfusionMatrixCalculator(gt_nested)
                list_metrics = nested_calculator.calculate_list_confusion_matrix(
                    field_name, pred_value
                )

                # Store the metrics for this nested field
                nested_metrics[nested_field_path] = {
                    key: value
                    for key, value in list_metrics.items()
                    if key != "nested_fields"
                }

                # Add nested field metrics if available
                if "nested_fields" in list_metrics:
                    for sub_field, sub_metrics in list_metrics["nested_fields"].items():
                        full_path = f"{nested_field_path}.{sub_field.split('.')[-1]}"
                        nested_metrics[full_path] = sub_metrics
            else:
                # Classify this field comparison
                nested_calculator = ConfusionMatrixCalculator(gt_nested)
                field_classification = nested_calculator.classify_field_for_confusion_matrix(
                    field_name, pred_value
                )

                # Store the metrics for this nested field
                nested_metrics[nested_field_path] = field_classification

                # Recursively calculate metrics for deeper nesting
                deeper_metrics = self.calculate_single_nested_field_metrics(
                    nested_field_path, gt_value, pred_value, is_child_aggregate
                )
                nested_metrics.update(deeper_metrics)

                # If this is an aggregate child field, we need to use its aggregated metrics
                # instead of the direct field comparison metrics
                if is_child_aggregate and nested_field_path in deeper_metrics:
                    # For an aggregate child field, we replace its direct metrics with
                    # the aggregation of its children's metrics
                    nested_metrics[nested_field_path] = deeper_metrics[
                        nested_field_path
                    ]

            # For parent aggregation, we need to be careful not to double count metrics
            if parent_is_aggregate:
                if is_child_aggregate:
                    # If child is an aggregate, use its aggregated metrics for parent
                    if nested_field_path in deeper_metrics:
                        child_agg_metrics = deeper_metrics[nested_field_path]
                        for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                            parent_metrics[metric] += child_agg_metrics.get(metric, 0)
                else:
                    # If child is not an aggregate, use its direct field metrics
                    field_metrics = nested_metrics[nested_field_path]
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        parent_metrics[metric] += field_metrics.get(metric, 0)

        # If parent is an aggregated field, add the aggregated metrics to the result
        if parent_is_aggregate:
            # Don't include metrics from child aggregate fields in the parent's metrics
            # as they've already been counted through their own aggregation
            for field_name in child_aggregate_fields:
                nested_field_path = f"{parent_field_name}.{field_name}"
                if nested_field_path in nested_metrics:
                    # Don't double count these metrics in the parent
                    field_metrics = nested_metrics[nested_field_path]
                    # Subtract these metrics from parent_metrics to avoid double counting
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        parent_metrics[metric] -= field_metrics.get(metric, 0)

            nested_metrics[parent_field_name] = parent_metrics
            # Add derived metrics
            nested_metrics[parent_field_name]["derived"] = (
                MetricsHelper().calculate_derived_metrics(parent_metrics)
            )

        return nested_metrics
