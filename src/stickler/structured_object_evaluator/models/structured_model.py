"""Structured model comparison using Pydantic models.

This module provides the StructuredModel class for defining structured data models
with comparison configuration and evaluation capabilities.
"""

from pydantic import BaseModel, Field
from typing import (
    Any,
    Dict,
    List,
    Type,
    Union,
    ClassVar,
    get_origin,
    get_args,
)
import inspect

from stickler.comparators.base import BaseComparator

from .comparable_field import ComparableField
from .non_match_field import NonMatchField
from .hungarian_helper import HungarianHelper
from .metrics_helper import MetricsHelper
from .field_helper import FieldHelper
from .configuration_helper import ConfigurationHelper
from .comparison_helper import ComparisonHelper
from .evaluator_format_helper import EvaluatorFormatHelper


class StructuredModel(BaseModel):
    """Base class for models with structured comparison capabilities.

    This class extends Pydantic's BaseModel with the ability to compare
    instances using configurable comparison metrics for each field.
    It supports:
    - Field-level comparison configuration
    - Nested model comparison
    - Integration with ANLS* comparators
    - JSON schema generation with comparison metadata
    - Unordered list comparison using Hungarian matching
    - Retention of extra fields not defined in the model
    """

    # Default match threshold - can be overridden in subclasses
    match_threshold: ClassVar[float] = 0.7

    extra_fields: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # Allow extra fields to be stored in extra_fields
    }

    def __init_subclass__(cls, **kwargs):
        """Validate field configurations when a StructuredModel subclass is defined."""
        super().__init_subclass__(**kwargs)

        # Validate field configurations using class annotations since model_fields isn't populated yet
        if hasattr(cls, "__annotations__"):
            for field_name, field_type in cls.__annotations__.items():
                if field_name == "extra_fields":
                    continue

                # Get the field default value if it exists
                field_default = getattr(cls, field_name, None)

                # Since ComparableField is now always a function that returns a Field,
                # we need to check if field_default has comparison metadata
                if hasattr(field_default, "json_schema_extra") and callable(
                    field_default.json_schema_extra
                ):
                    # Check for comparison metadata
                    temp_schema = {}
                    field_default.json_schema_extra(temp_schema)
                    if "x-comparison" in temp_schema:
                        # This field was created with ComparableField function - validate constraints
                        if cls._is_list_of_structured_model_type(field_type):
                            comparison_config = temp_schema["x-comparison"]

                            # Threshold validation - only flag if explicitly set to non-default value
                            threshold = comparison_config.get("threshold", 0.5)
                            if threshold != 0.5:  # Default threshold value
                                raise ValueError(
                                    f"Field '{field_name}' is a List[StructuredModel] and cannot have a "
                                    f"'threshold' parameter in ComparableField. Hungarian matching uses each "
                                    f"StructuredModel's 'match_threshold' class attribute instead. "
                                    f"Set 'match_threshold = {threshold}' on the list element class."
                                )

                            # Comparator validation - only flag if explicitly set to non-default type
                            comparator_type = comparison_config.get(
                                "comparator_type", "LevenshteinComparator"
                            )
                            if (
                                comparator_type != "LevenshteinComparator"
                            ):  # Default comparator type
                                raise ValueError(
                                    f"Field '{field_name}' is a List[StructuredModel] and cannot have a "
                                    f"'comparator' parameter in ComparableField. Object comparison uses each "
                                    f"StructuredModel's individual field comparators instead."
                                )

    @classmethod
    def _is_list_of_structured_model_type(cls, field_type) -> bool:
        """Check if a field type annotation represents List[StructuredModel].

        Args:
            field_type: The field type annotation

        Returns:
            True if the field is a List[StructuredModel] type
        """
        # Handle direct imports and typing constructs
        origin = get_origin(field_type)
        if origin is list or origin is List:
            args = get_args(field_type)
            if args:
                element_type = args[0]
                # Check if element type is a StructuredModel subclass
                try:
                    return inspect.isclass(element_type) and issubclass(
                        element_type, StructuredModel
                    )
                except (TypeError, AttributeError):
                    return False

        # Handle Union types (like Optional[List[StructuredModel]])
        elif origin is Union:
            args = get_args(field_type)
            for arg in args:
                if cls._is_list_of_structured_model_type(arg):
                    return True

        return False

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "StructuredModel":
        """Create a StructuredModel instance from JSON data.

        This method handles missing fields gracefully and stores extra fields
        in the extra_fields attribute.

        Args:
            json_data: Dictionary containing the JSON data

        Returns:
            StructuredModel instance created from the JSON data
        """
        return ConfigurationHelper.from_json(cls, json_data)

    @classmethod
    def model_from_json(cls, config: Dict[str, Any]) -> Type["StructuredModel"]:
        """Create a StructuredModel subclass from JSON configuration using Pydantic's create_model().

        This method leverages Pydantic's native dynamic model creation capabilities to ensure
        full compatibility with all Pydantic features while adding structured comparison
        functionality through inherited StructuredModel methods.

        The generated model inherits all StructuredModel capabilities:
        - compare_with() method for detailed comparisons
        - Field-level comparison configuration
        - Hungarian algorithm for list matching
        - Confusion matrix generation
        - JSON schema with comparison metadata

        Args:
            config: JSON configuration with fields, comparators, and model settings.
                   Required keys:
                   - fields: Dict mapping field names to field configurations
                   Optional keys:
                   - model_name: Name for the generated class (default: "DynamicModel")
                   - match_threshold: Overall matching threshold (default: 0.7)

                   Field configuration format:
                   {
                       "type": "str|int|float|bool|List[str]|etc.",  # Required
                       "comparator": "LevenshteinComparator|ExactComparator|etc.",  # Optional
                       "threshold": 0.8,  # Optional, default 0.5
                       "weight": 2.0,     # Optional, default 1.0
                       "required": true,  # Optional, default false
                       "default": "value", # Optional
                       "description": "Field description",  # Optional
                       "alias": "field_alias",  # Optional
                       "examples": ["example1", "example2"]  # Optional
                   }

        Returns:
            A fully functional StructuredModel subclass created with create_model()

        Raises:
            ValueError: If configuration is invalid or contains unsupported types/comparators
            KeyError: If required configuration keys are missing

        Examples:
            >>> config = {
            ...     "model_name": "Product",
            ...     "match_threshold": 0.8,
            ...     "fields": {
            ...         "name": {
            ...             "type": "str",
            ...             "comparator": "LevenshteinComparator",
            ...             "threshold": 0.8,
            ...             "weight": 2.0,
            ...             "required": True
            ...         },
            ...         "price": {
            ...             "type": "float",
            ...             "comparator": "NumericComparator",
            ...             "default": 0.0
            ...         }
            ...     }
            ... }
            >>> ProductClass = StructuredModel.model_from_json(config)
            >>> isinstance(ProductClass.model_fields, dict)  # Full Pydantic compatibility
            True
            >>> product = ProductClass(name="Widget", price=29.99)
            >>> product.name
            'Widget'
            >>> result = product.compare_with(ProductClass(name="Widget", price=29.99))
            >>> result["overall_score"]
            1.0
        """
        # Delegate to ModelFactory for dynamic model creation
        from .model_factory import ModelFactory
        
        return ModelFactory.create_model_from_json(config, base_class=cls)

    @classmethod
    def _is_structured_field_type(cls, field_info) -> bool:
        """Check if a field represents a structured type that needs special handling.

        Args:
            field_info: Pydantic field info object

        Returns:
            True if the field is a List[StructuredModel] or StructuredModel type
        """
        return ConfigurationHelper.is_structured_field_type(field_info)

    @classmethod
    def _get_comparison_info(cls, field_name: str) -> ComparableField:
        """Extract comparison info from a field.

        Args:
            field_name: Name of the field to get comparison info for

        Returns:
            ComparableField object with comparison configuration
        """
        return ConfigurationHelper.get_comparison_info(cls, field_name)

    # Remove legacy ComparableField handling since ComparableField is now always a function
    # that returns proper Pydantic Fields
    pass

    # No special __init__ needed since ComparableField is now always a function
    # that returns proper Pydantic Fields
    pass

    @classmethod
    def _is_aggregate_field(cls, field_name: str) -> bool:
        """Check if field is marked for confusion matrix aggregation.

        Args:
            field_name: Name of the field to check

        Returns:
            True if the field is marked for aggregation, False otherwise
        """
        return ConfigurationHelper.is_aggregate_field(cls, field_name)

    def _is_truly_null(self, val: Any) -> bool:
        """Check if a value is truly null (None).

        Args:
            val: Value to check

        Returns:
            True if the value is None, False otherwise
        """
        return val is None

    def _should_use_hierarchical_structure(self, val: Any, field_name: str) -> bool:
        """Check if a list value should maintain hierarchical structure.

        For lists, we need to check if they should maintain hierarchical structure
        based on their field type configuration.

        Args:
            val: Value to check (typically a list)
            field_name: Name of the field being evaluated

        Returns:
            True if the value should use hierarchical structure, False otherwise
        """
        if isinstance(val, list):
            # Check if this field is configured as List[StructuredModel]
            field_info = self.__class__.model_fields.get(field_name)
            if field_info and self._is_structured_field_type(field_info):
                return True
        return False

    def _is_effectively_null_for_lists(self, val: Any) -> bool:
        """Check if a list value is effectively null (None or empty list).

        Args:
            val: Value to check

        Returns:
            True if the value is None or an empty list, False otherwise
        """
        return val is None or (isinstance(val, list) and len(val) == 0)

    def _is_effectively_null_for_primitives(self, val: Any) -> bool:
        """Check if a primitive value is effectively null.

        Treats empty strings and None as equivalent for string fields.

        Args:
            val: Value to check

        Returns:
            True if the value is None or an empty string, False otherwise
        """
        return val is None or (isinstance(val, str) and val == "")

    def _is_list_field(self, field_name: str) -> bool:
        """Check if a field is ANY list type.

        Args:
            field_name: Name of the field to check

        Returns:
            True if the field is a list type (List[str], List[StructuredModel], etc.)
        """
        field_info = self.__class__.model_fields.get(field_name)
        if not field_info:
            return False

        field_type = field_info.annotation
        # Handle Optional types and direct List types
        if hasattr(field_type, "__origin__"):
            origin = field_type.__origin__
            if origin is list or origin is List:
                return True
            elif origin is Union:  # Optional[List[...]] case
                args = field_type.__args__
                for arg in args:
                    if hasattr(arg, "__origin__") and (
                        arg.__origin__ is list or arg.__origin__ is List
                    ):
                        return True
        return False

    def _handle_list_field_dispatch(
        self, gt_val: Any, pred_val: Any, weight: float
    ) -> dict:
        """Handle list field comparison using match statements.

        Args:
            gt_val: Ground truth list value
            pred_val: Predicted list value
            weight: Field weight for scoring

        Returns:
            Comparison result dictionary
        """
        gt_effectively_null = self._is_effectively_null_for_lists(gt_val)
        pred_effectively_null = self._is_effectively_null_for_lists(pred_val)

        match (gt_effectively_null, pred_effectively_null):
            case (True, True):
                # Both None or empty lists → True Negative
                return {
                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                    "fields": {},
                    "raw_similarity_score": 1.0,
                    "similarity_score": 1.0,
                    "threshold_applied_score": 1.0,
                    "weight": weight,
                }
            case (True, False):
                # GT=None/empty, Pred=populated list → False Alarm
                pred_list = pred_val if isinstance(pred_val, list) else []
                fa_count = (
                    len(pred_list) if pred_list else 1
                )  # At least 1 FA for the field itself
                return {
                    "overall": {
                        "tp": 0,
                        "fa": fa_count,
                        "fd": 0,
                        "fp": fa_count,
                        "tn": 0,
                        "fn": 0,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case (False, True):
                # GT=populated list, Pred=None/empty → False Negative
                gt_list = gt_val if isinstance(gt_val, list) else []
                fn_count = (
                    len(gt_list) if gt_list else 1
                )  # At least 1 FN for the field itself
                return {
                    "overall": {
                        "tp": 0,
                        "fa": 0,
                        "fd": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": fn_count,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case _:
                # Both non-null and non-empty, return None to continue processing
                return None

    def _create_true_negative_result(self, weight: float) -> dict:
        """Create a true negative result.

        Args:
            weight: Field weight for scoring

        Returns:
            True negative result dictionary
        """
        return {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
            "fields": {},
            "raw_similarity_score": 1.0,
            "similarity_score": 1.0,
            "threshold_applied_score": 1.0,
            "weight": weight,
        }

    def _create_false_alarm_result(self, weight: float) -> dict:
        """Create a false alarm result.

        Args:
            weight: Field weight for scoring

        Returns:
            False alarm result dictionary
        """
        return {
            "overall": {"tp": 0, "fa": 1, "fd": 0, "fp": 1, "tn": 0, "fn": 0},
            "fields": {},
            "raw_similarity_score": 0.0,
            "similarity_score": 0.0,
            "threshold_applied_score": 0.0,
            "weight": weight,
        }

    def _create_false_negative_result(self, weight: float) -> dict:
        """Create a false negative result.

        Args:
            weight: Field weight for scoring

        Returns:
            False negative result dictionary
        """
        return {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 1},
            "fields": {},
            "raw_similarity_score": 0.0,
            "similarity_score": 0.0,
            "threshold_applied_score": 0.0,
            "weight": weight,
        }

    def _handle_struct_list_empty_cases(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        weight: float,
    ) -> dict:
        """Handle empty list cases with beautiful match statements.

        Args:
            gt_list: Ground truth list (may be None)
            pred_list: Predicted list (may be None)
            weight: Field weight for scoring

        Returns:
            Result dictionary if early exit needed, None if should continue processing
        """
        # Normalize None to empty lists for consistent handling
        gt_len = len(gt_list or [])
        pred_len = len(pred_list or [])

        match (gt_len, pred_len):
            case (0, 0):
                # Both empty lists → True Negative
                return {
                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                    "fields": {},
                    "raw_similarity_score": 1.0,
                    "similarity_score": 1.0,
                    "threshold_applied_score": 1.0,
                    "weight": weight,
                }
            case (0, pred_len):
                # GT empty, pred has items → False Alarms
                return {
                    "overall": {
                        "tp": 0,
                        "fa": pred_len,
                        "fd": 0,
                        "fp": pred_len,
                        "tn": 0,
                        "fn": 0,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case (gt_len, 0):
                # GT has items, pred empty → False Negatives
                return {
                    "overall": {
                        "tp": 0,
                        "fa": 0,
                        "fd": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": gt_len,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case _:
                # Both non-empty, continue processing
                return None

    def _calculate_object_level_metrics(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        match_threshold: float,
    ) -> tuple:
        """Calculate object-level metrics using Hungarian matching.

        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            match_threshold: Threshold for considering objects as matches

        Returns:
            Tuple of (object_metrics_dict, matched_pairs, matched_gt_indices, matched_pred_indices)
        """
        # Use Hungarian matching for OBJECT-LEVEL counts - OPTIMIZED: Single call gets all info
        hungarian_helper = HungarianHelper()
        hungarian_info = hungarian_helper.get_complete_matching_info(gt_list, pred_list)
        matched_pairs = hungarian_info["matched_pairs"]

        # Count OBJECTS, not individual fields
        tp_objects = 0  # Objects with similarity >= match_threshold
        fd_objects = 0  # Objects with similarity < match_threshold
        for gt_idx, pred_idx, similarity in matched_pairs:
            if similarity >= match_threshold:
                tp_objects += 1
            else:
                fd_objects += 1

        # Count unmatched objects
        matched_gt_indices = {idx for idx, _, _ in matched_pairs}
        matched_pred_indices = {idx for _, idx, _ in matched_pairs}
        fn_objects = len(gt_list) - len(matched_gt_indices)  # Unmatched GT objects
        fa_objects = len(pred_list) - len(
            matched_pred_indices
        )  # Unmatched pred objects

        # Build list-level metrics counting OBJECTS (not fields)
        object_level_metrics = {
            "tp": tp_objects,
            "fa": fa_objects,
            "fd": fd_objects,
            "fp": fa_objects + fd_objects,  # Total false positives
            "tn": 0,  # No true negatives at object level for non-empty lists
            "fn": fn_objects,
        }

        return (
            object_level_metrics,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
        )

    def _calculate_struct_list_similarity(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        info: "ComparableField",
    ) -> float:
        """Calculate raw similarity score for structured list.

        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            info: Field comparison info

        Returns:
            Raw similarity score between 0.0 and 1.0
        """
        if len(pred_list) > 0:
            match_result = self._compare_unordered_lists(
                gt_list, pred_list, info.comparator, info.threshold
            )
            return match_result.get("overall_score", 0.0)
        else:
            return 0.0

    # Necessary/sufficient field methods removed - no longer used

    def _compare_unordered_lists(
        self,
        list1: List[Any],
        list2: List[Any],
        comparator: BaseComparator,
        threshold: float,
    ) -> Dict[str, Any]:
        """Compare two lists as unordered collections using Hungarian matching.

        Args:
            list1: First list
            list2: Second list
            comparator: Comparator to use for item comparison
            threshold: Minimum score to consider a match

        Returns:
            Dictionary with confusion matrix metrics including:
            - tp: True positives (matches >= threshold)
            - fd: False discoveries (matches < threshold)
            - fa: False alarms (unmatched prediction items)
            - fn: False negatives (unmatched ground truth items)
            - fp: Total false positives (fd + fa)
            - overall_score: Similarity score for backward compatibility
        """
        return ComparisonHelper.compare_unordered_lists(
            list1, list2, comparator, threshold
        )

    def compare_field(self, field_name: str, other_value: Any) -> float:
        """Compare a single field with a value using the configured comparator.

        Args:
            field_name: Name of the field to compare
            other_value: Value to compare with

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Get our field value
        my_value = getattr(self, field_name)

        # If both values are StructuredModel instances, use recursive compare_with
        if isinstance(my_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            # Use compare_with for rich comparison
            comparison_result = my_value.compare_with(
                other_value,
                include_confusion_matrix=False,
                document_non_matches=False,
                evaluator_format=False,
                recall_with_fd=False,
            )
            # Apply field-level threshold if configured
            info = self._get_comparison_info(field_name)
            raw_score = comparison_result["overall_score"]
            return (
                raw_score
                if raw_score >= info.threshold or not info.clip_under_threshold
                else 0.0
            )

        # CRITICAL FIX: For lists, don't clip under threshold for partial matches
        if isinstance(my_value, list) and isinstance(other_value, list):
            # Get field info
            info = self._get_comparison_info(field_name)

            # Use the raw comparison result without threshold clipping for lists
            result = ComparisonHelper.compare_unordered_lists(
                my_value, other_value, info.comparator, info.threshold
            )

            # Return the overall score directly (don't clip based on threshold for lists)
            return result["overall_score"]

        # For other fields, use existing logic
        return ComparisonHelper.compare_field_with_threshold(
            self, field_name, other_value
        )

    def compare_field_raw(self, field_name: str, other_value: Any) -> float:
        """Compare a single field with a value WITHOUT applying thresholds.

        This version is used by the compare method to get raw similarity scores.

        Args:
            field_name: Name of the field to compare
            other_value: Value to compare with

        Returns:
            Raw similarity score between 0.0 and 1.0 without threshold filtering
        """
        # Get our field value
        my_value = getattr(self, field_name)

        # If both values are StructuredModel instances, use recursive compare_with
        if isinstance(my_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            # Use compare_with for rich comparison, but extract the raw score
            comparison_result = my_value.compare_with(
                other_value,
                include_confusion_matrix=False,
                document_non_matches=False,
                evaluator_format=False,
                recall_with_fd=False,
            )
            return comparison_result["overall_score"]

        # For non-StructuredModel fields, use existing logic
        return ComparisonHelper.compare_field_raw(self, field_name, other_value)

    def compare_recursive(self, other: "StructuredModel") -> dict:
        """The ONE clean recursive function that handles everything.

        Enhanced to capture BOTH confusion matrix metrics AND similarity scores
        in a single traversal to eliminate double traversal inefficiency.

        Args:
            other: Another instance of the same model to compare with

        Returns:
            Dictionary with clean hierarchical structure:
            - overall: TP, FP, TN, FN, FD, FA counts + similarity_score + all_fields_matched
            - fields: Recursive structure for each field with scores
            - non_matches: List of non-matching items
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

        for field_name in self.__class__.model_fields:
            if field_name == "extra_fields":
                continue

            gt_val = getattr(self, field_name)
            pred_val = getattr(other, field_name, None)

            # Enhanced dispatch returns both metrics AND scores
            field_result = self._dispatch_field_comparison(field_name, gt_val, pred_val)

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
                info = self._get_comparison_info(field_name)
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
        model_fields_for_comparison = set(self.__class__.model_fields.keys()) - {
            "extra_fields"
        }
        result["overall"]["all_fields_matched"] = len(threshold_matched_fields) == len(
            model_fields_for_comparison
        )

        return result

    def _dispatch_field_comparison(
        self, field_name: str, gt_val: Any, pred_val: Any
    ) -> dict:
        """Enhanced case-based dispatch using match statements for clean logic flow."""

        # Get field configuration for scoring
        info = self._get_comparison_info(field_name)
        weight = info.weight
        threshold = info.threshold

        # Check if this field is ANY list type (including Optional[List[str]], Optional[List[StructuredModel]], etc.)
        is_list_field = self._is_list_field(field_name)

        # Get null states and hierarchical needs
        gt_is_null = self._is_truly_null(gt_val)
        pred_is_null = self._is_truly_null(pred_val)
        gt_needs_hierarchy = self._should_use_hierarchical_structure(gt_val, field_name)
        pred_needs_hierarchy = self._should_use_hierarchical_structure(
            pred_val, field_name
        )

        # Handle list fields with match statements
        if is_list_field:
            list_result = self._handle_list_field_dispatch(gt_val, pred_val, weight)
            if list_result is not None:
                return list_result
            # If None returned, continue to regular type-based dispatch

        # Handle non-hierarchical primitive null cases with match statements
        if not (gt_needs_hierarchy or pred_needs_hierarchy):
            gt_effectively_null_prim = self._is_effectively_null_for_primitives(gt_val)
            pred_effectively_null_prim = self._is_effectively_null_for_primitives(
                pred_val
            )

            match (gt_effectively_null_prim, pred_effectively_null_prim):
                case (True, True):
                    return self._create_true_negative_result(weight)
                case (True, False):
                    return self._create_false_alarm_result(weight)
                case (False, True):
                    return self._create_false_negative_result(weight)
                case _:
                    # Both non-null, continue to type-based dispatch
                    pass

        # Type-based dispatch
        if isinstance(gt_val, (str, int, float)) and isinstance(
            pred_val, (str, int, float)
        ):
            return self._compare_primitive_with_scores(gt_val, pred_val, field_name)
        elif isinstance(gt_val, list) and isinstance(pred_val, list):
            # Check if this should be structured list
            if gt_val and isinstance(gt_val[0], StructuredModel):
                return self._compare_struct_list_with_scores(
                    gt_val, pred_val, field_name
                )
            else:
                return self._compare_primitive_list_with_scores(
                    gt_val, pred_val, field_name
                )
        elif isinstance(gt_val, list) and len(gt_val) == 0:
            # Handle empty GT list - check if it should be structured
            field_info = self.__class__.model_fields.get(field_name)
            if field_info and self._is_structured_field_type(field_info):
                # Empty structured list - should still return hierarchical structure
                return self._compare_struct_list_with_scores(
                    gt_val, pred_val, field_name
                )
            else:
                return self._compare_primitive_list_with_scores(
                    gt_val, pred_val, field_name
                )
        elif isinstance(pred_val, list) and len(pred_val) == 0:
            # Handle empty pred list - check if it should be structured
            field_info = self.__class__.model_fields.get(field_name)
            if field_info and self._is_structured_field_type(field_info):
                # Empty structured list - should still return hierarchical structure
                return self._compare_struct_list_with_scores(
                    gt_val, pred_val, field_name
                )
            else:
                return self._compare_primitive_list_with_scores(
                    gt_val, pred_val, field_name
                )
        elif isinstance(gt_val, StructuredModel) and isinstance(
            pred_val, StructuredModel
        ):
            # CRITICAL FIX: For StructuredModel fields, object-level metrics should be based on
            # object similarity, not rollup of nested field metrics

            # Get object-level similarity score
            raw_score = gt_val.compare(pred_val)  # Overall object similarity

            # Apply object-level binary classification based on threshold
            if raw_score >= threshold:
                # Object matches threshold -> True Positive
                object_metrics = {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
                threshold_applied_score = raw_score
            else:
                # Object below threshold -> False Discovery
                object_metrics = {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0}
                threshold_applied_score = (
                    0.0 if info.clip_under_threshold else raw_score
                )

            # Still generate nested field details for debugging, but don't roll them up
            nested_details = gt_val.compare_recursive(pred_val)["fields"]

            # Return structure with object-level metrics and nested field details kept separate
            return {
                "overall": {
                    **object_metrics,
                    "similarity_score": raw_score,
                    "all_fields_matched": raw_score >= threshold,
                },
                "fields": nested_details,  # Nested details available for debugging
                "raw_similarity_score": raw_score,
                "similarity_score": raw_score,
                "threshold_applied_score": threshold_applied_score,
                "weight": weight,
                "non_matches": [],  # Add empty non_matches for consistency
            }
        else:
            # Mismatched types
            return {
                "overall": {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0},
                "fields": {},
                "raw_similarity_score": 0.0,
                "similarity_score": 0.0,
                "threshold_applied_score": 0.0,
                "weight": weight,
            }

    def _compare_primitive_with_scores(
        self, gt_val: Any, pred_val: Any, field_name: str
    ) -> dict:
        """Enhanced primitive comparison that returns both metrics AND scores."""
        info = self.__class__._get_comparison_info(field_name)
        raw_similarity = info.comparator.compare(gt_val, pred_val)
        weight = info.weight
        threshold = info.threshold

        # For binary classification metrics, always use threshold
        if raw_similarity >= threshold:
            metrics = {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
            threshold_applied_score = raw_similarity
        else:
            metrics = {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0}
            # For score calculation, respect clip_under_threshold setting
            threshold_applied_score = (
                0.0 if info.clip_under_threshold else raw_similarity
            )

        # UNIFIED STRUCTURE: Always use 'overall' for metrics
        # 'fields' key omitted for primitive leaf nodes (semantic meaning: not a parent container)
        return {
            "overall": metrics,
            "raw_similarity_score": raw_similarity,
            "similarity_score": raw_similarity,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
        }

    def _compare_primitive_list_with_scores(
        self, gt_list: List[Any], pred_list: List[Any], field_name: str
    ) -> dict:
        """Enhanced primitive list comparison that returns both metrics AND scores with hierarchical structure.

        DESIGN DECISION: Universal Hierarchical Structure
        ===============================================
        This method returns a hierarchical structure {"overall": {...}, "fields": {...}} even for
        primitive lists (List[str], List[int], etc.) to maintain API consistency across all field types.

        Why this approach:
        - CONSISTENCY: All list fields use the same access pattern: cm["fields"][name]["overall"]
        - TEST COMPATIBILITY: Multiple test files expect this pattern for both primitive and structured lists
        - PREDICTABLE API: Consumers don't need to check field type before accessing metrics

        Trade-offs:
        - Creates vestigial "fields": {} objects for primitive lists that will never be populated
        - Slightly more verbose structure than necessary for leaf nodes
        - Architecturally less pure than type-based structure (primitives flat, structured hierarchical)

        Alternative considered but rejected:
        - Type-based structure where List[primitive] → flat, List[StructuredModel] → hierarchical
        - Would require updating multiple test files and consumer code to handle mixed access patterns
        - More architecturally pure but breaks backward compatibility

        Future consideration: If we ever refactor the entire confusion matrix API, we could move to
        type-based structure where the presence of "fields" key indicates structured vs primitive.
        """
        # Get field configuration
        info = self.__class__._get_comparison_info(field_name)
        weight = info.weight
        threshold = info.threshold

        # CRITICAL FIX: Handle None values before checking length
        # Convert None to empty list for consistent handling
        if gt_list is None:
            gt_list = []
        if pred_list is None:
            pred_list = []

        # Handle empty/null list cases first - FIXED: Empty lists should be TN=1
        if len(gt_list) == 0 and len(pred_list) == 0:
            # Both empty lists should be TN=1
            return {
                "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                "fields": {},  # Empty for primitive lists
                "raw_similarity_score": 1.0,  # Perfect match
                "similarity_score": 1.0,
                "threshold_applied_score": 1.0,
                "weight": weight,
            }
        elif len(gt_list) == 0:
            # GT empty, pred has items → False Alarms
            return {
                "overall": {
                    "tp": 0,
                    "fa": len(pred_list),
                    "fd": 0,
                    "fp": len(pred_list),
                    "tn": 0,
                    "fn": 0,
                },
                "fields": {},
                "raw_similarity_score": 0.0,
                "similarity_score": 0.0,
                "threshold_applied_score": 0.0,
                "weight": weight,
            }
        elif len(pred_list) == 0:
            # GT has items, pred empty → False Negatives
            return {
                "overall": {
                    "tp": 0,
                    "fa": 0,
                    "fd": 0,
                    "fp": 0,
                    "tn": 0,
                    "fn": len(gt_list),
                },
                "fields": {},
                "raw_similarity_score": 0.0,
                "similarity_score": 0.0,
                "threshold_applied_score": 0.0,
                "weight": weight,
            }

        # For primitive lists, use the comparison logic from _compare_unordered_lists
        # which properly handles the threshold-based matching
        comparator = info.comparator
        match_result = self._compare_unordered_lists(
            gt_list, pred_list, comparator, threshold
        )

        # Extract the counts from the match result
        tp = match_result.get("tp", 0)
        fd = match_result.get("fd", 0)
        fa = match_result.get("fa", 0)
        fn = match_result.get("fn", 0)

        # Use the overall_score from the match result for raw similarity
        raw_similarity = match_result.get("overall_score", 0.0)

        # CRITICAL FIX: For lists, we NEVER clip under threshold - partial matches are important
        threshold_applied_score = raw_similarity  # Always use raw score for lists

        # Return hierarchical structure expected by tests
        return {
            "overall": {"tp": tp, "fa": fa, "fd": fd, "fp": fa + fd, "tn": 0, "fn": fn},
            "fields": {},  # Empty for primitive lists - no nested structure
            "raw_similarity_score": raw_similarity,
            "similarity_score": raw_similarity,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
        }

    def _compare_struct_list_with_scores(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        field_name: str,
    ) -> dict:
        """Enhanced structural list comparison that returns both metrics AND scores.

        PHASE 2: Delegates to StructuredListComparator while maintaining identical behavior.
        """
        # Import here to avoid circular imports
        from .structured_list_comparator import StructuredListComparator

        # Create comparator and delegate
        comparator = StructuredListComparator(self)
        return comparator.compare_struct_list_with_scores(
            gt_list, pred_list, field_name
        )

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
        for field_name in self.__class__.model_fields:
            if field_name == "extra_fields":
                continue

            gt_val = getattr(self, field_name, None)
            pred_val = getattr(other, field_name, None)

            # Check nested StructuredModel objects
            if isinstance(gt_val, StructuredModel) and isinstance(
                pred_val, StructuredModel
            ):
                fa_count += gt_val._count_extra_fields_as_false_alarms(pred_val)

            # Check lists of StructuredModel objects
            elif (
                isinstance(gt_val, list)
                and isinstance(pred_val, list)
                and gt_val
                and isinstance(gt_val[0], StructuredModel)
                and pred_val
                and isinstance(pred_val[0], StructuredModel)
            ):
                # For lists, we need to match them up properly using Hungarian matching - OPTIMIZED: Single call gets all info
                # to avoid double-counting in cases where the list comparison already
                # handles unmatched items as FA. For now, let's recursively check each item.
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
                        fa_count += gt_item._count_extra_fields_as_false_alarms(
                            pred_item
                        )

                # For unmatched prediction items, count their extra fields too
                matched_pred_indices = {pred_idx for _, pred_idx, _ in matched_pairs}
                for pred_idx, pred_item in enumerate(pred_val):
                    if pred_idx not in matched_pred_indices and isinstance(
                        pred_item, StructuredModel
                    ):
                        # For unmatched items, we need a dummy GT to compare against
                        if gt_val:  # Use first GT item as template
                            dummy_gt = gt_val[0]
                            fa_count += dummy_gt._count_extra_fields_as_false_alarms(
                                pred_item
                            )
                        else:
                            # If no GT items, count all extra fields in this pred item
                            if hasattr(pred_item, "__pydantic_extra__"):
                                fa_count += len(pred_item.__pydantic_extra__)

        return fa_count

    def _aggregate_to_overall(self, field_result: dict, overall: dict) -> None:
        """Simple aggregation to overall metrics."""
        for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
            if isinstance(field_result, dict):
                if metric in field_result:
                    overall[metric] += field_result[metric]
                elif "overall" in field_result and metric in field_result["overall"]:
                    overall[metric] += field_result["overall"][metric]

    def _calculate_aggregate_metrics(self, result: dict) -> dict:
        """Calculate aggregate metrics for all nodes in the result tree.

        CRITICAL FIX: Enhanced deep nesting traversal to handle arbitrary nesting depth.
        The aggregate field contains the sum of all primitive field confusion matrices
        below that node in the tree. This provides universal field-level granularity.

        Args:
            result: Result from compare_recursive with hierarchical structure

        Returns:
            Modified result with 'aggregate' fields added at each level
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
                    processed_field = self._calculate_aggregate_metrics(field_result)
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

    def _add_derived_metrics_to_result(self, result: dict) -> dict:
        """Walk through result and add 'derived' fields with F1, precision, recall, accuracy.

        Args:
            result: Result from compare_recursive with basic TP, FP, FN, etc. metrics

        Returns:
            Modified result with 'derived' fields added at each level
        """
        if not isinstance(result, dict):
            return result

        # Make a copy to avoid modifying the original
        result_copy = result.copy()

        # Add derived metrics to 'overall' if it exists and has basic metrics
        if "overall" in result_copy and isinstance(result_copy["overall"], dict):
            overall = result_copy["overall"]
            if self._has_basic_metrics(overall):
                metrics_helper = MetricsHelper()
                overall["derived"] = metrics_helper.calculate_derived_metrics(overall)

                # Also add derived metrics to aggregate if it exists
                if "aggregate" in overall and self._has_basic_metrics(
                    overall["aggregate"]
                ):
                    overall["aggregate"]["derived"] = (
                        metrics_helper.calculate_derived_metrics(overall["aggregate"])
                    )

        # Add derived metrics to top-level aggregate if it exists
        if "aggregate" in result_copy and self._has_basic_metrics(
            result_copy["aggregate"]
        ):
            metrics_helper = MetricsHelper()
            result_copy["aggregate"]["derived"] = (
                metrics_helper.calculate_derived_metrics(result_copy["aggregate"])
            )

        # Recursively process 'fields' if it exists
        if "fields" in result_copy and isinstance(result_copy["fields"], dict):
            fields_copy = {}
            for field_name, field_result in result_copy["fields"].items():
                if isinstance(field_result, dict):
                    # Check if this is a hierarchical field (has overall/fields) or a unified structure field
                    if "overall" in field_result and "fields" in field_result:
                        # Hierarchical field - process recursively
                        fields_copy[field_name] = self._add_derived_metrics_to_result(
                            field_result
                        )
                    elif "overall" in field_result and self._has_basic_metrics(
                        field_result["overall"]
                    ):
                        # Unified structure field - add derived metrics to overall
                        field_copy = field_result.copy()
                        metrics_helper = MetricsHelper()
                        field_copy["overall"]["derived"] = (
                            metrics_helper.calculate_derived_metrics(
                                field_result["overall"]
                            )
                        )

                        # Also add derived metrics to aggregate if it exists
                        if "aggregate" in field_copy and self._has_basic_metrics(
                            field_copy["aggregate"]
                        ):
                            field_copy["aggregate"]["derived"] = (
                                metrics_helper.calculate_derived_metrics(
                                    field_copy["aggregate"]
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
                            metrics_helper.calculate_derived_metrics(legacy_metrics)
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

    def _has_basic_metrics(self, metrics_dict: dict) -> bool:
        """Check if a dictionary has basic confusion matrix metrics.

        Args:
            metrics_dict: Dictionary to check

        Returns:
            True if it has the basic metrics (tp, fp, fn, etc.)
        """
        basic_metrics = ["tp", "fp", "fn", "tn", "fa", "fd"]
        return all(metric in metrics_dict for metric in basic_metrics)

    def _classify_field_for_confusion_matrix(
        self, field_name: str, other_value: Any, threshold: float = None
    ) -> Dict[str, Any]:
        """Classify a field comparison according to the confusion matrix rules.
        
        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        Args:
            field_name: Name of the field being compared
            other_value: Value to compare with
            threshold: Threshold for matching (uses field's threshold if None)

        Returns:
            Dictionary with TP, FP, TN, FN, FD counts and derived metrics
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator
        calculator = ConfusionMatrixCalculator(self)
        return calculator.classify_field_for_confusion_matrix(field_name, other_value, threshold)

    def _calculate_list_confusion_matrix(
        self, field_name: str, other_list: List[Any]
    ) -> Dict[str, Any]:
        """Calculate confusion matrix for a list field, including nested field metrics.
        
        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        Args:
            field_name: Name of the list field being compared
            other_list: Predicted list to compare with

        Returns:
            Dictionary with:
            - Top-level TP, FP, TN, FN, FD, FA counts and derived metrics for the list field
            - nested_fields: Dict with metrics for individual fields within list items (e.g., "transactions.date")
            - non_matches: List of individual object-level non-matches for detailed analysis
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator
        calculator = ConfusionMatrixCalculator(self)
        return calculator.calculate_list_confusion_matrix(field_name, other_list)

    def _calculate_nested_field_metrics(
        self,
        list_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        threshold: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for individual fields within list items.
        
        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        THRESHOLD-GATED RECURSION: Only perform recursive field analysis for object pairs
        with similarity >= StructuredModel.match_threshold. Poor matches and unmatched
        items are treated as atomic units.

        Args:
            list_field_name: Name of the parent list field (e.g., "transactions")
            gt_list: Ground truth list of StructuredModel objects
            pred_list: Predicted list of StructuredModel objects
            threshold: Matching threshold (not used for threshold-gating)

        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics
            E.g., {"transactions.date": {...}, "transactions.description": {...}}
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator
        calculator = ConfusionMatrixCalculator(self)
        return calculator.calculate_nested_field_metrics(list_field_name, gt_list, pred_list, threshold)

    def _calculate_single_nested_field_metrics(
        self,
        parent_field_name: str,
        gt_nested: "StructuredModel",
        pred_nested: "StructuredModel",
        parent_is_aggregate: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for fields within a single nested StructuredModel.
        
        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        Args:
            parent_field_name: Name of the parent field (e.g., "address")
            gt_nested: Ground truth nested StructuredModel
            pred_nested: Predicted nested StructuredModel
            parent_is_aggregate: Whether the parent field should aggregate child metrics

        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics
            E.g., {"address.street": {...}, "address.city": {...}}
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator
        calculator = ConfusionMatrixCalculator(self)
        return calculator.calculate_single_nested_field_metrics(
            parent_field_name, gt_nested, pred_nested, parent_is_aggregate
        )

    def _collect_enhanced_non_matches(
        self, recursive_result: dict, other: "StructuredModel"
    ) -> List[Dict[str, Any]]:
        """Collect enhanced non-matches with object-level granularity.
        
        This method delegates to NonMatchCollector for the actual implementation.

        Args:
            recursive_result: Result from compare_recursive containing field comparison details
            other: The predicted StructuredModel instance

        Returns:
            List of non-match dictionaries with enhanced object-level information
        """
        from .non_match_collector import NonMatchCollector
        collector = NonMatchCollector(self)
        return collector.collect_enhanced_non_matches(recursive_result, other)

    def _collect_non_matches(
        self, other: "StructuredModel", base_path: str = ""
    ) -> List[NonMatchField]:
        """Collect non-matches for detailed analysis.
        
        This method delegates to NonMatchCollector for the actual implementation.

        Args:
            other: Other model to compare with
            base_path: Base path for field naming (e.g., "address")

        Returns:
            List of NonMatchField objects documenting non-matches
        """
        from .non_match_collector import NonMatchCollector
        collector = NonMatchCollector(self)
        return collector.collect_non_matches(other, base_path)

    def compare(self, other: "StructuredModel") -> float:
        """Compare this model with another and return a scalar similarity score.

        Returns the overall weighted average score regardless of sufficient/necessary field matching.
        This provides a more nuanced score for use in comparators.

        Args:
            other: Another instance of the same model to compare with

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # We'll calculate the overall weighted score directly instead of using compare_with
        # This ensures that sufficient/necessary field rules don't cause a zero score
        # when at least some fields match

        total_score = 0.0
        total_weight = 0.0

        for field_name in self.__class__.model_fields:
            # Skip the extra_fields attribute in comparison
            if field_name == "extra_fields":
                continue
            if hasattr(other, field_name):
                # Get field configuration
                info = self.__class__._get_comparison_info(field_name)
                # Use weight from ComparableField object
                weight = info.weight

                # Compare field values WITHOUT applying thresholds
                field_score = self.compare_field_raw(
                    field_name, getattr(other, field_name)
                )

                # Update total score
                total_score += field_score * weight
                total_weight += weight

        # Calculate overall score
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0

    def compare_with(
        self,
        other: "StructuredModel",
        include_confusion_matrix: bool = False,
        document_non_matches: bool = False,
        evaluator_format: bool = False,
        recall_with_fd: bool = False,
        add_derived_metrics: bool = True,
    ) -> Dict[str, Any]:
        """Compare this model with another instance using SINGLE TRAVERSAL optimization.

        Args:
            other: Another instance of the same model to compare with
            include_confusion_matrix: Whether to include confusion matrix calculations
            document_non_matches: Whether to document non-matches for analysis
            evaluator_format: Whether to format results for the evaluator
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                            If False, use traditional recall (TP/(TP+FN))
            add_derived_metrics: Whether to add derived metrics to confusion matrix

        Returns:
            Dictionary with comparison results including:
            - field_scores: Scores for each field
            - overall_score: Weighted average score
            - all_fields_matched: Whether all fields matched
            - confusion_matrix: (optional) Confusion matrix data if requested
            - non_matches: (optional) Non-match documentation if requested
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
            confusion_matrix = recursive_result

            # Add universal aggregate metrics to all nodes
            confusion_matrix = self._calculate_aggregate_metrics(confusion_matrix)

            # Add derived metrics if requested
            if add_derived_metrics:
                confusion_matrix = self._add_derived_metrics_to_result(confusion_matrix)

            result["confusion_matrix"] = confusion_matrix

        # Add optional non-match documentation
        if document_non_matches:
            # Use NonMatchCollector for enhanced object-level non-matches
            from .non_match_collector import NonMatchCollector
            collector = NonMatchCollector(self)
            non_matches = collector.collect_enhanced_non_matches(recursive_result, other)
            result["non_matches"] = non_matches

        # If evaluator_format is requested, transform the result
        if evaluator_format:
            return self._format_for_evaluator(result, other, recall_with_fd)

        return result

    def _convert_score_to_binary_metrics(
        self, score: float, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Convert similarity score to binary classification metrics using MetricsHelper.

        Args:
            score: Similarity score [0-1]
            threshold: Threshold for considering a match

        Returns:
            Dictionary with TP, FP, FN, TN counts converted to metrics
        """
        metrics_helper = MetricsHelper()
        return metrics_helper.convert_score_to_binary_metrics(score, threshold)

    def _format_for_evaluator(
        self,
        result: Dict[str, Any],
        other: "StructuredModel",
        recall_with_fd: bool = False,
    ) -> Dict[str, Any]:
        """Format comparison results for evaluator compatibility.

        Args:
            result: Standard comparison result from compare_with
            other: The other model being compared
            recall_with_fd: Whether to include FD in recall denominator

        Returns:
            Dictionary in evaluator format with overall, fields, confusion_matrix
        """
        return EvaluatorFormatHelper.format_for_evaluator(
            self, result, other, recall_with_fd
        )

    def _calculate_list_item_metrics(
        self,
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
        return EvaluatorFormatHelper.calculate_list_item_metrics(
            field_name, gt_list, pred_list, recall_with_fd
        )

    @classmethod
    def model_json_schema(cls, **kwargs):
        """Override to add model-level comparison metadata.

        Extends the standard Pydantic JSON schema with comparison metadata
        at the field level.

        Args:
            **kwargs: Arguments to pass to the parent method

        Returns:
            JSON schema with added comparison metadata
        """
        schema = super().model_json_schema(**kwargs)

        # Add comparison metadata to each field in the schema
        for field_name, field_info in cls.model_fields.items():
            if field_name == "extra_fields":
                continue

            # Get the schema property for this field
            if field_name not in schema.get("properties", {}):
                continue

            field_props = schema["properties"][field_name]

            # Since ComparableField is now always a function, check for json_schema_extra
            if hasattr(field_info, "json_schema_extra") and callable(
                field_info.json_schema_extra
            ):
                # Fallback: Check for json_schema_extra function
                temp_schema = {}
                field_info.json_schema_extra(temp_schema)

                if "x-comparison" in temp_schema:
                    # Copy the comparison metadata from the temp schema to the real schema
                    field_props["x-comparison"] = temp_schema["x-comparison"]

        return schema
