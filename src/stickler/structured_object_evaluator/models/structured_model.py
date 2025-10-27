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
from .non_match_field import NonMatchField, NonMatchType
from .hungarian_helper import HungarianHelper
from .non_matches_helper import NonMatchesHelper
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
        from pydantic import create_model
        from .field_converter import convert_fields_config, validate_fields_config
        
        # Validate configuration structure
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        if "fields" not in config:
            raise ValueError("Configuration must contain 'fields' key")
        
        fields_config = config["fields"]
        if not isinstance(fields_config, dict) or len(fields_config) == 0:
            raise ValueError("'fields' must be a non-empty dictionary")
        
        # Validate all field configurations before proceeding (including nested schema validation)
        try:
            from .field_converter import get_global_converter

            converter = get_global_converter()
            
            # First validate basic field configurations
            validate_fields_config(fields_config)
            
            # Then validate nested schema rules
            for field_name, field_config in fields_config.items():
                converter.validate_nested_field_schema(field_name, field_config)
                
        except ValueError as e:
            raise ValueError(f"Invalid field configuration: {e}")
        
        # Extract model configuration
        model_name = config.get("model_name", "DynamicModel")
        match_threshold = config.get("match_threshold", 0.7)
        
        # Validate model name
        if not isinstance(model_name, str) or not model_name.isidentifier():
            raise ValueError(
                f"model_name must be a valid Python identifier, got: {model_name}"
            )
        
        # Validate match threshold
        if not isinstance(match_threshold, (int, float)) or not (
            0.0 <= match_threshold <= 1.0
        ):
            raise ValueError(
                f"match_threshold must be a number between 0.0 and 1.0, got: {match_threshold}"
            )
        
        # Convert field configurations to Pydantic field definitions
        try:
            field_definitions = convert_fields_config(fields_config)
        except ValueError as e:
            raise ValueError(f"Error converting field configurations: {e}")
        
        # Create the dynamic model extending StructuredModel
        try:
            DynamicClass = create_model(
                model_name,
                __base__=cls,  # Extend StructuredModel
                **field_definitions,
            )
        except Exception as e:
            raise ValueError(f"Error creating dynamic model: {e}")
        
        # Set class-level attributes
        DynamicClass.match_threshold = match_threshold
        
        # Add configuration metadata for debugging/introspection
        DynamicClass._model_config = config
        
        return DynamicClass
    
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
        """Check if a list value is effectively null (None, empty list, or empty string).
        
        For list fields, empty strings should be treated as equivalent to empty lists
        since both represent the absence of data.
        
        Args:
            val: Value to check
            
        Returns:
            True if the value is None, an empty list, or an empty string, False otherwise
        """
        return val is None or (isinstance(val, list) and len(val) == 0) or (isinstance(val, str) and val == "")

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

    def _get_nested_field_count(self, field_name: str) -> int:
        """Get the count of nested fields for List[StructuredModel] types.
        
        Args:
            field_name: Name of the field to analyze
            
        Returns:
            Number of nested fields in the StructuredModel, or 0 if not a hierarchical field
        """
        if not field_name:
            return 0
            
        field_info = self.__class__.model_fields.get(field_name)
        if not field_info or not self._is_structured_field_type(field_info):
            return 0
            
        from typing import get_origin, get_args, Union
        field_type = field_info.annotation if hasattr(field_info, 'annotation') else None
        if not field_type:
            return 0
            
        # Handle Union types (Optional[List[StructuredModel]])
        if get_origin(field_type) is Union:
            # Find the List type in the Union
            for arg in get_args(field_type):
                if get_origin(arg) is list:
                    field_type = arg
                    break
        
        # Check if it's a List[StructuredModel]
        if get_origin(field_type) is list:
            args = get_args(field_type)
            if args and hasattr(args[0], 'model_fields'):
                nested_model_class = args[0]
                # Count non-extra_fields
                return len([f for f in nested_model_class.model_fields.keys() if f != 'extra_fields'])
        
        return 0

    def _create_true_negative_result(self, weight: float, field_name: str = None) -> dict:
        """Create a true negative result.
        
        Args:
            weight: Field weight for scoring
            field_name: Name of the field (for hierarchical field detection)
            
        Returns:
            True negative result dictionary
        """
        tn_count = 1
        
        # For hierarchical fields, count nested field contributions
        nested_field_count = self._get_nested_field_count(field_name)
        if nested_field_count > 0:
            tn_count += nested_field_count  # Both GT and Pred are None, so count nested fields as TN
        
        return {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": tn_count, "fn": 0},
            "fields": {},
            "raw_similarity_score": 1.0,
            "similarity_score": 1.0,
            "threshold_applied_score": 1.0,
            "weight": weight,
        }

    def _create_false_alarm_result(self, weight: float, field_name: str = None, pred_val: Any = None) -> dict:
        """Create a false alarm result.
        
        Args:
            weight: Field weight for scoring
            field_name: Name of the field (for hierarchical field detection)
            pred_val: Predicted value (for counting nested fields)
            
        Returns:
            False alarm result dictionary
        """
        fa_count = 1
        fp_count = 1
        
        # For hierarchical fields, count nested field contributions
        nested_field_count = self._get_nested_field_count(field_name)
        if nested_field_count > 0 and pred_val is not None:
            # Count the list items and their nested fields
            if isinstance(pred_val, list):
                list_length = len(pred_val)
                fa_count += list_length * nested_field_count
                fp_count += list_length * nested_field_count
        
        return {
            "overall": {"tp": 0, "fa": fa_count, "fd": 0, "fp": fp_count, "tn": 0, "fn": 0},
            "fields": {},
            "raw_similarity_score": 0.0,
            "similarity_score": 0.0,
            "threshold_applied_score": 0.0,
            "weight": weight,
        }

    def _create_false_negative_result(self, weight: float, field_name: str = None, gt_val: Any = None) -> dict:
        """Create a false negative result.
        
        Args:
            weight: Field weight for scoring
            field_name: Name of the field (for hierarchical field detection)
            gt_val: Ground truth value (for counting nested fields)
            
        Returns:
            False negative result dictionary
        """
        fn_count = 1
        
        # For hierarchical fields, count nested field contributions
        nested_field_count = self._get_nested_field_count(field_name)
        if nested_field_count > 0 and gt_val is not None:
            # Count the list items and their nested fields
            if isinstance(gt_val, list):
                list_length = len(gt_val)
                fn_count += list_length * nested_field_count
        
        return {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": fn_count},
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
                    return self._create_true_negative_result(weight, field_name)
                case (True, False):
                    return self._create_false_alarm_result(weight, field_name, pred_val)
                case (False, True):
                    return self._create_false_negative_result(weight, field_name, gt_val)
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
    
    def _collect_all_primitive_metrics(self, node: dict, path: str = "") -> dict:
        """Collect ALL primitive field metrics from entire subtree.
        
        This method performs a comprehensive traversal to find all primitive confusion matrix
        metrics, including those from Hungarian matching results and deeply nested structures.
        It uses a comprehensive approach to ensure all metrics are collected.
        
        CRITICAL FIX: Now properly handles both 'nested_fields' from Hungarian matching results
        and regular 'fields' structures to ensure all primitive metrics are included in aggregates.
        
        Args:
            node: Dictionary node to traverse
            path: Current path for debugging/logging
            
        Returns:
            Dictionary with aggregated primitive counts: {tp, fa, fd, fp, tn, fn}
        """
        total_counts = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
        
        if not isinstance(node, dict):
            return total_counts
        
        # Check if this node has subfields
        has_subfields = ("fields" in node and 
                        isinstance(node["fields"], dict) and 
                        len(node["fields"]) > 0)
        
        # Check for nested_fields from Hungarian matching (legacy structure)
        has_nested_fields = ("nested_fields" in node and 
                           isinstance(node["nested_fields"], dict) and 
                           len(node["nested_fields"]) > 0)
        
        # CRITICAL FIX: For nodes with subfields, we need to recursively collect from all children
        if has_subfields:
            # This is a parent node - collect from children to avoid double counting
            for field_name, field_data in node["fields"].items():
                if isinstance(field_data, dict):
                    child_counts = self._collect_all_primitive_metrics(field_data, f"{path}.{field_name}" if path else field_name)
                    for metric in total_counts:
                        total_counts[metric] += child_counts[metric]
        
        # CRITICAL FIX: For nodes with nested_fields (Hungarian matching results), 
        # we should use nested_fields instead of fields for more accurate metrics
        if has_nested_fields:
            # This is a Hungarian matching result node - collect from nested_fields
            # These contain the individual field metrics from matched pairs and include TN counts
            for nested_field_name, nested_field_data in node["nested_fields"].items():
                if isinstance(nested_field_data, dict) and self._has_basic_metrics(nested_field_data):
                    # These are primitive metrics from individual fields within matched objects
                    for metric in total_counts:
                        total_counts[metric] += nested_field_data.get(metric, 0)
            # If we have nested_fields, don't also collect from fields to avoid double counting
            return total_counts
        
        # If this is a leaf node (no subfields and no nested_fields), collect its metrics
        if not has_subfields and not has_nested_fields:
            # CRITICAL FIX: For aggregate calculation, prefer "aggregate" section over "overall"
            # This ensures we collect metrics from ALL matches, not just those above threshold
            if "aggregate" in node and isinstance(node["aggregate"], dict) and self._has_basic_metrics(node["aggregate"]):
                # Hierarchical leaf node with aggregate: collect from aggregate (includes all matches)
                aggregate = node["aggregate"]
                for metric in total_counts:
                    total_counts[metric] += aggregate.get(metric, 0)
            elif "overall" in node and isinstance(node["overall"], dict) and self._has_basic_metrics(node["overall"]):
                # Hierarchical leaf node: collect from overall (fallback for nodes without aggregate)
                overall = node["overall"]
                for metric in total_counts:
                    total_counts[metric] += overall.get(metric, 0)
            elif self._has_basic_metrics(node):
                # Flat leaf node: collect directly
                for metric in total_counts:
                    total_counts[metric] += node.get(metric, 0)
        
        return total_counts

    def _calculate_aggregate_metrics(self, result: dict) -> dict:
        """Calculate aggregate metrics for all nodes in the result tree.
        
        CRITICAL FIX: Now uses comprehensive primitive metric collection to ensure
        all leaf-level metrics are included in aggregates, including Hungarian matching results.
        
        Args:
            result: Result from compare_recursive with hierarchical structure
            
        Returns:
            Modified result with 'aggregate' fields added at each level
        """
        if not isinstance(result, dict):
            return result
        
        # Make a copy to avoid modifying the original
        result_copy = result.copy()
        
        # Recursively process 'fields' first to get child aggregates
        if "fields" in result_copy and isinstance(result_copy["fields"], dict):
            fields_copy = {}
            for field_name, field_result in result_copy["fields"].items():
                if isinstance(field_result, dict):
                    # Recursively calculate aggregate for child field
                    processed_field = self._calculate_aggregate_metrics(field_result)
                    fields_copy[field_name] = processed_field
                else:
                    # Non-dict field - keep as is
                    fields_copy[field_name] = field_result
            result_copy["fields"] = fields_copy
        
        # CRITICAL FIX: Use comprehensive primitive metric collection
        # This ensures ALL primitive metrics are included, including Hungarian matching results
        aggregate_metrics = self._collect_all_primitive_metrics(result_copy)
        
        # Only add aggregate if this node has subfields (non-leaf nodes)
        has_subfields = ("fields" in result_copy and 
                        isinstance(result_copy["fields"], dict) and 
                        len(result_copy["fields"]) > 0)
        
        if has_subfields or any(aggregate_metrics.values()):
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
        
        Args:
            field_name: Name of the field being compared
            other_value: Value to compare with
            threshold: Threshold for matching (uses field's threshold if None)
            
        Returns:
            Dictionary with TP, FP, TN, FN, FD counts and derived metrics
        """
        # Get field values
        gt_value = getattr(self, field_name)
        pred_value = other_value
        
        # Get field configuration 
        info = self.__class__._get_comparison_info(field_name)
        if threshold is None:
            threshold = info.threshold
        comparator = info.comparator
        
        # Determine if values are null
        gt_is_null = FieldHelper.is_null_value(gt_value)
        pred_is_null = FieldHelper.is_null_value(pred_value)
        
        # Calculate similarity if both aren't null
        similarity = None
        if not gt_is_null and not pred_is_null:
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
            tn_count = 1
            # CRITICAL FIX: For hierarchical fields, count nested field contributions
            nested_field_count = self._get_nested_field_count(field_name)
            if nested_field_count > 0:
                tn_count += nested_field_count
            result = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": tn_count, "fn": 0}
        elif gt_is_null and not pred_is_null:
            # FA: GT null, prediction non-null (False Alarm)
            fa_count = 1
            fp_count = 1
            # CRITICAL FIX: For hierarchical fields, count nested field contributions
            nested_field_count = self._get_nested_field_count(field_name)
            if nested_field_count > 0 and isinstance(pred_value, list):
                list_length = len(pred_value)
                fa_count += list_length * nested_field_count
                fp_count += list_length * nested_field_count
            result = {"tp": 0, "fa": fa_count, "fd": 0, "fp": fp_count, "tn": 0, "fn": 0}
        elif not gt_is_null and pred_is_null:
            # FN: GT non-null, prediction null
            fn_count = 1
            # CRITICAL FIX: For hierarchical fields, count nested field contributions
            nested_field_count = self._get_nested_field_count(field_name)
            if nested_field_count > 0 and isinstance(gt_value, list):
                list_length = len(gt_value)
                fn_count += list_length * nested_field_count
            result = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": fn_count}
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
    
    def _calculate_list_confusion_matrix(
        self, field_name: str, other_list: List[Any]
    ) -> Dict[str, Any]:
        """Calculate confusion matrix for a list field, including nested field metrics.
        
        Args:
            field_name: Name of the list field being compared
            other_list: Predicted list to compare with
            
        Returns:
            Dictionary with:
            - Top-level TP, FP, TN, FN, FD, FA counts and derived metrics for the list field
            - nested_fields: Dict with metrics for individual fields within list items (e.g., "transactions.date")
            - non_matches: List of individual object-level non-matches for detailed analysis
        """
        gt_list = getattr(self, field_name)
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
            info = self.__class__._get_comparison_info(field_name)
            comparator = info.comparator
            threshold = info.threshold
            
            # Reuse existing Hungarian matching logic
            match_result = self._compare_unordered_lists(
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
                nested_metrics, hierarchical_fields = self._calculate_nested_field_metrics(
                    field_name, gt_list, pred_list, threshold
                )
                result["nested_fields"] = nested_metrics
                
                # Also preserve hierarchical fields structure if available
                if hierarchical_fields:
                    if "fields" not in result:
                        result["fields"] = {}
                    result["fields"].update(hierarchical_fields)

        # For List[StructuredModel], we should NOT aggregate nested fields to list level
        # List level metrics represent object-level matches from Hungarian algorithm
        # Nested field metrics represent field-level matches within those objects
        # They are separate concerns and should not be aggregated
        
        # Convert nested_fields to hierarchical fields structure for deep nesting support
        if "nested_fields" in result and result["nested_fields"]:
            hierarchical_fields = self._convert_nested_fields_to_hierarchical(
                result["nested_fields"], field_name
            )
            if hierarchical_fields:
                result["fields"] = hierarchical_fields

        
        # Add derived metrics  
        metrics_helper = MetricsHelper()
        result["derived"] = metrics_helper.calculate_derived_metrics(result)
        
        return result
    
    def _convert_nested_fields_to_hierarchical(
        self, nested_fields: Dict[str, Dict[str, Any]], parent_field_name: str
    ) -> Dict[str, Any]:
        """Convert flat nested_fields structure to hierarchical fields structure.
        
        Args:
            nested_fields: Flat structure like {"products.attributes.name": {...}, "products.attributes.properties.key": {...}}
            parent_field_name: Name of the parent field (e.g., "products")
            
        Returns:
            Hierarchical structure with nested "fields" sections
        """
        hierarchical = {}
        
        # First, collect all the paths and group them by their immediate parent
        path_groups = {}
        
        for field_path, metrics in nested_fields.items():
            # Remove the parent field name prefix to get the relative path
            if field_path.startswith(f"{parent_field_name}."):
                relative_path = field_path[len(f"{parent_field_name}."):]
                path_parts = relative_path.split(".")
                
                # Group by immediate parent path
                if len(path_parts) == 1:
                    # This is a direct child field
                    field_name = path_parts[0]
                    hierarchical[field_name] = {
                        "overall": metrics.copy(),
                        "aggregate": metrics.copy()
                    }
                else:
                    # This is a nested field - group by immediate parent
                    immediate_parent = path_parts[0]
                    remaining_path = ".".join(path_parts[1:])
                    
                    if immediate_parent not in path_groups:
                        path_groups[immediate_parent] = {}
                    
                    # Store with the remaining path
                    path_groups[immediate_parent][f"{immediate_parent}.{remaining_path}"] = metrics
        
        # Now recursively build the hierarchical structure for grouped paths
        for parent_field, child_paths in path_groups.items():
            if parent_field not in hierarchical:
                # Create the parent field structure
                hierarchical[parent_field] = {
                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
                    "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
                }
                
                # Aggregate metrics from all child paths for the parent
                for child_path, child_metrics in child_paths.items():
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        hierarchical[parent_field]["overall"][metric] += child_metrics.get(metric, 0)
                        hierarchical[parent_field]["aggregate"][metric] += child_metrics.get(metric, 0)
            
            # Recursively convert child paths
            child_hierarchical = self._convert_nested_fields_to_hierarchical(child_paths, parent_field)
            if child_hierarchical:
                hierarchical[parent_field]["fields"] = child_hierarchical
        
        return hierarchical
    
    def _calculate_nested_field_metrics(
        self,
        list_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        threshold: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for individual fields within list items.
        
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
        nested_metrics = {}
        
        if not gt_list or not isinstance(gt_list[0], StructuredModel):
            return nested_metrics
        
        # Get the model class from the first item
        model_class = gt_list[0].__class__
        
        # CRITICAL FIX: Use field's threshold, not class's match_threshold
        # Get the field info from the parent object to use the correct threshold
        parent_field_info = self.__class__._get_comparison_info(list_field_name)
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
                            list_classification = (
                                gt_item._calculate_list_confusion_matrix(
                                    field_name, pred_value
                                )
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
                            field_classification = (
                                gt_item._classify_field_for_confusion_matrix(
                                    field_name,
                                    pred_value,
                                    None,  # Use field's own threshold
                                )
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
                            list_classification = (
                                gt_item._calculate_list_confusion_matrix(
                                    field_name, dummy_empty_list
                                )
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
                                list_classification = (
                                    dummy_gt_item._calculate_list_confusion_matrix(
                                        field_name, pred_value
                                    )
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
        
        # Also collect hierarchical fields structures from recursive list comparisons
        hierarchical_fields = {}
        
        # Re-process the matched pairs to collect hierarchical structures
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
                    for field_name in model_class.model_fields:
                        if field_name == "extra_fields":
                            continue
                            
                        gt_value = getattr(gt_item, field_name, None)
                        pred_value = getattr(pred_item, field_name, None)
                        
                        # Check if this field is a List[StructuredModel] that needs recursive processing
                        if (
                            isinstance(gt_value, list)
                            and isinstance(pred_value, list)
                            and gt_value
                            and isinstance(gt_value[0], StructuredModel)
                        ):
                            # Get the hierarchical structure from the recursive call
                            list_classification = (
                                gt_item._calculate_list_confusion_matrix(
                                    field_name, pred_value
                                )
                            )
                            
                            # Preserve the hierarchical fields structure
                            if "fields" in list_classification and list_classification["fields"]:
                                if field_name not in hierarchical_fields:
                                    hierarchical_fields[field_name] = {}
                                
                                # Merge the hierarchical structure
                                for sub_field_name, sub_field_data in list_classification["fields"].items():
                                    if sub_field_name not in hierarchical_fields[field_name]:
                                        hierarchical_fields[field_name][sub_field_name] = {
                                            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
                                            "aggregate": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
                                        }
                                        if "fields" in sub_field_data:
                                            hierarchical_fields[field_name][sub_field_name]["fields"] = sub_field_data["fields"]
                                    
                                    # Aggregate the metrics
                                    for section in ["overall", "aggregate"]:
                                        if section in sub_field_data:
                                            for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                                                hierarchical_fields[field_name][sub_field_name][section][metric] += sub_field_data[section].get(metric, 0)
        
        return nested_metrics, hierarchical_fields
    
    def _calculate_single_nested_field_metrics(self, parent_field_name: str, gt_nested: 'StructuredModel', 
                        pred_nested: 'StructuredModel') -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for fields within a single nested StructuredModel.
        
        Args:
            parent_field_name: Name of the parent field (e.g., "address")
            gt_nested: Ground truth nested StructuredModel
            pred_nested: Predicted nested StructuredModel
            
        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics
            E.g., {"address.street": {...}, "address.city": {...}}
        """
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
        


        # For each field in the nested model
        for field_name in gt_nested.__class__.model_fields:
            if field_name == "extra_fields":
                continue
                
            nested_field_path = f"{parent_field_name}.{field_name}"
            

            
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
                list_metrics = gt_nested._calculate_list_confusion_matrix(
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
                field_classification = gt_nested._classify_field_for_confusion_matrix(
                    field_name, pred_value
                )
                
                # Store the metrics for this nested field
                nested_metrics[nested_field_path] = field_classification

                # Recursively calculate metrics for deeper nesting
                deeper_metrics = self._calculate_single_nested_field_metrics(
                    nested_field_path, gt_value, pred_value)
                nested_metrics.update(deeper_metrics)



        return nested_metrics
    
    

    
    def _collect_enhanced_non_matches(self, recursive_result: dict, other: 'StructuredModel') -> List[Dict[str, Any]]:
        """Collect enhanced non-matches with object-level granularity.
        
        Args:
            recursive_result: Result from compare_recursive containing field comparison details
            other: The predicted StructuredModel instance
            
        Returns:
            List of non-match dictionaries with enhanced object-level information
        """
        all_non_matches = []
        
        # Walk through the recursive result and collect non-matches
        for field_name, field_result in recursive_result.get("fields", {}).items():
            gt_val = getattr(self, field_name)
            pred_val = getattr(other, field_name, None)
            
            # Check if this is a list field that should use object-level collection
            if (
                isinstance(gt_val, list)
                and isinstance(pred_val, list)
                and gt_val
                and isinstance(gt_val[0], StructuredModel)
            ):
                # Use NonMatchesHelper for object-level collection
                helper = NonMatchesHelper()
                object_non_matches = helper.collect_list_non_matches(
                    field_name, gt_val, pred_val
                )
                all_non_matches.extend(object_non_matches)
                
                # NEW: Also collect field-level non-matches from within matched objects
                # Get the Hungarian matching information to find matched pairs
                hungarian_helper = HungarianHelper()
                if gt_val and pred_val:
                    hungarian_info = hungarian_helper.get_complete_matching_info(gt_val, pred_val)
                    matched_pairs = hungarian_info["matched_pairs"]
                    
                    # For each matched pair, collect field-level non-matches from the individual comparison
                    for gt_idx, pred_idx, similarity_score in matched_pairs:
                        if gt_idx < len(gt_val) and pred_idx < len(pred_val):
                            gt_item = gt_val[gt_idx]
                            pred_item = pred_val[pred_idx]
                            
                            # Get the match threshold for this object type
                            match_threshold = getattr(gt_item.__class__, 'match_threshold', 0.7)
                            
                            # Only collect field-level non-matches if this pair is above threshold (TP)
                            # or if it's below threshold but we want to see why it failed (FD)
                            if similarity_score >= match_threshold or similarity_score > 0.0:
                                # Perform individual comparison to get field-level details
                                individual_result = gt_item.compare_with(
                                    pred_item, 
                                    include_confusion_matrix=True, 
                                    document_non_matches=True
                                )
                                
                                # Collect field-level non-matches from the individual comparison
                                if "non_matches" in individual_result:
                                    for field_nm in individual_result["non_matches"]:
                                        # Prefix the field path with the list index
                                        indexed_field_path = f"{field_name}[{gt_idx}].{field_nm['field_path']}"
                                        field_nm_copy = field_nm.copy()
                                        field_nm_copy["field_path"] = indexed_field_path
                                        all_non_matches.append(field_nm_copy)
            
            # Handle null list cases  
            elif (
                (gt_val is None or (isinstance(gt_val, list) and len(gt_val) == 0))
                and isinstance(pred_val, list)
                and len(pred_val) > 0
            ):
                # GT empty, pred has items → use helper for FA entries
                helper = NonMatchesHelper()
                null_non_matches = helper.add_non_matches_for_null_cases(
                    field_name, gt_val, pred_val
                )
                all_non_matches.extend(null_non_matches)
                
            elif (
                isinstance(gt_val, list)
                and len(gt_val) > 0
                and (
                    pred_val is None
                    or (isinstance(pred_val, list) and len(pred_val) == 0)
                )
            ):
                # GT has items, pred empty → use helper for FN entries
                helper = NonMatchesHelper()
                null_non_matches = helper.add_non_matches_for_null_cases(
                    field_name, gt_val, pred_val
                )
                all_non_matches.extend(null_non_matches)
                
            else:
                # Use existing field-level logic for non-list fields
                # Extract metrics from field result to determine non-match type
                if isinstance(field_result, dict) and "overall" in field_result:
                    metrics = field_result["overall"]
                elif isinstance(field_result, dict):
                    metrics = field_result
                else:
                    continue  # Skip if we can't extract metrics
                
                # NEW: Collect leaf-level non_matches for primitive fields
                helper = NonMatchesHelper()
                
                # Check if this is a primitive field (not a StructuredModel or list)
                if not isinstance(gt_val, (StructuredModel, list)) and not isinstance(pred_val, (StructuredModel, list)):
                    # This is a primitive field - collect leaf-level non_matches ONLY
                    leaf_non_matches = helper.collect_primitive_field_non_matches(
                        field_name, gt_val, pred_val, field_result
                    )
                    all_non_matches.extend(leaf_non_matches)
                    # Skip object-level entries for primitive fields to avoid duplication
                
                # Check if this is a primitive list field
                elif (isinstance(gt_val, list) and isinstance(pred_val, list) and 
                      (not gt_val or not isinstance(gt_val[0], StructuredModel))):
                    # This is a primitive list - collect leaf-level non_matches ONLY
                    leaf_non_matches = helper.collect_primitive_list_non_matches(
                        field_name, gt_val, pred_val, field_result
                    )
                    all_non_matches.extend(leaf_non_matches)
                    # Skip object-level entries for primitive lists to avoid duplication
                
                else:
                    # For non-primitive fields, create object-level non-match entries (legacy format for backward compatibility)
                    if metrics.get("fa", 0) > 0:  # False Alarm
                        entry = {
                            "field_path": field_name,
                            "non_match_type": NonMatchType.FALSE_ALARM,  # Use enum value
                            "ground_truth_value": gt_val,
                            "prediction_value": pred_val,
                        "details": {"reason": "unmatched prediction"},
                        }
                        all_non_matches.append(entry)
                    elif metrics.get("fn", 0) > 0:  # False Negative
                        entry = {
                            "field_path": field_name,
                            "non_match_type": NonMatchType.FALSE_NEGATIVE,  # Use enum value
                            "ground_truth_value": gt_val,
                            "prediction_value": pred_val,
                        "details": {"reason": "unmatched ground truth"},
                        }
                        all_non_matches.append(entry)
                    elif metrics.get("fd", 0) > 0:  # False Discovery
                        similarity = field_result.get("raw_similarity_score")
                        entry = {
                            "field_path": field_name,
                            "non_match_type": NonMatchType.FALSE_DISCOVERY,  # Use enum value
                            "ground_truth_value": gt_val,
                            "prediction_value": pred_val,
                            "similarity_score": similarity,
                        "details": {"reason": "below threshold"},
                        }
                        if similarity is not None:
                            info = self._get_comparison_info(field_name)
                        entry["details"]["reason"] = (
                            f"below threshold ({similarity:.3f} < {info.threshold})"
                        )
                        all_non_matches.append(entry)
                
                # ADDITIONAL: Handle nested StructuredModel objects for detailed non-match collection
                if (
                    isinstance(gt_val, StructuredModel)
                    and isinstance(pred_val, StructuredModel)
                    and "fields" in field_result
                ):
                    # Recursively collect non-matches from nested objects
                    nested_non_matches = gt_val._collect_enhanced_non_matches(
                        field_result, pred_val
                    )
                    # Prefix nested field paths with the parent field name
                    for nested_nm in nested_non_matches:
                        nested_nm["field_path"] = (
                            f"{field_name}.{nested_nm['field_path']}"
                        )
                        all_non_matches.append(nested_nm)
        
        # NEW: Add hierarchical debugging support
        if all_non_matches:
            helper = NonMatchesHelper()
            
            # Add aggregate contribution tracing if we have aggregate metrics
            if "aggregate" in recursive_result:
                all_non_matches = helper.add_aggregate_contribution_tracing(
                    all_non_matches, recursive_result["aggregate"]
                )
            
            # Add hierarchical organization metadata
            hierarchical_info = helper.organize_non_matches_hierarchically(all_non_matches)
            
            # Add debugging metadata to the first non_match entry for easy access
            if all_non_matches:
                all_non_matches[0]["_debugging_metadata"] = {
                    "hierarchical_summary": hierarchical_info["summary"],
                    "field_paths_affected": list(hierarchical_info["by_field_path"].keys()),
                    "leaf_vs_object_breakdown": {
                        "leaf_level": hierarchical_info["summary"]["leaf_level_count"],
                        "object_level": hierarchical_info["summary"]["object_level_count"]
                    }
                }
        
        return all_non_matches
    
    def _collect_non_matches(
        self, other: "StructuredModel", base_path: str = ""
    ) -> List[NonMatchField]:
        """Collect non-matches for detailed analysis.
        
        Args:
            other: Other model to compare with
            base_path: Base path for field naming (e.g., "address")
            
        Returns:
            List of NonMatchField objects documenting non-matches
        """
        non_matches = []
        
        # Handle null cases
        if other is None:
            non_matches.append(
                NonMatchField(
                field_path=base_path or "root",
                non_match_type=NonMatchType.FALSE_NEGATIVE,
                ground_truth_value=self,
                    prediction_value=None,
                )
            )
            return non_matches
        
        # Compare each field
        for field_name in self.__class__.model_fields:
            if field_name == "extra_fields":
                continue
                
            field_path = f"{base_path}.{field_name}" if base_path else field_name
            gt_value = getattr(self, field_name)
            pred_value = getattr(other, field_name, None)
            
            # Use existing field classification logic
            if isinstance(pred_value, list):
                classification = self._calculate_list_confusion_matrix(
                    field_name, pred_value
                )
            else:
                classification = self._classify_field_for_confusion_matrix(
                    field_name, pred_value
                )
            
            # Document non-matches based on classification
            if classification["fa"] > 0:  # False Alarm
                non_matches.append(
                    NonMatchField(
                    field_path=field_path,
                    non_match_type=NonMatchType.FALSE_ALARM,
                    ground_truth_value=gt_value,
                    prediction_value=pred_value,
                        similarity_score=classification.get("similarity_score"),
                    )
                )
            elif classification["fn"] > 0:  # False Negative
                non_matches.append(
                    NonMatchField(
                    field_path=field_path,
                    non_match_type=NonMatchType.FALSE_NEGATIVE,
                    ground_truth_value=gt_value,
                        prediction_value=pred_value,
                    )
                )
            elif classification["fd"] > 0:  # False Discovery
                non_matches.append(
                    NonMatchField(
                    field_path=field_path,
                    non_match_type=NonMatchType.FALSE_DISCOVERY,
                    ground_truth_value=gt_value,
                    prediction_value=pred_value,
                        similarity_score=classification.get("similarity_score"),
                    )
                )
            
            # Handle nested models recursively
            if isinstance(gt_value, StructuredModel) and isinstance(
                pred_value, StructuredModel
            ):
                nested_non_matches = gt_value._collect_non_matches(
                    pred_value, field_path
                )
                non_matches.extend(nested_non_matches)
        
        return non_matches
    
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
            # NEW: Collect enhanced object-level non-matches
            non_matches = self._collect_enhanced_non_matches(recursive_result, other)
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
    
    def _calculate_aggregate_metrics(self, confusion_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate universal aggregate metrics by summing from aggregate sections.
        
        This method implements the universal aggregate functionality by:
        1. Summing field-level "aggregate" metrics to compute root-level aggregate totals
        2. Ensuring consistency between field-level and root-level aggregate calculations
        3. Using "aggregate" sections instead of "overall" sections for universal aggregation
        
        Args:
            confusion_matrix: The confusion matrix result from compare_recursive
            
        Returns:
            Updated confusion matrix with universal aggregate metrics added
        """
        # Create a copy to avoid modifying the original
        result = confusion_matrix.copy()
        
        # First, recursively calculate aggregate sections for all nested fields
        result = self._calculate_nested_aggregate_sections(result)
        
        # Calculate root-level aggregate metrics by summing field-level aggregate metrics
        root_aggregate = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
        
        # Process all fields to sum their aggregate contributions
        if "fields" in result:
            for field_name, field_data in result["fields"].items():
                if isinstance(field_data, dict) and "aggregate" in field_data:
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        root_aggregate[metric] += field_data["aggregate"].get(metric, 0)
        
        # Add the calculated aggregate metrics to the root level
        result["aggregate"] = root_aggregate
        
        return result
    
    def _calculate_nested_aggregate_sections(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate sections for all nested fields by summing their child contributions.
        
        This method recursively processes the confusion matrix structure and calculates
        aggregate sections for fields that have nested fields by summing the aggregate
        contributions from their children. It preserves existing aggregate sections that
        are already correctly set by the threshold-gated recursion logic.
        
        Args:
            result: The confusion matrix result structure
            
        Returns:
            Updated result with calculated aggregate sections
        """
        if "fields" in result:
            for field_name, field_data in result["fields"].items():
                if isinstance(field_data, dict):
                    # Recursively process nested fields first
                    if "fields" in field_data and len(field_data["fields"]) > 0:
                        field_data = self._calculate_nested_aggregate_sections(field_data)
                        
                        # For parent fields with non-empty nested fields, calculate aggregate by summing children's aggregates
                        nested_aggregate = self._expand_hierarchical_field_metrics(field_data["fields"])
                        
                        # Only use nested aggregate if it has meaningful data
                        if any(nested_aggregate.get(metric, 0) > 0 for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]):
                            field_data["aggregate"] = nested_aggregate
                        elif "aggregate" not in field_data and "overall" in field_data:
                            # Fallback to overall if nested fields don't contribute
                            field_data["aggregate"] = field_data["overall"].copy()
                        
                    elif "aggregate" not in field_data and "overall" in field_data:
                        # For leaf fields or fields with empty nested fields, aggregate should match overall
                        field_data["aggregate"] = field_data["overall"].copy()
                    
                    # Update the field data in the result
                    result["fields"][field_name] = field_data
        
        return result
    
    def _expand_hierarchical_field_metrics(self, fields_dict: Dict[str, Any]) -> Dict[str, int]:
        """Expand hierarchical field metrics by summing nested field contributions to aggregate section.
        
        This method recursively processes nested fields and sums their aggregate contributions
        to support universal aggregate calculation for hierarchical structures.
        
        Args:
            fields_dict: Dictionary of field data with potential nested structures
            
        Returns:
            Dictionary with summed aggregate metrics from all nested levels
        """
        aggregate_sum = {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}
        
        for field_name, field_data in fields_dict.items():
            if isinstance(field_data, dict):
                # If field has an aggregate section, use it (it already represents the sum of nested fields)
                if "aggregate" in field_data:
                    field_aggregate = field_data["aggregate"]
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        aggregate_sum[metric] += field_aggregate.get(metric, 0)
                
                # If field doesn't have aggregate but has nested fields, recurse into nested fields
                elif "fields" in field_data and len(field_data["fields"]) > 0:
                    nested_sum = self._expand_hierarchical_field_metrics(field_data["fields"])
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        aggregate_sum[metric] += nested_sum.get(metric, 0)
                
                # If field has neither aggregate nor nested fields, use overall
                elif "overall" in field_data:
                    field_overall = field_data["overall"]
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        aggregate_sum[metric] += field_overall.get(metric, 0)
        
        return aggregate_sum