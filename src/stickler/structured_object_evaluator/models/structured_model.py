"""StructuredModel: Pydantic BaseModel with configurable field comparison."""

import inspect
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field

from .comparable_field import ComparableField
from .comparison_helper import ComparisonHelper
from .confidence_helper import ConfidenceHelper
from .configuration_helper import ConfigurationHelper


class StructuredModel(BaseModel):
    """Base class for models with structured comparison capabilities.

    Extends Pydantic's BaseModel with configurable field-level comparison,
    confusion matrix metrics, and Hungarian matching for list fields.

    Comparison logic is delegated to specialized helper classes in this package.
    See the directory README.md for architecture details.

    Public API:
        compare(other) -> float: Weighted similarity score
        compare_with(other, **opts) -> dict: Detailed comparison with optional metrics
        compare_recursive(other) -> dict: Raw hierarchical comparison result
        compare_field_raw(field_name, value) -> float: Single field similarity
        from_json(data) -> instance: Create from JSON data
        model_from_json(config) -> class: Create dynamic subclass from config
        from_json_schema(schema) -> class: Create dynamic subclass from JSON Schema
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
        ConfigurationHelper.validate_subclass_fields(cls)

    def model_post_init(self, __context):
        """Initialize confidence storage after model creation."""
        # Use object.__setattr__ to bypass Pydantic field detection
        object.__setattr__(self, "field_confidences", {})

    @classmethod
    def _is_list_of_structured_model_type(cls, field_type) -> bool:
        """Check if a field type annotation represents List[StructuredModel]."""
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

    def get_field_confidence(self, field_name: str) -> Optional[float]:
        """Get confidence for a field."""
        # Don't create the attribute - just check if it exists
        if not hasattr(self, "field_confidences"):
            return None
        return self.field_confidences.get(field_name)

    def get_all_confidences(self) -> Dict[str, float]:
        """Get all confidences."""
        # Don't create the attribute - return empty dict if no confidence data
        if not hasattr(self, "field_confidences"):
            return {}
        return self.field_confidences.copy()

    @classmethod
    def from_json(
        cls, json_data: Dict[str, Any], process_confidence=True
    ) -> "StructuredModel":
        """Create a StructuredModel instance from JSON data, handling missing fields
        and storing extra fields. Processes confidence structures on top-level calls."""
        if process_confidence:
            # Only process confidence on the top-level call
            processed_data, confidences = (
                ConfidenceHelper.process_confidence_structures(json_data)
            )
            instance = ConfigurationHelper.from_json(cls, processed_data)
            if confidences:  # Only set if we have confidence data
                object.__setattr__(instance, "field_confidences", confidences)
        else:
            # Skip confidence processing for recursive calls
            instance = ConfigurationHelper.from_json(cls, json_data)
        return instance

    @classmethod
    def model_from_json(cls, config: Dict[str, Any]) -> Type["StructuredModel"]:
        """Create a StructuredModel subclass from JSON configuration.

        Uses Pydantic's create_model() under the hood. The generated model inherits
        all StructuredModel capabilities (compare, compare_with, etc.).

        Args:
            config: JSON configuration with 'fields' dict (required), plus optional
                    'model_name' (default: "DynamicModel") and 'match_threshold' (default: 0.7).
                    Each field config supports: type, comparator, threshold, weight,
                    required, default, description, alias, examples.

        Returns:
            A StructuredModel subclass.

        Raises:
            ValueError: If configuration is invalid.
            KeyError: If required keys are missing.
        """
        # Delegate to ModelFactory for dynamic model creation
        from .model_factory import ModelFactory

        return ModelFactory.create_model_from_json(config, base_class=cls)

    @classmethod
    def from_json_schema(cls, schema: Dict[str, Any]) -> Type["StructuredModel"]:
        """Create a StructuredModel subclass from a JSON Schema document (draft-07+).

        Customize comparison behavior with x-aws-stickler-* extension fields:
        - Field-level: x-aws-stickler-comparator, -threshold, -weight,
          -clip-under-threshold, -aggregate
        - Model-level: x-aws-stickler-model-name, -match-threshold

        Supports primitive types, nested objects, arrays, required fields,
        defaults, descriptions, and $ref references.

        Args:
            schema: JSON Schema document as a dictionary.

        Returns:
            StructuredModel subclass created from the schema.

        Raises:
            ValueError: If schema is invalid or contains unsupported features.
        """

        return cls._from_json_schema_internal(schema, field_path="")

    @classmethod
    def _from_json_schema_internal(
        cls, schema: Dict[str, Any], field_path: str
    ) -> Type["StructuredModel"]:
        """Internal recursive helper for from_json_schema with field path tracking."""
        from .model_factory import ModelFactory

        return ModelFactory.create_model_from_json_schema(
            schema=schema, field_path=field_path, base_class=cls
        )

    @classmethod
    def _is_structured_field_type(cls, field_info) -> bool:
        """Check if a field is a List[StructuredModel] or StructuredModel type."""
        return ConfigurationHelper.is_structured_field_type(field_info)

    @classmethod
    def _get_comparison_info(cls, field_name: str) -> ComparableField:
        """Get comparison configuration for a field."""
        return ConfigurationHelper.get_comparison_info(cls, field_name)

    @classmethod
    def _is_aggregate_field(cls, field_name: str) -> bool:
        """Check if field is marked for confusion matrix aggregation."""
        return ConfigurationHelper.is_aggregate_field(cls, field_name)


    def compare_field_raw(self, field_name: str, other_value: Any) -> float:
        """Compare a single field with a value, returning raw similarity without thresholds."""
        my_value = getattr(self, field_name)

        # Nested StructuredModel: use compare_with for rich comparison
        if isinstance(my_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            comparison_result = my_value.compare_with(
                other_value,
                include_confusion_matrix=False,
                document_non_matches=False,
                evaluator_format=False,
                recall_with_fd=False,
            )
            return comparison_result["overall_score"]

        return ComparisonHelper.compare_field_raw(self, field_name, other_value)

    def compare_recursive(self, other: "StructuredModel") -> dict:
        """Perform recursive comparison returning hierarchical metrics and scores.

        Returns dict with overall (TP/FP/TN/FN/FD/FA + similarity_score),
        fields (per-field results), and non_matches.
        """
        from .comparison_engine import ComparisonEngine

        engine = ComparisonEngine(self)
        return engine.compare_recursive(other)

    def compare(self, other: "StructuredModel") -> float:
        """Return overall weighted similarity score (0.0 to 1.0) without threshold filtering."""
        total_score = 0.0
        total_weight = 0.0

        for field_name in self.__class__.model_fields:
            if field_name == "extra_fields":
                continue
            if hasattr(other, field_name):
                info = self.__class__._get_comparison_info(field_name)
                weight = info.weight
                field_score = self.compare_field_raw(
                    field_name, getattr(other, field_name)
                )
                total_score += field_score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def compare_with(
        self,
        other: "StructuredModel",
        include_confusion_matrix: bool = False,
        document_non_matches: bool = False,
        evaluator_format: bool = False,
        recall_with_fd: bool = False,
        add_derived_metrics: bool = True,
        document_field_comparisons: bool = False,
        add_confidence_metrics: bool = False,
    ) -> Dict[str, Any]:
        """Compare this model with another, returning detailed results.

        Delegates to ComparisonEngine using single-traversal optimization.

        Returns dict with field_scores, overall_score, all_fields_matched,
        plus optional confusion_matrix, non_matches, field_comparisons,
        and auroc_confidence_metric based on flags.
        """
        from .comparison_engine import ComparisonEngine

        engine = ComparisonEngine(self)
        return engine.compare_with(
            other,
            include_confusion_matrix=include_confusion_matrix,
            document_non_matches=document_non_matches,
            evaluator_format=evaluator_format,
            recall_with_fd=recall_with_fd,
            add_derived_metrics=add_derived_metrics,
            document_field_comparisons=document_field_comparisons,
            add_confidence_metrics=add_confidence_metrics,
        )

    @classmethod
    def model_json_schema(cls, **kwargs):
        """Override to add x-comparison metadata to field schemas."""
        schema = super().model_json_schema(**kwargs)

        for field_name, field_info in cls.model_fields.items():
            if field_name == "extra_fields":
                continue
            if field_name not in schema.get("properties", {}):
                continue

            field_props = schema["properties"][field_name]
            if hasattr(field_info, "json_schema_extra") and callable(
                field_info.json_schema_extra
            ):
                temp_schema = {}
                field_info.json_schema_extra(temp_schema)
                if "x-comparison" in temp_schema:
                    field_props["x-comparison"] = temp_schema["x-comparison"]

        return schema
