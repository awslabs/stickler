"""Field module for structured model evaluation.

This module provides the ComparableField function for creating fields in structured models
with comparison configuration parameters.
"""

from typing import Any, Dict, Optional
from pydantic import Field
import warnings

from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator


def ComparableField(
    comparator: Optional[BaseComparator] = None,
    threshold: float = 0.5,
    weight: float = 1.0,
    default: Any = None,
    clip_under_threshold: bool = True,
    # Pydantic Field parameters (all optional, just like Field)
    alias: Optional[str] = None,
    description: Optional[str] = None,
    examples: Optional[list] = None,
    **field_kwargs,
):
    """Create a Pydantic Field with comparison metadata.

    This function creates a proper Pydantic Field with embedded comparison configuration,
    enabling both comparison functionality and native Pydantic features like aliases.

    Args:
        comparator: Comparator to use for field comparison (default: LevenshteinComparator)
        threshold: Minimum similarity score to consider a match (default: 0.5)
        weight: Weight of this field in overall score calculation (default: 1.0)
        default: Default value for the field (default: None)
        clip_under_threshold: Whether to zero out scores below threshold (default: True)
        alias: Pydantic field alias for serialization (default: None)
        description: Field description for documentation (default: None)
        examples: Example values for the field (default: None)
        **field_kwargs: Additional Pydantic Field arguments

    Returns:
        Pydantic Field with embedded comparison metadata

    Example:
        class MyModel(StructuredModel):
            # Basic usage (no alias):
            name: str = ComparableField(threshold=0.8)

            # With alias (new feature):
            email: str = ComparableField(
                threshold=0.9,
                alias="email_address",
                description="User's email",
                examples=["user@example.com"]
            )
    """

    # Create the actual comparator instance
    actual_comparator = comparator or LevenshteinComparator()

    # Create serializable metadata for JSON schema compatibility
    serializable_metadata = {
        "comparator_type": actual_comparator.__class__.__name__,
        "comparator_name": getattr(actual_comparator, "name", "unknown"),
        "comparator_config": getattr(actual_comparator, "config", {}),
        "threshold": threshold,
        "weight": weight,
        "clip_under_threshold": clip_under_threshold
    }

    # Create json_schema_extra function that stores runtime data
    def json_schema_extra_func(schema: Dict[str, Any]) -> None:
        schema["x-comparison"] = serializable_metadata

    # HYBRID APPROACH: Store runtime instances as function attributes
    # This works around FieldInfo's __slots__ restriction
    json_schema_extra_func._comparator_instance = actual_comparator
    json_schema_extra_func._threshold = threshold
    json_schema_extra_func._weight = weight
    json_schema_extra_func._clip_under_threshold = clip_under_threshold
    json_schema_extra_func._comparison_metadata = serializable_metadata

    # Merge with existing json_schema_extra if provided
    existing_json_schema_extra = field_kwargs.get("json_schema_extra", {})
    if callable(existing_json_schema_extra):

        def enhanced_json_schema_extra(schema: Dict[str, Any]) -> None:
            existing_json_schema_extra(schema)
            json_schema_extra_func(schema)

        # Copy our runtime data to the enhanced function
        enhanced_json_schema_extra._comparator_instance = actual_comparator
        enhanced_json_schema_extra._threshold = threshold
        enhanced_json_schema_extra._weight = weight
        enhanced_json_schema_extra._clip_under_threshold = clip_under_threshold
        enhanced_json_schema_extra._comparison_metadata = serializable_metadata
        final_json_schema_extra = enhanced_json_schema_extra
    elif isinstance(existing_json_schema_extra, dict):

        def enhanced_json_schema_extra(schema: Dict[str, Any]) -> None:
            schema.update(existing_json_schema_extra)
            json_schema_extra_func(schema)

        # Copy our runtime data to the enhanced function
        enhanced_json_schema_extra._comparator_instance = actual_comparator
        enhanced_json_schema_extra._threshold = threshold
        enhanced_json_schema_extra._weight = weight
        enhanced_json_schema_extra._clip_under_threshold = clip_under_threshold
        enhanced_json_schema_extra._comparison_metadata = serializable_metadata
        final_json_schema_extra = enhanced_json_schema_extra
    else:
        final_json_schema_extra = json_schema_extra_func

    # Remove json_schema_extra from field_kwargs to avoid duplication
    clean_field_kwargs = {
        k: v for k, v in field_kwargs.items() if k != "json_schema_extra"
    }

    # Create the Field
    field = Field(
        default=default,
        alias=alias,
        description=description,
        examples=examples,
        json_schema_extra=final_json_schema_extra,
        **clean_field_kwargs,
    )

    return field


def _reconstruct_comparator_from_type(
    comparator_type: str, config: Optional[Dict[str, Any]] = None
) -> BaseComparator:
    """Reconstruct a comparator instance from its type name and configuration.

    Args:
        comparator_type: Name of the comparator class
        config: Configuration dictionary for the comparator

    Returns:
        Reconstructed comparator instance
    """
    config = config or {}

    # Map of comparator type names to their classes
    comparator_map: Dict[str, type] = {
        "LevenshteinComparator": LevenshteinComparator,
    }

    # Import additional comparators as needed
    try:
        from stickler.comparators.exact import ExactComparator

        comparator_map["ExactComparator"] = ExactComparator
    except ImportError:
        pass

    try:
        from stickler.comparators.numeric import NumericComparator

        comparator_map["NumericComparator"] = NumericComparator
    except ImportError:
        pass

    try:
        from stickler.comparators.structured import StructuredModelComparator

        comparator_map["StructuredModelComparator"] = StructuredModelComparator
    except ImportError:
        pass

    # Get the comparator class and instantiate it
    comparator_class = comparator_map.get(comparator_type)
    if comparator_class:
        try:
            # Try to instantiate with config if the constructor accepts it
            return comparator_class(**config)
        except TypeError:
            # Fallback to parameterless constructor
            return comparator_class()

    # Default fallback
    return LevenshteinComparator()


# Backward compatibility: Keep some legacy helper functions if needed by existing code
def add_comparison_schema(schema: Dict[str, Any], info: Dict[str, Any]) -> None:
    """Add comparison info to a schema."""
    schema["x-comparison"] = info


__all__ = ["ComparableField"]
