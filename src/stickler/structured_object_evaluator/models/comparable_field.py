"""Field module for structured model evaluation.

This module provides the ComparableField function for creating fields in structured models
with comparison configuration parameters.
"""

import warnings
from typing import Any, Dict, Optional, Union

from pydantic import Field

from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator


class _Unset:
    """Sentinel distinguishing "argument omitted" from "argument passed".

    Needed because ``aggregate=False`` is both the historical default and a
    value a user may pass explicitly. Only the explicit case should warn.

    Detection is by identity (``is not _UNSET``). That is sound within one
    import namespace; a codebase importing stickler under two names (say
    ``stickler.x`` and ``src.stickler.x``) creates two distinct sentinels, and
    handing one namespace's to the other would read as an explicit argument.
    Harmless -- the result is a spurious deprecation warning about a parameter
    that is going away regardless.
    """

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        return False

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return "<unset>"


_UNSET = _Unset()


def ComparableField(
    comparator: Optional[BaseComparator] = None,
    threshold: float = 0.5,
    weight: float = 1.0,
    default: Any = None,
    aggregate: Union[bool, _Unset] = _UNSET,
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
        aggregate: DEPRECATED, has no effect, and will be removed in 0.8.0.
                  Passing it at all (either value) emits a DeprecationWarning.
                  Aggregation is applied at the comparison layer: every node in
                  compare_with() output already carries an 'aggregate' block
                  summing the primitive field metrics below it. Remove the
                  argument; there is no replacement to adopt.
                  See https://github.com/awslabs/stickler/issues/226
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
    # Warn on ANY explicit use, not just aggregate=True. Passing False was
    # silent before, so those callers had no signal that the parameter is going
    # away; they would have met a bare TypeError on upgrade. The value itself
    # has no effect either way: aggregation is applied at the comparison layer
    # and every node in compare_with() output carries an `aggregate` block.
    if aggregate is not _UNSET:
        warnings.warn(
            "The 'aggregate' parameter in ComparableField is deprecated, has no "
            "effect, and will be removed in 0.8.0. All nodes automatically "
            "include an 'aggregate' field in the compare_with() output that "
            "sums primitive field metrics below that node. Remove the argument; "
            "no replacement is needed. See "
            "https://github.com/awslabs/stickler/issues/226",
            DeprecationWarning,
            stacklevel=2,
        )
        aggregate = bool(aggregate)
    else:
        aggregate = False

    # Create the actual comparator instance
    actual_comparator = comparator or LevenshteinComparator()

    # Create serializable metadata for JSON schema compatibility
    serializable_metadata = {
        "comparator_type": actual_comparator.__class__.__name__,
        "comparator_name": getattr(actual_comparator, "name", "unknown"),
        "comparator_config": getattr(actual_comparator, "config", {}),
        "threshold": threshold,
        "weight": weight,
        "clip_under_threshold": clip_under_threshold,
        "aggregate": aggregate,
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
    json_schema_extra_func._aggregate = aggregate
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
        enhanced_json_schema_extra._aggregate = aggregate
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
        enhanced_json_schema_extra._aggregate = aggregate
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
