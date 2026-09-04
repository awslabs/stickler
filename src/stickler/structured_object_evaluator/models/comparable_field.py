"""Field module for structured model evaluation.

This module provides the ComparableField function for creating fields in structured models
with comparison configuration parameters.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator

# The threshold a field gets when it names neither a threshold nor a comparator
# threshold. Kept as a named constant rather than a literal because it is a
# placeholder: the contract says an unspecified field is inferred from its type
# and name, and this value stands in until inference owns that path (#239). It is
# not a meaningful default, it is the historical one.
_LEGACY_DEFAULT_THRESHOLD = 0.5


def _comparator_threshold_was_set(comparator: BaseComparator) -> Optional[float]:
    """Return the comparator's threshold if the caller named one, else ``None``.

    A field with no threshold of its own adopts one the caller put on the
    comparator, because ``LevenshteinComparator(threshold=0.9)`` is a clear
    statement of intent that was previously discarded in silence.

    It deliberately does NOT adopt a comparator's *default* threshold. Those
    defaults were only ever read by ``binary_compare()``, never as verdict
    thresholds, so they have not been audited as such and several are wrong for
    the job: ``DateComparator`` defaults to ``1.0`` while awarding partial credit
    of ``0.7`` for a year-less match, so adopting it would clip that feature to
    zero. Auditing every comparator's default is separate work (#246).

    The distinction comes from :attr:`BaseComparator.threshold_was_set`, recorded
    at construction. It cannot be recovered here: every comparator resolves its
    own default before calling ``super().__init__``, so ``DateComparator()`` and
    ``DateComparator(threshold=1.0)`` both arrive holding ``1.0``, and comparing
    against the signature default reads both as unset. That is why
    ``BaseComparator.__init__`` takes ``Optional[float] = None``.

    ``getattr`` with a default rather than a bare attribute read: a comparator
    that never chains to ``BaseComparator.__init__`` has no such attribute, and
    the safe reading of "cannot tell" is "not set".
    """
    if not getattr(comparator, "threshold_was_set", False):
        return None
    return getattr(comparator, "threshold", None)


def ComparableField(
    comparator: Optional[BaseComparator] = None,
    threshold: Optional[float] = None,
    weight: float = 1.0,
    default: Any = None,
    *,
    clip_under_threshold: Optional[bool] = None,
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
        threshold: Minimum similarity score to consider a match. ``None`` means
                  "not specified", in which case a threshold the caller set on
                  the comparator applies, since a threshold is only meaningful
                  next to the metric that produced the score. Otherwise 0.5
                  stands in until inference owns that case (#239)::

                      ComparableField(comparator=ExactComparator(threshold=0.9))
                      # 0.9, taken from the comparator

                      ComparableField(comparator=ExactComparator(), threshold=0.8)
                      # 0.8, stated on the field, which always wins

                      ComparableField(comparator=ExactComparator())
                      # 0.5. A comparator's *default* threshold is not adopted;
                      # see _comparator_threshold_was_set for why.
        weight: Weight of this field in overall score calculation (default: 1.0)
        default: Default value for the field (default: None)
        clip_under_threshold: Whether to zero out scores below threshold
                  (effective default: True). ``None`` means "not specified",
                  which lets a dict-annotated field default it to False so
                  partial credit survives, while an explicit True or False is
                  always honoured. See StructuredModel.__init_subclass__.
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

    if "aggregate" in field_kwargs:
        raise TypeError(
            "The 'aggregate' parameter was removed in 1.0; it had no effect. "
            "Aggregation is computed automatically for every node in compare_with() "
            "output. Remove the argument. "
            "See https://github.com/awslabs/stickler/issues/226"
        )

    # Create the actual comparator instance
    actual_comparator = comparator or LevenshteinComparator()
    # Whether the CALLER named a comparator. Recorded because the default is
    # resolved here, before the field's annotation is known, so this is the only
    # place the distinction survives. ConfigurationHelper needs it to give a
    # dict-annotated field a comparator that can actually score a mapping
    # without overriding a choice the user made deliberately.
    comparator_was_explicit = comparator is not None
    # Same reasoning for clip: the dict substitution turns it off so partial
    # credit survives, but must not overwrite a value the caller chose.
    clip_was_explicit = clip_under_threshold is not None
    if clip_under_threshold is None:
        clip_under_threshold = True

    # A threshold is only meaningful next to the metric that produced the score:
    # 0.85 means one thing on edit distance and another on a semantic embedding.
    # So a threshold the caller put on the comparator is a statement of intent
    # about this field, and it used to be discarded in silence. A comparator's
    # *default* threshold is not adopted; see _comparator_threshold_was_set.
    threshold_was_explicit = threshold is not None
    if threshold is None:
        from_comparator = (
            _comparator_threshold_was_set(actual_comparator)
            if comparator_was_explicit
            else None
        )
        threshold = (
            from_comparator
            if from_comparator is not None
            else _LEGACY_DEFAULT_THRESHOLD
        )

    # Create serializable metadata for JSON schema compatibility
    serializable_metadata = {
        "comparator_type": actual_comparator.__class__.__name__,
        "comparator_name": getattr(actual_comparator, "name", "unknown"),
        "comparator_config": getattr(actual_comparator, "config", {}),
        "threshold": threshold,
        "weight": weight,
        "clip_under_threshold": clip_under_threshold,
    }

    # Create json_schema_extra function that stores runtime data
    def json_schema_extra_func(schema: Dict[str, Any]) -> None:
        schema["x-comparison"] = serializable_metadata

    # HYBRID APPROACH: Store runtime instances as function attributes
    # This works around FieldInfo's __slots__ restriction
    json_schema_extra_func._comparator_instance = actual_comparator
    json_schema_extra_func._comparator_explicit = comparator_was_explicit
    json_schema_extra_func._clip_explicit = clip_was_explicit
    json_schema_extra_func._threshold_explicit = threshold_was_explicit
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
        enhanced_json_schema_extra._comparator_explicit = comparator_was_explicit
        enhanced_json_schema_extra._clip_explicit = clip_was_explicit
        enhanced_json_schema_extra._threshold_explicit = threshold_was_explicit
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
        enhanced_json_schema_extra._comparator_explicit = comparator_was_explicit
        enhanced_json_schema_extra._clip_explicit = clip_was_explicit
        enhanced_json_schema_extra._threshold_explicit = threshold_was_explicit
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
        from stickler.comparators.normalized import NormalizedComparator

        comparator_map["NormalizedComparator"] = NormalizedComparator
    except ImportError:
        pass

    try:
        from stickler.comparators.phone import PhoneComparator

        comparator_map["PhoneComparator"] = PhoneComparator
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
