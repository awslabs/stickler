"""Field converter for dynamic model creation.

This module provides utilities for converting JSON field configurations to
Pydantic Field instances with ComparableField functionality.
"""

from typing import Any, Dict, Optional, Tuple, Type

from pydantic import Field

from .comparable_field import _UNSET, ComparableField, _restore_deprecated_aggregate
from .comparator_registry import create_comparator
from .type_resolver import resolve_type_string


class FieldConverter:
    """Converter for JSON field configurations to Pydantic fields."""

    def __init__(self):
        """Initialize the field converter."""
        pass

    def convert_field_config(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified_fields: bool = False,
    ) -> Tuple[Type, Any]:
        """Convert a JSON field configuration to a Pydantic field definition.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the field
            infer_unspecified_fields: When True, a field without a
                ``comparator`` key routes through ``stickler.auto.inference``
                to pick a type + name-aware default (with the operator's
                explicit ``threshold`` / ``weight`` still winning per
                parameter). Off by default; when off, this method's behavior
                is unchanged from before the flag (the pre-existing
                Levenshtein fallback stays in place, though in practice the
                validation step at ``validate_nested_field_schema`` rejects
                missing-comparator configs before this method sees them).

        Returns:
            Tuple of (field_type, pydantic_field)

        Raises:
            ValueError: If configuration is invalid
        """
        # Extract field type
        type_string = field_config.get("type")
        if not type_string:
            raise ValueError(f"Field '{field_name}' missing required 'type' parameter")

        # Check if this is a structured model type
        if type_string in [
            "structured_model",
            "list_structured_model",
            "optional_structured_model",
        ]:
            return self._convert_nested_model_field(
                field_name,
                field_config,
                infer_unspecified_fields=infer_unspecified_fields,
            )

        # Handle primitive fields (existing logic)
        # Resolve the type
        try:
            field_type = resolve_type_string(type_string)
        except ValueError as e:
            raise ValueError(f"Invalid type for field '{field_name}': {e}")

        # Comparator selection. Three paths merge here:
        #   1. Author declared ``comparator`` — honor it verbatim.
        #   2. ``infer_unspecified_fields`` is ON and ``comparator`` is
        #      absent — route through ``stickler.auto.inference``. The
        #      resulting InferredSpec provides defaults for ``comparator``,
        #      ``threshold``, ``weight``, and ``clip_under_threshold`` that
        #      the author can still override per-parameter below.
        #   3. Flag is OFF and ``comparator`` is absent — legacy behavior:
        #      Levenshtein. In practice validate_nested_field_schema
        #      rejects this before we get here, so path 3 is only
        #      reachable when the caller bypassed validation.
        inferred_spec = None
        if "comparator" in field_config:
            comparator_name = field_config["comparator"]
            comparator_config = field_config.get("comparator_config", {})
        elif infer_unspecified_fields:
            from pydantic.fields import FieldInfo as _AutoFieldInfo

            from ...auto.inference import infer_field_config

            inferred_spec = infer_field_config(
                field_name, _AutoFieldInfo(annotation=field_type)
            )
            comparator_name = inferred_spec.comparator_name
            comparator_config = inferred_spec.comparator_config
        else:
            comparator_name = "LevenshteinComparator"
            comparator_config = field_config.get("comparator_config", {})

        # Create comparator instance
        try:
            comparator = create_comparator(comparator_name, comparator_config)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid comparator for field '{field_name}': {e}")

        # Extract other field parameters. When inference fired, each
        # parameter falls back to the InferredSpec value if the author
        # didn't declare it — per-parameter merge, matching Decision 2.
        # When inference didn't fire, fall back to the historical flat
        # defaults (0.5 / 1.0 / True).
        if inferred_spec is not None:
            threshold = field_config.get("threshold", inferred_spec.threshold)
            weight = field_config.get("weight", inferred_spec.weight)
            clip_under_threshold = field_config.get(
                "clip_under_threshold", inferred_spec.clip_under_threshold
            )
        else:
            threshold = field_config.get("threshold", 0.5)
            weight = field_config.get("weight", 1.0)
            clip_under_threshold = field_config.get("clip_under_threshold", True)
        # Reading a config is not an explicit use of the deprecated `aggregate`
        # parameter: `to_stickler_config()` writes the key for every field, so
        # passing it through would warn on every round trip -- blaming this
        # frame for a key the caller never wrote, and hard-failing under
        # `-W error::DeprecationWarning`. Construct with the sentinel, then
        # restore the config's value below so export stays faithful (issue #226).
        aggregate = _UNSET
        config_aggregate = field_config.get("aggregate")

        # Extract Pydantic field parameters
        default = field_config.get("default", ...)  # Use Ellipsis for required fields
        required = field_config.get("required", False)
        description = field_config.get("description")
        alias = field_config.get("alias")
        examples = field_config.get("examples")

        # Handle required vs default logic
        if required and default != ...:
            # If explicitly required, ignore default
            default = ...
        elif not required and default == ...:
            # If not required and no default specified, use None
            default = None

        # Widen the annotation to Optional[...] whenever the resolved default is
        # None, so the None default is a valid value. Mirrors the JSON-Schema path
        # (json_schema_field_converter.py) — issue #149: the rich-value path
        # round-trips through from_json(...).model_dump(), which materializes the
        # None default and re-validates it against the annotation. Without this,
        # a schema -> to_stickler_config -> model_from_json round-trip rebuilds a
        # broken model.
        if default is None:
            field_type = Optional[field_type]

        # Create ComparableField
        comparable_field = ComparableField(
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            default=default,
            aggregate=aggregate,
            clip_under_threshold=clip_under_threshold,
            alias=alias,
            description=description,
            examples=examples,
        )

        _restore_deprecated_aggregate(comparable_field, config_aggregate)

        # Attach provenance for retrieval when inference fired. Consumers
        # read ``model.model_fields[name].json_schema_extra._provenance``
        # to see the ordered rule trace, e.g.
        # ``["type:str -> LevenshteinComparator@0.7",
        #    "name-token:invoice_id -> ExactComparator@1.0"]``. Missing
        # attribute (or empty list) = author-configured comparator.
        if inferred_spec is not None:
            extra = comparable_field.json_schema_extra
            if extra is not None:
                extra._provenance = list(inferred_spec.provenance)

        return field_type, comparable_field

    def _convert_nested_model_field(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified_fields: bool = False,
    ) -> Tuple[Type, Any]:
        """Convert a nested structured model field configuration.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the nested field

        Returns:
            Tuple of (field_type, pydantic_field)

        Raises:
            ValueError: If configuration is invalid
        """
        from typing import List, Optional

        type_string = field_config["type"]
        nested_fields_config = field_config["fields"]

        # Recursively create the nested model class
        from .structured_model import StructuredModel

        # Create nested model configuration
        nested_config = {
            "model_name": f"{field_name.title()}Model",
            "fields": nested_fields_config,
            "match_threshold": field_config.get("match_threshold", 0.7),
        }

        # Create the nested model class. Thread the flag through so
        # nested un-annotated fields inherit the caller's inference choice.
        NestedModelClass = StructuredModel.model_from_json(
            nested_config,
            infer_unspecified_fields=infer_unspecified_fields,
        )

        # Determine the field type based on the type string
        if type_string == "structured_model":
            field_type = NestedModelClass
        elif type_string == "list_structured_model":
            field_type = List[NestedModelClass]
        elif type_string == "optional_structured_model":
            field_type = Optional[NestedModelClass]
        else:
            raise ValueError(f"Unknown structured model type: {type_string}")

        # Extract Pydantic field parameters
        default = field_config.get("default", ...)
        required = field_config.get("required", False)
        description = field_config.get("description")
        alias = field_config.get("alias")
        examples = field_config.get("examples")

        # Handle required vs default logic
        if required and default != ...:
            default = ...
        elif not required and default == ...:
            default = None

        # CRITICAL FIX: Create ComparableField for nested models to enable proper comparison
        # Extract threshold and weight from field configuration
        weight = field_config.get("weight", 1.0)  # Default weight
        clip_under_threshold = field_config.get("clip_under_threshold", True)
        # Reading a config is not an explicit use of the deprecated `aggregate`
        # parameter: `to_stickler_config()` writes the key for every field, so
        # passing it through would warn on every round trip -- blaming this
        # frame for a key the caller never wrote, and hard-failing under
        # `-W error::DeprecationWarning`. Construct with the sentinel, then
        # restore the config's value below so export stays faithful (issue #226).
        aggregate = _UNSET
        config_aggregate = field_config.get("aggregate")

        # For list_structured_model, don't set threshold (Hungarian matching uses model's match_threshold)
        # For single structured_model, use threshold from config
        if type_string == "list_structured_model":
            threshold = 0.5  # Use default threshold to avoid validation error
        else:
            threshold = field_config.get(
                "threshold", 0.7
            )  # Default threshold for single nested models

        # Create ComparableField with dummy comparator (never used for StructuredModel fields)
        # The comparator is required by ComparableField but StructuredModel uses recursive compare()
        from stickler.comparators.levenshtein import LevenshteinComparator

        dummy_comparator = LevenshteinComparator()  # Never actually called

        comparable_field = ComparableField(
            comparator=dummy_comparator,  # Required but unused for StructuredModel fields
            threshold=threshold,
            weight=weight,
            default=default,
            aggregate=aggregate,
            clip_under_threshold=clip_under_threshold,
            alias=alias,
            description=description,
            examples=examples,
        )

        _restore_deprecated_aggregate(comparable_field, config_aggregate)

        return field_type, comparable_field

    def convert_fields_config(
        self,
        fields_config: Dict[str, Dict[str, Any]],
        *,
        infer_unspecified_fields: bool = False,
    ) -> Dict[str, Tuple[Type, Field]]:
        """Convert multiple field configurations.

        Args:
            fields_config: Dictionary of field configurations
            infer_unspecified_fields: Threaded through to ``convert_field_config``.

        Returns:
            Dictionary mapping field names to (type, field) tuples

        Raises:
            ValueError: If any field configuration is invalid
        """
        field_definitions = {}

        for field_name, field_config in fields_config.items():
            try:
                field_type, pydantic_field = self.convert_field_config(
                    field_name,
                    field_config,
                    infer_unspecified_fields=infer_unspecified_fields,
                )
                field_definitions[field_name] = (field_type, pydantic_field)
            except ValueError as e:
                raise ValueError(f"Error processing field '{field_name}': {e}")

        return field_definitions

    def validate_field_config(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified_fields: bool = False,
    ) -> None:
        """Validate a field configuration without converting it.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the field

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        if "type" not in field_config:
            raise ValueError(f"Field '{field_name}' missing required 'type' parameter")

        # Validate type string
        type_string = field_config["type"]
        try:
            resolve_type_string(type_string)
        except ValueError as e:
            raise ValueError(f"Invalid type for field '{field_name}': {e}")

        # Validate comparator if specified
        if "comparator" in field_config:
            comparator_name = field_config["comparator"]
            comparator_config = field_config.get("comparator_config", {})
            try:
                create_comparator(comparator_name, comparator_config)
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid comparator for field '{field_name}': {e}")

        # Validate numeric parameters
        numeric_params = ["threshold", "weight"]
        for param in numeric_params:
            if param in field_config:
                value = field_config[param]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Field '{field_name}' parameter '{param}' must be numeric, got {type(value)}"
                    )
                if param == "threshold" and not (0.0 <= value <= 1.0):
                    raise ValueError(
                        f"Field '{field_name}' threshold must be between 0.0 and 1.0, got {value}"
                    )
                if param == "weight" and value < 0:
                    raise ValueError(
                        f"Field '{field_name}' weight must be non-negative, got {value}"
                    )

        # Validate boolean parameters
        boolean_params = ["required", "aggregate", "clip_under_threshold"]
        for param in boolean_params:
            if param in field_config:
                value = field_config[param]
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Field '{field_name}' parameter '{param}' must be boolean, got {type(value)}"
                    )

    def validate_fields_config(
        self,
        fields_config: Dict[str, Dict[str, Any]],
        *,
        infer_unspecified_fields: bool = False,
    ) -> None:
        """Validate multiple field configurations.

        Args:
            fields_config: Dictionary of field configurations
            infer_unspecified_fields: Threaded through to ``validate_field_config``.

        Raises:
            ValueError: If any field configuration is invalid
        """
        for field_name, field_config in fields_config.items():
            self.validate_field_config(
                field_name,
                field_config,
                infer_unspecified_fields=infer_unspecified_fields,
            )

    def validate_nested_field_schema(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified_fields: bool = False,
    ) -> None:
        """Validate schema for nested structured model fields.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the field

        Raises:
            ValueError: If schema is invalid for nested models
        """
        field_type = field_config.get("type", "")

        # Check if this is a structured model type
        if field_type in [
            "structured_model",
            "list_structured_model",
            "optional_structured_model",
        ]:
            # Structured model fields cannot have comparators
            if "comparator" in field_config:
                raise ValueError(
                    f"Field '{field_name}' with type '{field_type}' cannot have a 'comparator'. "
                    "Structured models use recursive comparison, not primitive comparators."
                )

            # Structured model fields must have 'fields'
            if "fields" not in field_config:
                raise ValueError(
                    f"Field '{field_name}' with type '{field_type}' requires a 'fields' configuration "
                    "defining the nested model structure."
                )

            # Recursively validate nested fields
            nested_fields = field_config["fields"]
            if not isinstance(nested_fields, dict):
                raise ValueError(
                    f"Field '{field_name}' 'fields' must be a dictionary, got {type(nested_fields)}"
                )

            # Validate each nested field
            for nested_field_name, nested_field_config in nested_fields.items():
                self.validate_nested_field_schema(
                    f"{field_name}.{nested_field_name}",
                    nested_field_config,
                    infer_unspecified_fields=infer_unspecified_fields,
                )

        else:
            # Primitive fields cannot have 'fields'
            if "fields" in field_config:
                raise ValueError(
                    f"Field '{field_name}' with primitive type '{field_type}' cannot have 'fields'. "
                    "Only structured_model types can have nested fields."
                )

            # Primitive fields must have comparators — unless the caller
            # opted into ``infer_unspecified_fields``, in which case an
            # absent comparator is filled in by ``stickler.auto.inference``
            # at convert time (see ``convert_field_config``). Flag off
            # preserves the original strict behavior, so every existing
            # test at ``test_model_from_json.py:792-804`` still passes.
            if "comparator" not in field_config and not infer_unspecified_fields:
                raise ValueError(
                    f"Field '{field_name}' with primitive type '{field_type}' requires a 'comparator'. "
                    "Primitive fields need comparators to define how they should be compared."
                )


# Global converter instance
_global_converter = FieldConverter()


def get_global_converter() -> FieldConverter:
    """Get the global field converter instance.

    Returns:
        Global FieldConverter instance
    """
    return _global_converter


def convert_field_config(
    field_name: str,
    field_config: Dict[str, Any],
    *,
    infer_unspecified_fields: bool = False,
) -> Tuple[Type, Field]:
    """Convert a field configuration using the global converter."""
    return _global_converter.convert_field_config(
        field_name,
        field_config,
        infer_unspecified_fields=infer_unspecified_fields,
    )


def convert_fields_config(
    fields_config: Dict[str, Dict[str, Any]],
    *,
    infer_unspecified_fields: bool = False,
) -> Dict[str, Tuple[Type, Field]]:
    """Convert multiple field configurations using the global converter."""
    return _global_converter.convert_fields_config(
        fields_config,
        infer_unspecified_fields=infer_unspecified_fields,
    )


def validate_field_config(
    field_name: str,
    field_config: Dict[str, Any],
    *,
    infer_unspecified_fields: bool = False,
) -> None:
    """Validate a field configuration using the global converter."""
    _global_converter.validate_field_config(
        field_name,
        field_config,
        infer_unspecified_fields=infer_unspecified_fields,
    )


def validate_fields_config(
    fields_config: Dict[str, Dict[str, Any]],
    *,
    infer_unspecified_fields: bool = False,
) -> None:
    """Validate multiple field configurations using the global converter."""
    _global_converter.validate_fields_config(
        fields_config,
        infer_unspecified_fields=infer_unspecified_fields,
    )
