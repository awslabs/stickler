"""Field converter for dynamic model creation.

This module provides utilities for converting JSON field configurations to
Pydantic Field instances with ComparableField functionality.
"""

from typing import Any, Dict, Optional, Tuple, Type

from pydantic import Field

from .comparable_field import ComparableField
from .comparator_registry import create_comparator
from .type_resolver import resolve_type_string

#: Field-level opt-in: `"comparator": "auto"` asks for the same inference
#: `stickler.evaluate()` uses, for this field only. Chosen over a separate key so a
#: reader of the config sees the decision on the line that would otherwise name a
#: comparator, and so it cannot be set alongside a real comparator name.
AUTO_COMPARATOR = "auto"


def _infer_spec(field_name: str, field_type: Any):
    """Infer a comparison spec for one config-driven field from its resolved type.

    ``stickler.auto`` infers from a live ``FieldInfo``, and a config-driven field
    has none, so one is synthesised around the resolved annotation. That is enough:
    ``infer_field_config`` reads only the annotation and the field name, so the
    name-token heuristics work here too.
    """
    from pydantic.fields import FieldInfo

    from stickler.auto.inference import infer_field_config

    return infer_field_config(field_name, FieldInfo(annotation=field_type))


def _wants_inference(field_config: Dict[str, Any], infer_unspecified: bool) -> bool:
    """Whether this field's unspecified parameters should be inferred.

    A field-level setting always wins over the model-level flag, in both
    directions: ``"comparator": "auto"`` opts one field in when the model did not,
    and naming a real comparator pins one field when the model opted everything in.

    Naming a comparator pins the field's threshold too, rather than inferring it.
    A threshold is only meaningful beside the metric that produced the score --
    0.85 means one thing on edit distance and another on numeric tolerance -- so
    inference's threshold belongs to the comparator inference would have chosen,
    not to the one the caller named. Per-parameter filling applies to a field that
    let inference pick the comparator.
    """
    declared = field_config.get("comparator")
    if declared == AUTO_COMPARATOR:
        return True
    if declared is not None:
        return False
    return infer_unspecified


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
        infer_unspecified: bool = False,
    ) -> Tuple[Type, Any]:
        """Convert a JSON field configuration to a Pydantic field definition.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the field
            infer_unspecified: When True, parameters the config does not name are
                inferred from the field's type and name rather than defaulted.
                Off by default, because turning it on moves reported metrics for
                any field that named no comparator.

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
                field_name, field_config, infer_unspecified=infer_unspecified
            )

        # Handle primitive fields (existing logic)
        # Resolve the type
        try:
            field_type = resolve_type_string(type_string)
        except ValueError as e:
            raise ValueError(f"Invalid type for field '{field_name}': {e}")

        # Resolve each parameter independently, so a partly-configured field keeps
        # what it wrote and infers only the rest. Treating any config as "fully
        # configured" left `{"type": "float", "threshold": 0.99}` on the type-blind
        # Levenshtein default -- a threshold tuned against edit distance over
        # "1000.00" and "1000.0".
        inferred = (
            _infer_spec(field_name, field_type)
            if _wants_inference(field_config, infer_unspecified)
            else None
        )
        provenance = list(inferred.provenance) if inferred else []

        declared_comparator = field_config.get("comparator")
        if declared_comparator == AUTO_COMPARATOR:
            declared_comparator = None
        if declared_comparator is None and inferred is not None:
            comparator_name = inferred.comparator_name
            # MERGED, not replaced. A `comparator_config` beside an inferred
            # comparator is a tweak to the comparator inference chose, so
            # replacing the whole dict silently dropped the rest of it:
            # NumericComparator's inferred `relative_tolerance` of 0.001 fell to
            # 0.0, turning a tolerance-based numeric comparison into an exact one
            # for anyone who set any other key.
            comparator_config = {
                **inferred.comparator_config,
                **field_config.get("comparator_config", {}),
            }
        else:
            comparator_name = declared_comparator or "LevenshteinComparator"
            comparator_config = field_config.get("comparator_config", {})

        # Create comparator instance
        try:
            comparator = create_comparator(comparator_name, comparator_config)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid comparator for field '{field_name}': {e}")

        # Extract other field parameters
        threshold = field_config.get(
            "threshold", inferred.threshold if inferred else 0.5
        )
        weight = field_config.get("weight", inferred.weight if inferred else 1.0)
        clip_under_threshold = field_config.get(
            "clip_under_threshold",
            inferred.clip_under_threshold if inferred else True,
        )


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
            clip_under_threshold=clip_under_threshold,
            alias=alias,
            description=description,
            examples=examples,
        )

        # Record what was inferred, so `explain()` reports the reason and a
        # `source` naming what drove the choice (`type`, `name-token`) rather than
        # claiming every config-driven field was explicitly configured. Written onto the closure `ComparableField` just
        # returned, which no caller has a reference to yet -- the shared-field
        # hazard `structured_model.py` warns about needs sharing.
        if provenance:
            extra_callable = comparable_field.json_schema_extra
            if callable(extra_callable):
                extra_callable._inferred_provenance = tuple(provenance)

        return field_type, comparable_field

    def _convert_nested_model_field(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified: bool = False,
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
            # Forwarded, because the nested model is built by re-entering
            # `model_from_json` with this synthesised config. Omitting it meant a
            # model-level flag stopped at the first nesting level, so a nested
            # field silently kept the type-blind default while its siblings one
            # level up were inferred.
            "infer_unspecified_fields": infer_unspecified,
        }

        # Create the nested model class
        NestedModelClass = StructuredModel.model_from_json(nested_config)

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
            clip_under_threshold=clip_under_threshold,
            alias=alias,
            description=description,
            examples=examples,
        )


        return field_type, comparable_field

    def convert_fields_config(
        self,
        fields_config: Dict[str, Dict[str, Any]],
        *,
        infer_unspecified: bool = False,
    ) -> Dict[str, Tuple[Type, Field]]:
        """Convert multiple field configurations.

        Args:
            fields_config: Dictionary of field configurations

        Returns:
            Dictionary mapping field names to (type, field) tuples

        Raises:
            ValueError: If any field configuration is invalid
        """
        field_definitions = {}

        for field_name, field_config in fields_config.items():
            try:
                field_type, pydantic_field = self.convert_field_config(
                    field_name, field_config, infer_unspecified=infer_unspecified
                )
                field_definitions[field_name] = (field_type, pydantic_field)
            except ValueError as e:
                raise ValueError(f"Error processing field '{field_name}': {e}")

        return field_definitions

    def validate_field_config(
        self, field_name: str, field_config: Dict[str, Any]
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

        # Validate comparator if specified. `"auto"` is a request to infer, not a
        # comparator name, so it is not resolved through the registry -- doing so
        # reported it as an unknown comparator and listed the built-ins.
        if (
            "comparator" in field_config
            and field_config["comparator"] != AUTO_COMPARATOR
        ):
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
        boolean_params = ["required", "clip_under_threshold"]
        for param in boolean_params:
            if param in field_config:
                value = field_config[param]
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Field '{field_name}' parameter '{param}' must be boolean, got {type(value)}"
                    )

    def validate_fields_config(self, fields_config: Dict[str, Dict[str, Any]]) -> None:
        """Validate multiple field configurations.

        Args:
            fields_config: Dictionary of field configurations

        Raises:
            ValueError: If any field configuration is invalid
        """
        for field_name, field_config in fields_config.items():
            self.validate_field_config(field_name, field_config)

    def validate_nested_field_schema(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified: bool = False,
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
                    infer_unspecified=infer_unspecified,
                )

        else:
            # Primitive fields cannot have 'fields'
            if "fields" in field_config:
                raise ValueError(
                    f"Field '{field_name}' with primitive type '{field_type}' cannot have 'fields'. "
                    "Only structured_model types can have nested fields."
                )

            # Primitive fields must have comparators, unless inference was asked
            # for -- at which point omission is the point, not an oversight.
            if "comparator" not in field_config and not infer_unspecified:
                raise ValueError(
                    f"Field '{field_name}' with primitive type '{field_type}' requires a 'comparator'. "
                    "Primitive fields need comparators to define how they should be compared. "
                    "Set 'infer_unspecified_fields': true on the model, or "
                    f"'comparator': '{AUTO_COMPARATOR}' on this field, to have it "
                    "chosen from the field's type and name."
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
    infer_unspecified: bool = False,
) -> Tuple[Type, Field]:
    """Convert a field configuration using the global converter.

    Args:
        field_name: Name of the field
        field_config: JSON configuration for the field

    Returns:
        Tuple of (field_type, pydantic_field)
    """
    return _global_converter.convert_field_config(
        field_name, field_config, infer_unspecified=infer_unspecified
    )


def convert_fields_config(
    fields_config: Dict[str, Dict[str, Any]],
    *,
    infer_unspecified: bool = False,
) -> Dict[str, Tuple[Type, Field]]:
    """Convert multiple field configurations using the global converter.

    Args:
        fields_config: Dictionary of field configurations

    Returns:
        Dictionary mapping field names to (type, field) tuples
    """
    return _global_converter.convert_fields_config(
        fields_config, infer_unspecified=infer_unspecified
    )


def validate_field_config(field_name: str, field_config: Dict[str, Any]) -> None:
    """Validate a field configuration using the global converter.

    Args:
        field_name: Name of the field
        field_config: JSON configuration for the field

    Raises:
        ValueError: If configuration is invalid
    """
    _global_converter.validate_field_config(field_name, field_config)


def validate_fields_config(fields_config: Dict[str, Dict[str, Any]]) -> None:
    """Validate multiple field configurations using the global converter.

    Args:
        fields_config: Dictionary of field configurations

    Raises:
        ValueError: If any field configuration is invalid
    """
    _global_converter.validate_fields_config(fields_config)
