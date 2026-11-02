"""Factory for creating dynamic StructuredModel subclasses from JSON configuration.

This module provides the ModelFactory class that encapsulates the logic for creating
dynamic StructuredModel subclasses from JSON configuration. It uses the factory pattern
to separate model creation concerns from the core StructuredModel class.
"""

from typing import Any, Dict, Optional, Type

from pydantic import create_model

from stickler.utils.deprecation import warn_once

from .field_converter import (
    convert_fields_config,
    get_global_converter,
    validate_fields_config,
)
from .threshold_helper import model_identity, warn_if_threshold_is_zero

#: Every key a top-level model config may carry, as read by ``create_model``.
ACCEPTED_MODEL_CONFIG_KEYS = frozenset(
    {"fields", "model_name", "match_threshold", "infer_unspecified_fields"}
)

#: Frames between ``warnings.warn`` and the user's ``model_from_json`` call, for
#: the model-level unknown-key warning. ``warn_once``'s default of 3 assumes one
#: wrapping call; this one is reached through four, so at 3 the warning was
#: attributed to this module instead of to the line the reader can fix. Counted
#: outward from ``warnings.warn``: warn_once, validate_config,
#: create_model_from_json, model_from_json, caller.
_UNKNOWN_KEY_STACKLEVEL = 5


class ModelFactory:
    """Factory for creating dynamic StructuredModel subclasses from JSON configuration.
    
    This class implements the factory pattern to create StructuredModel subclasses
    dynamically from JSON configuration. It handles:
    - Configuration validation
    - Field definition conversion
    - Dynamic model creation using Pydantic's create_model()
    - Class-level attribute configuration
    
    The factory ensures that all generated models are fully compatible with Pydantic
    while inheriting all StructuredModel comparison capabilities.
    """

    @staticmethod
    def create_model_from_json(
        config: Dict[str, Any],
        base_class: Type = None,
        *,
        path_prefix: str = "",
        model_id: Optional[str] = None,
    ) -> Type:
        """Create a StructuredModel subclass from JSON configuration.
        
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
                   - infer_unspecified_fields: When true, a primitive field that
                     names no comparator has its comparator and threshold inferred
                     from its type and name instead of being rejected. A field can
                     opt in on its own with "comparator": "auto", which wins over
                     this flag in both directions. Default false, because enabling
                     it moves reported metrics.

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
            base_class: The base class to extend (typically StructuredModel).
                       If None, will be imported to avoid circular dependency.
            path_prefix: Dotted path of the field this model hangs off, ``""``
                       for a root model and ``"billing."`` for the model built
                       from a nested ``billing`` field. Internal: set by the
                       nested-model recursion so a warning names the full path.
            model_id: ``model_identity`` of the ROOT model, computed here when
                       absent and passed down unchanged. Internal: it scopes the
                       warn-once bookkeeping to one model, so a second model in
                       the same process still reports its own misconfiguration.

        Returns:
            A fully functional StructuredModel subclass created with create_model()

        Raises:
            ValueError: If configuration is invalid or contains unsupported types/comparators
            KeyError: If required configuration keys are missing

        Examples:
            >>> from stickler import StructuredModel
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
            >>> ProductClass = ModelFactory.create_model_from_json(config, StructuredModel)
            >>> isinstance(ProductClass.model_fields, dict)  # Full Pydantic compatibility
            True
            >>> product = ProductClass(name="Widget", price=29.99)
            >>> product.name
            'Widget'
            >>> result = product.compare_with(ProductClass(name="Widget", price=29.99))
            >>> result["overall_score"]
            1.0
        """
        # Import here to avoid circular dependency
        if base_class is None:
            from .structured_model import StructuredModel
            base_class = StructuredModel

        # Computed once, at the root, and then passed down untouched: a nested
        # model's synthesised name is derived from the field it hangs off, so two
        # different configs each with a `billing` field would otherwise produce
        # the same identity and share a warn-once slot. Field names are stringified
        # because a non-string key must reach the ValueError below rather than
        # raising TypeError out of the join inside `model_identity`.
        if model_id is None:
            fields = config.get("fields") if isinstance(config, dict) else None
            model_id = model_identity(
                config.get("model_name", "DynamicModel")
                if isinstance(config, dict)
                else "DynamicModel",
                tuple(str(name) for name in fields) if isinstance(fields, dict) else (),
            )

        # Validate configuration structure
        ModelFactory.validate_config(config, path_prefix=path_prefix, model_id=model_id)

        # Extract configuration values
        fields_config = config["fields"]
        model_name = config.get("model_name", "DynamicModel")
        match_threshold = config.get("match_threshold", 0.7)
        # Off by default. Turning it on changes reported metrics for every field
        # that named no comparator -- a float previously compared as text starts
        # being compared as a number -- so it is never implied.
        infer_unspecified = config.get("infer_unspecified_fields", False)
        if not isinstance(infer_unspecified, bool):
            raise ValueError(
                "infer_unspecified_fields must be true or false, got: "
                f"{infer_unspecified!r}"
            )

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

        # Validate all field configurations before proceeding (including nested schema validation)
        try:
            converter = get_global_converter()

            # First validate basic field configurations
            validate_fields_config(
                fields_config, path_prefix=path_prefix, model_id=model_id
            )

            # Then validate nested schema rules
            for field_name, field_config in fields_config.items():
                converter.validate_nested_field_schema(
                    field_name, field_config, infer_unspecified=infer_unspecified
                )

        except ValueError as e:
            raise ValueError(f"Invalid field configuration: {e}")

        # Convert field configurations to Pydantic field definitions
        try:
            # `match_threshold` reaches inference too: a mapping field is judged
            # as an object, so it takes the object-level threshold rather than the
            # scalar default. Without it `{"type": "dict"}` sat at 0.7 while
            # `eval_for(cls, match_threshold=0.9)` gave 0.9 for the same field.
            field_definitions = convert_fields_config(
                fields_config,
                infer_unspecified=infer_unspecified,
                match_threshold=match_threshold,
                path_prefix=path_prefix,
                model_id=model_id,
            )
        except ValueError as e:
            raise ValueError(f"Error converting field configurations: {e}")

        # Create the dynamic model extending StructuredModel
        try:
            DynamicClass = create_model(
                model_name,
                __base__=base_class,  # Extend StructuredModel
                **field_definitions,
            )
        except Exception as e:
            raise ValueError(f"Error creating dynamic model: {e}")

        # Set class-level attributes. Assigned after create_model, so
        # StructuredModel.__init_subclass__ has already run and cannot see it --
        # check here too or a config-driven match_threshold=0.0 warns nowhere.
        DynamicClass.match_threshold = match_threshold
        # Dedup on name plus field names, not on `id(DynamicClass)`. `model_name`
        # defaults to "DynamicModel", so keying on the name alone would report
        # the first anonymous config and silence every later one. But keying on
        # object identity is worse: a new class object is created per call, so
        # loading the same config in a loop floods stderr (measured: 200 loads,
        # 192 warnings) and grows the process-global `_warned` set without bound
        # -- the exact behaviour `warn_once` exists to prevent. Field names are
        # stable across repeated loads of one config and differ between configs.
        warn_if_threshold_is_zero(
            match_threshold,
            model_identity(model_name, field_definitions),
            "match_threshold",
        )

        # Add configuration metadata for debugging/introspection
        DynamicClass._model_config = config

        return DynamicClass

    @staticmethod
    def create_model_from_fields(
        model_name: str,
        field_definitions: Dict[str, tuple],
        match_threshold: float = 0.7,
        base_class: Type = None,
    ) -> Type:
        """Create a StructuredModel subclass from pre-converted Pydantic fields.
        
        This method accepts field definitions that are already in Pydantic's format
        (type, Field) tuples, bypassing the need for intermediate configuration format.
        This is used by the JSON Schema converter and other advanced use cases where
        fields have already been converted to Pydantic format.
        
        Args:
            model_name: Name for the generated class. Must be a valid Python identifier.
            field_definitions: Dictionary mapping field names to (type, Field) tuples.
                             Each tuple contains:
                             - type: Python type annotation (str, int, List[str], etc.)
                             - Field: Pydantic Field instance (typically from ComparableField())
            match_threshold: Overall matching threshold for the model (default: 0.7).
                           Must be between 0.0 and 1.0.
            base_class: The base class to extend (typically StructuredModel).
                       If None, will be imported to avoid circular dependency.
            
        Returns:
            A fully functional StructuredModel subclass created with create_model()
            
        Raises:
            ValueError: If model_name is invalid, match_threshold is out of range,
                       or field_definitions are malformed
            
        Examples:
            >>> from pydantic import Field
            >>> from stickler import StructuredModel
            >>> from stickler import ComparableField
            >>> from stickler import LevenshteinComparator
            >>> 
            >>> # Create field definitions directly
            >>> field_defs = {
            ...     "name": (str, ComparableField(
            ...         comparator=LevenshteinComparator(),
            ...         threshold=0.8,
            ...         weight=2.0
            ...     )),
            ...     "age": (int, ComparableField(
            ...         comparator=NumericComparator(),
            ...         threshold=0.9,
            ...         default=0
            ...     ))
            ... }
            >>> 
            >>> PersonClass = ModelFactory.create_model_from_fields(
            ...     model_name="Person",
            ...     field_definitions=field_defs,
            ...     match_threshold=0.8,
            ...     base_class=StructuredModel
            ... )
            >>> 
            >>> person = PersonClass(name="Alice", age=30)
            >>> person.name
            'Alice'
        """
        # Import here to avoid circular dependency
        if base_class is None:
            from .structured_model import StructuredModel
            base_class = StructuredModel

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

        # Validate field_definitions structure
        if not isinstance(field_definitions, dict):
            raise ValueError("field_definitions must be a dictionary")

        if len(field_definitions) == 0:
            raise ValueError("field_definitions must contain at least one field")

        # Validate each field definition is a tuple with 2 elements
        for field_name, field_def in field_definitions.items():
            if not isinstance(field_def, tuple) or len(field_def) != 2:
                raise ValueError(
                    f"Field definition for '{field_name}' must be a tuple of (type, Field), "
                    f"got: {type(field_def)}"
                )

        # Create the dynamic model extending StructuredModel
        try:
            DynamicClass = create_model(
                model_name,
                __base__=base_class,
                **field_definitions,
            )
        except Exception as e:
            raise ValueError(f"Error creating dynamic model: {e}")

        # Set class-level attributes. Assigned after create_model, so
        # StructuredModel.__init_subclass__ has already run and cannot see it --
        # check here too or a config-driven match_threshold=0.0 warns nowhere.
        DynamicClass.match_threshold = match_threshold
        # Dedup on name plus field names, not on `id(DynamicClass)`. `model_name`
        # defaults to "DynamicModel", so keying on the name alone would report
        # the first anonymous config and silence every later one. But keying on
        # object identity is worse: a new class object is created per call, so
        # loading the same config in a loop floods stderr (measured: 200 loads,
        # 192 warnings) and grows the process-global `_warned` set without bound
        # -- the exact behaviour `warn_once` exists to prevent. Field names are
        # stable across repeated loads of one config and differ between configs.
        warn_if_threshold_is_zero(
            match_threshold,
            model_identity(model_name, field_definitions),
            "match_threshold",
        )

        return DynamicClass

    @staticmethod
    def validate_config(
        config: Dict[str, Any], *, path_prefix: str = "", model_id: str = ""
    ) -> None:
        """Validate model configuration before creation.
        
        This method performs structural validation of the configuration dictionary
        to ensure it contains all required keys and has the correct structure.
        It does not validate individual field configurations - that is handled
        by the field_converter module.
        
        Args:
            config: Configuration dictionary to validate
            path_prefix: Dotted path of the field this model hangs off, ``""``
                at the root. Part of the unknown-key warn-once key.
            model_id: ``model_identity`` of the root model. Part of the same key,
                so a second model carrying the same typo still reports it.

        Raises:
            ValueError: If configuration structure is invalid
            
        Examples:
            >>> config = {"fields": {"name": {"type": "str", "comparator": "ExactComparator"}}}
            >>> ModelFactory.validate_config(config)  # No exception raised
            
            >>> invalid_config = {"model_name": "Test"}  # Missing 'fields'
            >>> ModelFactory.validate_config(invalid_config)
            Traceback (most recent call last):
                ...
            ValueError: Configuration must contain 'fields' key
        """
        # Validate configuration is a dictionary
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        # Validate required 'fields' key exists
        if "fields" not in config:
            raise ValueError("Configuration must contain 'fields' key")

        # Same silent drop as at field level: 'match_threshhold' leaves the model
        # at 0.7 and 'model_nam' leaves the class named DynamicModel, neither
        # with any signal. Report and keep building.
        unknown = set(config) - ACCEPTED_MODEL_CONFIG_KEYS
        if unknown:
            warn_once(
                "model-config-unknown-keys",
                # Keyed on the model's identity as well as the keys. Keyed on the
                # key names alone, the first config in a process silenced every
                # later one carrying the same typo. ``key=str`` because a
                # non-string key must not turn this warning into a TypeError,
                # against a docstring that documents ``Raises: ValueError``.
                f"{model_id}|{path_prefix}|"
                f"{','.join(str(k) for k in sorted(unknown, key=str))}",
                f"Model config does not accept "
                f"{', '.join(repr(k) for k in sorted(unknown, key=str))}; "
                f"ignored. It accepts "
                f"{', '.join(sorted(ACCEPTED_MODEL_CONFIG_KEYS))}. "
                f"A misspelled key leaves the model at its default, so the "
                f"value you wrote is not the value being used.",
                category=UserWarning,
                stacklevel=_UNKNOWN_KEY_STACKLEVEL,
            )

        # Validate fields is a non-empty dictionary
        fields_config = config["fields"]
        if not isinstance(fields_config, dict) or len(fields_config) == 0:
            raise ValueError("'fields' must be a non-empty dictionary")
