"""Factory for creating dynamic StructuredModel subclasses from JSON configuration.

This module provides the ModelFactory class that encapsulates the logic for creating
dynamic StructuredModel subclasses from JSON configuration. It uses the factory pattern
to separate model creation concerns from the core StructuredModel class.
"""

from typing import Any, Dict, Type
from pydantic import create_model

from .field_converter import (
    convert_fields_config,
    validate_fields_config,
    get_global_converter,
)


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
        config: Dict[str, Any], base_class: Type = None
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

        Returns:
            A fully functional StructuredModel subclass created with create_model()

        Raises:
            ValueError: If configuration is invalid or contains unsupported types/comparators
            KeyError: If required configuration keys are missing

        Examples:
            >>> from stickler.structured_object_evaluator.models import StructuredModel
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

        # Validate configuration structure
        ModelFactory.validate_config(config)

        # Extract configuration values
        fields_config = config["fields"]
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

        # Validate all field configurations before proceeding (including nested schema validation)
        try:
            converter = get_global_converter()

            # First validate basic field configurations
            validate_fields_config(fields_config)

            # Then validate nested schema rules
            for field_name, field_config in fields_config.items():
                converter.validate_nested_field_schema(field_name, field_config)

        except ValueError as e:
            raise ValueError(f"Invalid field configuration: {e}")

        # Convert field configurations to Pydantic field definitions
        try:
            field_definitions = convert_fields_config(fields_config)
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

        # Set class-level attributes
        DynamicClass.match_threshold = match_threshold

        # Add configuration metadata for debugging/introspection
        DynamicClass._model_config = config

        return DynamicClass

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate model configuration before creation.
        
        This method performs structural validation of the configuration dictionary
        to ensure it contains all required keys and has the correct structure.
        It does not validate individual field configurations - that is handled
        by the field_converter module.
        
        Args:
            config: Configuration dictionary to validate
            
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

        # Validate fields is a non-empty dictionary
        fields_config = config["fields"]
        if not isinstance(fields_config, dict) or len(fields_config) == 0:
            raise ValueError("'fields' must be a non-empty dictionary")
