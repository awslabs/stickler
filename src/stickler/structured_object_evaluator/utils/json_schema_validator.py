"""JSON Schema validation utilities for StructuredModel creation.

This module provides validation functions to ensure JSON Schema documents
conform to the JSON Schema specification before being used to create
StructuredModel classes.
"""

from typing import Any, Dict

import jsonschema
from jsonschema import Draft7Validator, ValidationError


def validate_json_schema(schema: Dict[str, Any]) -> None:
    """Validate that a dictionary is a valid JSON Schema document.
    
    This function validates the schema against JSON Schema draft-07 specification.
    It checks that the schema structure is well-formed and follows JSON Schema rules.
    
    Args:
        schema: Dictionary representing a JSON Schema document
        
    Raises:
        ValueError: If schema is not a dictionary or is empty
        jsonschema.exceptions.SchemaError: If schema structure is malformed
        
    Examples:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"}
        ...     }
        ... }
        >>> validate_json_schema(schema)  # No error raised
        
        >>> invalid_schema = {"type": "invalid_type"}
        >>> validate_json_schema(invalid_schema)  # Raises SchemaError
    """
    if not isinstance(schema, dict):
        raise ValueError(
            f"Schema must be a dictionary, got {type(schema).__name__}"
        )
    
    if not schema:
        raise ValueError("Schema cannot be empty")
    
    # Validate against JSON Schema draft-07 specification
    # This will raise jsonschema.exceptions.SchemaError if invalid
    Draft7Validator.check_schema(schema)


def validate_instance_against_schema(
    instance: Dict[str, Any], schema: Dict[str, Any]
) -> None:
    """Validate that an instance conforms to a JSON Schema.
    
    This function validates that a data instance matches the structure
    and constraints defined in a JSON Schema document.
    
    Args:
        instance: Dictionary representing data to validate
        schema: JSON Schema document to validate against
        
    Raises:
        jsonschema.exceptions.ValidationError: If instance doesn't match schema
        
    Examples:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"}
        ...     },
        ...     "required": ["name"]
        ... }
        >>> instance = {"name": "Alice"}
        >>> validate_instance_against_schema(instance, schema)  # No error
        
        >>> invalid_instance = {"age": 30}
        >>> validate_instance_against_schema(invalid_instance, schema)  # Raises ValidationError
    """
    validator = Draft7Validator(schema)
    validator.validate(instance)
