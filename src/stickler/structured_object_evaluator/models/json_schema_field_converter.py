"""JSON Schema field converter for dynamic model creation.

This module provides utilities for converting JSON Schema properties to
Pydantic Field instances with ComparableField functionality.
"""

from typing import Dict, Any, Tuple, Type, List, Optional
from pydantic import Field

from .comparable_field import ComparableField
from .comparator_registry import create_comparator


# Type mapping from JSON Schema types to Python types
JSON_TYPE_TO_PYTHON_TYPE = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}

# Default comparator mapping from JSON Schema types to comparator class names
JSON_TYPE_TO_DEFAULT_COMPARATOR = {
    "string": "LevenshteinComparator",
    "number": "NumericComparator",
    "integer": "NumericComparator",
    "boolean": "ExactComparator",
}


class JsonSchemaFieldConverter:
    """Converter for JSON Schema properties to Pydantic fields with comparison capabilities.
    
    This class follows the same pattern as FieldConverter but reads from JSON Schema format.
    It extracts x-aws-stickler-* extensions and calls ComparableField() to create Pydantic Fields.
    """

    def __init__(self, schema: Dict[str, Any]):
        """Initialize with a JSON Schema document.
        
        Args:
            schema: JSON Schema document (already validated)
        """
        self.schema = schema
        self.definitions = schema.get("definitions", {})
        self.defs = schema.get("$defs", {})  # JSON Schema draft 2019-09+

    def convert_properties_to_fields(
        self, properties: Dict[str, Any], required: List[str]
    ) -> Dict[str, Tuple[Type, Any]]:
        """Convert JSON Schema properties to Pydantic field definitions.
        
        This is the main entry point, similar to FieldConverter.convert_fields_config().
        
        Args:
            properties: JSON Schema properties object
            required: List of required field names
            
        Returns:
            Dictionary mapping field names to (type, Field) tuples for create_model()
        """
        field_definitions = {}
        for field_name, property_schema in properties.items():
            is_required = field_name in required
            field_type, field = self.convert_property_to_field(
                field_name, property_schema, is_required
            )
            field_definitions[field_name] = (field_type, field)
        return field_definitions

    def convert_property_to_field(
        self, field_name: str, property_schema: Dict[str, Any], is_required: bool
    ) -> Tuple[Type, Any]:
        """Convert a single JSON Schema property to a Pydantic field.
        
        Similar to FieldConverter.convert_field_config(), but reads JSON Schema format.
        
        Args:
            field_name: Name of the field
            property_schema: JSON Schema for this property
            is_required: Whether this field is required
            
        Returns:
            Tuple of (field_type, pydantic_field) where pydantic_field is from ComparableField()
        """
        # Handle $ref
        if "$ref" in property_schema:
            property_schema = self._resolve_ref(property_schema["$ref"])
        
        # Get JSON Schema type
        json_type = property_schema.get("type")
        
        # Handle nested objects
        if json_type == "object":
            return self._handle_nested_object(field_name, property_schema, is_required)
        
        # Handle arrays
        if json_type == "array":
            return self._handle_array_type(field_name, property_schema, is_required)
        
        # Handle primitive types
        field_type = self._map_json_type_to_python_type(json_type)
        
        # Extract x-aws-stickler-* extensions
        extensions = self._extract_stickler_extensions(property_schema)
        
        # Get comparator (from extension or default)
        comparator = extensions.get("comparator") or self._get_default_comparator_for_type(json_type)
        
        # Get other parameters
        threshold = extensions.get("threshold", 0.5)
        weight = extensions.get("weight", 1.0)
        clip_under_threshold = extensions.get("clip_under_threshold", True)
        aggregate = extensions.get("aggregate", False)
        
        # Get Pydantic field parameters
        default = property_schema.get("default", ... if is_required else None)
        description = property_schema.get("description")
        examples = property_schema.get("examples")
        
        # Call ComparableField() to create the Pydantic Field
        field = ComparableField(
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            clip_under_threshold=clip_under_threshold,
            aggregate=aggregate,
            default=default,
            description=description,
            examples=examples
        )
        
        return field_type, field

    def _map_json_type_to_python_type(self, json_type: str) -> Type:
        """Map JSON Schema type to Python type.
        
        Args:
            json_type: JSON Schema type string
            
        Returns:
            Python type
            
        Raises:
            ValueError: If json_type is not supported
        """
        if json_type not in JSON_TYPE_TO_PYTHON_TYPE:
            raise ValueError(
                f"Unsupported JSON Schema type: {json_type}. "
                f"Supported types: {list(JSON_TYPE_TO_PYTHON_TYPE.keys())}"
            )
        return JSON_TYPE_TO_PYTHON_TYPE[json_type]

    def _get_default_comparator_for_type(self, json_type: str):
        """Get default comparator instance for a JSON Schema type.
        
        Args:
            json_type: JSON Schema type string
            
        Returns:
            Comparator instance
        """
        comparator_name = JSON_TYPE_TO_DEFAULT_COMPARATOR.get(
            json_type, "LevenshteinComparator"
        )
        return create_comparator(comparator_name, {})

    def _extract_stickler_extensions(
        self, property_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract x-aws-stickler-* extensions from property schema.
        
        Args:
            property_schema: JSON Schema property object
            
        Returns:
            Dictionary with extracted comparison configuration
            
        Raises:
            ValueError: If extension values are invalid
        """
        extensions = {}
        
        # Extract comparator
        if "x-aws-stickler-comparator" in property_schema:
            comparator_name = property_schema["x-aws-stickler-comparator"]
            try:
                extensions["comparator"] = self._create_comparator_from_name(comparator_name)
            except Exception as e:
                raise ValueError(
                    f"Invalid x-aws-stickler-comparator '{comparator_name}': {e}"
                )
        
        # Extract and validate threshold
        if "x-aws-stickler-threshold" in property_schema:
            threshold = property_schema["x-aws-stickler-threshold"]
            if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"x-aws-stickler-threshold must be a number between 0.0 and 1.0, got: {threshold}"
                )
            extensions["threshold"] = threshold
        
        # Extract and validate weight
        if "x-aws-stickler-weight" in property_schema:
            weight = property_schema["x-aws-stickler-weight"]
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(
                    f"x-aws-stickler-weight must be a positive number, got: {weight}"
                )
            extensions["weight"] = weight
        
        # Extract boolean parameters
        if "x-aws-stickler-clip-under-threshold" in property_schema:
            clip_value = property_schema["x-aws-stickler-clip-under-threshold"]
            if not isinstance(clip_value, bool):
                raise ValueError(
                    f"x-aws-stickler-clip-under-threshold must be a boolean, got: {type(clip_value)}"
                )
            extensions["clip_under_threshold"] = clip_value
        
        if "x-aws-stickler-aggregate" in property_schema:
            aggregate_value = property_schema["x-aws-stickler-aggregate"]
            if not isinstance(aggregate_value, bool):
                raise ValueError(
                    f"x-aws-stickler-aggregate must be a boolean, got: {type(aggregate_value)}"
                )
            extensions["aggregate"] = aggregate_value
        
        return extensions

    def _create_comparator_from_name(self, comparator_name: str):
        """Create a comparator instance from its class name.
        
        Args:
            comparator_name: Name of the comparator class
            
        Returns:
            Comparator instance
            
        Raises:
            KeyError: If comparator name is not registered
        """
        # Use existing comparator_registry
        return create_comparator(comparator_name, {})

    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        """Resolve a $ref reference within the schema.
        
        Args:
            ref: Reference string (e.g., "#/definitions/Address")
            
        Returns:
            Resolved schema object
            
        Raises:
            ValueError: If reference format is unsupported or reference not found
        """
        # Handle #/definitions/Name and #/$defs/Name
        if ref.startswith("#/definitions/"):
            name = ref.split("/")[-1]
            if name not in self.definitions:
                raise ValueError(
                    f"Reference '{ref}' not found in schema definitions. "
                    f"Available: {list(self.definitions.keys())}"
                )
            return self.definitions[name]
        elif ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name not in self.defs:
                raise ValueError(
                    f"Reference '{ref}' not found in schema $defs. "
                    f"Available: {list(self.defs.keys())}"
                )
            return self.defs[name]
        else:
            raise ValueError(
                f"Unsupported $ref format: {ref}. "
                "Only '#/definitions/' and '#/$defs/' references are supported."
            )

    def _handle_nested_object(
        self, field_name: str, property_schema: Dict[str, Any], is_required: bool
    ) -> Tuple[Type, Any]:
        """Handle nested object type (creates nested StructuredModel).
        
        Args:
            field_name: Name of the field
            property_schema: JSON Schema for the nested object
            is_required: Whether this field is required
            
        Returns:
            Tuple of (NestedModel, ComparableField)
        """
        from typing import List
        
        # Recursively create nested model from the nested schema
        # Import here to avoid circular dependency
        from .structured_model import StructuredModel
        NestedModel = StructuredModel.from_json_schema(property_schema)
        
        # Extract extensions for the field itself
        extensions = self._extract_stickler_extensions(property_schema)
        weight = extensions.get("weight", 1.0)
        clip_under_threshold = extensions.get("clip_under_threshold", True)
        aggregate = extensions.get("aggregate", False)
        
        # Get default value
        default = property_schema.get("default", ... if is_required else None)
        description = property_schema.get("description")
        
        # Create ComparableField with dummy comparator (not used for StructuredModel)
        from stickler.comparators.levenshtein import LevenshteinComparator
        field = ComparableField(
            comparator=LevenshteinComparator(),  # Not used for nested models
            threshold=0.7,  # Use model's match_threshold instead
            weight=weight,
            clip_under_threshold=clip_under_threshold,
            aggregate=aggregate,
            default=default,
            description=description
        )
        
        return NestedModel, field

    def _handle_array_type(
        self, field_name: str, property_schema: Dict[str, Any], is_required: bool
    ) -> Tuple[Type, Any]:
        """Handle array type (creates List field).
        
        Args:
            field_name: Name of the field
            property_schema: JSON Schema for the array
            is_required: Whether this field is required
            
        Returns:
            Tuple of (List[ElementType], ComparableField)
        """
        from typing import List
        
        items_schema = property_schema.get("items", {})
        
        # Handle $ref in items
        if "$ref" in items_schema:
            items_schema = self._resolve_ref(items_schema["$ref"])
        
        items_type = items_schema.get("type")
        
        # Array of objects -> List[StructuredModel]
        if items_type == "object":
            from .structured_model import StructuredModel
            ElementModel = StructuredModel.from_json_schema(items_schema)
            field_type = List[ElementModel]
            # Use default comparator for the element type
            comparator = self._get_default_comparator_for_type("string")
        else:
            # Array of primitives -> List[primitive]
            element_type = self._map_json_type_to_python_type(items_type)
            field_type = List[element_type]
            # Use default comparator for the element type
            comparator = self._get_default_comparator_for_type(items_type)
        
        # Extract extensions from the array property itself
        extensions = self._extract_stickler_extensions(property_schema)
        # Override comparator if specified in extensions
        if "comparator" in extensions:
            comparator = extensions["comparator"]
        
        threshold = extensions.get("threshold", 0.5)
        weight = extensions.get("weight", 1.0)
        clip_under_threshold = extensions.get("clip_under_threshold", True)
        aggregate = extensions.get("aggregate", False)
        
        # Get default
        default = property_schema.get("default", ... if is_required else None)
        description = property_schema.get("description")
        
        # Create ComparableField
        field = ComparableField(
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            clip_under_threshold=clip_under_threshold,
            aggregate=aggregate,
            default=default,
            description=description
        )
        
        return field_type, field
