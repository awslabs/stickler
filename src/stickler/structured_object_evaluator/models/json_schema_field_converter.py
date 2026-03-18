"""JSON Schema field converter for dynamic model creation.

This module provides utilities for converting JSON Schema properties to
Pydantic Field instances with ComparableField functionality.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic.fields import FieldInfo

from .comparable_field import ComparableField
from .comparator_registry import create_comparator

# Type mapping from JSON Schema types to Python types
JSON_TYPE_TO_PYTHON_TYPE = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}

# Bidirectional type mappings for export
PYTHON_TYPE_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

PYTHON_TYPE_TO_STICKLER_TYPE = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}

# Default comparator mapping from JSON Schema types to comparator class names
JSON_TYPE_TO_DEFAULT_COMPARATOR = {
    "string": "LevenshteinComparator",
    "number": "NumericComparator",
    "integer": "NumericComparator",
    "boolean": "ExactComparator",
}


class JsonSchemaFieldConverter:
    """Converter for JSON Schema properties to/from Pydantic fields with comparison capabilities.
    
    This class handles bidirectional conversion:
    - Import: JSON Schema → Pydantic Field (existing functionality)
    - Export: Pydantic Field → JSON Schema (new functionality)
    
    It extracts x-aws-stickler-* extensions and calls ComparableField() to create Pydantic Fields.
    """

    def __init__(self, schema: Dict[str, Any], field_path: str = ""):
        """Initialize with a JSON Schema document.
        
        Args:
            schema: JSON Schema document (already validated)
            field_path: Current field path for error messages (e.g., "address.street")
        """
        self.schema = schema
        self.definitions = schema.get("definitions", {})
        self.defs = schema.get("$defs", {})  # JSON Schema draft 2019-09+
        self.field_path = field_path

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
            # Build field path for nested error messages
            current_path = f"{self.field_path}.{field_name}" if self.field_path else field_name
            try:
                field_type, field = self.convert_property_to_field(
                    field_name, property_schema, is_required, current_path
                )
                field_definitions[field_name] = (field_type, field)
            except ValueError as e:
                # Re-raise with field path context if not already included
                if "field '" not in str(e).lower():
                    raise ValueError(f"Error in field '{current_path}': {e}")
                raise
        return field_definitions

    def convert_property_to_field(
        self, field_name: str, property_schema: Dict[str, Any], is_required: bool, field_path: str = None
    ) -> Tuple[Type, Any]:
        """Convert a single JSON Schema property to a Pydantic field.
        
        Similar to FieldConverter.convert_field_config(), but reads JSON Schema format.
        
        Args:
            field_name: Name of the field
            property_schema: JSON Schema for this property
            is_required: Whether this field is required
            field_path: Full path to this field for error messages
            
        Returns:
            Tuple of (field_type, pydantic_field) where pydantic_field is from ComparableField()
        """
        if field_path is None:
            field_path = field_name
        # Handle $ref
        if "$ref" in property_schema:
            property_schema = self._resolve_ref(property_schema["$ref"])
        
        # Get JSON Schema type
        json_type = property_schema.get("type")
        
        # Handle nested objects
        if json_type == "object":
            return self._handle_nested_object(field_name, property_schema, is_required, field_path)
        
        # Handle arrays
        if json_type == "array":
            return self._handle_array_type(field_name, property_schema, is_required, field_path)
        
        # Handle primitive types
        field_type = self._map_json_type_to_python_type(json_type)

        # Non-required fields accept None
        if not is_required:
            field_type = Optional[field_type]
        
        # Extract x-aws-stickler-* extensions
        extensions = self._extract_stickler_extensions(property_schema, field_path)
        
        # Get comparator (from extension or default)
        comparator = extensions.get("comparator") or self._get_default_comparator_for_type(json_type)
        
        # Get other parameters
        threshold = extensions.get("threshold", 0.5)
        weight = extensions.get("weight", 1.0)
        clip_under_threshold = extensions.get("clip_under_threshold", True)
        
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
        self, property_schema: Dict[str, Any], field_path: str = ""
    ) -> Dict[str, Any]:
        """Extract x-aws-stickler-* extensions from property schema.
        
        Args:
            property_schema: JSON Schema property object
            field_path: Full path to this field for error messages
            
        Returns:
            Dictionary with extracted comparison configuration
            
        Raises:
            ValueError: If extension values are invalid
        """
        extensions = {}
        
        # Extract comparator
        if "x-aws-stickler-comparator" in property_schema:
            comparator_name = property_schema["x-aws-stickler-comparator"]
            comparator_config = property_schema.get("x-aws-stickler-comparator-config", {})
            try:
                extensions["comparator"] = create_comparator(comparator_name, comparator_config)
            except Exception as e:
                field_info = f" in field '{field_path}'" if field_path else ""
                raise ValueError(
                    f"Invalid x-aws-stickler-comparator '{comparator_name}'{field_info}: {e}"
                )
        
        # Extract and validate threshold
        if "x-aws-stickler-threshold" in property_schema:
            threshold = property_schema["x-aws-stickler-threshold"]
            if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                field_info = f" for field '{field_path}'" if field_path else ""
                raise ValueError(
                    f"x-aws-stickler-threshold must be a number between 0.0 and 1.0{field_info}, got: {threshold}"
                )
            extensions["threshold"] = threshold
        
        # Extract and validate weight
        if "x-aws-stickler-weight" in property_schema:
            weight = property_schema["x-aws-stickler-weight"]
            if not isinstance(weight, (int, float)) or weight <= 0:
                field_info = f" for field '{field_path}'" if field_path else ""
                raise ValueError(
                    f"x-aws-stickler-weight must be a positive number{field_info}, got: {weight}"
                )
            extensions["weight"] = weight
        
        # Extract boolean parameters
        if "x-aws-stickler-clip-under-threshold" in property_schema:
            clip_value = property_schema["x-aws-stickler-clip-under-threshold"]
            if not isinstance(clip_value, bool):
                field_info = f" for field '{field_path}'" if field_path else ""
                raise ValueError(
                    f"x-aws-stickler-clip-under-threshold must be a boolean{field_info}, got: {type(clip_value).__name__}"
                )
            extensions["clip_under_threshold"] = clip_value
        
        if "x-aws-stickler-aggregate" in property_schema:
            aggregate_value = property_schema["x-aws-stickler-aggregate"]
            if not isinstance(aggregate_value, bool):
                field_info = f" for field '{field_path}'" if field_path else ""
                raise ValueError(
                    f"x-aws-stickler-aggregate must be a boolean{field_info}, got: {type(aggregate_value).__name__}"
                )
            extensions["aggregate"] = aggregate_value
        
        return extensions

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
        self, field_name: str, property_schema: Dict[str, Any], is_required: bool, field_path: str = None
    ) -> Tuple[Type, Any]:
        """Handle nested object type (creates nested StructuredModel).
        
        Args:
            field_name: Name of the field
            property_schema: JSON Schema for the nested object
            is_required: Whether this field is required
            field_path: Full path to this field for error messages
            
        Returns:
            Tuple of (NestedModel, ComparableField)
        """
        if field_path is None:
            field_path = field_name
        
        # Recursively create nested model from the nested schema
        # Import here to avoid circular dependency
        from .structured_model import StructuredModel
        
        # CRITICAL: Pass parent schema's definitions/defs to nested schema
        # so that nested $refs can be resolved
        enriched_schema = dict(property_schema)
        if self.definitions and "definitions" not in enriched_schema:
            enriched_schema["definitions"] = self.definitions
        if self.defs and "$defs" not in enriched_schema:
            enriched_schema["$defs"] = self.defs
        
        try:
            NestedModel = StructuredModel._from_json_schema_internal(enriched_schema, field_path=field_path)
        except ValueError:
            # Nested errors already have field path context
            raise
        
        # Extract extensions for the field itself
        extensions = self._extract_stickler_extensions(property_schema, field_path)
        weight = extensions.get("weight", 1.0)
        clip_under_threshold = extensions.get("clip_under_threshold", True)
        
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
            default=default,
            description=description
        )
        
        # Non-required nested objects accept None
        if not is_required:
            NestedModel = Optional[NestedModel]

        return NestedModel, field
    def _handle_array_type(
        self, field_name: str, property_schema: Dict[str, Any], is_required: bool, field_path: str = None
    ) -> Tuple[Type, Any]:
        """Handle array type (creates List field).
        
        Args:
            field_name: Name of the field
            property_schema: JSON Schema for the array
            is_required: Whether this field is required
            field_path: Full path to this field for error messages
            
        Returns:
            Tuple of (List[ElementType], ComparableField)
        """
        if field_path is None:
            field_path = field_name
        from typing import List
        
        items_schema = property_schema.get("items", {})
        
        # Handle $ref in items
        if "$ref" in items_schema:
            items_schema = self._resolve_ref(items_schema["$ref"])
        
        items_type = items_schema.get("type")
        
        # Array of objects -> List[StructuredModel]
        if items_type == "object":
            from .structured_model import StructuredModel
            try:
                ElementModel = StructuredModel._from_json_schema_internal(items_schema, field_path=f"{field_path}[]")
            except ValueError:
                # Nested errors already have field path context
                raise
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
        extensions = self._extract_stickler_extensions(property_schema, field_path)
        # Override comparator if specified in extensions
        if "comparator" in extensions:
            comparator = extensions["comparator"]
        
        threshold = extensions.get("threshold", 0.5)
        weight = extensions.get("weight", 1.0)
        clip_under_threshold = extensions.get("clip_under_threshold", True)
        
        # Get default
        default = property_schema.get("default", ... if is_required else None)
        description = property_schema.get("description")
        
        # Create ComparableField
        field = ComparableField(
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            clip_under_threshold=clip_under_threshold,
            default=default,
            description=description
        )

        # Non-required arrays accept None
        if not is_required:
            field_type = Optional[field_type]

        return field_type, field

    def field_to_property(self, field_type: Type, field_info: FieldInfo) -> Dict[str, Any]:
        """Convert Pydantic field to JSON Schema property.
        
        Extracts comparison metadata from the field's json_schema_extra attribute
        and formats it as x-aws-stickler-* extensions compatible with from_json_schema().
        
        Args:
            field_type: Python type annotation (e.g., str, int, float)
            field_info: Pydantic FieldInfo object containing field metadata
            
        Returns:
            JSON Schema property dict with x-aws-stickler-* extensions
        """
        json_type = PYTHON_TYPE_TO_JSON_TYPE.get(field_type, "string")
        property_schema = {"type": json_type}
        
        # Extract metadata and build extensions using consolidated helper
        metadata = self._extract_field_metadata(field_info)
        extensions = self._build_comparison_extensions(metadata, output_format="json_schema")
        property_schema.update(extensions)
        
        # Add Pydantic field params
        if field_info.description:
            property_schema["description"] = field_info.description
        if field_info.alias:
            property_schema["alias"] = field_info.alias
        if field_info.examples:
            property_schema["examples"] = field_info.examples
        
        return property_schema
    
    def field_to_stickler_config(self, field_type: Type, field_info: FieldInfo) -> Dict[str, Any]:
        """Convert Pydantic field to Stickler config format.
        
        Extracts comparison metadata and formats it as custom Stickler configuration
        compatible with model_from_json().
        
        Args:
            field_type: Python type annotation (e.g., str, int, float)
            field_info: Pydantic FieldInfo object containing field metadata
            
        Returns:
            Stickler field config dict with type, comparator, threshold, etc.
        """
        stickler_type = PYTHON_TYPE_TO_STICKLER_TYPE.get(field_type, "str")
        field_config = {"type": stickler_type}
        
        # Extract metadata and build extensions using consolidated helper
        metadata = self._extract_field_metadata(field_info)
        extensions = self._build_comparison_extensions(metadata, output_format="stickler_config")
        field_config.update(extensions)
        
        # Add Pydantic field params
        field_config["required"] = field_info.is_required()
        if not field_info.is_required():
            field_config["default"] = field_info.default
        if field_info.description:
            field_config["description"] = field_info.description
        if field_info.alias:
            field_config["alias"] = field_info.alias
        if field_info.examples:
            field_config["examples"] = field_info.examples
        
        return field_config
    
    def _build_comparison_extensions(
        self, 
        metadata: Dict[str, Any], 
        output_format: str = "json_schema"
    ) -> Dict[str, Any]:
        """Build comparison extensions in specified format.
        
        Consolidates duplicate logic from field_to_property() and field_to_stickler_config().
        
        Args:
            metadata: Extracted field metadata from _extract_field_metadata()
            output_format: Output format - "json_schema" or "stickler_config"
        
        Returns:
            Dictionary with comparison extensions in the specified format
        """
        extensions = {}
        if output_format not in ("json_schema", "stickler_config"):
            raise ValueError(f"Unsupported format: {output_format!r}. Use 'json_schema' or 'stickler_config'.")
        prefix = "x-aws-stickler-" if output_format == "json_schema" else ""
        
        # Export comparator class name and configuration
        if metadata.get("comparator"):
            comparator = metadata["comparator"]
            extensions[f"{prefix}comparator"] = comparator.__class__.__name__
            
            # Export comparator configuration (e.g., tolerance, case_sensitive)
            if hasattr(comparator, "config") and comparator.config:
                config_key = f"{prefix}comparator-config" if output_format == "json_schema" else "comparator_config"
                extensions[config_key] = comparator.config
        
        # Export comparison parameters
        if "threshold" in metadata:
            extensions[f"{prefix}threshold"] = metadata["threshold"]
        if "weight" in metadata:
            extensions[f"{prefix}weight"] = metadata["weight"]
        if metadata.get("clip_under_threshold") is not None:
            clip_key = f"{prefix}clip-under-threshold" if output_format == "json_schema" else "clip_under_threshold"
            extensions[clip_key] = metadata["clip_under_threshold"]
        if metadata.get("aggregate") is not None:
            extensions[f"{prefix}aggregate"] = metadata["aggregate"]
        
        return extensions
    
    def _extract_field_metadata(self, field_info: FieldInfo) -> Dict[str, Any]:
        """Extract comparison metadata from field's json_schema_extra.
        
        Only includes attributes that are explicitly set (no default values).
        
        Args:
            field_info: Pydantic FieldInfo object
            
        Returns:
            Dictionary with explicitly set comparator, threshold, weight, etc.
            Empty dict if no metadata found.
        """
        if not hasattr(field_info, "json_schema_extra"):
            return {}
        
        json_func = field_info.json_schema_extra
        if not callable(json_func):
            return {}
        
        # Only include attributes that are explicitly set
        metadata = {}
        
        if hasattr(json_func, "_comparator_instance"):
            metadata["comparator"] = json_func._comparator_instance
        
        if hasattr(json_func, "_threshold"):
            metadata["threshold"] = json_func._threshold
        
        if hasattr(json_func, "_weight"):
            metadata["weight"] = json_func._weight
        
        if hasattr(json_func, "_clip_under_threshold"):
            metadata["clip_under_threshold"] = json_func._clip_under_threshold
        
        if hasattr(json_func, "_aggregate"):
            metadata["aggregate"] = json_func._aggregate
        
        return metadata
