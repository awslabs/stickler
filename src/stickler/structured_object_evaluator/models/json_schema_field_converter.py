"""JSON Schema field converter for dynamic model creation.

This module provides utilities for converting JSON Schema properties to
Pydantic Field instances with ComparableField functionality.
"""

from typing import Any, Dict, List, Tuple, Type

from pydantic.fields import FieldInfo

from .configuration_helper import ConfigurationHelper
from .optional_annotation import unwrap_optional

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


class JsonSchemaFieldConverter:
    """Compatibility entry point for schema import and field export.

    Import delegates to :class:`JsonSchemaImporter`; this class retains the
    existing export helpers used by ``StructuredModel.to_json_schema()``.
    """

    def __init__(self, schema: Dict[str, Any], field_path: str = ""):
        """Initialize with a JSON Schema document.

        Args:
            schema: JSON Schema document (already validated)
            field_path: Current field path for error messages (e.g., "address.street")
        """
        self.schema = schema
        self.field_path = field_path

    def convert_properties_to_fields(
        self, properties: Dict[str, Any], required: List[str]
    ) -> Dict[str, Tuple[Type, Any]]:
        """Delegate JSON Schema parsing and keep only comparison adaptation here."""
        from .json_schema_importer import JsonSchemaImporter

        return JsonSchemaImporter(
            self.schema, field_path=self.field_path
        ).convert_properties_to_fields(properties, required)

    def field_to_property(
        self, field_type: Type, field_info: FieldInfo, is_nullable: bool = False
    ) -> Dict[str, Any]:
        """Convert Pydantic field to JSON Schema property.

        Extracts comparison metadata from the field's json_schema_extra attribute
        and formats it as x-aws-stickler-* extensions compatible with from_json_schema().

        Args:
            field_type: Python type annotation (e.g., str, int, float)
            field_info: Pydantic FieldInfo object containing field metadata
            is_nullable: Whether the source field was Optional[T]; emits the
                list-form ``["X", "null"]`` type so nullability round-trips.

        Returns:
            JSON Schema property dict with x-aws-stickler-* extensions
        """
        # Unwrap Optional[T] in case the caller passes the wrapped type so the
        # nullability still survives the round-trip as the ["X", "null"] idiom.
        # Every spelling, including `X | None`.
        unwrapped, was_optional = unwrap_optional(field_type)
        if was_optional:
            field_type = unwrapped
            is_nullable = True

        # A StructuredModel, or a union with a model arm, must never reach the
        # scalar fallback below. `PYTHON_TYPE_TO_JSON_TYPE.get(..., "string")`
        # would silently discard its schema, emitting one that rebuilds into a
        # structurally different model. Raise rather than produce a wrong
        # schema quietly.
        #
        # Deliberately scoped to models, not to every dict miss:
        # Decimal, UUID and enums legitimately export as "string".
        from .structured_model import StructuredModel

        StructuredModel._refuse_scalar_export(field_type)

        json_type = PYTHON_TYPE_TO_JSON_TYPE.get(field_type, "string")
        property_schema = {"type": [json_type, "null"] if is_nullable else json_type}

        # A plain BaseModel is scored as one object, as a mapping is
        # (`ConfigurationHelper.is_object_grade_annotation`), so it exports as a
        # free-form object, which re-imports as `dict`. As "string" it rebuilt
        # as `str` and rejected the original data (#349).
        if ConfigurationHelper.is_plain_model_annotation(field_type):
            property_schema["type"] = ["object", "null"] if is_nullable else "object"

        # Extract metadata and build extensions using consolidated helper
        metadata = self._extract_field_metadata(field_info)
        extensions = self._build_comparison_extensions(
            metadata, output_format="json_schema"
        )
        property_schema.update(extensions)

        # Add Pydantic field params
        if field_info.description:
            property_schema["description"] = field_info.description
        if field_info.alias:
            property_schema["alias"] = field_info.alias
        if field_info.examples:
            property_schema["examples"] = field_info.examples

        return property_schema

    def field_to_stickler_config(
        self, field_type: Type, field_info: FieldInfo
    ) -> Dict[str, Any]:
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
        # As in field_to_property(): a model or a union with a model arm cannot
        # be written as a scalar, and a plain BaseModel is written as `dict`.
        from .structured_model import StructuredModel

        StructuredModel._refuse_scalar_export(field_type)
        if ConfigurationHelper.is_plain_model_annotation(field_type):
            field_config["type"] = "dict"

        # Extract metadata and build extensions using consolidated helper
        metadata = self._extract_field_metadata(field_info)
        extensions = self._build_comparison_extensions(
            metadata, output_format="stickler_config"
        )
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
        self, metadata: Dict[str, Any], output_format: str = "json_schema"
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
            raise ValueError(
                f"Unsupported format: {output_format!r}. Use 'json_schema' or 'stickler_config'."
            )
        prefix = "x-aws-stickler-" if output_format == "json_schema" else ""

        # Export comparator class name and configuration
        if metadata.get("comparator"):
            comparator = metadata["comparator"]
            extensions[f"{prefix}comparator"] = comparator.__class__.__name__

            # Export comparator configuration (e.g., tolerance, case_sensitive)
            if hasattr(comparator, "config") and comparator.config:
                config_key = (
                    f"{prefix}comparator-config"
                    if output_format == "json_schema"
                    else "comparator_config"
                )
                extensions[config_key] = comparator.config

        # Export comparison parameters
        if "threshold" in metadata:
            extensions[f"{prefix}threshold"] = metadata["threshold"]
        if "weight" in metadata:
            extensions[f"{prefix}weight"] = metadata["weight"]
        if metadata.get("clip_under_threshold") is not None:
            clip_key = (
                f"{prefix}clip-under-threshold"
                if output_format == "json_schema"
                else "clip_under_threshold"
            )
            extensions[clip_key] = metadata["clip_under_threshold"]
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

        return metadata
