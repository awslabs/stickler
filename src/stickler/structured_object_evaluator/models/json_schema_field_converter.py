"""JSON Schema field converter for dynamic model creation.

This module provides utilities for converting JSON Schema properties to
Pydantic Field instances with ComparableField functionality.
"""

from typing import Any, Dict, List, Tuple, Type, get_args, get_origin

from pydantic.fields import FieldInfo, PydanticUndefined

from .configuration_helper import ConfigurationHelper
from .optional_annotation import is_optional_union, unwrap_annotated, unwrap_optional

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

# The three builtins ComparableField's own docstring recommends for a mutable
# default (see its `default` Args entry). Calling one to recover the value it
# produces is safe: each takes no arguments, is pure, and has no side effects,
# so invoking it here to export "default": [] cannot do anything a caller
# would not already expect model construction itself to do. Any other factory
# is left unexported rather than invoked blindly -- a caller-supplied callable
# might hit a database, a clock, or a counter, and calling it during export
# (not construction) would be a surprising side effect nobody asked for.
_SAFE_EXPORT_FACTORIES = (list, dict, set)


def _json_value(value: Any) -> Any:
    """Return ``value`` in a form JSON can hold: a set becomes a list.

    Both exporters write JSON, which has no set type, so a set default --
    from ``default_factory=set`` or written as ``default={"a"}`` -- made
    ``json.dumps()`` on the export raise ``TypeError: Object of type set is
    not JSON serializable``. Sorted where the elements allow it, so an
    exported file does not reorder between runs under hash randomisation.
    """
    if isinstance(value, (set, frozenset)):
        try:
            return sorted(value)
        except TypeError:
            return list(value)
    return value


def resolve_exportable_default(field_info: FieldInfo) -> Tuple[bool, Any]:
    """Return ``(True, value)`` to export, or ``(False, None)`` to omit the key.

    A field built from a real ``default_factory`` (see ``ComparableField``'s
    ``_DEFAULT_UNSET`` handling) reports ``field_info.default is
    PydanticUndefined`` even though ``field_info.is_required()`` is False --
    the factory, not a static value, supplies each instance's default.
    Exporting ``PydanticUndefined`` itself is not JSON serializable, and
    omitting the key rebuilt the field as required (``to_stickler_config()``)
    or with ``None`` (``to_json_schema()``) (awslabs/stickler#360).

    For ``list``/``dict``/``set`` -- the factories ComparableField's own
    docstring recommends -- calling the factory to recover its produced value
    is safe and lets the exported ``"default": []`` round-trip correctly,
    which is what a reader of the exported config or schema expects. Any
    other factory returns ``(False, None)``: guessing at an arbitrary
    callable's value would be worse than omitting the key. Matched by
    identity, because ``in`` compares with ``==`` and would consult a
    caller's factory object's own ``__eq__`` during an export.

    Either kind of value goes through :func:`_json_value`.
    """
    if field_info.default is not PydanticUndefined:
        return True, _json_value(field_info.default)
    factory = field_info.default_factory
    if any(factory is safe for safe in _SAFE_EXPORT_FACTORIES):
        return True, _json_value(factory())
    return False, None


def _exports_as_container(annotation: Any) -> bool:
    """Whether both exporters write this annotation as a list or a mapping.

    Those are the only exported types a factory's ``[]`` or ``{}`` fits.
    Anything else exports as a scalar (``Set[str]`` included, as ``"str"``),
    and a collection default written into a scalar field rebuilt a model
    that rejected every value except its own default. ``Annotated`` is
    stripped too: pydantic keeps it inside a union, so
    ``Optional[Annotated[List[str], ...]]`` read as a scalar and exported no
    default.
    """
    inner = ConfigurationHelper.strip_annotation_wrappers(annotation)
    return get_origin(inner) is list or ConfigurationHelper.is_mapping_annotation(inner)


def add_exportable_default(
    target: Dict[str, Any],
    field_info: FieldInfo,
    *,
    include_literal: bool = False,
    stickler_config: bool = False,
) -> None:
    """Set ``target["default"]`` to the field's exportable default, if it has one.

    Every export branch calls this, so the rules live in one place: nothing
    for a required field; a factory's value only where the exported type can
    hold it (:func:`_exports_as_container`) and the factory is one
    :func:`resolve_exportable_default` will call. A literal ``default=...`` is
    written only with ``include_literal``: ``to_stickler_config()`` has always
    exported it and ``to_json_schema()`` never has.

    With ``stickler_config``, an empty ``[]`` or ``{}`` default on a nullable
    field is not written. A Stickler config has no nullable flag, and
    ``model_from_json()`` widens a field to ``Optional`` only when its default
    is ``None``, so ``Optional[List[int]] = ComparableField(default_factory=list)``
    rebuilt as ``List[int]`` and rejected the ``None`` the original accepts.
    Omitted, the default rebuilds as ``None``, which an empty list or mapping
    scores identically to. Only an empty default is dropped: a non-empty one
    does not score like ``None``, so it is written and still rebuilds
    non-nullable, as before #306. The JSON Schema path keeps the default, and
    its rebuild still accepts ``None`` because the importer
    (``JsonSchemaImporter``) widens every non-required field to ``Optional``,
    not because the exported ``type`` names ``null``.
    """
    if field_info.is_required():
        return
    if field_info.default is PydanticUndefined:
        if not _exports_as_container(field_info.annotation):
            return
    elif not include_literal:
        return
    has_default, value = resolve_exportable_default(field_info)
    if not has_default:
        return
    if (
        stickler_config
        and isinstance(value, (list, dict))
        and not value
        and is_optional_union(unwrap_annotated(field_info.annotation))
    ):
        return
    target["default"] = value


def _mapping_value_type(annotation: Any) -> Any:
    """The value type a mapping annotation names, or ``Any`` if it names none."""
    args = get_args(annotation)
    return args[1] if len(args) == 2 else Any


def _stickler_mapping_type(annotation: Any) -> str:
    """``Dict[str, int]`` for a mapping of named primitives, else ``dict``.

    ``model_from_json`` resolves both spellings. A key or value type it cannot
    be handed by name (a model, a nested container) falls back to ``dict``,
    which keeps the field a mapping and drops only the per-value check.
    """
    names = [
        "Any" if arg is Any else PYTHON_TYPE_TO_STICKLER_TYPE.get(arg)
        for arg in get_args(annotation)
    ]
    if len(names) == 2 and all(names):
        return f"Dict[{names[0]}, {names[1]}]"
    return "dict"


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

        # A mapping exports as an object, not the "string" fallback below: as
        # "string" it re-imported as `Optional[str]` and could not hold one.
        # Object keys are strings in JSON, so only the value type is carried,
        # and only when it is a primitive JSON has a name for.
        is_mapping = ConfigurationHelper.is_mapping_annotation(field_type)
        if is_mapping:
            json_type = "object"
        else:
            json_type = PYTHON_TYPE_TO_JSON_TYPE.get(field_type, "string")
        property_schema = {"type": [json_type, "null"] if is_nullable else json_type}
        if is_mapping:
            value_type = PYTHON_TYPE_TO_JSON_TYPE.get(_mapping_value_type(field_type))
            if value_type:
                property_schema["additionalProperties"] = {"type": value_type}
        add_exportable_default(property_schema, field_info)

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
        # A mapping fell back to "str" here, so it rebuilt as a string field
        # that could hold no mapping, including the `{}` its factory exports.
        if ConfigurationHelper.is_mapping_annotation(field_type):
            stickler_type = _stickler_mapping_type(field_type)
        else:
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
        add_exportable_default(
            field_config, field_info, include_literal=True, stickler_config=True
        )
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
