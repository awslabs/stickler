"""Import JSON Schema types while keeping Stickler comparison semantics separate."""

from __future__ import annotations

import re
from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from json_schema_to_pydantic import create_model as create_pydantic_model
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .comparable_field import ComparableField
from .comparator_registry import create_comparator
from .model_factory import ModelFactory
from .optional_annotation import is_union, unwrap_optional

_JSON_DEFAULTS = {
    str: ("LevenshteinComparator", 0.5),
    float: ("NumericComparator", 0.5),
    int: ("NumericComparator", 0.5),
    bool: ("ExactComparator", 0.5),
}

_PRESERVED_EXAMPLES_KEY = "x-aws-stickler-internal-examples"
_COMBINERS = frozenset({"allOf", "anyOf", "oneOf"})
_ANNOTATION_KEYWORDS = frozenset(
    {
        "$comment",
        "$id",
        "$schema",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
        _PRESERVED_EXAMPLES_KEY,
    }
)


class JsonSchemaImporter:
    """Turn a JSON Schema into field definitions for ``StructuredModel``.

    ``json-schema-to-pydantic`` owns JSON Schema parsing, reference resolution,
    combiners, constraints, defaults, aliases, and format types. This adapter
    only maps the resulting live Pydantic fields to Stickler comparison fields.
    """

    def __init__(self, schema: Dict[str, Any], field_path: str = ""):
        self.schema = schema
        self.field_path = field_path

    def convert_properties_to_fields(
        self, properties: Dict[str, Any], required: List[str]
    ) -> Dict[str, Tuple[Type, Any]]:
        """Return ``create_model`` field tuples for a schema's named properties."""
        schema = deepcopy(self.schema)
        if "properties" in schema or properties:
            schema["properties"] = properties
        if "required" in schema or required:
            schema["required"] = required

        from ..utils.json_schema_validator import validate_json_schema

        try:
            validate_json_schema(schema)
        except Exception as exc:
            raise ValueError(f"Invalid JSON Schema: {exc}") from exc

        self._validate_supported_shapes(schema)
        if (
            not properties
            and "properties" in self.schema
            and not any(keyword in self.schema for keyword in (*_COMBINERS, "$ref"))
        ):
            return {}
        schema = self._prepare_schema(schema)

        try:
            source_model = create_pydantic_model(schema)
        except Exception as exc:
            location = f" at '{self.field_path}'" if self.field_path else ""
            message = self._translate_import_error(str(exc), schema)
            raise ValueError(
                f"Could not import JSON Schema{location}: {message}"
            ) from exc

        if not source_model.model_fields:
            if "patternProperties" in schema:
                raise ValueError(
                    "JSON Schema patternProperties cannot be represented as named "
                    "StructuredModel fields. Define explicit properties instead."
                )
            return {}

        return self._convert_model_fields(
            source_model,
            field_path=self.field_path,
            building=set(),
        )

    def _convert_model_fields(
        self,
        source_model: Type[BaseModel],
        *,
        field_path: str,
        building: set[Type[BaseModel]],
    ) -> Dict[str, Tuple[Type, Any]]:
        if source_model in building:
            location = f" at '{field_path}'" if field_path else ""
            raise ValueError(
                "Recursive JSON Schema models are not supported by the comparison "
                f"engine{location}. Use a bounded, non-recursive schema."
            )

        building.add(source_model)
        try:
            fields: Dict[str, Tuple[Type, Any]] = {}
            for field_name, field_info in source_model.model_fields.items():
                current_path = (
                    f"{field_path}.{field_name}" if field_path else field_name
                )
                fields[field_name] = self._convert_field(
                    field_name,
                    field_info,
                    field_path=current_path,
                    building=building,
                )
            return fields
        finally:
            building.discard(source_model)

    def _convert_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        *,
        field_path: str,
        building: set[Type[BaseModel]],
    ) -> Tuple[Type, Any]:
        annotation, annotation_nullable = unwrap_optional(field_info.annotation)
        nullable = annotation_nullable or not field_info.is_required()
        origin = get_origin(annotation)

        if self._is_model(annotation):
            nested = self._build_nested_model(
                annotation,
                field_path=field_path,
                building=building,
            )
            final_type = Optional[nested] if nullable else nested
            extensions = self._extract_extensions(field_info, field_path)
            comparison_field = self._make_comparison_field(
                field_info,
                comparator_name="LevenshteinComparator",
                threshold=0.7,
                weight=extensions.get("weight", 1.0),
                clip_under_threshold=extensions.get("clip_under_threshold", True),
            )
            return final_type, comparison_field

        if origin in (list, List, set):
            element = get_args(annotation)[0] if get_args(annotation) else Any
            element, element_nullable = unwrap_optional(element)
            if self._is_model(element):
                nested = self._build_nested_model(
                    element,
                    field_path=f"{field_path}[]",
                    building=building,
                )
                final_element = Optional[nested] if element_nullable else nested
                final_list = List[final_element]
                comparison_field = self._comparison_from_extensions(
                    field_info,
                    field_path,
                    comparator_name="LevenshteinComparator",
                    threshold=0.5,
                )
            else:
                element = self._adapt_union_models(
                    element,
                    field_path=f"{field_path}[]",
                    building=building,
                )
                final_list = List[Optional[element] if element_nullable else element]
                comparator_name, threshold = self._default_comparison(element)
                comparison_field = self._comparison_from_extensions(
                    field_info,
                    field_path,
                    comparator_name=comparator_name,
                    threshold=threshold,
                )
            return (Optional[final_list] if nullable else final_list), comparison_field

        annotation = self._adapt_union_models(
            annotation,
            field_path=field_path,
            building=building,
        )
        comparator_name, threshold = self._default_comparison(annotation)
        comparison_field = self._comparison_from_extensions(
            field_info,
            field_path,
            comparator_name=comparator_name,
            threshold=threshold,
        )
        final_type = Optional[annotation] if nullable else annotation
        return final_type, comparison_field

    def _adapt_union_models(
        self,
        annotation: Any,
        *,
        field_path: str,
        building: set[Type[BaseModel]],
    ) -> Any:
        """Replace Pydantic model arms inside unions with StructuredModels."""
        if not is_union(annotation):
            return annotation
        converted = []
        for arm in get_args(annotation):
            if arm is type(None):
                converted.append(arm)
            elif self._is_model(arm):
                converted.append(
                    self._build_nested_model(
                        arm,
                        field_path=field_path,
                        building=building,
                    )
                )
            else:
                converted.append(arm)
        return Union[tuple(converted)]

    def _build_nested_model(
        self,
        source_model: Type[BaseModel],
        *,
        field_path: str,
        building: set[Type[BaseModel]],
    ):
        from .structured_model import StructuredModel

        model_extra = source_model.model_config.get("json_schema_extra") or {}
        if not isinstance(model_extra, dict):
            model_extra = {}
        model_name = model_extra.get("x-aws-stickler-model-name", source_model.__name__)
        match_threshold = model_extra.get("x-aws-stickler-match-threshold", 0.7)
        self._validate_model_config(model_name, match_threshold, field_path)

        fields = self._convert_model_fields(
            source_model,
            field_path=field_path,
            building=building,
        )
        return ModelFactory.create_model_from_fields(
            model_name=model_name,
            field_definitions=fields,
            match_threshold=match_threshold,
            base_class=StructuredModel,
        )

    def _comparison_from_extensions(
        self,
        field_info: FieldInfo,
        field_path: str,
        *,
        comparator_name: str,
        threshold: float,
    ) -> FieldInfo:
        extensions = self._extract_extensions(field_info, field_path)
        comparator = extensions.get("comparator")
        if comparator is None:
            comparator = create_comparator(comparator_name, {})
        return self._make_comparison_field(
            field_info,
            comparator=comparator,
            threshold=extensions.get("threshold", threshold),
            weight=extensions.get("weight", 1.0),
            clip_under_threshold=extensions.get("clip_under_threshold", True),
        )

    @staticmethod
    def _make_comparison_field(
        source: FieldInfo,
        *,
        comparator=None,
        comparator_name: Optional[str] = None,
        threshold: float,
        weight: float,
        clip_under_threshold: bool,
    ) -> FieldInfo:
        if comparator is None:
            comparator = create_comparator(comparator_name, {})
        default = ... if source.is_required() else source.default
        source_extra = source.json_schema_extra
        examples = source.examples
        if isinstance(source_extra, dict):
            source_extra = dict(source_extra)
            examples = source_extra.pop(_PRESERVED_EXAMPLES_KEY, examples)

        comparison = ComparableField(
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            clip_under_threshold=clip_under_threshold,
            default=default,
            json_schema_extra=source_extra,
        )

        # Keep every constraint/default/alias produced by the schema library;
        # replace only the schema-extra hook that carries comparison runtime data.
        field = copy(source)
        field.default = comparison.default
        field.examples = examples
        field.json_schema_extra = comparison.json_schema_extra
        return field

    @classmethod
    def _prepare_schema(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize library gaps without taking ownership of schema parsing."""

        def prepare(node: Any) -> Any:
            if isinstance(node, list):
                return [prepare(value) for value in node]
            if not isinstance(node, dict):
                return node

            prepared = {key: prepare(value) for key, value in node.items()}
            if "examples" in prepared:
                prepared[_PRESERVED_EXAMPLES_KEY] = deepcopy(prepared["examples"])

            raw_type = prepared.get("type")
            if (
                "properties" in prepared
                and raw_type is None
                and not any(keyword in prepared for keyword in (*_COMBINERS, "$ref"))
            ):
                prepared["type"] = "object"
                raw_type = "object"

            # The library supports primitive list-form types natively, but routes
            # object/array arms through ``dict``. Express those unions as anyOf so
            # its recursive model builder sees the structural arm.
            if isinstance(raw_type, list) and any(
                value in {"object", "array"}
                for value in raw_type
                if isinstance(value, str)
            ):
                metadata = {
                    key: value
                    for key, value in prepared.items()
                    if key in _ANNOTATION_KEYWORDS
                    or key.startswith("x-aws-stickler-")
                    or key == "x-comparison"
                }
                non_metadata = {
                    key: value
                    for key, value in prepared.items()
                    if key not in metadata and key != "type"
                }
                metadata["anyOf"] = [
                    {**deepcopy(non_metadata), "type": value} for value in raw_type
                ]
                prepared = metadata

            return prepared

        return prepare(schema)

    @classmethod
    def _validate_supported_shapes(cls, schema: Dict[str, Any]) -> None:
        """Reject shapes the comparison model cannot represent faithfully."""

        def walk(node: Any, path: str) -> None:
            if isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")
                return
            if not isinstance(node, dict):
                return

            if "patternProperties" in node:
                location = f" at '{path}'" if path else ""
                raise ValueError(
                    "JSON Schema patternProperties cannot be represented as named "
                    f"StructuredModel fields{location}. Define explicit properties instead."
                )

            combiners = [keyword for keyword in _COMBINERS if keyword in node]
            if combiners:
                allowed = set(_ANNOTATION_KEYWORDS)
                allowed.update({"$defs", "definitions"})
                allowed.update(combiners)
                allowed.update(
                    key
                    for key in node
                    if key.startswith("x-aws-stickler-") or key == "x-comparison"
                )
                unsupported = sorted(key for key in node if key not in allowed)
                if len(combiners) > 1:
                    unsupported.extend(sorted(combiners[1:]))
                if unsupported:
                    location = f" for field '{path}'" if path else ""
                    raise ValueError(
                        f"Unsupported {combiners[0]}{location}: sibling schema "
                        f"keywords are not supported: {sorted(set(unsupported))}."
                    )

            for key, value in node.items():
                child_path = path
                if key == "properties" and isinstance(value, dict):
                    for name, child in value.items():
                        walk(child, f"{path}.{name}" if path else name)
                    continue
                if key == "items":
                    child_path = f"{path}[]" if path else "[]"
                walk(value, child_path)

        walk(schema, "")

    @staticmethod
    def _translate_import_error(message: str, schema: Dict[str, Any]) -> str:
        if message.startswith("Only local references"):
            return (
                "Unsupported $ref format. Only '#/definitions/' and '#/$defs/' "
                "references are supported"
            )
        match = re.search(r"Invalid reference path: (\S+)", message)
        if match:
            ref = match.group(1)
            namespace = "$defs" if ref.startswith("#/$defs/") else "definitions"
            available = sorted((schema.get(namespace) or {}).keys())
            return f"Reference '{ref}' not found. Available: {available}"
        if "Circular reference" in message or "recursion" in message.lower():
            return (
                "Recursive JSON Schema models are not supported by the comparison "
                "engine. Use a bounded, non-recursive schema."
            )
        return message

    @staticmethod
    def _default_comparison(annotation: Any) -> Tuple[str, float]:
        annotation, _ = unwrap_optional(annotation)
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return "ExactComparator", 1.0
        return _JSON_DEFAULTS.get(annotation, ("ExactComparator", 1.0))

    @staticmethod
    def _is_model(annotation: Any) -> bool:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)

    @staticmethod
    def _extract_extensions(field_info: FieldInfo, field_path: str) -> Dict[str, Any]:
        extra = field_info.json_schema_extra
        if not isinstance(extra, dict):
            return {}
        extensions: Dict[str, Any] = {}

        if "x-aws-stickler-comparator" in extra:
            comparator_name = extra["x-aws-stickler-comparator"]
            comparator_config = extra.get("x-aws-stickler-comparator-config", {})
            try:
                extensions["comparator"] = create_comparator(
                    comparator_name, comparator_config
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid x-aws-stickler-comparator '{comparator_name}' "
                    f"in field '{field_path}': {exc}"
                ) from exc

        if "x-aws-stickler-threshold" in extra:
            threshold = extra["x-aws-stickler-threshold"]
            if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "x-aws-stickler-threshold must be a number between 0.0 "
                    f"and 1.0 for field '{field_path}', got: {threshold}"
                )
            extensions["threshold"] = threshold

        if "x-aws-stickler-weight" in extra:
            weight = extra["x-aws-stickler-weight"]
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(
                    "x-aws-stickler-weight must be a positive number for field "
                    f"'{field_path}', got: {weight}"
                )
            extensions["weight"] = weight

        clip_key = "x-aws-stickler-clip-under-threshold"
        if clip_key in extra:
            value = extra[clip_key]
            if not isinstance(value, bool):
                raise ValueError(
                    f"{clip_key} must be a boolean for field '{field_path}', "
                    f"got: {type(value).__name__}"
                )
            extensions["clip_under_threshold"] = value

        # ``x-aws-stickler-aggregate`` was removed in #226. Accept and ignore
        # it here so schemas exported by older Stickler versions still import.

        return extensions

    @staticmethod
    def _validate_model_config(
        model_name: Any, match_threshold: Any, field_path: str
    ) -> None:
        if not isinstance(model_name, str) or not model_name.isidentifier():
            raise ValueError(
                "x-aws-stickler-model-name must be a valid Python identifier "
                f"at '{field_path}', got: {model_name}"
            )
        if not isinstance(match_threshold, (int, float)):
            raise ValueError(
                "x-aws-stickler-match-threshold must be a number "
                f"at '{field_path}', got: {type(match_threshold).__name__}"
            )
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError(
                "x-aws-stickler-match-threshold must be between 0.0 and 1.0 "
                f"at '{field_path}', got: {match_threshold}"
            )
