"""Import JSON Schema types while keeping Stickler comparison semantics separate."""

from __future__ import annotations

import difflib
import re
from copy import copy, deepcopy
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)
from uuid import UUID

from json_schema_to_pydantic import create_model as create_pydantic_model
from pydantic import AnyUrl, BaseModel, TypeAdapter
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

# Every extension key this importer honours, on a field. Anything else that LOOKS
# like one is a typo or a wrong prefix, and is rejected rather than dropped.
#
# The asymmetry this closes: a bad extension VALUE already raised a clear error,
# so the schema path taught users it validated their input, then said nothing
# about the case that actually bites. A dropped key leaves a plausible model that
# scores wrong in the direction of over-reporting accuracy, which is the worst
# combination for an evaluation library.
_KNOWN_FIELD_EXTENSIONS = frozenset(
    {
        "x-aws-stickler-comparator",
        "x-aws-stickler-comparator-config",
        "x-aws-stickler-threshold",
        "x-aws-stickler-weight",
        "x-aws-stickler-clip-under-threshold",
        # Removed in #226. Accepted and ignored so schemas exported by older
        # versions still import.
        "x-aws-stickler-aggregate",
        _PRESERVED_EXAMPLES_KEY,
    }
)

# The subset worth suggesting to an author. `aggregate` is removed and
# `internal-examples` is ours, so offering either as a fix would be misleading.
_DOCUMENTED_FIELD_EXTENSIONS = frozenset(
    {
        "x-aws-stickler-comparator",
        "x-aws-stickler-comparator-config",
        "x-aws-stickler-threshold",
        "x-aws-stickler-weight",
        "x-aws-stickler-clip-under-threshold",
    }
)

# Model-level keys, read elsewhere but valid, so they must not be flagged.
_KNOWN_MODEL_EXTENSIONS = frozenset(
    {
        "x-aws-stickler-model-name",
        "x-aws-stickler-match-threshold",
    }
)

# Prefixes that mean "the author was reaching for a Stickler extension". The
# second appears in our own README, so it is a mistake we taught.
_EXTENSION_PREFIXES = ("x-aws-stickler-", "x-stickler-")


def _is_object_schema(node: Any) -> bool:
    """Whether this node generates a StructuredModel of its own.

    Such a node reads both key sets: the field keys describe how its parent
    scores it, the model keys configure the class it generates. Checked
    structurally rather than by `type` alone so the list form
    (``["object", "null"]``) and a bare ``properties`` both count.
    """
    if not isinstance(node, dict):
        return False
    if "properties" in node or "patternProperties" in node:
        return True
    declared = node.get("type")
    if declared == "object":
        return True
    return isinstance(declared, list) and "object" in declared


def _closest_known_key(key: str, candidates: FrozenSet[str]) -> Optional[str]:
    """The valid key a typo was probably reaching for, or None.

    Compares the SUFFIX after the prefix, not the whole key. Every valid key
    starts with `x-aws-stickler-`, so whole-string similarity is dominated by
    that shared prefix and returns a confident match for anything at all:
    `x-aws-stickler-zzzzzzzz` scored as a near-miss on `-weight`. Comparing
    suffixes makes both a real suggestion and no suggestion possible, and a wrong
    suggestion is worse than none.

    `candidates` is scoped to the position being checked. Suggesting a key that
    is valid *somewhere* is worse than suggesting nothing: a field-position typo
    of `-match-threshold` used to be answered with "did you mean
    `x-aws-stickler-match-threshold`", and taking that advice imported cleanly
    and did nothing, which is the silent drop this module exists to prevent.
    """
    known = {}
    for full in sorted(candidates):
        for prefix in _EXTENSION_PREFIXES:
            if full.startswith(prefix):
                known[full[len(prefix) :]] = full
                break

    suffix = key
    for prefix in _EXTENSION_PREFIXES:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            break

    # An exact suffix match is the wrong-prefix case: `x-stickler-comparator`.
    if suffix in known:
        return known[suffix]
    matches = difflib.get_close_matches(suffix, sorted(known), n=1, cutoff=0.75)
    return known[matches[0]] if matches else None


def _reject_unknown_extensions(
    extra: Any,
    field_path: str,
    *,
    scope: str = "field",
) -> None:
    """Raise for any extension-shaped key this importer does not honour here.

    Raising rather than warning, deliberately, and unlike the mapping-comparator
    case elsewhere: a schema is authored once and read deterministically, so there
    is no risk of failing on document N of a corpus after succeeding on N-1. The
    author is present, the mistake is in a file they can edit, and the alternative
    is a model that builds and reports the wrong number.

    `scope` decides which keys are honoured at this position, because the two sets
    are not interchangeable. `x-aws-stickler-match-threshold` on an object is read;
    the same key on a scalar field is dropped. Accepting both everywhere made the
    check pass on keys that do nothing, so a valid-but-misplaced key was exactly
    as silent as the typo it was mistaken for.
    """
    if not isinstance(extra, dict):
        return

    if scope == "model":
        accepted = _KNOWN_MODEL_EXTENSIONS
        suggestible = _KNOWN_MODEL_EXTENSIONS
        other_scope, other_keys = "field", _DOCUMENTED_FIELD_EXTENSIONS
        where = f"object '{field_path}'"
    elif scope == "field_or_model":
        # An object-typed property honours both sets, so there is no misplacement
        # to report and nothing to route elsewhere.
        accepted = _KNOWN_FIELD_EXTENSIONS | _KNOWN_MODEL_EXTENSIONS
        suggestible = _DOCUMENTED_FIELD_EXTENSIONS | _KNOWN_MODEL_EXTENSIONS
        other_scope, other_keys = "", frozenset()
        where = f"field '{field_path}'"
    else:
        accepted = _KNOWN_FIELD_EXTENSIONS
        suggestible = _DOCUMENTED_FIELD_EXTENSIONS
        other_scope, other_keys = "object", _KNOWN_MODEL_EXTENSIONS
        where = f"field '{field_path}'"

    for key in extra:
        if not isinstance(key, str):
            continue
        if key in accepted:
            continue
        if not key.startswith(_EXTENSION_PREFIXES):
            continue  # an unrelated x-* extension is none of our business

        # A real key in the wrong position. Naming the position is the whole
        # answer here, so say that instead of offering a spelling correction.
        if key in other_keys:
            raise ValueError(
                f"Stickler extension '{key}' is not read on {where}. It belongs "
                f"on the {other_scope}. Left in place it would be dropped "
                f"silently. Valid keys here: {', '.join(sorted(suggestible))}."
            )

        suggestion = _closest_known_key(key, suggestible)
        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
        raise ValueError(
            f"Unrecognized Stickler extension '{key}' on {where}."
            f"{hint} It would otherwise be dropped silently, leaving the "
            "configuration to fall back while its correctly-spelled siblings "
            f"were honoured. Valid keys here: {', '.join(sorted(suggestible))}."
        )


_COMBINERS = frozenset({"allOf", "anyOf", "oneOf"})
_NON_VALIDATING_SCHEMA_KEYWORDS = frozenset(
    {
        "const",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maximum",
        "maxLength",
        "minItems",
        "minimum",
        "minLength",
        "multipleOf",
        "pattern",
        "uniqueItems",
    }
)
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
    uses the parsed types to select comparison behavior, then widens validation
    annotations so malformed predictions remain scoreable.
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
                comparator_name, threshold = self._default_comparison(element)
                evaluation_element = self._evaluation_annotation(element)
                final_list = List[
                    Optional[evaluation_element]
                    if element_nullable
                    else evaluation_element
                ]
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
        evaluation_annotation = self._evaluation_annotation(annotation)
        final_type = (
            Optional[evaluation_annotation] if nullable else evaluation_annotation
        )
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

        parsed_schema = TypeAdapter(source.rebuild_annotation()).json_schema()
        schema_metadata = {
            key: deepcopy(value)
            for key, value in parsed_schema.items()
            if key in _NON_VALIDATING_SCHEMA_KEYWORDS
        }
        if isinstance(source_extra, dict):
            schema_metadata.update(source_extra)
        source_extra = schema_metadata or None

        comparison = ComparableField(
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            clip_under_threshold=clip_under_threshold,
            default=default,
            json_schema_extra=source_extra,
        )

        # Keep descriptive/default/alias data, but not validation constraints.
        # Stickler scores imperfect predictions; a constraint violation must reach
        # the comparator rather than abort model construction.
        field = copy(source)
        field.metadata = []
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
        """Reject shapes the comparison model cannot represent faithfully.

        Also where extension keys are checked, because this walk sees every node
        of the RAW schema before `_prepare_schema` rewrites list-form types into
        `anyOf`. Checking per-field at extraction time reached only the positions
        that produce a field, so a typo on the root object, on `items`, or on a
        `["object", "null"]` node was silently dropped while the same typo one
        level down raised.
        """

        def walk(node: Any, path: str, scope: Optional[str] = "model") -> None:
            if isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]", scope)
                return
            if not isinstance(node, dict):
                return

            if scope is not None:
                _reject_unknown_extensions(node, path or "(root)", scope=scope)

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
                child_scope: Optional[str] = None
                if key == "properties" and isinstance(value, dict):
                    for name, child in value.items():
                        # A property is scored as a field of this object, and if
                        # it is itself an object it also configures its own
                        # generated class, so both key sets are honoured there.
                        walk(
                            child,
                            f"{path}.{name}" if path else name,
                            "field_or_model" if _is_object_schema(child) else "field",
                        )
                    continue
                if key == "items":
                    child_path = f"{path}[]" if path else "[]"
                    # Field keys are NOT read on `items`: weight, threshold and
                    # comparator there are all dropped, since the field they would
                    # configure is the array itself, one level up. Only the element
                    # class's own settings are read.
                    child_scope = "model"
                elif key in _COMBINERS or key in ("$defs", "definitions"):
                    child_scope = "model"
                walk(value, child_path, child_scope)

        walk(schema, "", "model")

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
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        if annotation in (date, datetime):
            return "DateComparator", 1.0
        if annotation is Decimal:
            return "NumericComparator", 0.5
        if get_origin(annotation) is Literal:
            return "ExactComparator", 1.0
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return "ExactComparator", 1.0
        return _JSON_DEFAULTS.get(annotation, ("ExactComparator", 1.0))

    @classmethod
    def _evaluation_annotation(cls, annotation: Any) -> Any:
        """Return a scoreable annotation while retaining parsed type semantics.

        The schema library deliberately produces strict Pydantic annotations for
        formats, enums, literals, and constrained collection elements. Those are
        useful for comparator selection but wrong for an evaluation model: an
        invalid extraction is an ordinary zero-score candidate, not a model
        construction error. Widen only to the corresponding JSON value type and
        keep primitive type boundaries that the previous importer enforced.
        """
        if get_origin(annotation) is Annotated:
            return cls._evaluation_annotation(get_args(annotation)[0])

        if is_union(annotation):
            widened = []
            for arm in get_args(annotation):
                value = cls._evaluation_annotation(arm)
                if value not in widened:
                    widened.append(value)
            if len(widened) == 1:
                return widened[0]
            return Union[tuple(widened)]

        origin = get_origin(annotation)
        if origin is Literal:
            value_types = []
            for value in get_args(annotation):
                if isinstance(value, Enum):
                    value = value.value
                value_type = type(value)
                if value_type not in value_types:
                    value_types.append(value_type)
            if len(value_types) == 1:
                return value_types[0]
            return Union[tuple(value_types)]

        if origin in (list, List, set):
            args = get_args(annotation)
            element = cls._evaluation_annotation(args[0] if args else Any)
            return List[element]

        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return cls._evaluation_annotation(
                Literal[tuple(item.value for item in annotation)]
            )

        if annotation in (date, datetime, time, UUID, AnyUrl):
            return str
        if annotation is Decimal:
            return float
        return annotation

    @staticmethod
    def _is_model(annotation: Any) -> bool:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)

    @staticmethod
    def _extract_extensions(field_info: FieldInfo, field_path: str) -> Dict[str, Any]:
        extra = field_info.json_schema_extra
        if not isinstance(extra, dict):
            return {}
        # Backstop for entry points that do not walk a raw schema, such as an
        # already-parsed pydantic model. `_validate_supported_shapes` is the
        # precise gate: it sees each node's position, so it is what distinguishes
        # a model key on a scalar field from the same key on an object. Here the
        # position is no longer visible, so both sets are accepted rather than
        # re-deciding it from an annotation and disagreeing with the walker.
        _reject_unknown_extensions(extra, field_path, scope="field_or_model")
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
