"""Import JSON Schema types while keeping Stickler comparison semantics separate."""

from __future__ import annotations

import difflib
import re
import warnings
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
from .comparator_registry import create_comparator, normalize_comparator_config
from .field_converter import AUTO_COMPARATOR
from .model_factory import ModelFactory
from .optional_annotation import is_union, unwrap_optional

_JSON_DEFAULTS = {
    str: ("LevenshteinComparator", 0.5),
    float: ("NumericComparator", 0.5),
    int: ("NumericComparator", 0.5),
    bool: ("ExactComparator", 0.5),
}

_PRESERVED_EXAMPLES_KEY = "x-aws-stickler-internal-examples"

# The value every released `to_json_schema()` wrote for
# `x-aws-stickler-threshold` on a `List[StructuredModel]` property. It could
# write nothing else: a named threshold on that shape is refused at class
# definition. A threshold equal to it therefore carries no authorial intent and
# must not draw a warning. This release stops emitting the key at all, so only
# legacy artifacts reach it.
_EXPORTED_ARRAY_THRESHOLD_SENTINEL = 0.5

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
        "x-aws-stickler-infer-unspecified",
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
    merges: bool = True,
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

    `merges` says whether a one-arm `allOf` of this definition imports, so
    whether the advice for a field key may offer one. It does only when the
    definition, and each alias hop to its end, is typed `"object"` or untyped.
    """
    if not isinstance(extra, dict):
        return

    # Each entry names a set of real keys that are not read here and where they
    # belong instead, so a misplaced key is answered with its position.
    misplaced: Tuple[Tuple[str, FrozenSet[str]], ...]
    # Field keys are read on the property, never on a combiner arm or `items`,
    # so the message names the property for each way a definition is used. An
    # array of objects reads only the weight there, and pairs its elements by
    # the definition's match threshold. The root is no field, so it reads none.
    # A one-arm `allOf` fails to import unless it names an object typed
    # `"object"` or untyped, so only such a definition, or an alias of one, is
    # offered the `allOf`.
    beside = "property that uses this definition: beside its '$ref', or beside the "
    arm = "when the '$ref' is one of its arms"
    no_root = ". A definition used as the root reads no field keys"
    beside_ref = (
        f"{beside}'anyOf' or 'oneOf' {arm}, or on the array property when the "
        f"definition is its 'items' or an arm of them{no_root}"
    )
    combiners = "'anyOf', 'oneOf' or 'allOf'" if merges else "'anyOf' or 'oneOf'"
    of_objects = (
        f"{beside}{combiners} {arm}. When the definition is an "
        "array's 'items' or an arm of them, the array property reads only "
        "'x-aws-stickler-weight', and the elements pair by "
    )
    beside_ref_object = (
        f"{of_objects}this definition's 'x-aws-stickler-match-threshold'{no_root}"
    )
    beside_ref_alias = (
        f"{of_objects}the 'x-aws-stickler-match-threshold' of the object it "
        f"aliases{no_root}"
    )
    # Removed in #226 and ignored wherever it appears, so an older schema that
    # carries it in a definition still imports, as it does on a field.
    removed = frozenset({"x-aws-stickler-aggregate"})
    if scope in ("model", "definition"):
        # A definition is a type that several positions can `$ref`, and not all
        # of them read field keys (`items` does not). Honouring them there would
        # make one key work at one reference and vanish at the next, so they go
        # on the property that uses the definition.
        accepted = _KNOWN_MODEL_EXTENSIONS
        suggestible = _KNOWN_MODEL_EXTENSIONS
        if scope == "model":
            misplaced = (("field", _DOCUMENTED_FIELD_EXTENSIONS),)
            where = f"object '{field_path}'"
        else:
            accepted = accepted | removed
            misplaced = ((beside_ref_object, _DOCUMENTED_FIELD_EXTENSIONS),)
            where = f"definition '{field_path}'"
    elif scope in ("value_definition", "alias_definition"):
        # A definition that generates no class (a scalar, an array, a combiner,
        # an alias `$ref`, a mapping) reads neither set. Accepting the model keys
        # here passed the check and then dropped them. An alias of an object is
        # used wherever the object is, so its field keys go where the object's do.
        accepted = removed
        suggestible = frozenset()
        misplaced = (
            (
                beside_ref_alias if scope == "alias_definition" else beside_ref,
                _DOCUMENTED_FIELD_EXTENSIONS,
            ),
            ("object definition with 'properties'", _KNOWN_MODEL_EXTENSIONS),
        )
        where = f"definition '{field_path}'"
    elif scope == "field_or_model":
        # An object-typed property honours both sets, so there is no misplacement
        # to report and nothing to route elsewhere.
        accepted = _KNOWN_FIELD_EXTENSIONS | _KNOWN_MODEL_EXTENSIONS
        suggestible = _DOCUMENTED_FIELD_EXTENSIONS | _KNOWN_MODEL_EXTENSIONS
        misplaced = ()
        where = f"field '{field_path}'"
    else:
        accepted = _KNOWN_FIELD_EXTENSIONS
        suggestible = _DOCUMENTED_FIELD_EXTENSIONS
        misplaced = (("object", _KNOWN_MODEL_EXTENSIONS),)
        where = f"field '{field_path}'"
    valid_here = ", ".join(sorted(suggestible)) or "none"

    for key in extra:
        if not isinstance(key, str):
            continue
        if key in accepted:
            continue
        if not key.startswith(_EXTENSION_PREFIXES):
            continue  # an unrelated x-* extension is none of our business

        # A real key in the wrong position. Naming the position is the whole
        # answer here, so say that instead of offering a spelling correction.
        for other_scope, other_keys in misplaced:
            if key in other_keys:
                raise ValueError(
                    f"Stickler extension '{key}' is not read on {where}. It "
                    f"belongs on the {other_scope}. Left in place it would be "
                    f"dropped silently. Valid keys here: {valid_here}."
                )

        suggestion = _closest_known_key(key, suggestible)
        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
        raise ValueError(
            f"Unrecognized Stickler extension '{key}' on {where}."
            f"{hint} It would otherwise be dropped silently, leaving the "
            "configuration to fall back while its correctly-spelled siblings "
            f"were honoured. Valid keys here: {valid_here}."
        )


def _resolve_local_ref(
    schema: Any, ref: Any
) -> Tuple[Any, Tuple[str, ...], Optional[Tuple[str, ...]], Any]:
    """Resolve a local `$ref`: its node, its pointer, and the innermost
    definition entry that contains it, with that entry's body."""
    if not isinstance(ref, str) or not ref.startswith("#"):
        return None, (), None, None
    pointer = ref[1:]
    if pointer and not pointer.startswith("/"):
        return None, (), None, None
    parts = (
        tuple(
            part.replace("~1", "/").replace("~0", "~")
            for part in pointer[1:].split("/")
        )
        if pointer
        else ()
    )
    node: Any = schema
    entry: Optional[Tuple[str, ...]] = None
    entry_body: Any = None
    readings = {"schema"}
    for index, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            return None, (), None, None
        # An entry only where the segment before was read as the keyword,
        # not as a property NAMED `$defs` or `definitions`.
        if readings & {"$defs", "definitions"}:
            entry, entry_body = parts[: index + 1], node[part]
        readings = _next_readings(readings, part)
        node = node[part]
    return node, parts, entry, entry_body


def _ref_label(target: Tuple[str, ...], entry: Optional[Tuple[str, ...]]) -> str:
    """A whole entry is a definition; a node inside one is an object."""
    kind = "definition" if entry is not None and target == entry else "object"
    return f"{kind} '{'.'.join(target)}'"


def _ref_chain(
    schema: Any, node: Any, where: str, pointer: Optional[Tuple[str, ...]] = None
):
    """The node, then each alias it `$ref`s: every hop is the same class. Each
    hop comes with its label and its JSON pointer (`pointer` for the first,
    None when that node is not a `$ref` target)."""
    seen: set = set()
    while isinstance(node, dict) and id(node) not in seen:
        seen.add(id(node))
        yield node, where, pointer
        node, pointer, entry, _ = _resolve_local_ref(schema, node.get("$ref"))
        where = _ref_label(pointer, entry)


# Keywords whose value maps names to schemas, and those whose value lists them.
_NAMED_SCHEMA_KEYWORDS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)
_LISTED_SCHEMA_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
# Keywords whose value is one schema.
_SINGLE_SCHEMA_KEYWORDS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)


def _is_property_pointer(pointer: Optional[Tuple[str, ...]]) -> bool:
    """Whether a JSON pointer ends at a property: a schema under some node's
    `properties`. Read from the root, so a property NAMED `properties` is a
    name, not the keyword.

    Under a key no schema has, such as `components` or an `x-*` payload, the
    value may be a schema or a map of them, and a member of that map may be
    named like a keyword (`#/components/schemas/properties/...`). There both
    readings are kept, and the pointer is a property if either ends at one."""
    if not pointer:
        return False
    # What the next segment can be in each reading: a key of a schema
    # ("schema"), a member of a map this reader does not know ("member"), an
    # index ("index"), or a name under the keyword held.
    readings = {"schema"}
    for part in pointer[:-1]:
        readings = _next_readings(readings, part)
    return "properties" in readings


def _walk_position(pointer: Tuple[str, ...]) -> Tuple[str, Optional[str], bool]:
    """Where the walk would meet the node at `pointer` if it entered every
    key: the path it would name, "property" or "items" when the node is one,
    and whether the pointer passes an `x-*` key of a schema.

    Each segment is read as the walk reads it, a keyword wherever a schema
    holds it, so a property NAMED `items` or `x-a` is a name."""
    path, kind, under_x, reading = "", None, False, "schema"
    for part in pointer:
        kind = None
        if reading == "index":
            path = f"{path}[{part}]"
        elif reading == "schema" and part in ("properties", *_COMBINERS):
            pass
        elif reading == "schema" and part == "items":
            path, kind = f"{path}[]", "items"
        else:
            if reading == "properties":
                kind = "property"
            elif reading == "schema" and part.startswith("x-"):
                under_x = True
            path = f"{path}.{part}" if path else part
        # The walk reads every value as a schema, never as a map of them.
        reading = min(_next_readings({reading}, part) - {"member"})
    return path, kind, under_x


def _next_readings(readings: set, part: str) -> set:
    """What the segment after `part` can be, in each reading of `part`."""
    following = set()
    for expect in readings:
        if expect == "member" or (
            expect == "schema"
            and part not in _NAMED_SCHEMA_KEYWORDS
            and part not in _LISTED_SCHEMA_KEYWORDS
            and part not in _SINGLE_SCHEMA_KEYWORDS
        ):
            following.update(("schema", "member"))
        elif expect == "schema" and part in _NAMED_SCHEMA_KEYWORDS:
            following.add(part)
        elif expect == "schema" and part in _LISTED_SCHEMA_KEYWORDS:
            following.add("index")
        else:
            following.add("schema")
    return following


_COMBINERS = frozenset({"allOf", "anyOf", "oneOf"})
# Keywords whose values are instance data, not schemas.
_DATA_KEYWORDS = frozenset({"const", "default", "enum", "examples"})
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

    def __init__(
        self,
        schema: Dict[str, Any],
        field_path: str = "",
        *,
        infer_unspecified: bool = False,
    ):
        self.schema = schema
        self.field_path = field_path
        # The object-level match threshold, which inference uses as the FIELD
        # threshold for a mapping field (a dict declines to name its keys, so it
        # is judged as an object). Re-read here rather than passed in because
        # `_build_nested_model` swaps it per subtree, exactly like the flag below.
        # Range and type are validated by the caller
        # (`StructuredModel._from_json_schema_internal`, `_validate_model_config`);
        # a value that got here unvalidated must not reach inference, so anything
        # that is not a plain number falls back to the default.
        declared_threshold = schema.get("x-aws-stickler-match-threshold", 0.7)
        self.match_threshold = (
            declared_threshold
            if isinstance(declared_threshold, (int, float))
            and not isinstance(declared_threshold, bool)
            else 0.7
        )
        # Read from the ROOT object and inherited by every nested model, so one
        # flag covers a whole document rather than being repeated at each level.
        # A nested object may override it for its own subtree; see
        # `_build_nested_model`. A per-field `"comparator": "auto"` is the way to
        # be selective about a single scalar property.
        declared = schema.get("x-aws-stickler-infer-unspecified", infer_unspecified)
        if not isinstance(declared, bool):
            raise ValueError(
                "x-aws-stickler-infer-unspecified must be true or false"
                + (f" at '{field_path}'" if field_path else "")
                + f", got: {declared!r}"
            )
        self.infer_unspecified = declared

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

        unread_warnings = self._validate_supported_shapes(schema)
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

        # A root typed `["object", "null"]`, or a root `anyOf`/`oneOf`, parses to
        # a union rather than a class, and crashed below with
        # `AttributeError: model_fields`. The root is the document being scored,
        # so it cannot be null or one of several shapes.
        if not (isinstance(source_model, type) and issubclass(source_model, BaseModel)):
            raise ValueError(
                "The root of a JSON Schema must be a single object. A root typed "
                "['object', 'null'], a root anyOf/oneOf, or a root '$ref' to "
                "either, is a union, and the document being scored cannot be null "
                "or one of several shapes. Type the root, or what its '$ref' "
                "names, 'object'."
            )

        if not source_model.model_fields:
            if "patternProperties" in schema:
                raise ValueError(
                    "JSON Schema patternProperties cannot be represented as named "
                    "StructuredModel fields. Define explicit properties instead."
                )
            return {}

        fields = self._convert_model_fields(
            source_model,
            field_path=self.field_path,
            building=set(),
        )
        # Issued only once the import builds: one that fails anyway, such as a
        # root `$ref` to a property typed `["object", "null"]`, would otherwise
        # warn that a key "is ignored" with advice that cannot be followed. No
        # fields fails the import at the caller.
        if fields:
            for message in unread_warnings:
                warnings.warn(message, UserWarning)
        return fields

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
            if extensions.get("infer"):
                raise self._auto_on_container_error(
                    field_path,
                    "an object",
                    "recursive field-by-field comparison",
                    "this object",
                )
            comparison_field = self._make_comparison_field(
                field_info,
                comparator_name="LevenshteinComparator",
                # Read the declared threshold, like `weight` and
                # `clip_under_threshold` beside it. The literal used to stand
                # here unconditionally, so `x-aws-stickler-threshold` on an
                # object-typed property was dropped in silence while the other
                # two field-level keys on the SAME node were honoured.
                #
                # Not inert in this position, which is why carrying it is the
                # right fix rather than rejecting it the way #312 rejects a
                # misplaced key. A nested-model field's threshold gates the
                # subtree mean: measured on a two-leaf child scoring 0.5, a
                # threshold of 0.6 with clipping on reports 0.0. Set through a
                # `StructuredModel` class it worked; set in a schema it did not,
                # so the two configuration paths disagreed about what is
                # configurable. See #317.
                threshold=extensions.get("threshold", 0.7),
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
                    # `None`, not the legacy 0.5. Both resolve to the same
                    # number, but passing a value marks the field as having
                    # named a threshold, and a list-of-model field is not
                    # allowed to: Hungarian matching reads each element class's
                    # `match_threshold` instead.
                    threshold=None,
                    # And a threshold the SCHEMA names here is dropped rather
                    # than forwarded, because forwarding it now raises.
                    #
                    # Every `to_json_schema()` on a released version emitted
                    # `x-aws-stickler-threshold` on an array-of-model property,
                    # so every schema artifact already written to disk carries
                    # it. Import used to pass the value through and the old
                    # `threshold != 0.5` proxy let the placeholder slide; with
                    # the proxy replaced by an explicitness marker, forwarding
                    # it makes `__init_subclass__` refuse the class and no
                    # previously exported schema containing a list of models
                    # can be read back.
                    #
                    # Warned rather than raised: which is a legacy placeholder
                    # and which a human wrote is not decidable from the value,
                    # and refusing would break persisted artifacts to catch a
                    # misconfiguration whose only cost is being ignored.
                    ignore_threshold=True,
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
                    # The ELEMENT type, because a primitive list is scored per
                    # element. Passing nothing here left `List[number]` on the
                    # shallow default with the flag on, while the config path
                    # inferred the same shape -- one flag, two answers.
                    annotation=element,
                    element_of_list=True,
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
            annotation=annotation,
        )
        evaluation_annotation = self._evaluation_annotation(annotation)
        final_type = (
            Optional[evaluation_annotation] if nullable else evaluation_annotation
        )
        return final_type, comparison_field

    @staticmethod
    def _auto_on_container_error(
        field_path: str, shape: str, how_it_is_scored: str, where_the_flag_goes: str
    ) -> ValueError:
        """The error for ``x-aws-stickler-comparator: "auto"`` on a container.

        Inference chooses a comparator from a scalar's type and name. A container
        has no comparator of its own -- it is scored structurally -- so there is
        nothing to infer, and the key cannot be honoured at this position.

        Raising rather than ignoring, because every other unusable value of this
        key raises here: `"Nonsense"` on the same property is rejected by name.
        Accepting `"auto"` and doing nothing turned that loud error into the silent
        drop #210 and #312 exist to remove, through a value those checks consider
        valid everywhere.
        """
        return ValueError(
            f"x-aws-stickler-comparator '{AUTO_COMPARATOR}' cannot be applied to "
            f"field '{field_path}': it is {shape}, scored by "
            f"{how_it_is_scored}, so there is no comparator to infer. Set "
            f"'x-aws-stickler-infer-unspecified': true on {where_the_flag_goes} to "
            "infer the fields inside it instead."
        )

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
        # A nested object may opt its own subtree in or out. Reading the key only
        # at the root meant writing it on a nested object was accepted and then
        # silently ignored -- the same silent-drop #210 and #312 exist to remove,
        # reintroduced by a key those checks now consider valid everywhere.
        nested_infer = model_extra.get(
            "x-aws-stickler-infer-unspecified", self.infer_unspecified
        )
        if not isinstance(nested_infer, bool):
            raise ValueError(
                "x-aws-stickler-infer-unspecified must be true or false at "
                f"'{field_path}', got: {nested_infer!r}"
            )
        self._validate_model_config(model_name, match_threshold, field_path)

        # Restored in `finally`, because nested models are built through this same
        # instance: leaving the nested value in place would leak it onto the
        # OUTER object's remaining fields, so a sibling declared after a nested
        # object would silently inherit that subtree's setting. `match_threshold`
        # is swapped alongside it for the same reason: a mapping field inside this
        # object is judged against THIS object's threshold.
        outer_infer = self.infer_unspecified
        outer_threshold = self.match_threshold
        self.infer_unspecified = nested_infer
        self.match_threshold = match_threshold
        try:
            fields = self._convert_model_fields(
                source_model,
                field_path=field_path,
                building=building,
            )
        finally:
            self.infer_unspecified = outer_infer
            self.match_threshold = outer_threshold
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
        threshold: Optional[float],
        ignore_threshold: bool = False,
        annotation: Any = None,
        element_of_list: bool = False,
    ) -> FieldInfo:
        extensions = self._extract_extensions(field_info, field_path)
        if ignore_threshold and "threshold" in extensions:
            declared = extensions.pop("threshold")
            # Ignored either way; warned about only when the author can act on it.
            #
            # Every `to_json_schema()` on a released version wrote `0.5` here and
            # could write nothing else, since `__init_subclass__` refuses a named
            # threshold on this shape. Warning on the key's mere PRESENCE therefore
            # warned on every legacy artifact -- precisely the schemas the
            # ignore-instead-of-raise behaviour exists to rescue -- telling the
            # author to change a key the library itself wrote. A value equal to the
            # export sentinel carries no authorial intent; any other value does.
            #
            # `warnings.warn`, not `warn_once`: the latter memoises on
            # `(id, field_path)` for the life of the process, so a SECOND schema
            # declaring the same property name imported silently and its author
            # never heard about their dead key. A schema import happens once per
            # call, so ordinary dedup by message is correct here.
            #
            # The message names the key to write but not `declared` as its value:
            # the element class's gate is a different number, so echoing it would
            # tell the reader to overwrite a working configuration with the value
            # being discarded.
            #
            # When `items` is a `$ref`, "inside that property's 'items'" read
            # as beside the `$ref`, where the key is not read either and now
            # raises, so the message names the class the `$ref` builds from.
            # Where the property is not found in the raw schema, the advice
            # must hold for either shape.
            if declared != _EXPORTED_ARRAY_THRESHOLD_SENTINEL:
                sources = self._element_class_sources(field_path)
                if sources:
                    where = (
                        f"declared on {' or '.join(sources)}, where its 'items' "
                        "'$ref' leads"
                    )
                elif sources is None:
                    where = (
                        "declared on the class its 'items' builds, which for a "
                        "'$ref' is the definition it leads to"
                    )
                else:
                    where = "declared inside that property's 'items'"
                warnings.warn(
                    f"'x-aws-stickler-threshold': {declared} has no effect on "
                    f"array property '{field_path}' and is ignored. Pairing of "
                    f"array elements is gated by the element class's own "
                    f"'x-aws-stickler-match-threshold', {where}; set it there "
                    "if that is what you meant.",
                    UserWarning,
                )

        if annotation is None and extensions.get("infer"):
            # A list of objects is matched element-wise by Hungarian matching, so
            # there is no comparator for inference to choose. Raising rather than
            # dropping: every other unusable value of this key already raises, and
            # accepting `auto` here silently would leave the field on the shallow
            # default while the schema says it was inferred.
            raise self._auto_on_container_error(
                field_path,
                "an array of objects",
                "Hungarian matching per element",
                "its 'items' object",
            )

        # Inference fills only the parameters the schema left unnamed, so a
        # property that states a threshold keeps it and still gets a comparator
        # suited to its type. `annotation` is None on the container paths, which
        # are not scalar fields and have nothing to infer from.
        wants_inference = extensions.get("infer") or (
            self.infer_unspecified
            and extensions.get("comparator") is None
            and annotation is not None
        )
        inferred = None
        if wants_inference and annotation is not None:
            from .field_converter import LIST_ELEMENT_PROVENANCE, _infer_spec

            inferred = _infer_spec(
                field_path.rsplit(".", 1)[-1] or "value",
                annotation,
                self.match_threshold,
                # The array branch above already unwrapped one list level, so
                # `_infer_spec` must not unwrap another. A NESTED array otherwise
                # descended to the innermost scalar and handed the field's
                # comparator a `list`, scoring an IDENTICAL
                # `array<array<number>>` pair 0.0 where the flag-off default and
                # `stickler.eval_for` both say 1.0.
                already_element=bool(element_of_list),
            )
            # `_infer_spec` prepends this itself when it is handed a list
            # annotation, but the array branch above passes the ELEMENT type, so it
            # cannot see that this is a list. Without it, `explain()` reported an
            # array field with a scalar-looking trail here while the config path
            # and `stickler.evaluate()` both said "spec applies to each element".
            # Exactly one insert: `already_element` suppresses the one inside, and
            # the duplicated trail was the visible tell for the double unwrap.
            if element_of_list:
                inferred.provenance.insert(0, LIST_ELEMENT_PROVENANCE)

        # MERGED over the inferred config, author's keys winning, matching the
        # config path (`field_converter.convert_field_config`). Reading
        # `x-aws-stickler-comparator-config` only in the branch that also names a
        # comparator dropped it entirely for an inferred field, so
        # `absolute_tolerance: 0.5` beside `"auto"` built rel=0.001/abs=0.0 while
        # the identical Stickler config built abs=0.5.
        author_config = normalize_comparator_config(
            extensions.get("comparator_config"), f"field '{field_path}'"
        )
        comparator = extensions.get("comparator")
        if comparator is None and inferred is not None:
            comparator = create_comparator(
                inferred.comparator_name,
                {**inferred.comparator_config, **author_config},
            )
        if comparator is None:
            # No comparator named and nothing inferred: the config still belongs
            # to the type default, as it does on the config path. Dropping it was
            # the same silent drop one branch over.
            comparator = create_comparator(comparator_name, author_config)

        field = self._make_comparison_field(
            field_info,
            comparator=comparator,
            threshold=extensions.get(
                "threshold", inferred.threshold if inferred else threshold
            ),
            weight=extensions.get("weight", inferred.weight if inferred else 1.0),
            clip_under_threshold=extensions.get(
                "clip_under_threshold",
                inferred.clip_under_threshold if inferred else True,
            ),
        )
        if inferred is not None and inferred.provenance:
            extra_callable = field.json_schema_extra
            if callable(extra_callable):
                extra_callable._inferred_provenance = tuple(inferred.provenance)
        return field

    @staticmethod
    def _make_comparison_field(
        source: FieldInfo,
        *,
        comparator=None,
        comparator_name: Optional[str] = None,
        threshold: Optional[float],
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
                # The object arm generates the class, and `_build_nested_model`
                # reads the class's own keys off that arm. Leaving them only as
                # siblings of `anyOf` built `DynamicModel` at 0.7 for a node that
                # declared a name and threshold, while its field keys, read off
                # the property, were honoured. See #349.
                model_keys = {
                    key: deepcopy(value)
                    for key, value in metadata.items()
                    if key in _KNOWN_MODEL_EXTENSIONS
                }

                def arm(value: Any) -> Dict[str, Any]:
                    # The null arm takes no structural keywords: a `$ref` copied
                    # onto it parsed as a second object arm, so the node became
                    # a union of two classes scored with ExactComparator.
                    if value == "null":
                        return {"type": "null"}
                    if value != "object":
                        return {**deepcopy(non_metadata), "type": value}
                    # An arm `type` beside a `$ref` makes the parser merge the
                    # two into a new class. A property typed `"object"` builds
                    # the referenced class, so the arm keeps only the `$ref`.
                    typed = {} if "$ref" in non_metadata else {"type": value}
                    return {**deepcopy(non_metadata), **model_keys, **typed}

                metadata["anyOf"] = [arm(value) for value in raw_type]
                prepared = metadata

            return prepared

        return prepare(schema)

    @classmethod
    def _validate_supported_shapes(cls, schema: Dict[str, Any]) -> List[str]:
        """Reject shapes the comparison model cannot represent faithfully.

        Also where extension keys are checked, because this walk sees every node
        of the RAW schema before `_prepare_schema` rewrites list-form types into
        `anyOf`. Checking per-field at extraction time reached only the positions
        that produce a field, so a typo on the root object, on `items`, or on a
        `["object", "null"]` node was silently dropped while the same typo one
        level down raised.

        A `$defs`/`definitions` entry is checked as a definition once a `$ref`
        reaches it. A dereferenced schema keeps the originals beside their
        inlined copies, and an entry nothing references builds nothing, so
        checking its own keys raised for keys that were read on the copies. Such
        an entry is walked as before, for its contents only, and so are the
        targets of its `$ref`s. Data keywords and `x-*` payloads are values, not
        schemas, and are not walked, except where a `$ref` points into one and
        for an entry under `$defs` inside one, which was walked before as well.

        Every `$ref` target is walked from the reference as well, so a class
        built inside it reads model keys wherever it sits: under `not` or an
        `x-*` key inside a definition, or outside `$defs` (`#/components/...`).
        Only a whole definition entry is checked as a definition. Any other
        target has its own keys, model keys beside its `$ref` included, checked
        only where the walk around it checks them: on a property or `items`
        node anywhere in a definition entry, under `not` or `then` too. Under
        an `x-*` key, which the walk does not enter, such a node has its own
        keys checked from the reference, but not model keys beside its `$ref`.
        A target that is no property or `items` node (`#/$defs/W/not`,
        `#/x-models/T`) has only its contents checked. A field key on a
        property node is valid there, but only the object holding the node
        reads it: another property that `$ref`s the node does not. The rest of
        the entry around a target builds nothing, as an unused entry does.

        Model keys no class reads raise, except on an object-typed property,
        a position the docs call valid for them: beside its `$ref`, or reached
        only through the root `$ref` or an `allOf` arm. There they are dropped,
        as before, with a warning until 1.1.0. The warnings are returned, for
        the caller to issue once the import builds.
        """
        # Nodes whose model keys some class reads, by identity: every position
        # except an `allOf` arm and the root's own `$ref` target. Over-counting
        # here can only skip an error, never raise one.
        read: set = set()
        # `allOf` arms at positions that build a class, checked once every
        # reader is known.
        merged: List[Tuple[Any, str, str]] = []
        # `$ref` targets, by JSON pointer, and whether a position that builds a
        # class reached them. A target inside a definition entry is queued
        # itself, and the entry that holds it is queued as built nothing, so the
        # entry is checked as a definition but its other contents read nothing.
        walked: Dict[Tuple[str, ...], bool] = {}
        pending: List[Tuple[Tuple[str, ...], Any, bool, bool]] = []
        # Model keys beside a `$ref` at a position that reads them but where
        # nothing is built, by identity; the nodes inside a definition entry
        # that a `$ref` targets; and every hop after the first of a `$ref`
        # chain from a position that builds a class. A target is walked with
        # no scope, so a node in all three is reported once the walk is done.
        unbuilt: Dict[int, Tuple[Any, ...]] = {}
        inner: set = set()
        passed: List[int] = []
        # Warnings for model keys no class reads, by node and key. Returned, and
        # issued only once the import builds, so an import that raises here or
        # later gives none.
        deferred: Dict[Tuple[int, str], str] = {}
        # Definition entries a `$ref` reached, and later every body walked as
        # building nothing, by identity; and every `$defs`/`definitions`
        # mapping passed, so the rest can be walked last.
        entries: set = set()
        mappings: List[Tuple[str, Dict[str, Any]]] = []

        def resolve(
            ref: Any,
        ) -> Tuple[Any, Tuple[str, ...], Optional[Tuple[str, ...]], Any]:
            return _resolve_local_ref(schema, ref)

        label = _ref_label

        def chain(node: Any, where: str, pointer: Optional[Tuple[str, ...]] = None):
            return _ref_chain(schema, node, where, pointer)

        def unread(
            node: Any,
            declared: List[str],
            subject: str,
            reason: str,
            advice: str,
            warn: bool,
        ) -> None:
            """Raise for model keys no class reads. On an object-typed
            property, where the docs call them valid, record a warning instead,
            until 1.1.0, and let the import go on: it builds what it builds
            without the keys, and warns, or it still fails and does not warn."""
            if not warn:
                raise ValueError(
                    f"Stickler extension '{declared[0]}' {subject} is not "
                    f"read{reason}. {advice} Left in place it would be dropped "
                    "silently."
                )
            for key in declared:
                deferred.setdefault(
                    (id(node), key),
                    f"Stickler extension '{key}' {subject} is not read{reason}. "
                    f"{advice} It is ignored, and this becomes a ValueError in "
                    "1.1.0.",
                )

        def collect(value: Any, path: str) -> None:
            """The `$defs`/`definitions` mappings inside a value not walked."""
            if isinstance(value, list):
                for index, item in enumerate(value):
                    collect(item, f"{path}[{index}]")
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key in ("$defs", "definitions") and isinstance(item, dict):
                        mappings.append((f"{path}.{key}", item))
                    else:
                        collect(item, f"{path}.{key}")

        def walk(
            node: Any,
            path: str,
            scope: Optional[str] = "model",
            built: bool = True,
            reads: bool = True,
        ) -> None:
            if isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]", scope, built, reads)
                return
            if not isinstance(node, dict):
                return

            if scope is not None:
                _reject_unknown_extensions(node, path or "(root)", scope=scope)

            ref = node.get("$ref")
            target, target_parts, entry, entry_body = resolve(ref)
            if entry is not None and target_parts == entry:
                pending.append((entry, entry_body, built, True))
            elif entry is not None:
                # The reference builds the target, not the entry around it. The
                # entry is still checked, but its other contents build nothing
                # here, and the target may sit where the entry's walk does not
                # build it (`not`, `x-*`), so it is walked from here as well.
                pending.append((entry, entry_body, False, True))
                pending.append((target_parts, target, built, False))
                inner.add(id(target))
            elif target is not None and target_parts:
                # Outside `$defs` the target may sit where the walk does not
                # build it (`components`) or does not go (`x-*`), so classes
                # built inside it would not count as readers.
                pending.append((target_parts, target, built, False))
            # Only a position that builds a class reads the model keys it
            # reaches: a `$ref` under `not` or `additionalProperties` builds
            # nothing, so it must not excuse an `allOf` that drops them.
            if reads and built:
                hops = list(chain(node, ""))
                read.update(id(hop) for hop, _, _ in hops)
                passed.extend(id(hop) for hop, _, _ in hops[1:])
            # The class is built from the `$ref` target, which reads its own
            # model keys. Only the root reads the ones beside its `$ref`.
            if reads and isinstance(ref, str) and scope in ("model", "field_or_model"):
                declared = sorted(k for k in node if k in _KNOWN_MODEL_EXTENSIONS)
                if declared:
                    hops = list(chain(node, ""))
                    # Through an alias, the class is the end of the chain.
                    source = hops[-1][1] if len(hops) > 1 else f"'{ref}'"
                    # `items` and a combiner arm are positions no doc names
                    # for these keys, so they raise. An object-typed property
                    # is one the docs call valid for them, so it warns.
                    found = (
                        node,
                        declared,
                        f"beside the '$ref' at '{path}'",
                        f": the class is built from {source}, which reads only "
                        "its own keys",
                        "Put it there.",
                        scope == "field_or_model",
                    )
                    if built:
                        unread(*found)
                    else:
                        # Nothing is built here, but a `$ref` from a position
                        # that builds a class may point at this node. Such a
                        # reference walks it with no scope, so it is checked
                        # once the walk is done.
                        unbuilt.setdefault(id(node), found)

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
                if key in _DATA_KEYWORDS or (
                    isinstance(key, str) and key.startswith("x-")
                ):
                    # A value, not a schema. An entry under `$defs` inside one
                    # is still walked last, as before.
                    collect(value, f"{path}.{key}" if path else key)
                    continue
                if key in ("$defs", "definitions") and isinstance(value, dict):
                    mappings.append((f"{path}.{key}" if path else key, value))
                    continue
                if key == "properties" and isinstance(value, dict):
                    for name, child in value.items():
                        # A property is scored as a field of this object, and if
                        # it is itself an object it also configures its own
                        # generated class, so both key sets are honoured there.
                        walk(
                            child,
                            f"{path}.{name}" if path else name,
                            "field_or_model" if _is_object_schema(child) else "field",
                            built,
                        )
                    continue
                if key == "allOf" and isinstance(value, list):
                    # The arms are merged into one `AllOfModel`, so no arm, and
                    # nothing an arm `$ref`s, has its model keys read here.
                    for index, arm in enumerate(value):
                        if built:
                            merged.append(
                                (
                                    arm,
                                    f"{path}.allOf[{index}]"
                                    if path
                                    else f"allOf[{index}]",
                                    path or "(root)",
                                )
                            )
                        walk(arm, f"{path}[{index}]", "model", built, reads=False)
                    continue
                # Named in the path, so an error under `not` or `then` points
                # at a node that exists (`$defs.W.not[]`, not `$defs.W[]`).
                child_path = f"{path}.{key}" if path else key
                child_scope: Optional[str] = None
                # Only these positions build part of the model. Anything else,
                # such as `not` or `if`, is walked for its shape but builds no
                # class whose keys could be dropped.
                child_built = False
                if key == "items":
                    child_path = f"{path}[]" if path else "[]"
                    # Field keys are NOT read on `items`: weight, threshold and
                    # comparator there are all dropped, since the field they would
                    # configure is the array itself, one level up. Only the element
                    # class's own settings are read.
                    child_scope, child_built = "model", built
                elif key in _COMBINERS or key in ("$defs", "definitions"):
                    # A `$defs` that is not a mapping is not a set of definitions;
                    # it is still walked, as before, rather than skipped.
                    child_path = path
                    child_scope, child_built = "model", built
                walk(value, child_path, child_scope, child_built)

        def own_keys(pointer: Tuple[str, ...], body: Any) -> None:
            """Check a property or `items` target under an `x-*` key, which the
            walk does not enter, with the path and scope it would have there."""
            at, kind, under_x = _walk_position(pointer)
            if not under_x or not isinstance(body, dict):
                return
            if kind == "property":
                scope = "field_or_model" if _is_object_schema(body) else "field"
            elif kind == "items":
                scope = "model"
            else:
                return
            _reject_unknown_extensions(body, at, scope=scope)

        read.add(id(schema))
        walk(schema, "", "model", reads=False)
        while pending:
            pointer, body, built, definition = pending.pop()
            if pointer in walked and (walked[pointer] or not built):
                continue
            walked[pointer] = built
            at = ".".join(pointer)
            if definition and isinstance(body, dict):
                entries.add(id(body))
                # Checked like any other node. Walking a definition with no scope
                # let a typo there import silently while the same typo inline
                # raised. See #340. Only a body with `properties` generates a
                # class: a `type: "object"` mapping becomes a `dict` field, judged
                # against its parent's threshold, so it reads no model keys.
                hops = [hop for hop, _, _ in chain(body, "")]
                scope = "value_definition"
                if "properties" in body:
                    scope = "definition"
                elif "$ref" in body and hops[-1] is not body:
                    # An alias reads no keys either, but where it ends at an
                    # object it can be used as one, so its advice says so.
                    if "properties" in hops[-1]:
                        scope = "alias_definition"
                # A one-arm `allOf` builds the object only if no hop is typed
                # otherwise: `["object", "null"]` fails to import there.
                merges = all(hop.get("type", "object") == "object" for hop in hops)
                _reject_unknown_extensions(body, at, scope=scope, merges=merges)
            if not definition:
                own_keys(pointer, body)
            walk(body, at, None, built, reads=False)

        # A node inside an entry, such as `#/$defs/List/items`, is built from
        # where it is referenced, but the entry's walk built nothing there, so
        # model keys beside the node's own `$ref` passed unread.
        for hop in passed:
            if hop in inner and hop in unbuilt:
                unread(*unbuilt[hop])

        # An entry no `$ref` reaches is walked as it was before definitions were
        # checked: its own keys are left alone, and its contents are checked
        # like any other node, so a typo in one of its properties still raises.
        # It builds nothing, but the `$ref`s it holds were followed before, so
        # their targets are walked for their contents too: a typo in an `x-*`
        # payload that only an unused entry references still raises.
        while mappings or pending:
            if mappings:
                at, mapping = mappings.pop()
                for name, body in mapping.items():
                    if id(body) not in entries:
                        entries.add(id(body))
                        walk(body, f"{at}.{name}", None, False, reads=False)
                continue
            pointer, body, _, _ = pending.pop()
            if pointer in walked or id(body) in entries:
                continue
            walked[pointer] = False
            entries.add(id(body))
            own_keys(pointer, body)
            walk(body, ".".join(pointer), None, False, reads=False)

        # An object's model keys are read only where its class is built. An
        # `allOf` arm is merged into `AllOfModel`, and a root `$ref` builds the
        # root class from the root's own keys, so keys there are dropped unless
        # another position builds the same object. See #340. An object-typed
        # property is a position the docs call valid for these keys, so there
        # it warns until 1.1.0.
        def check(
            node: Any,
            where: str,
            pointer: Optional[Tuple[str, ...]],
            reached: str,
            advice: str,
        ) -> None:
            for hop, hop_where, hop_pointer in chain(node, where, pointer):
                declared = sorted(key for key in hop if key in _KNOWN_MODEL_EXTENSIONS)
                if declared and id(hop) not in read:
                    unread(
                        hop,
                        declared,
                        f"on {hop_where}",
                        f" when it is reached through {reached}",
                        advice,
                        _is_property_pointer(hop_pointer) and _is_object_schema(hop),
                    )

        for arm, arm_path, at in merged:
            check(
                arm,
                f"object '{arm_path}'",
                None,
                f"'allOf' at '{at}'",
                "The arms are merged into one class with the default name and "
                "match threshold, and no position sets them. Reference the object "
                "directly, without 'allOf', to configure it.",
            )
        root_target, root_target_parts, root_entry, _ = resolve(schema.get("$ref"))
        if isinstance(root_target, dict):
            check(
                root_target,
                label(root_target_parts, root_entry),
                root_target_parts,
                "the root '$ref'",
                "The root class reads the root's own keys: put it on the root, "
                "beside the '$ref'.",
            )

        return list(deferred.values())

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

    def _as_object(self, node: Any) -> Any:
        """The object a raw node describes: through its `$ref` chain, and into
        the one arm of an `anyOf`/`oneOf` that is not `null`."""
        seen: set = set()
        while isinstance(node, dict) and id(node) not in seen:
            seen.add(id(node))
            if "$ref" in node:
                node = _resolve_local_ref(self.schema, node["$ref"])[0]
                continue
            arms = node.get("anyOf", node.get("oneOf"))
            if "properties" in node or not isinstance(arms, list):
                return node
            arms = [arm for arm in arms if arm != {"type": "null"}]
            if len(arms) != 1:
                return node
            node = arms[0]
        return node if isinstance(node, dict) else None

    def _object_properties(self, node: Any) -> List[Dict[str, Any]]:
        """The `properties` of the object a raw node describes, its own and
        those of its `allOf` arms, which the merged class holds as well."""
        found: List[Dict[str, Any]] = []
        stack, seen = [node], set()
        while stack:
            current = self._as_object(stack.pop())
            if not isinstance(current, dict) or id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current.get("properties"), dict):
                found.append(current["properties"])
            if isinstance(current.get("allOf"), list):
                stack.extend(reversed(current["allOf"]))
        return found

    def _properties_at(self, node: Any, path: str):
        """The raw schema of each property at `path` below `node`.

        `path` joins property names with '.' and marks an array's `items` with
        '[]', so each name is matched whole: a name holding '.' is found, not
        split into two. A path can then name more than one property (`a.b` as
        the property `a.b`, or as `b` inside `a`), so every match is given."""
        for properties in self._object_properties(node):
            for name, child in properties.items():
                if not isinstance(name, str) or not path.startswith(name):
                    continue
                rest = path[len(name) :]
                while True:
                    if not rest:
                        yield child
                        break
                    if rest.startswith("."):
                        yield from self._properties_at(child, rest[1:])
                    if not rest.startswith("[]"):
                        break
                    child = (self._as_object(child) or {}).get("items")
                    rest = rest[2:]

    def _element_class_sources(self, field_path: str) -> Optional[List[str]]:
        """Where the element class of the array property at `field_path` reads
        its own keys, when its `items`, or an arm of them, is a `$ref`: the end
        of each `$ref` chain. Empty when `items` is inline, and None when no
        single array property is found at that path in the raw schema."""
        path = field_path
        if self.field_path:
            if not path.startswith(self.field_path):
                return None
            path = path[len(self.field_path) :].lstrip(".")
        arrays = {
            id(array): array
            for array in (
                self._as_object(node) for node in self._properties_at(self.schema, path)
            )
            if isinstance(array, dict) and "items" in array
        }
        if len(arrays) != 1:
            return None
        items = next(iter(arrays.values()))["items"]
        if not isinstance(items, dict):
            return []
        arms = items.get("anyOf", items.get("oneOf"))
        refs = [items] if "$ref" in items else arms if isinstance(arms, list) else []
        sources: List[str] = []
        for ref in refs:
            if not isinstance(ref, dict) or "$ref" not in ref:
                continue
            hops = list(_ref_chain(self.schema, ref, ""))
            if len(hops) > 1 and hops[-1][1] not in sources:
                sources.append(hops[-1][1])
        return sources

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

        # Kept whether or not a comparator is named beside it. Reading it only in
        # the branch that resolves a named comparator meant an inferred field's
        # config was read into a local and thrown away, so the schema path and the
        # Stickler-config path built different comparators from the same input.
        # Normalised as it is read, so a non-mapping is named for what it is.
        # Deferring the check to the merge below meant a bad CONFIG was reported as
        # `Invalid x-aws-stickler-comparator 'NumericComparator'` -- blaming a
        # perfectly valid comparator name, and telling the author to look at the
        # wrong key.
        comparator_config = normalize_comparator_config(
            extra.get("x-aws-stickler-comparator-config"),
            f"field '{field_path}' (as 'x-aws-stickler-comparator-config')",
        )
        if comparator_config:
            extensions["comparator_config"] = comparator_config

        if "x-aws-stickler-comparator" in extra:
            comparator_name = extra["x-aws-stickler-comparator"]
            if comparator_name == AUTO_COMPARATOR:
                # A request to infer this field, not a comparator name. Resolving
                # it through the registry reported it as an unknown comparator and
                # listed the built-ins.
                extensions["infer"] = True
            else:
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
