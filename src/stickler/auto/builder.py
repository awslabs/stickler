"""Build a comparable ``StructuredModel`` from a vanilla pydantic class.

:func:`structured_model_for` is the bridge between a user's plain
``pydantic.BaseModel`` and stickler's comparison engine. It walks the live
``cls.model_fields``, asks :mod:`.inference` for a comparison spec per primitive
field, recurses into nested ``BaseModel`` / ``List[BaseModel]`` fields, and
delegates the actual class creation to the existing
``ModelFactory.create_model_from_fields``.

Why walk ``model_fields`` instead of ``model_json_schema()``? The JSON-schema
path is lossy and crashes on real models: ``Optional[str]`` becomes an
``anyOf`` with no ``type`` (raising "Unsupported JSON Schema type: None"),
enums degrade to bare strings, and ``datetime`` loses its type. The live
annotations keep every signal inference needs.

Wire form: instances are normalized via ``model_dump(mode="json")`` before
comparison. Fields whose JSON form is not a scalar (dict, tuple, set, Any,
multi-arm unions, unparameterized containers) are declared with a shadow type
that canonicalizes the incoming JSON value to a deterministic string
(``json.dumps`` with sorted keys), so "compared as their JSON string form" is
actually true rather than a validation error.

Results are cached in a ``WeakKeyDictionary`` keyed on
``(cls, weight_hints, match_threshold)`` so repeated ``evaluate`` calls in a
loop are cheap without pinning user classes in memory.
"""

from __future__ import annotations

import datetime
import enum
import json
import weakref
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    get_args,
    get_origin,
)

from pydantic import BaseModel, BeforeValidator
from pydantic.fields import FieldInfo

from ..comparators.structured import StructuredModelComparator
from ..structured_object_evaluator.models.comparable_field import ComparableField
from ..structured_object_evaluator.models.comparator_registry import (
    create_comparator,
    get_global_registry,
)
from ..structured_object_evaluator.models.model_factory import ModelFactory
from ..structured_object_evaluator.models.structured_model import StructuredModel
from .inference import InferredSpec, _is_literal, infer_field_config, unwrap_optional

# Cache of built shadow classes. Keyed by the source pydantic class (weak) ->
# {cache_key: shadow_class}. Weak keys let user classes be garbage-collected.
_CACHE: "weakref.WeakKeyDictionary[type, Dict[Any, Type[StructuredModel]]]" = (
    weakref.WeakKeyDictionary()
)

# StructuredModel reserves this field name for its own extra-field capture;
# a user field with the same name would be silently dropped from scoring.
_RESERVED_FIELD_NAMES = frozenset({"extra_fields"})


def structured_model_for(
    cls: Type[BaseModel],
    *,
    weight_hints: bool = False,
    match_threshold: float = 0.7,
) -> Type[StructuredModel]:
    """Return a ``StructuredModel`` subclass mirroring ``cls`` with inferred config.

    Args:
        cls: A pydantic ``BaseModel`` subclass (e.g. a Strands ``response_model``).
        weight_hints: Enable name-token weight heuristics (default off).
        match_threshold: Overall match threshold for the generated model.

    Returns:
        A cached ``StructuredModel`` subclass whose fields carry inferred
        comparators, ready for ``compare_with``.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(
            f"structured_model_for expects a pydantic BaseModel subclass, got {cls!r}"
        )

    key = _cache_key(weight_hints, match_threshold)
    per_class = _CACHE.get(cls)
    if per_class is not None and key in per_class:
        return per_class[key]

    shadow = _build(
        cls,
        weight_hints=weight_hints,
        match_threshold=match_threshold,
        _building=set(),
    )

    _CACHE.setdefault(cls, {})[key] = shadow
    return shadow


def specs_for(
    cls: Type[BaseModel],
    *,
    weight_hints: bool = False,
    match_threshold: float = 0.7,
) -> Dict[str, InferredSpec]:
    """Return the inferred spec per field (for ``.explain()``), keyed by
    dotted path.

    Nested ``BaseModel`` / ``List[BaseModel]`` fields appear twice: once as a
    structural row (recursive / Hungarian handling) and once per sub-field
    under a dotted path (``lines.sku``), so every comparator decision at every
    depth is auditable. Primitive-list fields report the spec of the ELEMENT
    type, matching what the builder actually installs.
    """
    registry = get_global_registry()
    result: Dict[str, InferredSpec] = {}
    _collect_specs(cls, "", weight_hints, match_threshold, registry, result, set())
    return result


def _collect_specs(
    cls: Type[BaseModel],
    prefix: str,
    weight_hints: bool,
    match_threshold: float,
    registry,
    result: Dict[str, InferredSpec],
    _visiting: set,
) -> None:
    if cls in _visiting:
        return
    _visiting.add(cls)
    try:
        for name, field_info in cls.model_fields.items():
            path = f"{prefix}{name}"
            annotation, _ = unwrap_optional(field_info.annotation)
            kind = _field_kind(annotation)
            if kind == "model":
                result[path] = InferredSpec(
                    comparator_name="StructuredModelComparator",
                    threshold=match_threshold,
                    clip_under_threshold=False,
                    provenance=["type:nested BaseModel -> recursive comparison"],
                )
                if not issubclass(annotation, StructuredModel):
                    _collect_specs(
                        annotation,
                        f"{path}.",
                        weight_hints,
                        match_threshold,
                        registry,
                        result,
                        _visiting,
                    )
            elif kind == "model_list":
                result[path] = InferredSpec(
                    comparator_name="Hungarian (per-element StructuredModel)",
                    threshold=match_threshold,
                    provenance=["type:List[BaseModel] -> Hungarian object matching"],
                )
                element, _ = unwrap_optional(_list_element(annotation))
                if not issubclass(element, StructuredModel):
                    _collect_specs(
                        element,
                        f"{path}.",
                        weight_hints,
                        match_threshold,
                        registry,
                        result,
                        _visiting,
                    )
            elif kind == "primitive_list":
                # Report the element spec — the same one _field_definition
                # installs — so the audit trail matches the built model.
                element, _ = unwrap_optional(_list_element(annotation))
                spec = _primitive_spec(name, element, weight_hints, registry)
                spec.provenance.insert(0, "list: spec applies to each element")
                result[path] = spec
            else:
                result[path] = infer_field_config(
                    name, field_info, weight_hints=weight_hints, registry=registry
                )
    finally:
        _visiting.discard(cls)


# --- internals --------------------------------------------------------------


def _build(
    cls: Type[BaseModel],
    *,
    weight_hints: bool,
    match_threshold: float,
    _building: set,
) -> Type[StructuredModel]:
    """Recursively assemble the shadow class for ``cls``.

    Recursive model graphs (self-referential or mutually recursive) cannot be
    mirrored as shadow classes: the shadow for ``cls`` would need itself as a
    field type before it exists. Rather than overflow the stack, raise a clear
    error naming the cycle.
    """
    if cls in _building:
        raise TypeError(
            f"evaluate() does not support recursive models: {cls.__name__} "
            "refers back to itself (directly or via another model). "
            "Evaluate the non-recursive parts, or define a StructuredModel "
            "subclass for this shape explicitly."
        )
    _building.add(cls)
    try:
        registry = get_global_registry()
        field_definitions: Dict[str, Tuple[Any, Any]] = {}

        for name, field_info in cls.model_fields.items():
            if name in _RESERVED_FIELD_NAMES:
                raise TypeError(
                    f"evaluate() cannot mirror field {cls.__name__}.{name}: "
                    f"'{name}' is reserved by stickler's comparison engine. "
                    "Rename the field (e.g. with a pydantic alias) to evaluate "
                    "this model."
                )
            field_definitions[name] = _field_definition(
                name,
                field_info,
                weight_hints=weight_hints,
                match_threshold=match_threshold,
                registry=registry,
                _building=_building,
            )

        if not field_definitions:
            raise TypeError(
                f"evaluate() needs at least one field to score; "
                f"{cls.__name__} defines none."
            )

        model_name = _valid_identifier(f"{cls.__name__}Eval")
        shadow = ModelFactory.create_model_from_fields(
            model_name=model_name,
            field_definitions=field_definitions,
            match_threshold=match_threshold,
            base_class=StructuredModel,
        )
        return shadow
    finally:
        _building.discard(cls)


def _field_definition(
    name: str,
    field_info: FieldInfo,
    *,
    weight_hints: bool,
    match_threshold: float,
    registry,
    _building: set,
) -> Tuple[Any, Any]:
    """Produce a ``(type, Field)`` tuple for one field.

    Nullability of the shadow field comes from the SOURCE annotation
    (``Optional[X]`` / ``X | None``), not from required-ness: a required
    ``Optional[str]`` must still accept None, and a non-required plain ``str``
    gets a None default only because the shadow model needs a placeholder for
    an omitted value.
    """
    annotation, was_optional = unwrap_optional(field_info.annotation)
    kind = _field_kind(annotation)
    # An omitted optional field materializes as None after model_dump, so the
    # shadow must accept None whenever the source annotation does OR the field
    # can be absent. Any admits None by definition.
    nullable = was_optional or not field_info.is_required() or annotation is Any

    if kind == "model":
        child = _shadow_for(
            annotation,
            weight_hints=weight_hints,
            match_threshold=match_threshold,
            _building=_building,
        )
        # A single nested model may carry an explicit StructuredModelComparator
        # (unlike List[StructuredModel], which __init_subclass__ forbids).
        # clip_under_threshold=False so a mostly-right nested object gets
        # partial credit, consistent with how the same object scores as a
        # one-element List[Model] via Hungarian matching.
        return (
            Optional[child],
            ComparableField(
                comparator=StructuredModelComparator(),
                threshold=match_threshold,
                clip_under_threshold=False,
                weight=1.0,
                default=None,
            ),
        )

    if kind == "model_list":
        element, element_optional = unwrap_optional(_list_element(annotation))
        child = _shadow_for(
            element,
            weight_hints=weight_hints,
            match_threshold=match_threshold,
            _building=_building,
        )
        element_type = Optional[child] if element_optional else child
        list_type = List[element_type]
        # List[StructuredModel] must use default threshold/comparator; Hungarian
        # matching uses each element's match_threshold. Weight-only is allowed.
        return (
            Optional[list_type] if nullable else list_type,
            ComparableField(weight=1.0, default=None),
        )

    if kind == "primitive_list":
        element, element_optional = unwrap_optional(_list_element(annotation))
        spec = _primitive_spec(name, element, weight_hints, registry)
        wire = _scalar_wire_type(element)
        element_type = Optional[wire] if element_optional else wire
        list_type = List[element_type]
        return (
            Optional[list_type] if nullable else list_type,
            _comparable_from_spec(spec, default=None),
        )

    # Primitive scalar.
    spec = infer_field_config(
        name, field_info, weight_hints=weight_hints, registry=registry
    )
    wire = _scalar_wire_type(annotation)
    default = ... if field_info.is_required() else None
    return (
        Optional[wire] if nullable else wire,
        _comparable_from_spec(spec, default=default),
    )


def _shadow_for(
    annotation: Any,
    *,
    weight_hints: bool,
    match_threshold: float,
    _building: set,
) -> Type[StructuredModel]:
    """Shadow class for a nested model annotation.

    A field typed as a ``StructuredModel`` subclass already carries its own
    comparators; use it directly instead of re-inferring (which would discard
    the user's explicit configuration).
    """
    if isinstance(annotation, type) and issubclass(annotation, StructuredModel):
        return annotation
    return _build(
        annotation,
        weight_hints=weight_hints,
        match_threshold=match_threshold,
        _building=_building,
    )


def _primitive_spec(
    name: str, element: Any, weight_hints: bool, registry
) -> InferredSpec:
    """Infer a spec for the *element* type of a primitive list."""
    dummy = FieldInfo(annotation=element)
    return infer_field_config(name, dummy, weight_hints=weight_hints, registry=registry)


def _comparable_from_spec(spec: InferredSpec, *, default: Any):
    return ComparableField(
        comparator=create_comparator(spec.comparator_name, spec.comparator_config),
        threshold=spec.threshold,
        weight=spec.weight,
        clip_under_threshold=spec.clip_under_threshold,
        default=default,
    )


def _field_kind(annotation: Any) -> str:
    """Classify a (already-optional-unwrapped) annotation."""
    annotation, _ = unwrap_optional(annotation)
    if _is_model(annotation):
        return "model"
    origin = get_origin(annotation)
    if origin in (list, List):
        args = get_args(annotation)
        if not args:
            # Unparameterized List: elements have unknown type; treat as a
            # list of exotic scalars (canonicalized JSON strings).
            return "primitive_list"
        element, _ = unwrap_optional(args[0])
        if _is_model(element):
            return "model_list"
        return "primitive_list"
    return "primitive"


def _list_element(annotation: Any) -> Any:
    """Element annotation of a list type; Any when unparameterized."""
    args = get_args(annotation)
    return args[0] if args else Any


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _canonicalize_json(value: Any) -> Any:
    """Normalize a JSON value to a deterministic string (None passes through).

    Applied as a BeforeValidator on shadow fields whose source type has no
    scalar JSON form. Both GT and prediction pass through the same
    canonicalization, so key order and container spelling never affect scores.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return json.dumps(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _canonicalize_json_sorted(value: Any) -> Any:
    """Like :func:`_canonicalize_json`, also sorting top-level arrays.

    Used for ``Set``/``FrozenSet`` sources, whose ``model_dump`` order is not
    deterministic across processes.
    """
    if isinstance(value, (list, tuple)):
        value = sorted(value, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    return _canonicalize_json(value)


# Shadow types for fields whose JSON wire form is not a scalar.
_WireJson = Annotated[str, BeforeValidator(_canonicalize_json)]
_WireJsonSorted = Annotated[str, BeforeValidator(_canonicalize_json_sorted)]


def _scalar_wire_type(annotation: Any) -> Any:
    """The type the shadow field should declare.

    Instances are normalized via ``model_dump(mode="json")`` before comparison,
    so enums/dates arrive in their JSON form. String-form sources declare
    ``str``; enum/Literal members may be non-str (IntEnum, Literal[1, 2]), so
    they canonicalize through ``_WireJson``. Types with no scalar JSON form
    (dict, tuple, set, Any, multi-arm unions) canonicalize to a deterministic
    JSON string so they are genuinely "compared as their JSON string form".
    """
    annotation, _ = unwrap_optional(annotation)

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        # model_dump emits the member VALUE, which is int for IntEnum.
        return _WireJson
    if _is_literal(annotation):
        return _WireJson
    if isinstance(annotation, type) and issubclass(annotation, datetime.date):
        return str
    if annotation in (str, int, float, bool):
        return annotation
    if isinstance(annotation, type) and issubclass(annotation, (set, frozenset)):
        return _WireJsonSorted
    origin = get_origin(annotation)
    if origin in (set, frozenset):
        return _WireJsonSorted
    # Everything else (dict, tuple, Any, multi-arm unions, unknown objects)
    # is compared as its canonical JSON string form.
    return _WireJson


def _valid_identifier(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not cleaned or not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"M_{cleaned}"
    return cleaned or "DynamicModelEval"


def _cache_key(weight_hints: bool, match_threshold: float) -> Tuple:
    return (weight_hints, match_threshold)
