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

Results are cached in a ``WeakKeyDictionary`` keyed on
``(cls, overrides-signature, weight_hints, match_threshold)`` so repeated
``evaluate`` calls in a loop are cheap without pinning user classes in memory.
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional, Tuple, Type, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ..comparators.structured import StructuredModelComparator
from ..structured_object_evaluator.models.comparable_field import ComparableField
from ..structured_object_evaluator.models.comparator_registry import (
    create_comparator,
    get_global_registry,
)
from ..structured_object_evaluator.models.model_factory import ModelFactory
from ..structured_object_evaluator.models.structured_model import StructuredModel
from .inference import InferredSpec, infer_field_config, unwrap_optional

# Cache of built shadow classes. Keyed by the source pydantic class (weak) ->
# {cache_key: shadow_class}. Weak keys let user classes be garbage-collected.
_CACHE: "weakref.WeakKeyDictionary[type, Dict[Any, Type[StructuredModel]]]" = (
    weakref.WeakKeyDictionary()
)


def structured_model_for(
    cls: Type[BaseModel],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    weight_hints: bool = False,
    match_threshold: float = 0.7,
) -> Type[StructuredModel]:
    """Return a ``StructuredModel`` subclass mirroring ``cls`` with inferred config.

    Args:
        cls: A pydantic ``BaseModel`` subclass (e.g. a Strands ``response_model``).
        overrides: Optional ``{field_name: ComparableField(...)}`` honored verbatim
            at the highest precedence. Fields not present are inferred.
        weight_hints: Enable name-token weight heuristics (default off).
        match_threshold: Overall match threshold for the generated model.

    Returns:
        A cached ``StructuredModel`` subclass whose fields carry inferred (or
        overridden) comparators, ready for ``compare_with``.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(
            f"structured_model_for expects a pydantic BaseModel subclass, got {cls!r}"
        )

    key = _cache_key(overrides, weight_hints, match_threshold)
    per_class = _CACHE.get(cls)
    if per_class is not None and key in per_class:
        return per_class[key]

    shadow = _build(
        cls,
        overrides=overrides or {},
        weight_hints=weight_hints,
        match_threshold=match_threshold,
        _seen={},
    )

    _CACHE.setdefault(cls, {})[key] = shadow
    return shadow


def specs_for(
    cls: Type[BaseModel],
    *,
    weight_hints: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, InferredSpec]:
    """Return the inferred spec per primitive field (for ``.explain()``).

    Nested-model and list fields are reported with a synthetic spec describing
    the structural handling rather than a primitive comparator. Overridden
    fields report ``source="override"`` rather than the inferred choice.
    """
    registry = get_global_registry()
    overrides = overrides or {}
    result: Dict[str, InferredSpec] = {}
    for name, field_info in cls.model_fields.items():
        if name in overrides:
            result[name] = InferredSpec(
                comparator_name="(user override)",
                provenance=["override: user-supplied ComparableField"],
            )
            continue
        kind = _field_kind(field_info.annotation)
        if kind == "model":
            result[name] = InferredSpec(
                comparator_name="StructuredModelComparator",
                threshold=0.9,
                provenance=["type:nested BaseModel -> recursive comparison"],
            )
        elif kind == "model_list":
            result[name] = InferredSpec(
                comparator_name="Hungarian (per-element StructuredModel)",
                provenance=["type:List[BaseModel] -> Hungarian object matching"],
            )
        else:
            result[name] = infer_field_config(
                name, field_info, weight_hints=weight_hints, registry=registry
            )
    return result


# --- internals --------------------------------------------------------------


def _build(
    cls: Type[BaseModel],
    *,
    overrides: Dict[str, Any],
    weight_hints: bool,
    match_threshold: float,
    _seen: Dict[type, Type[StructuredModel]],
) -> Type[StructuredModel]:
    """Recursively assemble the shadow class for ``cls`` (cycle-safe)."""
    if cls in _seen:
        return _seen[cls]

    # Placeholder registration for self-referential models. Pydantic can't build
    # a truly recursive dynamic class in one pass, so a direct cycle degrades to
    # a generic structured comparison at that edge (documented limitation).
    registry = get_global_registry()
    field_definitions: Dict[str, Tuple[Any, Any]] = {}

    for name, field_info in cls.model_fields.items():
        if name in overrides:
            comparable = overrides[name]
            wire_type, _ = _wire_type_for(field_info, cls, overrides, weight_hints, match_threshold, _seen)
            field_definitions[name] = (wire_type, comparable)
            continue

        field_definitions[name] = _field_definition(
            name,
            field_info,
            cls=cls,
            overrides=overrides,
            weight_hints=weight_hints,
            match_threshold=match_threshold,
            registry=registry,
            _seen=_seen,
        )

    model_name = _valid_identifier(f"{cls.__name__}Eval")
    shadow = ModelFactory.create_model_from_fields(
        model_name=model_name,
        field_definitions=field_definitions,
        match_threshold=match_threshold,
        base_class=StructuredModel,
    )
    _seen[cls] = shadow
    return shadow


def _field_definition(
    name: str,
    field_info: FieldInfo,
    *,
    cls: Type[BaseModel],
    overrides: Dict[str, Any],
    weight_hints: bool,
    match_threshold: float,
    registry,
    _seen: Dict[type, Type[StructuredModel]],
) -> Tuple[Any, Any]:
    """Produce a ``(type, Field)`` tuple for one field."""
    annotation, _ = unwrap_optional(field_info.annotation)
    kind = _field_kind(annotation)

    if kind == "model":
        child = _build(
            annotation,
            overrides={},
            weight_hints=weight_hints,
            match_threshold=match_threshold,
            _seen=_seen,
        )
        # A single nested model may carry an explicit StructuredModelComparator
        # (unlike List[StructuredModel], which __init_subclass__ forbids).
        return (
            Optional[child],
            ComparableField(
                comparator=StructuredModelComparator(),
                threshold=0.9,
                weight=1.0,
                default=None,
            ),
        )

    if kind == "model_list":
        element = get_args(annotation)[0]
        child = _build(
            element,
            overrides={},
            weight_hints=weight_hints,
            match_threshold=match_threshold,
            _seen=_seen,
        )
        # List[StructuredModel] must use default threshold/comparator; Hungarian
        # matching uses each element's match_threshold. Weight-only is allowed.
        return (List[child], ComparableField(weight=1.0, default=None))

    if kind == "primitive_list":
        element = get_args(annotation)[0]
        spec = _primitive_spec(name, element, weight_hints, registry)
        return (
            List[element],
            _comparable_from_spec(spec, default=None),
        )

    # Primitive scalar.
    spec = infer_field_config(
        name, field_info, weight_hints=weight_hints, registry=registry
    )
    wire = _scalar_wire_type(annotation)
    default = ... if field_info.is_required() else None
    return (Optional[wire] if default is None else wire, _comparable_from_spec(spec, default=default))


def _primitive_spec(name: str, element: Any, weight_hints: bool, registry) -> InferredSpec:
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


def _wire_type_for(
    field_info: FieldInfo,
    cls: Type[BaseModel],
    overrides: Dict[str, Any],
    weight_hints: bool,
    match_threshold: float,
    _seen: Dict[type, Type[StructuredModel]],
) -> Tuple[Any, None]:
    """Resolve just the wire type for an overridden field (config comes from the override)."""
    annotation, _ = unwrap_optional(field_info.annotation)
    kind = _field_kind(annotation)
    if kind == "model":
        child = _build(annotation, overrides={}, weight_hints=weight_hints, match_threshold=match_threshold, _seen=_seen)
        return Optional[child], None
    if kind == "model_list":
        element = get_args(annotation)[0]
        child = _build(element, overrides={}, weight_hints=weight_hints, match_threshold=match_threshold, _seen=_seen)
        return List[child], None
    if kind == "primitive_list":
        return List[get_args(annotation)[0]], None
    wire = _scalar_wire_type(annotation)
    return (Optional[wire] if not field_info.is_required() else wire), None


def _field_kind(annotation: Any) -> str:
    """Classify a (already-optional-unwrapped) annotation."""
    annotation, _ = unwrap_optional(annotation)
    if _is_model(annotation):
        return "model"
    origin = get_origin(annotation)
    if origin in (list, List):
        args = get_args(annotation)
        if args and _is_model(args[0]):
            return "model_list"
        return "primitive_list"
    return "primitive"


def _is_model(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and not _is_stickler_model(annotation)
    )


def _is_stickler_model(annotation: Any) -> bool:
    # Already a StructuredModel? Treat as model but no rebuild needed upstream;
    # for simplicity we still recurse (it will just re-wrap fields).
    return isinstance(annotation, type) and issubclass(annotation, StructuredModel)


def _scalar_wire_type(annotation: Any) -> Any:
    """The type the shadow field should declare.

    Instances are normalized via ``model_dump(mode="json")`` before comparison,
    so enums/dates arrive as strings. Declaring ``str`` for those keeps pydantic
    validation from rejecting the JSON wire form.
    """
    import datetime
    import enum

    from .inference import _is_literal

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return str
    if _is_literal(annotation):
        return str
    if isinstance(annotation, type) and issubclass(annotation, datetime.date):
        return str
    if annotation in (str, int, float, bool):
        return annotation
    # Exotic types are compared as their JSON string form.
    return str


def _valid_identifier(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not cleaned or not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"M_{cleaned}"
    return cleaned or "DynamicModelEval"


def _cache_key(
    overrides: Optional[Dict[str, Any]], weight_hints: bool, match_threshold: float
) -> Tuple:
    override_sig = tuple(sorted((overrides or {}).keys()))
    return (override_sig, weight_hints, match_threshold, len(overrides or {}))
