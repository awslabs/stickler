"""Zero-config comparator inference for vanilla pydantic fields.

This module is the "brain" behind :func:`stickler.evaluate`. Given a single
pydantic ``FieldInfo`` (the live one from ``cls.model_fields``, never a
JSON-schema round-trip) it decides which comparator/threshold/weight the field
should be evaluated with, so an *unconfigured* model still gets a sensible,
type-aware evaluation instead of the blind string-edit-distance fallback the
raw comparison engine applies to unannotated fields.

Design rules (see ``auto/README.md``):

* **Type first.** The python annotation is the highest-value, always-present
  signal. ``bool``/``Enum``/``Literal`` -> Exact, ``int`` -> Numeric(exact),
  ``float`` -> Numeric(tolerant), ``date``/``datetime`` -> Date, ``str`` ->
  Levenshtein.
* **Name tokens refine.** Field-name tokens (``id``, ``amount``, ``email`` ...)
  sharpen the comparator/threshold on top of the type default.
* **Weights are honest.** Business-criticality is not encoded in a vanilla
  model, so weights default to ``1.0``. Name-token weight bumps are opt-in via
  ``weight_hints=True`` and always recorded in provenance.
* **Never surprise.** Semantic/BERT/LLM comparators are never auto-selected;
  a comparator that is unavailable in this environment degrades to
  Levenshtein/Exact and the degrade is recorded.

The public entry point is :func:`infer_field_config`, which returns an
:class:`InferredSpec`. The builder turns that spec into a ``(type, Field)``
tuple for ``ModelFactory.create_model_from_fields``.
"""

from __future__ import annotations

import datetime
import enum
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from pydantic.fields import FieldInfo

from ..structured_object_evaluator.models.comparator_registry import (
    ComparatorRegistry,
    get_global_registry,
)

# Comparators we will never auto-select: they need models/embeddings/creds and
# would fail or be surprising as a silent default. A field only ever gets one
# of these if the user configures it explicitly.
_NEVER_AUTO = frozenset(
    {"SemanticComparator", "BERTComparator", "LLMComparator"}
)

# Relative tolerance applied to inferred float comparisons. A bare Numeric
# comparator defaults to exact matching, so a field threshold alone is inert;
# the tolerance must live on the comparator instance.
_FLOAT_RELATIVE_TOLERANCE = 0.001


@dataclass
class InferredSpec:
    """Resolved comparison configuration for a single field.

    Everything the builder needs to emit a ``ComparableField`` plus the
    provenance surfaced through ``EvalResult.explain()``.
    """

    comparator_name: str
    comparator_config: Dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    weight: float = 1.0
    clip_under_threshold: bool = True
    # Human-readable trail of how this spec was chosen, e.g.
    # ["type:float -> NumericComparator@0.95", "name-token:amount -> weight 2.5"].
    provenance: List[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        """Coarse origin label for the final comparator decision."""
        for entry in self.provenance:
            if entry.startswith("override"):
                return "override"
        for entry in reversed(self.provenance):
            if entry.startswith("degrade"):
                return "degrade"
        for entry in self.provenance:
            if entry.startswith("name-token"):
                return "name-token"
        return "type"


# --- Type detection helpers -------------------------------------------------


def unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    """Strip ``Optional[X]`` / ``Union[X, None]`` down to ``X``.

    Returns ``(inner, was_optional)``. A multi-arm union that is not simply
    ``X | None`` is returned unchanged (the caller treats it as ambiguous and
    falls back to a string comparator).
    """
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        # Ambiguous multi-arm union: leave as-is, mark optional if None present.
        return annotation, type(None) in get_args(annotation)
    return annotation, False


def _is_enum(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, enum.Enum)


def _is_literal(annotation: Any) -> bool:
    return get_origin(annotation) is Literal


def _is_date(annotation: Any) -> bool:
    # datetime is a subclass of date; both route to DateComparator.
    return isinstance(annotation, type) and issubclass(annotation, datetime.date)


# --- Name-token rules -------------------------------------------------------

# Each rule: token-set -> (comparator, config, threshold, weight_hint).
# ``weight_hint`` is only applied when ``weight_hints=True``. Rules are checked
# in order; the first whose token appears as a whole word in the field name
# wins. Ambiguous tokens (type/state/number/key/level/...) are intentionally
# absent so they stay at the safe type default.
_NAME_TOKEN_RULES: List[Tuple[frozenset, str, Dict[str, Any], float, float]] = [
    (
        frozenset({"id", "uuid", "guid", "sku", "code", "ref", "isbn", "ssn"}),
        "ExactComparator",
        {},
        1.0,
        3.0,
    ),
    (
        frozenset(
            {"amount", "price", "total", "subtotal", "cost", "balance", "fee",
             "tax", "salary", "revenue"}
        ),
        "NumericComparator",
        {"relative_tolerance": _FLOAT_RELATIVE_TOLERANCE},
        0.95,
        2.5,
    ),
    (
        frozenset({"quantity", "qty", "count", "units", "num"}),
        "NumericComparator",
        {},
        1.0,
        1.2,
    ),
    (
        frozenset(
            {"date", "dob", "expiry", "expires", "created", "updated",
             "issued", "due", "birthdate"}
        ),
        "DateComparator",
        {},
        0.95,
        1.5,
    ),
    (
        frozenset({"email", "url", "uri", "zip", "postal", "postcode", "phone"}),
        "ExactComparator",
        {},
        1.0,
        1.5,
    ),
    (
        frozenset(
            {"name", "customer", "vendor", "company", "contact", "author",
             "recipient"}
        ),
        "LevenshteinComparator",
        {},
        0.85,
        1.5,
    ),
    (
        frozenset({"address", "addr", "street", "city", "location"}),
        "FuzzyComparator",
        {"method": "token_sort_ratio"},
        0.8,
        1.5,
    ),
    (
        frozenset(
            {"description", "desc", "summary", "notes", "note", "comment",
             "comments", "remarks", "memo", "body", "text"}
        ),
        "FuzzyComparator",
        {"method": "token_set_ratio"},
        0.6,
        0.3,
    ),
]

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _field_tokens(field_name: str) -> frozenset:
    """Split a field name into lowercase word tokens (snake/camel-friendly)."""
    # Insert boundaries for camelCase before lowercasing.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field_name)
    return frozenset(t for t in _TOKEN_SPLIT.split(spaced.lower()) if t)


# --- Availability gate ------------------------------------------------------


def _gate(
    comparator_name: str,
    config: Dict[str, Any],
    registry: ComparatorRegistry,
    provenance: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """Ensure ``comparator_name`` can actually be built in this environment.

    Falls back to Levenshtein (or Exact for Date) and records the degrade if
    the comparator is unregistered or fails to instantiate (e.g. rapidfuzz or
    python-dateutil not installed). ``_NEVER_AUTO`` comparators should never
    reach here, but are also downgraded defensively.
    """
    if comparator_name in _NEVER_AUTO:
        provenance.append(f"degrade:{comparator_name}->LevenshteinComparator (never auto-selected)")
        return "LevenshteinComparator", {}

    if not registry.is_registered(comparator_name):
        fallback = "ExactComparator" if comparator_name == "DateComparator" else "LevenshteinComparator"
        provenance.append(f"degrade:{comparator_name}->{fallback} (not registered)")
        return fallback, {}

    try:
        # Probe construction so a missing optional backend degrades now, not at
        # compare time.
        registry.create_instance(comparator_name, config)
    except Exception as exc:  # noqa: BLE001 - any construction failure degrades
        fallback = "ExactComparator" if comparator_name == "DateComparator" else "LevenshteinComparator"
        provenance.append(
            f"degrade:{comparator_name}->{fallback} ({type(exc).__name__})"
        )
        return fallback, {}

    return comparator_name, config


# --- Main entry point -------------------------------------------------------


def infer_field_config(
    field_name: str,
    field_info: FieldInfo,
    *,
    weight_hints: bool = False,
    registry: Optional[ComparatorRegistry] = None,
) -> InferredSpec:
    """Infer a comparison spec for one pydantic field.

    Args:
        field_name: The field's name (drives name-token heuristics).
        field_info: The live ``FieldInfo`` from ``cls.model_fields``.
        weight_hints: When True, apply name-token weight bumps. When False
            (default) all weights stay 1.0 so precision/recall are not skewed by
            guessed business-criticality.
        registry: Comparator registry for the availability gate. Defaults to the
            global registry.

    Returns:
        An :class:`InferredSpec`. Nested ``BaseModel`` / ``List[BaseModel]``
        fields are NOT resolved here (the builder detects and recurses on them);
        this function only handles primitive and primitive-list fields.
    """
    registry = registry or get_global_registry()
    annotation, was_optional = unwrap_optional(field_info.annotation)
    provenance: List[str] = []
    if was_optional:
        provenance.append("optional: unwrapped to inner type")

    # 1) Type signal (safe, always on).
    comparator, config, threshold, clip = _type_default(annotation, provenance)

    # 2) Name-token refinement layered on the type default.
    weight = 1.0
    token_match = _match_name_token(field_name)
    if token_match is not None:
        t_comparator, t_config, t_threshold, t_weight = token_match
        comparator, config, threshold = t_comparator, dict(t_config), t_threshold
        provenance.append(
            f"name-token:{field_name} -> {t_comparator}@{t_threshold}"
        )
        if weight_hints and t_weight != 1.0:
            weight = t_weight
            provenance.append(f"name-token:{field_name} -> weight {t_weight}")
        # Free-text fields keep partial credit rather than clipping to zero.
        if t_comparator == "FuzzyComparator" and t_config.get("method") == "token_set_ratio":
            clip = False

    # 3) Availability gate (degrade unavailable comparators).
    comparator, config = _gate(comparator, config, registry, provenance)

    return InferredSpec(
        comparator_name=comparator,
        comparator_config=config,
        threshold=threshold,
        weight=weight,
        clip_under_threshold=clip,
        provenance=provenance,
    )


def _type_default(
    annotation: Any, provenance: List[str]
) -> Tuple[str, Dict[str, Any], float, bool]:
    """Comparator/threshold/clip from the python type alone."""
    if annotation is bool:
        provenance.append("type:bool -> ExactComparator@1.0")
        return "ExactComparator", {}, 1.0, True
    if _is_enum(annotation) or _is_literal(annotation):
        kind = "enum" if _is_enum(annotation) else "literal"
        provenance.append(f"type:{kind} -> ExactComparator@1.0")
        return "ExactComparator", {}, 1.0, True
    if _is_date(annotation):
        provenance.append("type:date -> DateComparator@0.95")
        return "DateComparator", {}, 0.95, True
    if annotation is int:
        provenance.append("type:int -> NumericComparator(exact)@1.0")
        return "NumericComparator", {}, 1.0, True
    if annotation is float:
        provenance.append(
            f"type:float -> NumericComparator(rel_tol={_FLOAT_RELATIVE_TOLERANCE})@0.95"
        )
        return (
            "NumericComparator",
            {"relative_tolerance": _FLOAT_RELATIVE_TOLERANCE},
            0.95,
            True,
        )
    if annotation is str:
        provenance.append("type:str -> LevenshteinComparator@0.7")
        return "LevenshteinComparator", {}, 0.7, True

    # Unknown / exotic (dict, tuple, bytes, multi-arm union): safe string default.
    provenance.append(
        f"type:{_annotation_label(annotation)} -> LevenshteinComparator@0.7 (fallback)"
    )
    return "LevenshteinComparator", {}, 0.7, True


def _match_name_token(
    field_name: str,
) -> Optional[Tuple[str, Dict[str, Any], float, float]]:
    tokens = _field_tokens(field_name)
    for token_set, comparator, config, threshold, weight_hint in _NAME_TOKEN_RULES:
        if tokens & token_set:
            return comparator, config, threshold, weight_hint
    return None


def _annotation_label(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))
