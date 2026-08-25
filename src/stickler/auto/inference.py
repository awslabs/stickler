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
import decimal
import enum
import re
import types
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
_NEVER_AUTO = frozenset({"SemanticComparator", "BERTComparator", "LLMComparator"})

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
        for entry in reversed(self.provenance):
            if entry.startswith("degrade"):
                return "degrade"
        for entry in self.provenance:
            if entry.startswith("name-token"):
                return "name-token"
        return "type"


# --- Type detection helpers -------------------------------------------------


def unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    """Strip ``Optional[X]`` / ``Union[X, None]`` / ``X | None`` down to ``X``.

    Returns ``(inner, was_optional)``. A multi-arm union that is not simply
    ``X | None`` is returned unchanged (the caller treats it as ambiguous and
    falls back to a string comparator).

    One of three deliberate copies of this logic, kept separate to avoid
    cross-package dependencies. The others are
    ``stickler.reporting.html.utils.data_extractors`` and
    ``stickler.structured_object_evaluator.models.optional_annotation`` (which
    additionally exposes the permissive ``is_union`` / ``union_args`` pair, for
    sites that search a union's arms rather than unwrapping it). Keep them in
    sync; unifying all three is tracked for 0.8.0.
    """
    origin = get_origin(annotation)
    # PEP 604 unions (X | None) have origin types.UnionType, not typing.Union.
    if origin is Union or origin is types.UnionType:
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


def _is_datetime(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, datetime.datetime)


# --- Name-token rules -------------------------------------------------------

# Type families a token rule may apply to. A rule whose comparator cannot
# parse the field's actual type must never fire: DateComparator on a bool or
# NumericComparator on a free-text str silently scores identical values 0.0,
# which is the worst failure mode for a metrics API. bool/Enum/Literal always
# keep their Exact type default (nothing sharpens an exact match).
_STRINGY = "stringy"  # str
_NUMERIC = "numeric"  # int, float
_TEMPORAL = "temporal"  # date, datetime


@dataclass(frozen=True)
class _TokenRule:
    tokens: frozenset
    comparator: str
    config: Dict[str, Any]
    threshold: float
    weight_hint: float
    # Which type families this rule is allowed to refine. A field whose type
    # is outside the set keeps its type default (conflict noted in provenance).
    applies_to: frozenset


# Rules are checked in order; the first whose token appears as a whole word in
# the field name AND whose type family matches wins. Ambiguous tokens
# (type/state/number/key/level/num/...) are intentionally absent so they stay
# at the safe type default.
_NAME_TOKEN_RULES: List[_TokenRule] = [
    # Phone and postal rules precede the identifier rule deliberately.
    # `_match_name_token` returns the first matching rule, and the identifier
    # token set contains "code" -- so "zip_code" and "postal_code" (tokens
    # {zip, code} and {postal, code}) would match the identifier rule first and
    # never reach the rule written for them. Scoring was unaffected (both are
    # case-sensitive Exact@1.0) but the weight hint and the provenance trail
    # were wrong, and the postal rule silently never fired for the two names it
    # most obviously targets (#243 review).
    # Phone numbers differ by punctuation and country-code prefix rather than
    # case, so PhoneComparator parses both sides instead of comparing strings.
    # Ahead of the numeric rules deliberately: NumericComparator strips
    # non-digits and would report a reformatted number as 0.0 while claiming a
    # numeric comparison.
    _TokenRule(
        frozenset({"phone"}),
        "PhoneComparator",
        {},
        1.0,
        1.5,
        frozenset({_STRINGY}),
    ),
    # Postal codes stay strictly exact, so a formatting-only difference
    # ("98101-1234" vs "98101 1234") scores 0.0. This is deliberate: postal
    # formats are country-specific in ways a generic normalizer gets wrong --
    # a UK postcode's internal space is significant ("SW1A 1AA"), and Dutch
    # codes mix letters and digits -- so stripping punctuation would be correct
    # for the US and quietly wrong elsewhere.
    #
    # Levenshtein is not a fallback here either: it scores a *different* postal
    # code (98101-1234 vs 98102-1234, 0.9) the same as the same code reformatted
    # (0.9), so no threshold separates them.
    #
    # See docs/docs/Guides/Comparators/postal-codes.md for how to handle this
    # for a specific country.
    _TokenRule(
        frozenset({"zip", "postal", "postcode"}),
        "ExactComparator",
        {"case_sensitive": True},
        1.0,
        1.5,
        frozenset({_STRINGY}),
    ),
    # Identifiers stay case-sensitive on purpose: an ID that differs in case may
    # well be a different ID, which is what #199 is about. Stated here rather
    # than inherited from ExactComparator's default, so reading this table tells
    # you what these rules mean. (The value does not appear in an exported
    # config -- ExactComparator only serializes `case_sensitive` when it is
    # False -- so an exported model still tracks the constructor default.)
    _TokenRule(
        frozenset({"id", "uuid", "guid", "sku", "code", "ref", "isbn", "ssn"}),
        "ExactComparator",
        {"case_sensitive": True},
        1.0,
        3.0,
        frozenset({_STRINGY, _NUMERIC}),
    ),
    # Email and URL are case-insensitive by specification -- a domain is
    # case-insensitive, and extraction routinely varies the casing of both. So
    # these compare case-insensitively but are otherwise exact: `a@b.com` and
    # `a@c.com` must still be a non-match, which a similarity comparator does
    # not give (Levenshtein scores that pair 0.857).
    #
    # A URL path *is* case-sensitive, so this over-normalizes it. Accepted for
    # now: host casing is the drift that actually occurs, and the alternative is
    # scoring every host-case difference 0.0.
    _TokenRule(
        frozenset({"email", "url", "uri"}),
        "ExactComparator",
        {"case_sensitive": False},
        1.0,
        1.5,
        frozenset({_STRINGY}),
    ),
    _TokenRule(
        frozenset(
            {
                "amount",
                "price",
                "total",
                "subtotal",
                "cost",
                "balance",
                "fee",
                "tax",
                "salary",
                "revenue",
            }
        ),
        "NumericComparator",
        {"relative_tolerance": _FLOAT_RELATIVE_TOLERANCE},
        0.95,
        2.5,
        frozenset({_NUMERIC}),
    ),
    _TokenRule(
        frozenset({"quantity", "qty", "count", "units"}),
        "NumericComparator",
        {},
        1.0,
        1.2,
        frozenset({_NUMERIC}),
    ),
    _TokenRule(
        frozenset(
            {
                "date",
                "dob",
                "expiry",
                "expires",
                "created",
                "updated",
                "issued",
                "due",
                "birthdate",
            }
        ),
        "DateComparator",
        {},
        0.95,
        1.5,
        frozenset({_TEMPORAL}),
    ),
    _TokenRule(
        frozenset(
            {"name", "customer", "vendor", "company", "contact", "author", "recipient"}
        ),
        "LevenshteinComparator",
        {},
        0.85,
        1.5,
        frozenset({_STRINGY}),
    ),
    _TokenRule(
        frozenset({"address", "addr", "street", "city", "location"}),
        "FuzzyComparator",
        {"method": "token_sort_ratio"},
        0.8,
        1.5,
        frozenset({_STRINGY}),
    ),
    _TokenRule(
        frozenset(
            {
                "description",
                "desc",
                "summary",
                "notes",
                "note",
                "comment",
                "comments",
                "remarks",
                "memo",
                "body",
                "text",
            }
        ),
        "FuzzyComparator",
        {"method": "token_set_ratio"},
        0.6,
        0.3,
        frozenset({_STRINGY}),
    ),
]

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _field_tokens(field_name: str) -> frozenset:
    """Split a field name into lowercase word tokens (snake/camel-friendly).

    Trailing digits are shed (isbn13 -> isbn) and simple plurals are
    singularized (ids -> id, addresses -> address) so idiomatic field names
    reach their intended rules.
    """
    # Insert boundaries for camelCase and letter->digit runs before lowercasing.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field_name)
    spaced = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", "_", spaced)
    tokens = set(t for t in _TOKEN_SPLIT.split(spaced.lower()) if t)
    expanded = set(tokens)
    for t in tokens:
        if t.endswith("es") and len(t) > 3:
            expanded.add(t[:-2])
        if t.endswith("s") and len(t) > 2:
            expanded.add(t[:-1])
    return frozenset(expanded)


def _type_family(annotation: Any) -> Optional[str]:
    """Type family used to gate token rules; None -> never token-refined."""
    if annotation is bool or _is_enum(annotation) or _is_literal(annotation):
        # Exact by type; no token rule improves on an exact match.
        return None
    if annotation in (int, float, decimal.Decimal):
        return _NUMERIC
    if _is_date(annotation):
        return _TEMPORAL
    if annotation is str:
        return _STRINGY
    return None


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
        provenance.append(
            f"degrade:{comparator_name}->LevenshteinComparator (never auto-selected)"
        )
        return "LevenshteinComparator", {}

    if not registry.is_registered(comparator_name):
        fallback = (
            "ExactComparator"
            if comparator_name == "DateComparator"
            else "LevenshteinComparator"
        )
        provenance.append(f"degrade:{comparator_name}->{fallback} (not registered)")
        return fallback, {}

    try:
        # Probe construction so a missing optional backend degrades now, not at
        # compare time.
        registry.create_instance(comparator_name, config)
    except Exception as exc:  # noqa: BLE001 - any construction failure degrades
        fallback = (
            "ExactComparator"
            if comparator_name == "DateComparator"
            else "LevenshteinComparator"
        )
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

    # 2) Name-token refinement layered on the type default, gated on type
    #    compatibility: a rule whose comparator cannot parse this type keeps
    #    the type default (recorded in provenance) instead of silently
    #    scoring identical values 0.0.
    weight = 1.0
    family = _type_family(annotation)
    rule = _match_name_token(field_name)
    if rule is not None:
        if family is not None and family in rule.applies_to:
            comparator, config, threshold = (
                rule.comparator,
                dict(rule.config),
                rule.threshold,
            )
            provenance.append(
                f"name-token:{field_name} -> {rule.comparator}@{rule.threshold}"
            )
            if weight_hints and rule.weight_hint != 1.0:
                weight = rule.weight_hint
                provenance.append(
                    f"name-token:{field_name} -> weight {rule.weight_hint}"
                )
            # Free-text fields keep partial credit rather than clipping to zero.
            if (
                rule.comparator == "FuzzyComparator"
                and rule.config.get("method") == "token_set_ratio"
            ):
                clip = False
        else:
            provenance.append(
                f"name-token:{field_name} matched {rule.comparator} but "
                f"type {_annotation_label(annotation)} is incompatible; "
                "keeping type default"
            )

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
    if _is_datetime(annotation):
        # Sub-day tolerance so timestamps hours apart do not silently score
        # 1.0 (DateComparator's default floors both sides to the calendar
        # day). One minute absorbs clock-precision noise, nothing more.
        provenance.append("type:datetime -> DateComparator(tolerance=1min)@0.95")
        return "DateComparator", {"tolerance": 1.0 / (24 * 60)}, 0.95, True
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
    if annotation is decimal.Decimal:
        # Numeric, not string: Decimal('100') must equal Decimal('100.00').
        # Exact (no tolerance): Decimal is chosen precisely when precision
        # matters (money).
        provenance.append("type:Decimal -> NumericComparator(exact)@1.0")
        return "NumericComparator", {}, 1.0, True
    if annotation is str:
        provenance.append("type:str -> LevenshteinComparator@0.7")
        return "LevenshteinComparator", {}, 0.7, True

    # Unknown / exotic (dict, tuple, set, Any, multi-arm union). These have no
    # scalar JSON form, so the wire layer canonicalizes them to a deterministic
    # JSON string (sorted keys; sets also sort elements). Comparison over that
    # string is all-or-nothing: edit distance on a JSON blob would award ~0.95
    # to a substantively wrong value, which is indefensible in a metrics API.
    provenance.append(
        f"type:{_annotation_label(annotation)} -> ExactComparator@1.0 "
        "(canonical JSON string)"
    )
    return "ExactComparator", {}, 1.0, True


def _match_name_token(field_name: str) -> Optional[_TokenRule]:
    tokens = _field_tokens(field_name)
    for rule in _NAME_TOKEN_RULES:
        if tokens & rule.tokens:
            return rule
    return None


def _annotation_label(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))
