"""The rule a threshold must meet, and the one error every path raises for it.

A threshold that decides a match, a field's ``threshold`` or a model's
``match_threshold``, is a real number (not ``bool``) from 0.0 to 1.0 inclusive,
so NaN and infinity fail. An ``int`` or ``float`` is kept as given; a
``Decimal``, ``Fraction`` or numpy number is converted to ``float``
(:func:`as_threshold`). The Python path checks it when the class is defined,
and the Stickler-config and JSON Schema importers when they load. All three
raise ``ValueError`` with a message built here, so they word it alike and
differ only in how they quote keys and values (:data:`PYTHON`, :data:`CONFIG`,
:data:`SCHEMA`). The ``match_threshold`` passed to ``evaluate()`` or
``eval_for()`` is not checked here.

Only a value that becomes a threshold is checked. A comparator's constructor is
not: a custom comparator may use ``threshold`` for something else, and the
evaluator reads it only when a field adopts it.
"""

import inspect
import json
import math
import numbers
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from .threshold_helper import THRESHOLD_DOCS_URL

#: ``src/stickler/``, with a trailing separator so ``stickler_ext/`` does not match.
STICKLER_DIR = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + os.sep
)

_RULE = "A threshold is the minimum similarity score for a match, from 0.0 to 1.0."
_MATCH_RULE = (
    "A match threshold is the minimum overall score for two objects to match, "
    "from 0.0 to 1.0."
)


#: Set on the ``ValueError`` this module builds; see `is_threshold_error`.
_MARKER = "_stickler_invalid_threshold"


def is_threshold_error(error: BaseException) -> bool:
    """Whether ``error`` is one this module built.

    The config importer's wrappers ("Invalid field configuration: ...") re-raise
    such an error unchanged: its message already names the model and the field,
    and nesting stacked one prefix per level. A marker on a plain
    ``ValueError`` rather than a subclass, so the traceback reads
    ``ValueError: Invalid threshold ...``, as the docs show it.
    """
    return getattr(error, _MARKER, False)


def _error(message: str) -> ValueError:
    error = ValueError(message)
    setattr(error, _MARKER, True)
    return error


def as_threshold(value: Any) -> Optional[float]:
    """``value`` as the threshold it sets, or ``None`` if it breaks the rule.

    An ``int`` or ``float`` is returned as given. Another real number, such as
    a ``Decimal``, a ``Fraction`` or a numpy scalar, means the same number, so
    it is converted to ``float``, which every reader of a threshold and both
    exporters handle. ``bool`` is refused: ``True`` is not a similarity.
    """
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, Decimal)):
        return None
    number = value if type(value) in (int, float) else _as_float(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return number


def is_valid_threshold(value: Any) -> bool:
    """Whether ``value`` can decide a match: a real number from 0.0 to 1.0."""
    return as_threshold(value) is not None


def normalize_threshold(value: Any) -> Any:
    """A valid threshold as :func:`as_threshold` gives it; anything else unchanged.

    For a value that is checked later, by a caller that knows the model and the
    field: an invalid one must reach that check, and its message, as written.
    """
    number = as_threshold(value)
    return value if number is None else number


@dataclass(frozen=True)
class Spelling:
    """How one path quotes the keys and values a threshold message names.

    Templates take the rendered value as ``{}``; ``list_fix`` and ``list_key``
    also take the element class as ``{element}``.
    """

    as_json: bool
    threshold: str  # the field's own threshold
    threshold_key: str  # the same, with no value
    on_field: str  # where that key goes, after a suggestion
    unset: str  # the field states no threshold
    weight: str
    match_threshold: str
    nested: str  # what this path calls a nested model
    list_note: str  # why a threshold does nothing on a list of models
    list_fix: str  # where the pairing threshold goes instead
    list_key: str  # the same, with no value
    unnamed: str  # how to name an unnamed model


PYTHON = Spelling(
    as_json=False,
    threshold="ComparableField(threshold={})",
    threshold_key="ComparableField(threshold=...)",
    on_field="",
    unset="the field sets no threshold",
    weight="ComparableField(weight={})",
    match_threshold="match_threshold = {}",
    nested="a nested model",
    list_note=(
        "A List[StructuredModel] field takes no threshold: its items are paired "
        "on the element class's match_threshold."
    ),
    list_fix="match_threshold = {} on {element}",
    list_key="match_threshold on {element}",
    unnamed="",
)

CONFIG = Spelling(
    as_json=True,
    threshold='"threshold": {}',
    threshold_key='"threshold"',
    on_field=" on the field",
    unset='the field sets no "threshold"',
    weight='"weight": {}',
    match_threshold='"match_threshold": {}',
    nested="a nested model",
    list_note=(
        'A list_structured_model field reads no "threshold": its items are '
        'paired on "match_threshold".'
    ),
    list_fix='"match_threshold": {} on the field',
    list_key='"match_threshold" on the field',
    unnamed=" (set model_name to name it)",
)

SCHEMA = Spelling(
    as_json=True,
    threshold='"x-aws-stickler-threshold": {}',
    threshold_key='"x-aws-stickler-threshold"',
    on_field=" on the property",
    unset='the field sets no "x-aws-stickler-threshold"',
    weight='"x-aws-stickler-weight": {}',
    match_threshold='"x-aws-stickler-match-threshold": {}',
    nested="an object with properties",
    list_note=(
        'An array of objects reads no "x-aws-stickler-threshold": its items are '
        'paired on "x-aws-stickler-match-threshold".'
    ),
    list_fix='"x-aws-stickler-match-threshold": {} in its "items"',
    list_key='"x-aws-stickler-match-threshold" in its "items"',
    unnamed=" (set title or x-aws-stickler-model-name to name it)",
)


@dataclass(frozen=True)
class ComparatorSource:
    """Where a threshold the field adopts from its comparator was written.

    ``written`` quotes it as the author did, with ``{}`` for the value, or is
    ``None`` when that is not known. ``name`` is the comparator as the author
    named it. ``carrier`` is what has no effect on a nested model, where only
    the threshold is read, and ``remove`` what to delete there to move the
    threshold to the field: ``"it"`` for the carrier itself. ``takes_threshold``
    says the author can pass the comparator a threshold, though ``written`` is
    ``None`` because its constructor may also set one itself.
    """

    written: Optional[str]
    name: str
    carrier: str
    remove: str = "it"
    takes_threshold: bool = False


def python_comparator_source(comparator: Any) -> ComparatorSource:
    """``LevenshteinComparator(threshold={})`` for a comparator known to pass it through.

    That is one whose constructor is stickler's own: each passes ``threshold``
    to ``BaseComparator`` unchanged. Another may set the threshold itself, with
    or without a ``threshold`` parameter, so quoting ``Strict(threshold=85)``
    could name a call the author never wrote; the message says the
    comparator's threshold is 85 instead.
    """
    cls = type(comparator)
    name = cls.__name__
    try:
        takes_it = "threshold" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        takes_it = False
    owner = next((k for k in cls.__mro__ if "__init__" in k.__dict__), object)
    passes_it_through = takes_it and owner.__module__.split(".")[0] == "stickler"
    return ComparatorSource(
        written=f"{name}(threshold={{}})" if passes_it_through else None,
        name=name,
        carrier=name,
        takes_threshold=takes_it,
    )


def caller_location() -> Optional[str]:
    """``models.py:14`` for the first frame outside stickler, if it is a real file.

    Called by ``ComparableField``, so the Python path's error can point at the
    field line rather than the ``class`` line a traceback ends on. ``None`` when
    the caller is generated code (``<string>``, which is how pydantic evaluates
    a string annotation) or stickler itself (an importer building a field).
    """
    frame = sys._getframe(2)
    filename = frame.f_code.co_filename
    if filename.startswith("<") or filename.startswith(STICKLER_DIR):
        return None
    return f"{os.path.basename(filename)}:{frame.f_lineno}"


def threshold_error(
    value: Any,
    *,
    spelling: Spelling,
    field: str,
    model: Optional[str],
    comparator: Optional[ComparatorSource] = None,
    nested: bool = False,
    list_element: Optional[str] = None,
    list_unresolved: bool = False,
    defined_at: Optional[str] = None,
    set_at: Optional[str] = None,
) -> ValueError:
    """The error for a field threshold ``value`` that breaks the rule.

    ``comparator`` is given when the field states no threshold and adopts the
    comparator's. ``nested`` when the field is a nested model, whose comparator
    is never called. ``list_element`` (the element class, or ``""``) when the
    field is a list of models, which pairs on ``match_threshold`` instead, and
    ``list_unresolved`` when it is a list of a class not defined yet, which may
    or may not be a model. ``defined_at`` is the JSON pointer of a property
    reached through a ``$ref``, and ``set_at`` the ``file:line`` of a
    ``ComparableField`` call.
    """
    found = _diagnose(value, spelling, pair="pair")
    shown = _render(value, spelling)
    head = (
        f"Invalid threshold for field {field!r}"
        + (f" (defined at {defined_at})" if defined_at else "")
        + _of_model(model, spelling)
        + ": "
    )
    if comparator is None:
        parts = [f"{spelling.threshold.format(shown)} {found.verb}.", _RULE]
    else:
        if comparator.written:
            uses = comparator.written.format(shown)
        elif comparator.takes_threshold:
            uses = f"its comparator's threshold, which for {comparator.name} is {shown}"
        else:
            uses = f"the threshold its comparator {comparator.name} sets, {shown}"
        parts = [f"{spelling.unset}, so it uses {uses}.", _RULE]
        if found.verb != _OUT_OF_RANGE:
            parts.append(f"{shown} {found.verb}.")
    if found.detail:
        parts.append(found.detail)
    if list_element is not None and comparator is None and list_unresolved:
        parts.append(_unresolved_list_fix(found, spelling, list_element or _ELEMENT))
    elif list_element is not None and comparator is None:
        parts.append(_list_fix(found, spelling, list_element))
    elif comparator is not None and nested:
        parts.append(_nested_fix(found, spelling, comparator))
    elif comparator is not None:
        parts.append(_comparator_fix(found, spelling, comparator))
    else:
        parts.append(_own_fix(found, spelling))
    if set_at:
        parts.append(f"(Set at {set_at}.)")
    return _error(head + " ".join(p for p in parts if p))


def list_threshold_error(
    value: Any,
    *,
    spelling: Spelling,
    field: str,
    model: Optional[str],
    element: str,
    set_at: Optional[str] = None,
) -> ValueError:
    """The error for a valid threshold written on a list of models.

    It would decide nothing: Hungarian matching pairs the items on the element
    class's ``match_threshold``. So the fix moves the number there, except 0.0,
    which as a ``match_threshold`` makes every paired object a true positive.
    """
    shown = _render(value, spelling)
    if value == 0:
        fix = (
            f"Set a positive {spelling.list_key.format(element=element)} instead: "
            "0.0 would classify every paired object as a true positive. "
            f"See {THRESHOLD_DOCS_URL}"
        )
    else:
        where = spelling.list_fix.format(_plain(value, spelling), element=element)
        fix = f"Set {where} instead."
    parts = [
        f"{spelling.threshold.format(shown)} would have no effect.",
        spelling.list_note,
        fix,
    ]
    if set_at:
        parts.append(f"(Set at {set_at}.)")
    return _error(
        f"Invalid threshold for field {field!r}{_of_model(model, spelling)}: "
        + " ".join(parts)
    )


def match_threshold_error(
    value: Any,
    *,
    spelling: Spelling,
    model: Optional[str],
    field: Optional[str] = None,
) -> ValueError:
    """The error for a ``match_threshold`` that breaks the rule.

    ``field`` names the nested-model field it was set on, ``None`` for the model
    itself.
    """
    found = _diagnose(value, spelling, pair="pair of objects")
    where = (
        f"field {field!r}{_of_model(model, spelling)}"
        if field
        else _of_model(model, spelling)[len(" of ") :]
    )
    parts = [
        f"{spelling.match_threshold.format(_render(value, spelling))} {found.verb}.",
        _MATCH_RULE,
        found.detail,
        _own_fix(found, spelling, match=True),
    ]
    return _error(
        f"Invalid match threshold for {where}: " + " ".join(p for p in parts if p)
    )


_OUT_OF_RANGE = "is out of range"
_BOOLEAN = "is a boolean, not a number"
#: What a list-of-models fix calls an element class it cannot name.
_ELEMENT = "the element class"


@dataclass(frozen=True)
class _Diagnosis:
    """What is wrong with a value, and the number most likely meant.

    ``verb`` completes "<value> ...". ``reason`` says why ``suggestion`` is
    likely: ``percent`` (85 for 0.85), ``exact`` (just above 1, or ``true``),
    ``unquote`` (a numeric string), ``percent_text`` ("85%"), ``null``.
    ``subject`` is the number the reason quotes.
    """

    verb: str
    detail: Optional[str] = None
    suggestion: Optional[str] = None
    reason: Optional[str] = None
    subject: str = ""


def _diagnose(value: Any, spelling: Spelling, *, pair: str) -> _Diagnosis:
    if isinstance(value, bool):
        # `true` loaded as 1 on the importers, so 1.0 keeps what it did.
        if value:
            return _Diagnosis(_BOOLEAN, suggestion="1.0", reason="exact")
        return _Diagnosis(_BOOLEAN)
    if value is None:
        return _Diagnosis("is not a number", reason="null")
    if isinstance(value, str):
        return _diagnose_text(value)
    if isinstance(value, (numbers.Real, Decimal)):
        if type(value) in (int, float):
            return _diagnose_number(value, spelling, pair=pair)
        number = _as_float(value)
        if number is not None:
            return _diagnose_number(
                number, spelling, pair=pair, text=_number_text(value)
            )
    return _Diagnosis(f"is {_a_kind(value, spelling)}, not a number")


def _diagnose_number(
    value: float, spelling: Spelling, *, pair: str, text: Optional[str] = None
) -> _Diagnosis:
    text = text or _plain(value, spelling)
    if not isinstance(value, float):
        # An int too large for a float would overflow `math.isnan`.
        number = _as_float(value)
        value = math.inf if number is None else number
    if math.isnan(value):
        detail = f"NaN fails every comparison, so no {pair} would match."
    elif value < 0:
        detail = f"{text} is below 0.0, so every {pair} would match."
    else:
        detail = f"{text} is above 1.0, so no {pair} would match."
    if math.isfinite(value) and 2 <= value <= 100:
        return _Diagnosis(
            _OUT_OF_RANGE,
            detail,
            _plain(round(value / 100, 12), spelling),
            "percent",
            text,
        )
    if math.isfinite(value) and 1 < value < 2:
        return _Diagnosis(_OUT_OF_RANGE, detail, "1.0", "exact", text)
    return _Diagnosis(_OUT_OF_RANGE, detail)


def _diagnose_text(value: str) -> _Diagnosis:
    verb = "is a string, not a number"
    body = value.strip()
    percent = body.endswith("%")
    try:
        number = float(body[:-1] if percent else body)
    except ValueError:
        return _Diagnosis(verb)
    if not math.isfinite(number):
        return _Diagnosis(verb)
    if percent:
        if 0 <= number <= 100:
            return _Diagnosis(
                verb,
                suggestion=repr(round(number / 100, 12)),
                reason="percent_text",
                subject=body,
            )
        return _Diagnosis(verb)
    if is_valid_threshold(number):
        return _Diagnosis(verb, suggestion=repr(number), reason="unquote")
    if 2 <= number <= 100:
        return _Diagnosis(
            verb,
            suggestion=repr(round(number / 100, 12)),
            reason="percent",
            subject=body,
        )
    return _Diagnosis(verb, reason="unquote")


def _own_fix(found: _Diagnosis, spelling: Spelling, *, match: bool = False) -> str:
    """The fix when the value was written where the threshold is read."""
    s, reason = found.suggestion, found.reason
    if reason == "null":
        return "Remove the key to use the default, or write a number such as 0.8."
    if reason == "percent_text":
        return f"For {found.subject}, use {s}."
    if reason == "unquote":
        return (
            f"Write {s} without quotes."
            if s
            else "Write a number from 0.0 to 1.0 without quotes."
        )
    if reason == "percent":
        unquote = " without quotes" if found.verb.startswith("is a string") else ""
        verb = "write" if unquote else "use"
        return f"If {found.subject} is a percentage, {verb} {s}{unquote}."
    if reason == "exact":
        weight = _as_float(found.subject)
        if match or weight is None or round(weight, 2) != weight:
            return "Use 1.0 to require an exact match."
        return (
            "Use 1.0 to require an exact match; if you meant the field's weight, "
            f"set {spelling.weight.format(found.subject)}."
        )
    return "Use a number from 0.0 to 1.0 instead."


def _prefix(found: _Diagnosis) -> str:
    """How a fix built on ``found.suggestion`` opens: why that number."""
    if found.reason == "percent":
        return f"If {found.subject} is a percentage, "
    if found.reason == "exact":
        return "To require an exact match, "
    if found.reason == "percent_text":
        return f"For {found.subject}, "
    return ""


def _suggestion(found: _Diagnosis, comparator: ComparatorSource) -> Optional[str]:
    """The number to suggest for a comparator's value, if any.

    None for a value the comparator's constructor set: the author did not write
    it, so there is no telling what they meant. A comparator the author can
    pass a threshold to may have been given this one, so it gets the guess.
    """
    if found.reason == "null" or not (
        comparator.written or comparator.takes_threshold
    ):
        return None
    return found.suggestion


def _comparator_fix(
    found: _Diagnosis, spelling: Spelling, comparator: ComparatorSource
) -> str:
    """Fix the comparator's value, or set the field's own, which takes precedence."""
    s = _suggestion(found, comparator)
    if comparator.written is None and not comparator.takes_threshold:
        return (
            f"Set {spelling.threshold_key}{spelling.on_field} to a number from "
            "0.0 to 1.0, which takes precedence."
        )
    if s is None:
        there = (
            "write a number from 0.0 to 1.0 there"
            if spelling.as_json
            else f"pass a number from 0.0 to 1.0 to {comparator.name}"
        )
        override = f"{spelling.threshold_key}{spelling.on_field}"
    else:
        if spelling.as_json:
            there = f"write {s} there"
        elif comparator.written:
            there = f"write {comparator.written.format(s)}"
        else:
            there = f"pass threshold={s} to {comparator.name}"
        override = f"{spelling.threshold.format(s)}{spelling.on_field}"
    return _sentence(
        f"{_prefix(found) if s else ''}{there}, or set {override}, which takes "
        "precedence."
    )


def _nested_fix(
    found: _Diagnosis, spelling: Spelling, comparator: ComparatorSource
) -> str:
    """A nested model never calls its comparator: move the threshold to the field."""
    note = (
        f"{comparator.carrier} has no effect on {spelling.nested} except for its "
        "threshold."
    )
    s = _suggestion(found, comparator)
    remove = comparator.remove
    if s is None:
        return (
            f"{note} Remove {remove} and set "
            f"{spelling.threshold_key}{spelling.on_field} to a number from 0.0 to 1.0."
        )
    return (
        note
        + " "
        + _sentence(
            f"{_prefix(found)}remove {remove} and set "
            f"{spelling.threshold.format(s)}{spelling.on_field}."
        )
    )


#: The reasons whose suggestion is a number worth moving to ``match_threshold``.
_LIST_REASONS = ("percent", "percent_text", "unquote")


def _list_match_threshold(found: _Diagnosis, spelling: Spelling, element: str) -> str:
    """``set match_threshold = 0.85 on Item instead.``, opened by why that number."""
    if found.suggestion and found.reason in _LIST_REASONS:
        where = spelling.list_fix.format(found.suggestion, element=element)
        return f"{_prefix(found)}set {where} instead."
    return f"set {spelling.list_key.format(element=element)} instead."


def _list_fix(found: _Diagnosis, spelling: Spelling, element: str) -> str:
    """A list of models pairs on ``match_threshold``: point there."""
    fix = _sentence(_list_match_threshold(found, spelling, element))
    return f"{spelling.list_note} {fix}"


def _unresolved_list_fix(found: _Diagnosis, spelling: Spelling, element: str) -> str:
    """A list of a class not defined yet: point at ``match_threshold`` if a model.

    The field takes no threshold if the class is a ``StructuredModel``, and then
    a number moved onto the field would load and decide nothing. Otherwise the
    usual fix applies.
    """
    pairing = _list_match_threshold(found, spelling, element)
    otherwise = _own_fix(found, spelling).replace(" instead.", ".")
    return (
        f"If {element} is a StructuredModel, this field takes no threshold: its "
        f"items are paired on {element}'s match_threshold, so "
        f"{_lower_first(pairing)} Otherwise, {_lower_first(otherwise)}"
    )


def _of_model(model: Optional[str], spelling: Spelling) -> str:
    if model:
        return f" of model {model!r}"
    return f" of an unnamed model{spelling.unnamed}"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _number_text(value: Any) -> Optional[str]:
    """``85`` for ``Decimal('85')``, ``85.0`` for ``np.float32(85.0)``: its ``str``.

    ``None`` when that is not a plain number, as for ``Fraction(17, 2)``.
    """
    text = str(value)
    try:
        float(text)
    except ValueError:
        return None
    return text if math.isfinite(float(text)) else None


def _as_float(value: Any) -> Optional[float]:
    """A Decimal, Fraction or numpy scalar as the float it most likely meant.

    Via ``str`` first, so ``np.float32(0.9)`` reads 0.9 and not 0.8999999761.
    """
    for convert in (lambda v: float(str(v)), float):
        try:
            return convert(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _plain(value: float, spelling: Spelling) -> str:
    """A number as a reader writes it: ``85``, ``0.85``, ``inf``, ``NaN``."""
    if isinstance(value, int):
        return str(int(value))
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        sign = "-" if value < 0 else ""
        return sign + ("Infinity" if spelling.as_json else "inf")
    return repr(value)


def _render(value: Any, spelling: Spelling) -> str:
    """``value`` as the author wrote it: JSON on the importers, code in Python."""
    if isinstance(value, float) and not math.isfinite(value):
        if spelling.as_json:
            return _plain(value, spelling)
        text = "nan" if math.isnan(value) else ("-inf" if value < 0 else "inf")
        return f"float({text!r})"
    if spelling.as_json:
        if (
            isinstance(value, (numbers.Real, Decimal))
            and not isinstance(value, bool)
            and type(value) not in (int, float)
        ):
            # A Decimal, Fraction or numpy scalar, as the JSON number it means.
            text = _number_text(value)
            number = _as_float(value)
            if text is not None or number is not None:
                return text if text is not None else _plain(number, spelling)
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return repr(value)
    return repr(value)


def _a_kind(value: Any, spelling: Spelling) -> str:
    """``a string``, ``an array``, ``a Decimal``, ``a numpy.float32``."""
    if spelling.as_json and isinstance(value, (list, tuple)):
        return "an array"
    if spelling.as_json and isinstance(value, dict):
        return "an object"
    cls = type(value)
    name = cls.__name__
    if cls.__module__.split(".")[0] == "numpy":
        name = f"numpy.{name}"
    article = "an" if name[:1].lower() in "aeiou" else "a"
    return f"{article} {name}"
