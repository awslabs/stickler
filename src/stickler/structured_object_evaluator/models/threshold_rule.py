"""The rule a threshold must meet, and the one error every path raises for it.

A threshold that decides a match, a field's ``threshold`` or a model's
``match_threshold``, is an ``int`` or ``float`` (not ``bool``) from 0.0 to 1.0
inclusive, so NaN and infinity fail. The Python path checks it when the class
is defined, and the Stickler-config and JSON Schema importers when they load,
never during evaluation. All three raise ``ValueError`` with a message built
here, so they word it alike and differ only in how they quote keys and values
(:data:`PYTHON`, :data:`CONFIG`, :data:`SCHEMA`).

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


def is_valid_threshold(value: Any) -> bool:
    """Whether ``value`` can decide a match: an int or float from 0.0 to 1.0."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 1.0
    )


@dataclass(frozen=True)
class Spelling:
    """How one path quotes the keys and values a threshold message names.

    Templates take the rendered value as ``{}``; ``list_fix`` also takes the
    element class as ``{element}``.
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
    ``None`` when the comparator's own constructor set it. ``name`` is the
    comparator as the author named it. ``carrier`` is the key or object to
    remove from a nested model, where only the threshold is read.
    """

    written: Optional[str]
    name: str
    carrier: str


def python_comparator_source(comparator: Any) -> ComparatorSource:
    """``LevenshteinComparator(threshold={})``, if its constructor takes ``threshold``.

    A constructor without that parameter set the threshold itself, so quoting
    ``MaxDist(threshold=5)`` would name a call the author never wrote.
    """
    name = type(comparator).__name__
    try:
        takes_it = "threshold" in inspect.signature(type(comparator)).parameters
    except (TypeError, ValueError):
        takes_it = False
    return ComparatorSource(
        written=f"{name}(threshold={{}})" if takes_it else None,
        name=name,
        carrier=name,
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
    defined_at: Optional[str] = None,
    set_at: Optional[str] = None,
) -> ValueError:
    """The error for a field threshold ``value`` that breaks the rule.

    ``comparator`` is given when the field states no threshold and adopts the
    comparator's. ``nested`` when the field is a nested model, whose comparator
    is never called. ``list_element`` (the element class, or ``""``) when the
    field is a list of models, which pairs on ``match_threshold`` instead.
    ``defined_at`` is the JSON pointer of a property reached through a
    ``$ref``, and ``set_at`` the ``file:line`` of a ``ComparableField`` call.
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
        uses = (
            comparator.written.format(shown)
            if comparator.written
            else f"the threshold its comparator {comparator.name} sets, {shown}"
        )
        parts = [f"{spelling.unset}, so it uses {uses}.", _RULE]
        if found.verb != _OUT_OF_RANGE:
            parts.append(f"{shown} {found.verb}.")
    if found.detail:
        parts.append(found.detail)
    if list_element is not None and comparator is None:
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


@dataclass(frozen=True)
class _Diagnosis:
    """What is wrong with a value, and the number most likely meant.

    ``verb`` completes "<value> ...". ``reason`` says why ``suggestion`` is
    likely: ``percent`` (85 for 0.85), ``exact`` (just above 1), ``unquote``
    (a numeric string), ``percent_text`` ("85%"), ``float`` (a Decimal or a
    numpy scalar), ``null``. ``subject`` is the number the reason quotes, and
    ``convert`` says the suggestion must be passed as a float.
    """

    verb: str
    detail: Optional[str] = None
    suggestion: Optional[str] = None
    reason: Optional[str] = None
    subject: str = ""
    convert: bool = False


def _diagnose(value: Any, spelling: Spelling, *, pair: str) -> _Diagnosis:
    if isinstance(value, bool):
        return _Diagnosis("is a boolean, not a number")
    if value is None:
        return _Diagnosis("is not a number", reason="null")
    if isinstance(value, str):
        return _diagnose_text(value)
    if isinstance(value, (int, float)):
        return _diagnose_number(value, spelling, pair=pair)
    if isinstance(value, (numbers.Real, Decimal)):
        verb = f"is {_a_kind(value, spelling)}, not an int or float"
        number = _as_float(value)
        if number is None:
            return _Diagnosis(verb)
        if is_valid_threshold(number):
            return _Diagnosis(verb, suggestion=_plain(number, spelling), reason="float")
        inner = _diagnose_number(number, spelling, pair=pair, text=_number_text(value))
        return _Diagnosis(
            verb,
            detail=inner.detail,
            suggestion=inner.suggestion,
            reason=inner.reason,
            subject=inner.subject,
            convert=True,
        )
    return _Diagnosis(f"is {_a_kind(value, spelling)}, not a number")


def _diagnose_number(
    value: float, spelling: Spelling, *, pair: str, text: Optional[str] = None
) -> _Diagnosis:
    text = text or _plain(value, spelling)
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
            convert=False,
        )
    return _Diagnosis(verb, reason="unquote")


def _own_fix(found: _Diagnosis, spelling: Spelling, *, match: bool = False) -> str:
    """The fix when the value was written where the threshold is read."""
    s, reason = found.suggestion, found.reason
    use = "pass the float" if found.convert else "use"
    if reason == "null":
        return "Remove the key to use the default, or write a number such as 0.8."
    if reason == "float":
        return f"Pass a float: {s}."
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
        verb = "write" if unquote else use
        return f"If {found.subject} is a percentage, {verb} {s}{unquote}."
    if reason == "exact":
        weight = _as_float(found.subject)
        if match or weight is None or round(weight, 2) != weight:
            return "Use 1.0 to require an exact match."
        return (
            "Use 1.0 to require an exact match; if you meant the field's weight, "
            f"set {spelling.weight.format(found.subject)}."
        )
    return ""


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
    it, so there is no telling what they meant.
    """
    if found.reason == "null" or comparator.written is None:
        return None
    return found.suggestion


def _comparator_fix(
    found: _Diagnosis, spelling: Spelling, comparator: ComparatorSource
) -> str:
    """Fix the comparator's value, or set the field's own, which takes precedence."""
    s = _suggestion(found, comparator)
    if comparator.written is None:
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
        there = (
            f"write {s} there"
            if spelling.as_json
            else f"write {comparator.written.format(s)}"
        )
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
    if s is None:
        return (
            f"{note} Remove it and set {spelling.threshold_key}{spelling.on_field} "
            "to a number from 0.0 to 1.0."
        )
    return (
        note
        + " "
        + _sentence(
            f"{_prefix(found)}remove it and set "
            f"{spelling.threshold.format(s)}{spelling.on_field}."
        )
    )


def _list_fix(found: _Diagnosis, spelling: Spelling, element: str) -> str:
    """A list of models pairs on ``match_threshold``: point there."""
    if found.suggestion and found.reason in (
        "percent",
        "percent_text",
        "unquote",
        "float",
    ):
        where = spelling.list_fix.format(found.suggestion, element=element)
        fix = _sentence(f"{_prefix(found)}set {where} instead.")
    else:
        fix = f"Set {spelling.list_key.format(element=element)} instead."
    return f"{spelling.list_note} {fix}"


def _of_model(model: Optional[str], spelling: Spelling) -> str:
    if model:
        return f" of model {model!r}"
    return f" of an unnamed model{spelling.unnamed}"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


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
