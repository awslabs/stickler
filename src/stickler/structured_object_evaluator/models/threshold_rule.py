"""The rule a threshold must meet, and the one error every path raises for it.

A threshold that decides a match, a field's ``threshold`` or a model's
``match_threshold``, is a real number (not ``bool``) from 0.0 to 1.0 inclusive,
so NaN and infinity fail. An ``int`` or ``float`` is kept as given; a
``Decimal``, ``Fraction``, numpy number or 0-d numpy array is converted with
``float()`` (:func:`as_threshold`). The Python path checks it when the class is
defined, and the Stickler-config and JSON Schema importers when they load. All
three raise ``ValueError`` with a message built here, so they word it alike and
differ only in how they quote keys and values (:data:`PYTHON`, :data:`CONFIG`,
:data:`SCHEMA`).

The ``match_threshold`` passed to ``evaluate()`` or ``eval_for()`` is outside
this rule: for a plain pydantic model ``stickler.auto.builder`` keeps its own
check, and for a ``StructuredModel`` it is not checked.

Only a value that becomes a threshold is checked. A comparator's constructor is
not: a custom comparator may use ``threshold`` for something else, and the
evaluator reads it only when a field adopts it.
"""

import abc
import inspect
import json
import math
import numbers
import os
import sys
import typing
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import pydantic

from .threshold_helper import THRESHOLD_DOCS_URL

#: ``src/stickler/``, with a trailing separator so ``stickler_ext/`` does not match.
STICKLER_DIR = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + os.sep
)

#: Frames a location or a warning skips to reach the author's code: stickler,
#: pydantic, and the class machinery a ``class`` statement runs through.
_MACHINERY = (
    STICKLER_DIR,
    os.path.dirname(os.path.abspath(pydantic.__file__)) + os.sep,
    os.path.abspath(getattr(abc, "__file__", "<frozen abc>")),
    os.path.abspath(typing.__file__),
    "<frozen ",
)

_RULE = "A threshold is the minimum similarity score for a match, from 0.0 to 1.0."
_MATCH_RULE = (
    "A match threshold is the minimum overall score for two objects to match, "
    "from 0.0 to 1.0."
)
_NAN_SOURCE = "A NaN usually comes from an empty cell in a DataFrame or CSV"


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
    a ``Decimal``, a ``Fraction``, a numpy scalar or a 0-d numpy array, means
    the same number, so it is converted with ``float()``, which every reader of
    a threshold handles. ``float()`` and not a round trip through ``str``:
    ``np.float32(0.9)`` is 0.8999999761581421, and that is the number it
    compared as before, so verdicts do not move. ``bool``, numpy booleans
    included, is refused: ``True`` is not a similarity.
    """
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

    Templates take the rendered value as ``{}``. The ``list_*`` templates also
    take the element class as ``{element}`` (Python only).
    """

    as_json: bool
    threshold: str  # the field's own threshold
    threshold_key: str  # the same, with no value
    on_field: str  # where that key goes, after a suggestion
    unset: str  # opens the sentence for a field that adopts its comparator's
    leave_unset: str  # how to drop the field's threshold
    clip: str  # clip_under_threshold set to true
    weight: str
    match_threshold: str
    leave_match_unset: str
    nested: str  # what this path calls a nested model
    list_shape: str  # a list of models, as this path names it
    list_pairing: str  # what pairs its items
    list_remove: str  # what to remove from it
    list_add: str  # where the pairing threshold goes, with a value
    list_add_key: str  # the same, with no value
    unnamed: str  # how to name an unnamed model


PYTHON = Spelling(
    as_json=False,
    threshold="ComparableField(threshold={})",
    threshold_key="ComparableField(threshold=...)",
    on_field="",
    unset="ComparableField sets no threshold, so the field adopts",
    leave_unset="leave threshold unset",
    clip="ComparableField(clip_under_threshold=True)",
    weight="ComparableField(weight={})",
    match_threshold="match_threshold = {}",
    leave_match_unset="remove match_threshold",
    nested="a nested model",
    list_shape="a List[{element}] field",
    list_pairing="{element}.match_threshold",
    list_remove="threshold={}",
    list_add="add match_threshold = {} to the body of class {element}",
    list_add_key="add match_threshold to the body of class {element}",
    unnamed="",
)

CONFIG = Spelling(
    as_json=True,
    threshold='"threshold": {}',
    threshold_key='"threshold"',
    on_field=" on the field",
    unset='the field sets no "threshold", so it adopts',
    leave_unset='remove "threshold"',
    clip='"clip_under_threshold": true',
    weight='"weight": {}',
    match_threshold='"match_threshold": {}',
    leave_match_unset='remove "match_threshold"',
    nested="a nested model",
    list_shape="a list_structured_model field",
    list_pairing='"match_threshold"',
    list_remove='"threshold"',
    list_add='write "match_threshold": {} on the field',
    list_add_key='write "match_threshold" on the field',
    unnamed=" (set model_name to name it)",
)

SCHEMA = Spelling(
    as_json=True,
    threshold='"x-aws-stickler-threshold": {}',
    threshold_key='"x-aws-stickler-threshold"',
    on_field=" on the property",
    unset='the field sets no "x-aws-stickler-threshold", so it adopts',
    leave_unset='remove "x-aws-stickler-threshold"',
    clip='"x-aws-stickler-clip-under-threshold": true',
    weight='"x-aws-stickler-weight": {}',
    match_threshold='"x-aws-stickler-match-threshold": {}',
    leave_match_unset='remove "x-aws-stickler-match-threshold"',
    nested="an object with properties",
    list_shape="an array of objects",
    list_pairing='"x-aws-stickler-match-threshold" in its "items"',
    list_remove='"x-aws-stickler-threshold"',
    list_add='write "x-aws-stickler-match-threshold": {} in its "items"',
    list_add_key='write "x-aws-stickler-match-threshold" in its "items"',
    unnamed=" (set title or x-aws-stickler-model-name to name it)",
)


@dataclass(frozen=True)
class ComparatorSource:
    """Where a threshold the field adopts from its comparator was written.

    ``written`` quotes it as the author did, with ``{}`` for the value, or is
    ``None`` when that is not known. ``name`` is the comparator as the author
    named it. ``carrier`` is what a nested model ignores but for its threshold,
    and ``remove`` what to delete there to move the threshold to the field:
    ``"it"`` for the carrier itself. ``takes_threshold`` says the author can
    pass the comparator a threshold, though ``written`` is ``None`` because its
    constructor may also set one itself. ``defined_at`` is the ``file:line``
    of the comparator's class, when it is not stickler's own.
    """

    written: Optional[str]
    name: str
    carrier: str
    remove: str = "it"
    takes_threshold: bool = False
    defined_at: Optional[str] = None


def python_comparator_source(comparator: Any) -> ComparatorSource:
    """``LevenshteinComparator(threshold={})`` for a comparator known to pass it through.

    That is one whose constructor is stickler's own: each passes ``threshold``
    to ``BaseComparator`` unchanged. Another may set the threshold itself, with
    or without a ``threshold`` parameter, so quoting ``Strict(threshold=85)``
    could name a call the author never wrote; the message says the
    comparator's threshold is 85 instead, and where the class is defined.
    """
    cls = type(comparator)
    name = cls.__name__
    try:
        takes_it = "threshold" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        takes_it = False
    owner = next((k for k in cls.__mro__ if "__init__" in k.__dict__), object)
    own = owner.__module__.split(".")[0] == "stickler"
    return ComparatorSource(
        written=f"{name}(threshold={{}})" if takes_it and own else None,
        name=name,
        carrier=name,
        takes_threshold=takes_it,
        defined_at=None if own else _class_location(owner),
    )


def _class_location(cls: type) -> Optional[str]:
    """Where ``cls`` is defined, as :func:`location` gives it, or ``None``."""
    try:
        filename = inspect.getsourcefile(cls)
        line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        return None
    return location(filename, line) if filename else None


def location(filename: str, line: int) -> Optional[str]:
    """``models/invoice.py:14``, relative to the working directory, for a message.

    The full path when the file is outside it. A notebook cell reads ``line 4
    of the notebook cell``, since its file name means nothing to the author.
    ``None`` for generated code (``<string>``, which is how pydantic evaluates
    a string annotation).
    """
    if filename.startswith("<ipython-input-"):
        return f"line {line} of the notebook cell"
    if filename.startswith("<"):
        return None
    if os.path.basename(os.path.dirname(filename)).startswith("ipykernel_"):
        return f"line {line} of the notebook cell"
    try:
        relative = os.path.relpath(filename)
    except ValueError:  # another drive, on Windows
        relative = filename
    shown = filename if relative.startswith(os.pardir) else relative
    return f"{shown}:{line}"


def caller_location() -> Optional[str]:
    """Where the caller of the caller was called from, if that is the author's code.

    Called by ``ComparableField``, so the Python path's error can point at the
    field line rather than the ``class`` line a traceback ends on. ``None``
    when that caller is stickler itself (an importer building a field).
    """
    frame = sys._getframe(2)
    if frame.f_code.co_filename.startswith(STICKLER_DIR):
        return None
    return location(frame.f_code.co_filename, frame.f_lineno)


def outside_location() -> Optional[str]:
    """Where the first frame outside stickler, pydantic and class machinery is.

    For a ``class`` statement that is its line, so a ``match_threshold`` error
    can point at the class.
    """
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_MACHINERY):
        frame = frame.f_back
    if frame is None:
        return None
    return location(frame.f_code.co_filename, frame.f_lineno)


def stacklevel_outside(*, through_warn_once: bool = False) -> int:
    """The ``stacklevel`` that attributes a warning to the author's code.

    Call it from the function that calls ``warnings.warn``, or ``warn_once``
    (pass ``through_warn_once``, which adds its own frame). A fixed count
    landed in ``<frozen abc>`` for a warning raised while a class is defined.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_MACHINERY):
        frame = frame.f_back
        level += 1
    return level + 1 if through_warn_once else level


def threshold_error(
    value: Any,
    *,
    spelling: Spelling,
    field: str,
    model: Optional[str],
    comparator: Optional[ComparatorSource] = None,
    nested: bool = False,
    nested_model: Optional[str] = None,
    list_element: Optional[str] = None,
    list_unresolved: bool = False,
    defined_at: Optional[str] = None,
    set_at: Optional[str] = None,
) -> ValueError:
    """The error for a field threshold ``value`` that breaks the rule.

    ``comparator`` is given when the field states no threshold and adopts the
    comparator's. ``nested`` when the field is a nested model, whose comparator
    is never called, and ``nested_model`` names its class. ``list_element``
    (the element class, or ``""``) when the field is a list of models, which
    matches its items on ``match_threshold`` instead, and ``list_unresolved``
    when it is a list of a class not defined yet, which may or may not be a
    model. ``defined_at`` is the JSON pointer of a property reached through a
    ``$ref``, and ``set_at`` the location of a ``ComparableField`` call.
    """
    if list_element is not None and comparator is None and not list_unresolved:
        return _list_error(
            value,
            spelling=spelling,
            field=field,
            model=model,
            element=list_element,
            valid=False,
            defined_at=defined_at,
            set_at=set_at,
        )
    found = _diagnose(value, spelling, things="values")
    shown = _render(value, spelling)
    if comparator is None:
        parts = [f"{spelling.threshold.format(shown)} {found.verb}.", _RULE]
    else:
        parts = [f"{_adopts(spelling, comparator, shown)}.", _RULE]
        if found.verb != _OUT_OF_RANGE:
            parts.append(f"{shown} {found.verb}.")
    if found.detail:
        parts.append(found.detail)
    if list_element is not None and list_unresolved:
        element = list_element or _ELEMENT
        if comparator is None:
            parts.append(_unresolved_list_fix(found, spelling, element, shown))
        else:
            parts.append(
                _unresolved_list_comparator_fix(found, spelling, element, comparator)
            )
    elif comparator is not None and nested:
        parts.append(_nested_fix(found, spelling, comparator, nested_model))
    elif comparator is not None:
        parts.append(_comparator_fix(found, spelling, comparator, shown))
    else:
        parts.append(_own_fix(found, spelling))
    parts.append(_locations(set_at, comparator))
    return _error(_head("Invalid", field, model, spelling, defined_at) + _join(parts))


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

    It would decide nothing: the items are matched up on the element class's
    ``match_threshold``. So the fix moves the number there, except 0.0, which as
    a ``match_threshold`` makes every paired object a true positive.
    """
    return _list_error(
        value,
        spelling=spelling,
        field=field,
        model=model,
        element=element,
        valid=True,
        set_at=set_at,
    )


def match_threshold_error(
    value: Any,
    *,
    spelling: Spelling,
    model: Optional[str],
    field: Optional[str] = None,
    set_at: Optional[str] = None,
) -> ValueError:
    """The error for a ``match_threshold`` that breaks the rule.

    ``field`` names the nested-model field it was set on, ``None`` for the model
    itself. ``set_at`` is the location of the class, on the Python path.
    """
    found = _diagnose(value, spelling, things="objects")
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
        _locations(set_at, None),
    ]
    return _error(f"Invalid match threshold for {where}: " + _join(parts))


_OUT_OF_RANGE = "is out of range"
_BOOLEAN = "is a boolean, not a number"
_NOT_A_NUMBER = "is not a number"
#: What a list-of-models fix calls an element class it cannot name.
_ELEMENT = "the element class"


@dataclass(frozen=True)
class _Diagnosis:
    """What is wrong with a value, and the number most likely meant.

    ``verb`` completes "<value> ...". ``reason`` says why ``suggestion`` is
    likely: ``percent`` (85 for 0.85), ``exact`` (just above 1), ``true``,
    ``false``, ``nan``, ``unquote`` (a numeric string), ``percent_text``
    ("85%"), ``null``. ``subject`` is the number the reason quotes.
    """

    verb: str
    detail: Optional[str] = None
    suggestion: Optional[str] = None
    reason: Optional[str] = None
    subject: str = ""


def _diagnose(value: Any, spelling: Spelling, *, things: str) -> _Diagnosis:
    if _is_boolean(value):
        # `true` loaded as 1 on the importers, so 1.0 keeps what it did.
        if value:
            return _Diagnosis(_BOOLEAN, suggestion="1.0", reason="true")
        return _Diagnosis(_BOOLEAN, reason="false")
    if value is None:
        return _Diagnosis(_NOT_A_NUMBER, reason="null")
    if isinstance(value, str):
        return _diagnose_text(value)
    if type(value) in (int, float):
        return _diagnose_number(value, spelling, things=things)
    number = _as_float(value)
    if number is not None:
        return _diagnose_number(
            number, spelling, things=things, text=_number_text(value)
        )
    return _Diagnosis(f"is {_a_kind(value, spelling)}, not a number")


def _diagnose_number(
    value: float, spelling: Spelling, *, things: str, text: Optional[str] = None
) -> _Diagnosis:
    text = text or _plain(value, spelling)
    if not isinstance(value, float):
        # An int too large for a float would overflow `math.isnan`.
        number = _float_or_none(value)
        value = math.inf if number is None else number
    if math.isnan(value):
        return _Diagnosis(
            _NOT_A_NUMBER,
            f"NaN fails every comparison, so even identical {things} would not match.",
            reason="nan",
        )
    if value < 0:
        detail = (
            f"{text} is below 0.0, so even completely different {things} would match."
        )
    else:
        detail = f"{text} is above 1.0, so even identical {things} would not match."
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
    key = spelling.match_threshold if match else spelling.threshold
    perfect = "a perfect overall score" if match else "a perfect similarity score"
    unset = spelling.leave_match_unset if match else spelling.leave_unset
    if reason == "null":
        return "Remove the key to use the default, or write a number such as 0.8."
    if reason == "percent_text":
        return f"For {found.subject}, write {key.format(s)}."
    if reason == "unquote":
        return (
            f"Write {key.format(s)}, without quotes."
            if s
            else "Write a number from 0.0 to 1.0, without quotes."
        )
    if reason == "percent":
        unquote = ", without quotes" if found.verb.startswith("is a string") else ""
        return f"If {found.subject} is a percentage, write {key.format(s)}{unquote}."
    if reason == "exact":
        weight = _float_or_none(found.subject)
        fix = f"Write {key.format('1.0')} to require {perfect}"
        if match or weight is None or round(weight, 2) != weight:
            return f"{fix}."
        return (
            f"{fix}; if you meant the field's weight, write "
            f"{spelling.weight.format(found.subject)}."
        )
    if reason == "true":
        fix = f"Write {key.format('1.0')} to require {perfect}."
        if match:
            return fix
        return (
            f"{fix} If you meant to zero scores below the threshold, that is "
            f"{spelling.clip}."
        )
    if reason == "false":
        every = "every pair of objects" if match else "every pair"
        return (
            f"{_sentence(unset)} to use the default, or write "
            f"{key.format('0.0')} to match {every}."
        )
    if reason == "nan":
        return f"{_NAN_SOURCE}: fill it in, or {unset} to use the default."
    return "Write a number from 0.0 to 1.0 instead."


def _prefix(found: _Diagnosis) -> str:
    """How a fix built on ``found.suggestion`` opens: why that number."""
    if found.reason == "percent":
        return f"If {found.subject} is a percentage, "
    if found.reason in ("exact", "true"):
        return "To require a perfect similarity score, "
    if found.reason == "percent_text":
        return f"For {found.subject}, "
    return ""


def _suffix(found: _Diagnosis) -> str:
    """:func:`_prefix` as it reads at the end of a sentence."""
    if found.reason == "percent":
        return f" if {found.subject} is a percentage"
    if found.reason in ("exact", "true"):
        return " to require a perfect similarity score"
    if found.reason == "percent_text":
        return f" for {found.subject}"
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


def _adopts(spelling: Spelling, comparator: ComparatorSource, shown: str) -> str:
    """``ComparableField sets no threshold, so the field adopts its comparator's: ...``"""
    if comparator.written:
        return f"{spelling.unset} its comparator's: {comparator.written.format(shown)}"
    if comparator.takes_threshold:
        return f"{spelling.unset} its comparator's, which for {comparator.name} is {shown}"
    return f"{spelling.unset} the threshold its comparator {comparator.name} sets, {shown}"


_WINS = "which wins over the comparator's"


def _comparator_options(
    found: _Diagnosis, spelling: Spelling, comparator: ComparatorSource
) -> str:
    """``write X(threshold=0.85), or ComparableField(threshold=0.85), which wins ...``

    Lower-case and without the reason, for a caller to open.
    """
    s = _suggestion(found, comparator)
    field_key = f"{spelling.threshold_key}{spelling.on_field}"
    if comparator.written is None and not comparator.takes_threshold:
        return f"write {field_key} with a number from 0.0 to 1.0, {_WINS}"
    if s is None:
        if comparator.written:
            return (
                f"write {comparator.written.format('...')} with a number from 0.0 "
                f"to 1.0, or {field_key}, {_WINS}"
            )
        return (
            f"pass {comparator.name} a number from 0.0 to 1.0, or write "
            f"{field_key}, {_WINS}"
        )
    field_fix = f"{spelling.threshold.format(s)}{spelling.on_field}"
    if comparator.written:
        return f"write {comparator.written.format(s)}, or {field_fix}, {_WINS}"
    return f"write {field_fix}, {_WINS}, or pass threshold={s} to {comparator.name}"


def _comparator_fix(
    found: _Diagnosis, spelling: Spelling, comparator: ComparatorSource, shown: str
) -> str:
    """Fix the comparator's value, or set the field's own, which wins."""
    s = _suggestion(found, comparator)
    lead = _NAN_SOURCE + ". " if found.reason == "nan" else ""
    fix = lead + _sentence(
        (_prefix(found) if s else "") + _comparator_options(found, spelling, comparator)
    )
    fix += "."
    if comparator.takes_threshold and not comparator.written:
        fix += (
            f" If {comparator.name} sets {shown} itself, pass threshold "
            "to super().__init__() unchanged and put its default in "
            "DEFAULT_THRESHOLD."
        )
    return fix


def _nested_fix(
    found: _Diagnosis,
    spelling: Spelling,
    comparator: ComparatorSource,
    nested_model: Optional[str],
) -> str:
    """A nested model never calls its comparator: the field takes only its threshold."""
    s = _suggestion(found, comparator)
    lead = _NAN_SOURCE + ". " if found.reason == "nan" else ""
    if not spelling.as_json:
        subject = nested_model or "A nested model"
        note = (
            f"{subject} is scored field by field, so {comparator.name} is never "
            "called; the field takes only its threshold."
        )
        if s is None:
            return (
                f"{note} {lead}Write {spelling.threshold_key} with a number from 0.0 "
                "to 1.0 and drop the comparator."
            )
        other = (
            f"write {comparator.written.format(s)}"
            if comparator.written
            else f"pass threshold={s} to {comparator.name}"
        )
        return (
            f"{note} {lead}"
            + _sentence(
                f"{_prefix(found)}write {spelling.threshold.format(s)} and drop the "
                f"comparator, or {other}."
            )
        )
    target = comparator.carrier if comparator.remove == "it" else comparator.remove
    note = (
        f"{target} is ignored on {spelling.nested}, which is scored field by "
        "field, except for its threshold."
    )
    if s is None:
        return (
            f"{note} {lead}Delete {target}, and write "
            f"{spelling.threshold_key}{spelling.on_field} with a number from 0.0 "
            "to 1.0."
        )
    return (
        f"{note} {lead}Delete {target}, and write {spelling.threshold.format(s)}"
        f"{spelling.on_field}{_suffix(found)}."
    )


#: The reasons whose suggestion is a number worth moving to ``match_threshold``.
_LIST_REASONS = ("percent", "percent_text", "unquote")


def _list_words(spelling: Spelling, element: str) -> dict:
    """The list templates of ``spelling`` filled in for ``element``.

    An element class that cannot be named reads ``a List[StructuredModel]
    field`` and ``the element class's match_threshold``.
    """
    if spelling.as_json or element not in ("", _ELEMENT):
        name = element or _ELEMENT
        return {
            "shape": spelling.list_shape.format(element=name),
            "pairing": spelling.list_pairing.format(element=name),
            "add": spelling.list_add.replace("{element}", name),
            "add_key": spelling.list_add_key.format(element=name),
        }
    return {
        "shape": "a List[StructuredModel] field",
        "pairing": "the element class's match_threshold",
        "add": "add match_threshold = {} to the element class",
        "add_key": "add match_threshold to the element class",
    }


def _list_error(
    value: Any,
    *,
    spelling: Spelling,
    field: str,
    model: Optional[str],
    element: str,
    valid: bool,
    defined_at: Optional[str] = None,
    set_at: Optional[str] = None,
) -> ValueError:
    """A threshold on a list of models has no effect, whatever its value.

    The items are matched up on the element class's ``match_threshold``, so the
    fix is to remove it, and to set that instead if the author meant to change
    how items match. ``valid`` heads it "Unused" rather than "Invalid".
    """
    words = _list_words(spelling, element)
    shown = _render(value, spelling)
    add = _list_add(value, spelling, words, shown, valid=valid)
    remove = (
        spelling.list_remove.format(shown)
        if "{}" in spelling.list_remove
        else spelling.list_remove
    )
    parts = [
        f"{spelling.threshold.format(shown)} has no effect on {words['shape']}: its "
        f"items are matched up using {words['pairing']}.",
        f"Remove {remove}.",
        f"To change how items match, {add}",
        _locations(set_at, None),
    ]
    head = "Unused" if valid else "Invalid"
    return _error(_head(head, field, model, spelling, defined_at) + _join(parts))


def _list_add(value: Any, spelling: Spelling, words: dict, shown: str, *, valid: bool) -> str:
    """``add match_threshold = 0.85 to the body of class Item.`` for a list fix.

    A valid value moves as it is, except 0.0, which is not echoed back.
    """
    if valid:
        number = as_threshold(value)
        if number == 0:
            return (
                words["add_key"].replace(" match_threshold", " a positive match_threshold", 1)
                + ": 0.0 would classify every paired object as a true positive. "
                f"See {THRESHOLD_DOCS_URL}"
            )
        return words["add"].format(_plain(number, spelling)) + "."
    found = _diagnose(value, spelling, things="values")
    if found.suggestion and found.reason in _LIST_REASONS:
        add = words["add"].format(found.suggestion)
        if found.reason != "unquote":
            add += f" (it must be from 0.0 to 1.0, so {shown} would fail there too)"
        return add + "."
    return f"{words['add_key']}, as a number from 0.0 to 1.0."


def list_comparator_warning(
    value: Any,
    *,
    comparator: ComparatorSource,
    field: str,
    model: Optional[str],
    element: str,
) -> str:
    """The warning for a comparator threshold on a ``List[StructuredModel]`` field.

    Warned, not raised: the comparator is never called there, whatever its
    threshold, and a comparator instance can be shared with fields that do
    use it.
    """
    words = _list_words(PYTHON, element)
    shown = _render(value, PYTHON)
    held = (
        comparator.written.format(shown)
        if comparator.written
        else f"{comparator.name}, whose threshold is {shown},"
    )
    valid = is_valid_threshold(value)
    return (
        f"Field {field!r}{_of_model(model, PYTHON)} is {words['shape']}, so {held} "
        f"is never used: its items are matched up using {words['pairing']}. "
        "Remove the comparator. To change how items match, "
        + _list_add(value, PYTHON, words, shown, valid=valid)
    )


def _unresolved_list_fix(
    found: _Diagnosis, spelling: Spelling, element: str, shown: str
) -> str:
    """A field threshold on a list of a class not defined yet.

    The field takes no threshold if the class is a ``StructuredModel``, and then
    a number moved onto the field would load and decide nothing. Otherwise the
    usual fix applies.
    """
    words = _list_words(spelling, element)
    s = found.suggestion if found.reason in _LIST_REASONS else None
    add = words["add"].format(s) if s else f"{words['add_key']}, as a number from 0.0 to 1.0"
    otherwise = (
        f"write {spelling.threshold.format(found.suggestion)}"
        if found.suggestion
        else "write a number from 0.0 to 1.0 instead"
    )
    return (
        f"{_not_defined_yet(element)} If it is, remove "
        f"{spelling.list_remove.format(shown)} and {add}: a threshold on a list of "
        f"models is ignored. If it is not, {otherwise}."
    )


def _unresolved_list_comparator_fix(
    found: _Diagnosis, spelling: Spelling, element: str, comparator: ComparatorSource
) -> str:
    """An adopted comparator threshold on a list of a class not defined yet.

    A list of models never calls its comparator. Otherwise the field adopts the
    threshold, so it is checked.
    """
    words = _list_words(spelling, element)
    return (
        f"{_not_defined_yet(element)} If it is, remove the comparator: a list of "
        f"models matches its items using {words['pairing']} and never calls it. "
        f"If it is not, {_comparator_options(found, spelling, comparator)}."
    )


def _not_defined_yet(element: str) -> str:
    if element in ("", _ELEMENT):
        return (
            "The list's element class is not defined yet, so stickler cannot tell "
            "whether it is a StructuredModel."
        )
    return (
        f"{element} is not defined yet, so stickler cannot tell whether it is a "
        "StructuredModel."
    )


def _head(
    kind: str,
    field: str,
    model: Optional[str],
    spelling: Spelling,
    defined_at: Optional[str],
) -> str:
    return (
        f"{kind} threshold for field {field!r}"
        + (f" (defined at {defined_at})" if defined_at else "")
        + _of_model(model, spelling)
        + ": "
    )


def _locations(set_at: Optional[str], comparator: Optional[ComparatorSource]) -> str:
    """``(Set at models.py:14.)``, plus where a custom comparator is defined."""
    defined = comparator.defined_at if comparator is not None else None
    if set_at and defined:
        return f"(Set at {set_at}; {comparator.name} is defined at {defined}.)"
    if set_at:
        return f"(Set at {set_at}.)"
    if defined:
        return f"({comparator.name} is defined at {defined}.)"
    return ""


def _of_model(model: Optional[str], spelling: Spelling) -> str:
    if model:
        return f" of model {model!r}"
    return f" of an unnamed model{spelling.unnamed}"


def _join(parts) -> str:
    return " ".join(p for p in parts if p)


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _is_boolean(value: Any) -> bool:
    """``True`` or ``False``, or a numpy boolean, scalar or 0-d array."""
    if isinstance(value, bool):
        return True
    return (
        getattr(getattr(value, "dtype", None), "kind", None) == "b"
        and getattr(value, "ndim", None) == 0
    )


def _is_numpy(value: Any) -> bool:
    return type(value).__module__.split(".")[0] == "numpy"


def _number_text(value: Any) -> Optional[str]:
    """``85`` for ``Decimal('85')``, ``85.0`` for ``np.float32(85.0)``: its ``str``.

    ``None`` when that is not a plain number, as for ``Fraction(17, 2)``.
    """
    text = str(value)
    number = _float_or_none(text)
    return text if number is not None and math.isfinite(number) else None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(value: Any) -> Optional[float]:
    """A real number other than ``int`` or ``float`` as ``float()`` gives it.

    A ``Decimal``, a ``Fraction``, a numpy scalar, or a 0-d numpy array of
    numbers. ``None`` for anything else, booleans included.
    """
    if _is_boolean(value) or isinstance(value, str):
        return None
    if isinstance(value, (numbers.Real, Decimal)):
        return _float_or_none(value)
    dtype = getattr(value, "dtype", None)
    if getattr(value, "ndim", None) == 0 and getattr(dtype, "kind", None) in (
        "i",
        "u",
        "f",
    ):
        return _float_or_none(value)
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
        if _is_boolean(value):
            return "true" if value else "false"
        if type(value) not in (int, float):
            number = _as_float(value)
            if number is not None:
                # A Decimal, Fraction or numpy number, as the JSON number it means.
                text = _number_text(value)
                return text if text is not None else _plain(number, spelling)
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return repr(value)
    return repr(value)


def _a_kind(value: Any, spelling: Spelling) -> str:
    """``a string``, ``an array``, ``a Decimal``, ``a numpy.ndarray``."""
    if spelling.as_json and isinstance(value, (list, tuple)):
        return "an array"
    if spelling.as_json and isinstance(value, dict):
        return "an object"
    name = type(value).__name__
    if _is_numpy(value):
        name = f"numpy.{name}"
    article = "an" if name[:1].lower() in "aeiou" else "a"
    return f"{article} {name}"
