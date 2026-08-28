"""Deterministic JSON canonicalization for values with no scalar form.

A ``dict`` (and a ``set``, whose iteration order is not stable across
processes) has no comparator: ``LevenshteinComparator`` raises for a dict, and
``str(dict)`` makes key order significant. Sorted-key JSON removes both
problems.

Shared by :mod:`stickler.auto`, which applies these as ``BeforeValidator`` on
shadow fields whose source type has no scalar JSON form, and by
:class:`~stickler.algorithms.hungarian.HungarianMatcher`, which falls back to
this form when a comparator refuses a list item. Both paths must produce the
same string or a dict field would score differently by hand than generated.
"""

import json
from typing import Any

from pydantic_core import to_jsonable_python


def canonicalize_json(value: Any) -> Any:
    """Normalize a value to a deterministic JSON string (None passes through).

    ``to_jsonable_python`` first converts native Python values (sets, enum
    members, Decimal, UUID, date dict-keys, ...) to their pydantic JSON
    representation, so plain ``model_dump()`` and ``model_dump(mode="json")``
    canonicalize identically; key order and container spelling never affect
    scores. ``fallback=str`` covers types pydantic does not know at all --
    NumPy scalars and arrays, and arbitrary objects reachable through
    ``Dict[str, Any]`` -- which would otherwise raise
    ``PydanticSerializationError``.
    """
    if value is None:
        return value
    value = to_jsonable_python(value, fallback=str)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return json.dumps(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def jsonable_preserving_tuples(value: Any) -> Any:
    """JSON-normalize a value but keep tuples as tuples.

    ``to_jsonable_python`` turns a tuple into a list, which is exactly wrong for
    an ANLS* ground truth: a tuple there means "any one of these is correct"
    (1-of-n alternatives), and a list means "all of these, in any order". Passing
    a ground truth through the plain conversion silently reinterpreted the first
    as the second, so ``anls_score`` and ``ANLSStarComparator`` disagreed by 1.0
    versus 0.0 on the same input.

    Everything else is normalized as usual, so arbitrary objects reachable
    through ``Dict[str, Any]`` still become comparable rather than raising.
    """
    if isinstance(value, tuple):
        return tuple(jsonable_preserving_tuples(v) for v in value)
    if isinstance(value, dict):
        return {
            to_jsonable_python(k, fallback=str): jsonable_preserving_tuples(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [jsonable_preserving_tuples(v) for v in value]
    return to_jsonable_python(value, fallback=str)


def jsonable_mapping(value: Any) -> Any:
    """Normalize a mapping to its JSON form while KEEPING it a mapping.

    Same normalization as :func:`canonicalize_json` -- ``to_jsonable_python``
    converts dates, enums, Decimal, UUID and non-str dict keys to their JSON
    representation -- but stops short of ``json.dumps``, so the result is still
    a dict that :class:`~stickler.comparators.anls.ANLSStarComparator` can walk
    structurally.

    Needed because a dict is the one container scored by structure rather than
    as a canonical string. Without this step, ``model_dump()`` and
    ``model_dump(mode="json")`` produce different key types for a
    ``Dict[date, X]`` (native ``date`` versus ISO string), and the two forms of
    the same document would score 0.0 against each other.

    Non-mappings pass through untouched, so a field annotated ``dict`` that
    somehow holds something else is left for the dispatcher to classify.
    """
    if value is None or not isinstance(value, dict):
        return value
    return to_jsonable_python(value, fallback=str)


def canonicalize_json_sorted(value: Any) -> Any:
    """Like :func:`canonicalize_json`, also sorting top-level arrays.

    Used for ``Set``/``FrozenSet`` sources, whose iteration order is not
    deterministic across processes (and whose plain ``model_dump`` form is a
    native set). ``sort_keys`` orders mapping keys only, never array elements,
    so the sort here is what actually makes a set canonical.
    """
    if value is None:
        return value
    value = to_jsonable_python(value, fallback=str)
    if isinstance(value, (list, tuple)):
        value = sorted(value, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    return canonicalize_json(value)
