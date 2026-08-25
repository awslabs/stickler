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
