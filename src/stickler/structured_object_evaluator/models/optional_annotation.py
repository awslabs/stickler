"""Recognising an optional annotation, in every spelling Python allows.

``Optional[T]``, ``Union[T, None]`` and ``T | None`` are the same type, but they
do not look the same to ``get_origin``:

===========================  =====================  ==================
annotation                   ``get_origin(...)``    ``__origin__``
===========================  =====================  ==================
``Optional[T]``              ``typing.Union``       present
``Union[T, None]``           ``typing.Union``       present
``T | None``                 ``types.UnionType``    **absent**
===========================  =====================  ==================

Testing ``get_origin(x) is Union`` therefore silently answers "no" for the PEP
604 spelling, and testing ``hasattr(x, "__origin__")`` does too. Ten sites in
this package each hand-rolled one of those checks and each got ``X | None``
wrong; three consecutive reviews found only some of them. Ask through this
module instead of rebuilding the check.

``requires-python = ">=3.10"`` means ``T | None`` is valid across the whole
supported range, and it is the spelling modern tooling emits.

.. note::
   Two other copies of this logic exist deliberately, and are not imported
   from here: ``stickler.auto.inference.unwrap_optional`` and the check in
   ``stickler.reporting.html.utils.data_extractors``. Both live in packages
   that do not depend on the evaluator models — importing across would pull
   pydantic model building into the report path (see #268). Keep them in sync
   by hand; unifying all three is tracked for 1.0.

.. note::
   On Python 3.14 the two union representations are unified, so the
   ``types.UnionType`` arm of :func:`is_optional_union` becomes redundant
   rather than wrong. It is kept for 3.10-3.13.
"""

import types
from typing import Annotated, Any, Tuple, Union, get_args, get_origin

__all__ = [
    "is_union",
    "union_args",
    "is_optional_union",
    "unwrap_optional",
    "unwrap_annotated",
]

# Both origins a union annotation can report. `types.UnionType` is what the
# `|` operator produces; `typing.Union` is what the subscript form produces.
_UNION_ORIGINS = (Union, types.UnionType)


def unwrap_annotated(annotation: Any) -> Any:
    """Strip ``Annotated[T, ...]`` down to ``T``, leaving anything else alone.

    Pydantic strips ``Annotated`` when it wraps a whole annotation, so
    ``Annotated[List[str], "m"]`` arrives at these readers already unwrapped. It
    does **not** strip it inside a union, so ``Optional[Annotated[List[str],
    "m"]]`` keeps the wrapper on the arm -- and ``Annotated[List[str], "m"] |
    None`` normalises to exactly that. Any reader that destructures a union arm
    has to unwrap it itself or the arm answers for ``Annotated`` rather than for
    the type inside.

    ``get_origin`` reports ``Annotated`` here rather than the wrapped type, so
    an origin test alone reads the wrapper. ``get_args(...)[0]`` is the wrapped
    type and the remaining args are the metadata.

    Applies once, not repeatedly: ``typing`` flattens nested ``Annotated``, so
    ``Annotated[Annotated[T, "a"], "b"]`` is stored as ``Annotated[T, "a", "b"]``.

    Args:
        annotation: Type annotation to unwrap.

    Returns:
        The wrapped type when ``annotation`` is ``Annotated``, otherwise
        ``annotation`` unchanged.
    """
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def is_union(annotation: Any) -> bool:
    """Whether ``annotation`` is a union of any arity, in any spelling.

    Broader than :func:`is_optional_union`: true for ``A | B``, ``A | B | None``
    and ``Optional[T]`` alike. Use this when the question is "does any arm of
    this union look like X", and :func:`is_optional_union` when the question is
    "is there a single inner type to descend into".

    Args:
        annotation: Type annotation to inspect.

    Returns:
        True if the annotation is a union.
    """
    return get_origin(annotation) in _UNION_ORIGINS


def union_args(annotation: Any) -> Tuple[Any, ...]:
    """The non-``None`` arms of a union, in any spelling.

    Returns an empty tuple when ``annotation`` is not a union, so a caller can
    loop unconditionally.

    Args:
        annotation: Type annotation to inspect.

    Returns:
        The union's arms with ``NoneType`` removed, or ``()`` if not a union.
    """
    if not is_union(annotation):
        return ()
    return tuple(arg for arg in get_args(annotation) if arg is not type(None))


def is_optional_union(annotation: Any) -> bool:
    """Whether ``annotation`` is a union of exactly one type with ``None``.

    True for ``Optional[T]``, ``Union[T, None]`` and ``T | None``. False for a
    genuine multi-arm union such as ``A | B | None``, which has no single inner
    type to descend into, and False for a union with no ``None`` arm.

    Args:
        annotation: Type annotation to inspect.

    Returns:
        True if the annotation is an optional wrapper around a single type.
    """
    if get_origin(annotation) not in _UNION_ORIGINS:
        return False
    args = get_args(annotation)
    non_none = [arg for arg in args if arg is not type(None)]
    return len(non_none) == 1 and type(None) in args


def unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    """Strip an optional wrapper, reporting whether one was present.

    Args:
        annotation: Type annotation to unwrap.

    Returns:
        ``(inner_type, True)`` when ``annotation`` is an optional wrapper
        around a single type, otherwise ``(annotation, False)`` unchanged. A
        multi-arm union is returned as-is rather than unwrapped to an arbitrary
        arm.
    """
    if not is_optional_union(annotation):
        return annotation, False
    inner = next(arg for arg in get_args(annotation) if arg is not type(None))
    return inner, True
