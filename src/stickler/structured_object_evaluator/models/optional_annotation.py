"""Recognising an optional annotation, in every spelling Python allows.

Re-exports the shared implementation from
:mod:`stickler.utils.optional_annotation` for callers inside
``stickler.structured_object_evaluator``.
"""

from ...utils.optional_annotation import (
    is_optional_union,
    is_union,
    union_args,
    unwrap_annotated,
    unwrap_optional,
)

__all__ = [
    "is_union",
    "union_args",
    "is_optional_union",
    "unwrap_optional",
    "unwrap_annotated",
]
