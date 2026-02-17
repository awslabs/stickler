"""
stickler: Structured object comparison and evaluation library.

This library provides tools for comparing complex structured objects
with configurable comparison strategies and detailed evaluation metrics.
"""

from .structured_object_evaluator import (
    ComparableField,
    NonMatchField,
    NonMatchType,
    StructuredModel,
    aggregate_from_comparisons,
    anls_score,
    compare_json,
    compare_structured_models,
)

__version__ = "0.1.5"

__all__ = [
    "StructuredModel",
    "ComparableField",
    "NonMatchField",
    "NonMatchType",
    "compare_structured_models",
    "anls_score",
    "compare_json",
    "aggregate_from_comparisons",
]
