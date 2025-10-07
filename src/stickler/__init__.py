"""
stickler: Structured object comparison and evaluation library.

This library provides tools for comparing complex structured objects
with configurable comparison strategies and detailed evaluation metrics.
"""

from .structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    NonMatchField,
    NonMatchType,
    StructuredModelEvaluator,
    compare_structured_models,
    anls_score,
    compare_json,
)

__version__ = "0.1.0"

__all__ = [
    "StructuredModel",
    "ComparableField",
    "NonMatchField",
    "NonMatchType",
    "StructuredModelEvaluator",
    "compare_structured_models",
    "anls_score",
    "compare_json",
]
