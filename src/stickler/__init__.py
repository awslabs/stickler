"""
stickler: Structured object comparison and evaluation library.
This library provides tools for comparing complex structured objects
with configurable comparison strategies and detailed evaluation metrics.
"""

# Always-available comparators (core deps)
from .comparators.base import BaseComparator
from .comparators.exact import ExactComparator
from .comparators.fuzzy import FuzzyComparator
from .comparators.levenshtein import LevenshteinComparator
from .comparators.llm import STRANDS_AVAILABLE as _HAS_LLM
from .comparators.numeric import NumericComparator
from .comparators.semantic import SemanticComparator
from .comparators.structured import StructuredModelComparator
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

# Optional: LLM comparator (requires strands-agents)
if _HAS_LLM:
    from .comparators.llm import LLMComparator  # noqa: F401

# Optional: BERT comparator (import fails without evaluate package)
try:
    from .comparators.bert import BERTComparator  # noqa: F401

    _HAS_BERT = True
except ModuleNotFoundError:
    _HAS_BERT = False

__version__ = "0.4.0"
__all__ = [
    # Models and evaluation
    "StructuredModel",
    "ComparableField",
    "NonMatchField",
    "NonMatchType",
    "compare_structured_models",
    "anls_score",
    "compare_json",
    "aggregate_from_comparisons",
    # Comparators (always available)
    "BaseComparator",
    "ExactComparator",
    "FuzzyComparator",
    "LevenshteinComparator",
    "NumericComparator",
    "SemanticComparator",
    "StructuredModelComparator",
]

# Conditionally add optional comparators to __all__
if _HAS_LLM:
    __all__.append("LLMComparator")

if _HAS_BERT:
    __all__.append("BERTComparator")
