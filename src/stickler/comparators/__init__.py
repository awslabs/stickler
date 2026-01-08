"""Common comparators for key information evaluation.

This package contains comparators that are shared between the traditional
and ANLS Star evaluation systems. These comparators implement a unified
interface that works with both systems.
"""

from stickler.comparators.base import BaseComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.llm import LLMComparator
from stickler.comparators.numeric import NumericComparator, NumericExactC
from stickler.comparators.semantic import SemanticComparator
from stickler.comparators.structured import StructuredModelComparator
from stickler.comparators.utils import generate_bedrock_embedding

# Import LLMComparator if strands-agents is available
try:
    from stickler.comparators.llm import LLMComparator  # noqa: F401

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import BERTComparator if evaluate is available
try:
    from stickler.comparators.bert import BERTComparator  # noqa: F401

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Import FuzzyComparator and Fuzz alias only if rapidfuzz is available
try:
    from stickler.comparators.fuzzy import (  # noqa: F401
        RAPIDFUZZ_AVAILABLE,
        Fuzz,
        FuzzyComparator,
    )
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


__all__ = [
    "BaseComparator",
    "LevenshteinComparator",
    "NumericComparator",
    "NumericExactC",
    "ExactComparator",
    "StructuredModelComparator",
    "SemanticComparator",
    "generate_bedrock_embedding",
]

# Add LLMComparator to __all__ if available
if LLM_AVAILABLE:
    __all__.append("LLMComparator")

# Add BERTComparator to __all__ if available
if BERT_AVAILABLE:
    __all__.append("BERTComparator")

# Add FuzzyComparator and Fuzz to __all__ if available
if RAPIDFUZZ_AVAILABLE:
    __all__.append("FuzzyComparator")
    __all__.append("Fuzz")
