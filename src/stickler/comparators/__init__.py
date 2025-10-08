"""Common comparators for key information evaluation.

This package contains comparators that are shared between the traditional
and ANLS Star evaluation systems. These comparators implement a unified
interface that works with both systems.
"""

from stickler.comparators.utils import generate_bedrock_embedding
from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator, NumericExactC
from stickler.comparators.exact import ExactComparator
from stickler.comparators.llm import LLMComparator
from stickler.comparators.structured import StructuredModelComparator

# Import BERTComparator if evaluate is available
try:
    from stickler.comparators.bert import BERTComparator

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Import FuzzyComparator and Fuzz alias only if rapidfuzz is available
try:
    from stickler.comparators.fuzzy import FuzzyComparator, Fuzz, RAPIDFUZZ_AVAILABLE
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from stickler.comparators.semantic import SemanticComparator

__all__ = [
    "BaseComparator",
    "LevenshteinComparator",
    "NumericComparator",
    "NumericExactC",
    "ExactComparator",
    "LLMComparator",
    "StructuredModelComparator",
    "generate_bedrock_embedding",
]

# Add BERTComparator to __all__ if available
if BERT_AVAILABLE:
    __all__.append("BERTComparator")

# Add FuzzyComparator and Fuzz to __all__ if available
if RAPIDFUZZ_AVAILABLE:
    __all__.append("FuzzyComparator")
    __all__.append("Fuzz")

# Add SemanticComparator to __all__
__all__.append("SemanticComparator")
