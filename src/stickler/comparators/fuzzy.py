"""Fuzzy string matching comparator implementation.

This module provides a comparator for fuzzy string matching using the rapidfuzz
library, which implements Levenshtein distance with additional optimizations
for better string matching. RapidFuzz is a high-performance alternative to
thefuzz/fuzzywuzzy with the same API.
"""

from typing import Any, Optional, Dict
from stickler.comparators.base import BaseComparator

# Check if rapidfuzz is available
try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class FuzzyComparator(BaseComparator):
    """Comparator for fuzzy string matching.

    This comparator uses the rapidfuzz library to calculate similarity between
    strings using advanced Levenshtein distance calculations. It provides better
    fuzzy matching than basic Levenshtein for many use cases.

    If rapidfuzz is not available, this will raise an ImportError when instantiated.
    """

    def __init__(
        self, method: str = "ratio", normalize: bool = True, threshold: float = 0.7
    ):
        """Initialize the fuzzy comparator.

        Args:
            method: The fuzzy matching method to use. Options:
                - "ratio": Standard Levenshtein distance ratio
                - "partial_ratio": Partial string matching
                - "token_sort_ratio": Token-based matching with sorting
                - "token_set_ratio": Token-based matching with set operations
            normalize: Whether to normalize input strings before comparison
                      (strip whitespace, lowercase)
            threshold: Similarity threshold (default 0.7)

        Raises:
            ImportError: If rapidfuzz library is not available
        """
        super().__init__(threshold=threshold)

        if not RAPIDFUZZ_AVAILABLE:
            raise ImportError(
                "The rapidfuzz library is required for FuzzyComparator. "
                "Install it with: pip install rapidfuzz"
            )

        self._method = method
        self._normalize = normalize

        # Select the appropriate fuzzy matching function
        self._fuzzy_func = {
            "ratio": fuzz.ratio,
            "partial_ratio": fuzz.partial_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
        }.get(method, fuzz.ratio)

    @property
    def name(self) -> str:
        """Return the name of the comparator."""
        return f"fuzzy_{self._method}"

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return configuration parameters."""
        return {"method": self._method, "normalize": self._normalize}

    def compare(self, value1: Any, value2: Any) -> float:
        """Compare two strings using fuzzy matching.

        Args:
            value1: First string or value
            value2: Second string or value

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Handle None values
        if value1 is None and value2 is None:
            return 1.0
        elif value1 is None or value2 is None:
            return 0.0

        # Convert to strings
        s1 = str(value1)
        s2 = str(value2)

        # Normalize if enabled
        if self._normalize:
            s1 = s1.strip().lower()
            s2 = s2.strip().lower()

        # Calculate fuzzy match score and normalize to 0.0-1.0
        if s1 == "" and s2 == "":
            return 1.0

        # Use the selected fuzzy matching function
        try:
            return self._fuzzy_func(s1, s2) / 100.0
        except Exception:
            # Fall back to basic comparison if fuzzy match fails
            return 1.0 if s1 == s2 else 0.0


# Class alias for backward compatibility with Fuzz() in evaluation metrics
# This makes Fuzz directly available as a class alias to FuzzyComparator
Fuzz = FuzzyComparator
