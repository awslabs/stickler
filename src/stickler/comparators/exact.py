"""Exact string comparison comparator."""

from typing import Any

from stickler.utils.text_normalizers import strip_punctuation_space, lowercase
from stickler.comparators.base import BaseComparator


class ExactComparator(BaseComparator):
    """Comparator that checks for exact string matching.

    This comparator removes whitespace and punctuation before comparison.
    It returns 1.0 for exact matches and 0.0 otherwise.

    Example:
        ```python
        comparator = ExactComparator()

        # Returns 1.0 (exact match after normalization)
        comparator.compare("hello, world!", "hello world")

        # Returns 0.0 (different strings)
        comparator.compare("hello", "goodbye")
        ```
    """

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0)
            case_sensitive: Whether comparison is case sensitive (default False)
        """
        super().__init__(threshold=threshold)
        self.case_sensitive = case_sensitive

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two values with exact string matching.

        Args:
            str1: First value
            str2: Second value

        Returns:
            1.0 if the strings match exactly after normalization, 0.0 otherwise
        """
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0

        # Convert to strings if they aren't already
        str1 = str(str1)
        str2 = str(str2)

        # Apply case normalization if needed
        if not self.case_sensitive:
            str1 = lowercase(str1)
            str2 = lowercase(str2)

        # Remove whitespace and punctuation
        normalized1 = strip_punctuation_space(str1)
        normalized2 = strip_punctuation_space(str2)

        # Compare normalized strings
        return 1.0 if normalized1 == normalized2 else 0.0
