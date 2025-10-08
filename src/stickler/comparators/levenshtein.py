"""Levenshtein distance comparator implementation."""

from typing import Any, Optional, Dict
from stickler.comparators.base import BaseComparator


class LevenshteinComparator(BaseComparator):
    """Comparator using Levenshtein distance for string similarity.

    This class implements the Levenshtein distance algorithm for measuring
    the difference between two strings. It calculates a normalized similarity
    score between 0 and 1.
    """

    def __init__(self, normalize: bool = True, threshold: float = 0.7):
        """Initialize the comparator.

        Args:
            normalize: Whether to normalize input strings
                      (strip whitespace, lowercase) before comparison
            threshold: Similarity threshold (default 0.7)
        """
        super().__init__(threshold=threshold)
        self._normalize = normalize

    @property
    def name(self) -> str:
        """Return the name of the comparator."""
        return "levenshtein"

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return configuration parameters."""
        return {"normalize": self._normalize}

    def compare(self, s1: Any, s2: Any) -> float:
        """
        Compare two strings using Levenshtein distance.

        Args:
            s1: First string or value
            s2: Second string or value

        Returns:
            Similarity score between 0.0 and 1.0, with 1.0 indicating identical

        Raises:
            TypeError: If either input is a dictionary, as dictionaries are not suitable
                      for Levenshtein distance comparison and should be handled through
                      structured models instead.
        """
        # Reject dictionaries - they should be broken down into proper StructuredModel subclasses
        if isinstance(s1, dict) or isinstance(s2, dict):
            raise TypeError(
                "Dictionary objects cannot be compared using LevenshteinComparator. "
                "Use a StructuredModel subclass with properly defined fields instead."
            )

        # Convert to strings and handle None values
        s1 = "" if s1 is None else str(s1)
        s2 = "" if s2 is None else str(s2)

        # Normalize strings if enabled
        if self._normalize:
            s1 = " ".join(s1.strip().lower().split())
            s2 = " ".join(s2.strip().lower().split())

        # Handle empty strings
        if not s1 and not s2:
            return 1.0

        # Calculate Levenshtein distance
        dist = self._levenshtein_distance(s1, s2)
        str_length = max(len(s1), len(s2))

        if str_length == 0:
            return 1.0

        # Convert distance to similarity (1.0 - normalized_distance)
        return 1.0 - (float(dist) / float(str_length))

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            The Levenshtein distance as an integer
        """
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                    )
            distances = distances_
        return distances[-1]
