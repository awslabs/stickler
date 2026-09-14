"""Levenshtein distance comparator implementation."""

from typing import Any, Dict, Optional

from stickler.comparators.base import BaseComparator


class LevenshteinComparator(BaseComparator):
    """Comparator using Levenshtein distance for string similarity.

    This class implements the Levenshtein distance algorithm for measuring
    the difference between two strings. It calculates a normalized similarity
    score between 0 and 1.
    """

    DEFAULT_THRESHOLD = 0.7

    def __init__(self, normalize: bool = True, threshold: Optional[float] = None):
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

    def _compare(self, s1: Any, s2: Any) -> float:
        """
        Compare two strings using Levenshtein distance.

        Args:
            s1: First string or value, guaranteed not None
            s2: Second string or value, guaranteed not None

        Returns:
            Similarity score between 0.0 and 1.0, with 1.0 indicating identical

        Raises:
            TypeError: If either input is a dictionary, as dictionaries are not suitable
                      for Levenshtein distance comparison and should be handled through
                      structured models instead. Both inputs are guaranteed non-None
                      here (BaseComparator.compare handles None before delegating), so
                      this only ever fires for an actual dict value.
        """
        # Reject dictionaries. Edit distance over str(dict) makes key order
        # significant, so two mappings with identical content can score well
        # below 1.0. Two remedies exist and the message names both, since which
        # one applies depends on whether the keys are known up front.
        if isinstance(s1, dict) or isinstance(s2, dict):
            raise TypeError(
                "Dictionary objects cannot be compared using LevenshteinComparator. "
                "If you know the keys, declare a nested StructuredModel: each field "
                "then carries its own comparator and threshold, and results are "
                "reported per field. If the keys are not known when the model is "
                "written, use ANLSStarComparator, which scores the mapping "
                "structurally and gives partial credit."
            )

        # Convert to strings
        s1 = str(s1)
        s2 = str(s2)

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
