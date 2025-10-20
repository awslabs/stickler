"""Base class for comparators."""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseComparator(ABC):
    """Base class for all comparators.

    This class defines the interface that all comparators must implement.
    Comparators are used to compare two values and return a similarity score
    between 0.0 and 1.0, where 1.0 means the values are identical.
    """

    def __init__(self, threshold: float = 0.7):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (0.0-1.0)
        """
        self.threshold = threshold

    @abstractmethod
    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two values and return a similarity score.

        Args:
            str1: First value
            str2: Second value

        Returns:
            Similarity score between 0.0 and 1.0
        """
        pass

    def __call__(self, str1: Any, str2: Any) -> float:
        """Make the comparator callable.

        Args:
            str1: First value
            str2: Second value

        Returns:
            Similarity score between 0.0 and 1.0
        """
        return self.compare(str1, str2)

    def binary_compare(self, str1: Any, str2: Any) -> Tuple[int, int]:
        """Compare two values and return a binary result as (tp, fp) tuple.

        This method converts the continuous similarity score to a binary decision
        based on the threshold. If the similarity is greater than or equal to the
        threshold, it returns (1, 0) indicating true positive. Otherwise, it returns
        (0, 1) indicating false positive.

        Args:
            str1: First value
            str2: Second value

        Returns:
            Tuple of (tp, fp) where tp is 1 if similar, 0 otherwise,
            and fp is the opposite
        """
        score = self.compare(str1, str2)
        if score >= self.threshold:
            return (1, 0)  # True positive
        else:
            return (0, 1)  # False positive

    def __str__(self) -> str:
        """String representation for serialization."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(threshold={self.threshold})"
