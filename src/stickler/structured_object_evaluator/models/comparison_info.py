"""
Comparison configuration for structured model fields.
"""

from typing import Any, Dict, Optional

from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator


class ComparisonInfo:
    """Container for comparison configuration.

    This class holds the configuration for how a field should be compared,
    including which comparator to use, the threshold for considering a match,
    and the weight in scoring.

    Attributes:
        comparator: The comparator to use for string similarity
        threshold: Minimum score to consider a match (like ANLS threshold)
        weight: Weight of this field in the overall score calculation
    """

    def __init__(
        self,
        comparator: Optional[BaseComparator] = None,
        threshold: float = 0.5,
        weight: float = 1.0,
    ):
        """Initialize comparison configuration.

        Args:
            comparator: Comparator to use (default: LevenshteinComparator)
            threshold: Minimum similarity score to consider a match (default: 0.5)
            weight: Weight of this field in the overall score (default: 1.0)
        """
        self.comparator = comparator or LevenshteinComparator()
        self.threshold = threshold
        self.weight = weight

    def compare(self, value1: Any, value2: Any) -> float:
        """Compare two values and return a similarity score between 0 and 1.

        Args:
            value1: First value to compare
            value2: Second value to compare

        Returns:
            Similarity score between 0.0 and 1.0, with 0.0 if below threshold
        """
        # Handle None values
        if value1 is None or value2 is None:
            return 1.0 if value1 == value2 else 0.0

        # Use the comparator to calculate similarity
        similarity = self.comparator.compare(value1, value2)

        # Apply threshold (if below threshold, return 0)
        return 0.0 if similarity < self.threshold else similarity

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ComparisonInfo(comparator={self.comparator}, threshold={self.threshold}, weight={self.weight})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary for JSON schema."""
        return {
            "comparator_type": self.comparator.__class__.__name__,
            "comparator_name": getattr(self.comparator, "name", "unknown"),
            "comparator_config": getattr(self.comparator, "config", {}),
            "threshold": self.threshold,
            "weight": self.weight,
        }


class ComparableFieldConfig:
    """Container for field comparison configuration.

    This class holds the configuration for how a field should be compared,
    including which comparator to use, the threshold for considering a match,
    and the weight in scoring.

    Attributes:
        comparator: The comparator to use for string similarity
        threshold: Minimum score to consider a match (like ANLS threshold)
        weight: Weight of this field in the overall score calculation
        aggregate: Whether to aggregate metrics from child fields
        clip_under_threshold: Whether to zero out scores below threshold
    """

    def __init__(
        self,
        comparator: Optional[BaseComparator] = None,
        threshold: float = 0.5,
        weight: float = 1.0,
        clip_under_threshold: bool = True,
    ):
        """Initialize comparison configuration.

        Args:
            comparator: Comparator to use (default: LevenshteinComparator)
            threshold: Minimum similarity score to consider a match (default: 0.5)
            weight: Weight of this field in the overall score (default: 1.0)
            clip_under_threshold: Whether to zero out scores below threshold (default: True)
        """
        self.comparator = comparator or LevenshteinComparator()
        self.threshold = threshold
        self.weight = weight
        self.clip_under_threshold = clip_under_threshold

    def compare(self, value1: Any, value2: Any) -> float:
        """Compare two values and return a similarity score between 0 and 1.

        Args:
            value1: First value to compare
            value2: Second value to compare

        Returns:
            Similarity score between 0.0 and 1.0, with 0.0 if below threshold
        """
        # Handle None values
        if value1 is None or value2 is None:
            return 1.0 if value1 == value2 else 0.0

        # Use the comparator to calculate similarity
        similarity = self.comparator.compare(value1, value2)

        # Apply threshold if clip_under_threshold is enabled
        if self.clip_under_threshold and similarity < self.threshold:
            return 0.0

        return similarity

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ComparableFieldConfig(comparator={self.comparator}, threshold={self.threshold}, weight={self.weight}, clip_under_threshold={self.clip_under_threshold})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary for JSON schema."""
        return {
            "comparator_type": self.comparator.__class__.__name__,
            "comparator_name": getattr(self.comparator, "name", "unknown"),
            "comparator_config": getattr(self.comparator, "config", {}),
            "threshold": self.threshold,
            "weight": self.weight,
            "clip_under_threshold": self.clip_under_threshold,
        }


def add_comparison_schema(schema: Dict[str, Any], info: ComparisonInfo) -> None:
    """Add comparison info to a schema."""
    schema["x-comparison"] = info.to_dict()
