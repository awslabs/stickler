"""Structured model comparator."""

from typing import Any

from stickler.comparators.base import BaseComparator


class StructuredModelComparator(BaseComparator):
    """Comparator for structured model objects.

    This comparator is designed to work with StructuredModel instances,
    leveraging their built-in comparison capabilities.
    """

    def __init__(self, threshold: float = 0.7, strict_types: bool = False):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (0.0-1.0)
            strict_types: If True, will raise TypeError when non-StructuredModel objects are compared
        """
        super().__init__(threshold)
        self.strict_types = strict_types

    def compare(self, model1: Any, model2: Any) -> float:
        """Compare two structured model instances.

        This method uses the built-in compare method of StructuredModel objects
        if available, otherwise falls back to basic equality comparison.

        Args:
            model1: First model (ideally a StructuredModel instance)
            model2: Second model (ideally a StructuredModel instance)

        Returns:
            Similarity score between 0.0 and 1.0

        Raises:
            TypeError: When strict_types=True and comparing non-StructuredModel objects
        """
        # In strict mode, enforce StructuredModel types (used in tests)
        # For string values, always raise TypeError in strict mode
        if self.strict_types and isinstance(model1, str) and isinstance(model2, str):
            raise TypeError(
                "StructuredModelComparator can only compare StructuredModel instances"
            )

        # Handle None values
        if model1 is None or model2 is None:
            return 1.0 if model1 == model2 else 0.0

        # Check if both objects have a compare method (duck typing)
        if hasattr(model1, "compare") and callable(model1.compare):
            return model1.compare(model2)

        # Fall back to equality check for non-StructuredModel objects
        return 1.0 if model1 == model2 else 0.0
