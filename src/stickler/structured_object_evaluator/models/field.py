"""
Field module for structured model evaluation.

This module contains the ComparableField class used to define fields in structured models
with comparison configuration parameters.
"""

from typing import Any, Dict, Optional, Callable, List, Type, Union
from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo

from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator

DEFAULT_THRESHOLD = 0.7
DEFAULT_WEIGHT = 1.0


class CustomField(FieldInfo):
    """
    Field with comparable properties for structured model evaluation.

    This extends pydantic's Field with additional attributes that control how the field
    is compared during evaluation.

    Attributes:
        comparator: The comparator to use for this field
        threshold: The threshold for determining if values match
        weight: The weight of this field in the overall score
        description: Human-readable description of the field
    """

    def __init__(
        self,
        default: Any = ...,
        *,
        comparator: Optional[BaseComparator] = None,
        threshold: float = DEFAULT_THRESHOLD,
        weight: float = DEFAULT_WEIGHT,
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize comparable field.

        Args:
            default: Default value for the field
            comparator: The comparator to use for this field
            threshold: The threshold for determining if values match
            weight: The weight of this field in the overall score
            description: Human-readable description of the field
            **kwargs: Additional field parameters
        """
        # Fix: Pass all kwargs together with default as keyword args,
        # since pydantic expects a specific format
        kwargs["default"] = default
        super().__init__(**kwargs)

        # Store comparison configuration
        self.comparator = comparator or LevenshteinComparator()
        self.threshold = threshold
        self.weight = weight
        self.description = description

    def get_config(self) -> Dict[str, Any]:
        """
        Get field configuration.

        Returns:
            Dictionary with field configuration
        """
        return {
            "comparator": self.comparator,
            "threshold": self.threshold,
            "weight": self.weight,
            "description": self.description,
        }

    def __repr__(self) -> str:
        """
        String representation.

        Returns:
            String representation
        """
        return (
            f"ComparableField("
            f"comparator={self.comparator}, "
            f"threshold={self.threshold}, "
            f"weight={self.weight})"
        )
