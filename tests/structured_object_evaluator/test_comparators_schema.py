"""
Test that custom comparators are correctly reflected in the schema metadata.
"""

import pytest
from typing import Any

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator


# Create a custom comparator for special handling
class CaseInsensitiveComparator(LevenshteinComparator):
    """A comparator that performs case-insensitive comparisons."""

    @property
    def name(self) -> str:
        """Return the name of the comparator."""
        return "case_insensitive"

    def compare(self, a: Any, b: Any) -> float:
        """Compare strings in a case-insensitive way."""
        if a is None or b is None:
            return 1.0 if a == b else 0.0

        # Convert both to strings and lowercase
        a_str = str(a).lower()
        b_str = str(b).lower()

        # Use the parent Levenshtein implementation
        return super().compare(a_str, b_str)


class SpecializedComparatorModel(StructuredModel):
    """Model with specialized comparators for specific needs."""

    # Standard field with default comparator (which normalizes case)
    standard_field: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )

    # Case-insensitive field
    insensitive_field: str = ComparableField(
        comparator=CaseInsensitiveComparator(), threshold=0.7, weight=1.0
    )


def test_custom_comparator_in_schema():
    """Test that custom comparators are correctly reflected in the schema."""
    # Get schema for model with custom comparator
    schema = SpecializedComparatorModel.model_json_schema()

    # Check standard field
    std_comp_info = schema["properties"]["standard_field"]["x-comparison"]
    assert std_comp_info["comparator_type"] == "LevenshteinComparator"
    assert std_comp_info["comparator_name"] == "levenshtein"

    # Check case insensitive field
    case_comp_info = schema["properties"]["insensitive_field"]["x-comparison"]
    assert case_comp_info["comparator_type"] == "CaseInsensitiveComparator"
    assert case_comp_info["comparator_name"] == "case_insensitive"
