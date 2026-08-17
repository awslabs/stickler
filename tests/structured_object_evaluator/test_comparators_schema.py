"""
Test that custom comparators are correctly reflected in the schema metadata.
"""

from typing import Any

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


# Create a custom comparator for special handling
class CaseInsensitiveComparator(LevenshteinComparator):
    """A comparator that performs case-insensitive comparisons."""

    @property
    def name(self) -> str:
        """Return the name of the comparator."""
        return "case_insensitive"

    def _compare(self, a: Any, b: Any) -> float:
        """Compare strings in a case-insensitive way."""
        # Convert both to strings and lowercase
        a_str = str(a).lower()
        b_str = str(b).lower()

        # Use the parent Levenshtein implementation
        return super()._compare(a_str, b_str)


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
    """Custom comparators are recorded on the field, not the rendered schema.

    ``model_json_schema()`` describes the shape only (issue #188); the
    comparison config is read from ``json_schema_extra`` the way the engine
    reads it.
    """
    # The rendered schema must not leak comparison config
    schema = SpecializedComparatorModel.model_json_schema()
    assert "x-comparison" not in schema["properties"]["standard_field"]
    assert "x-comparison" not in schema["properties"]["insensitive_field"]

    def comparison_metadata(field_name):
        extra = {}
        SpecializedComparatorModel.model_fields[field_name].json_schema_extra(extra)
        return extra["x-comparison"]

    # Check standard field
    std_comp_info = comparison_metadata("standard_field")
    assert std_comp_info["comparator_type"] == "LevenshteinComparator"
    assert std_comp_info["comparator_name"] == "levenshtein"

    # Check case insensitive field
    case_comp_info = comparison_metadata("insensitive_field")
    assert case_comp_info["comparator_type"] == "CaseInsensitiveComparator"
    assert case_comp_info["comparator_name"] == "case_insensitive"
