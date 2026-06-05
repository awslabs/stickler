"""
Regression tests for PEP 604 union type handling (list[X] | None syntax).

Validates that fields declared with Python 3.10+ PEP 604 syntax (types.UnionType)
are correctly recognized by the type-detection methods, ensuring proper comparison
dispatch for nested list fields.

See: https://github.com/awslabs/stickler/issues/139
"""

from typing import List, Optional

import pytest

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.configuration_helper import (
    ConfigurationHelper,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


# ---------------------------------------------------------------------------
# PEP 604 models (list[X] | None syntax)
# ---------------------------------------------------------------------------


class InnerPep604(StructuredModel):
    match_threshold = 0.7

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class OuterPep604(StructuredModel):
    match_threshold = 0.7

    key: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    items: list[InnerPep604] | None = ComparableField(weight=1.0)


class RootPep604(StructuredModel):
    match_threshold = 0.7

    outer: list[OuterPep604] | None = ComparableField(weight=1.0)


class OuterPrimitivePep604(StructuredModel):
    match_threshold = 0.7

    key: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    items: list[str] | None = ComparableField(weight=1.0)


class RootPrimitivePep604(StructuredModel):
    match_threshold = 0.7

    outer: list[OuterPrimitivePep604] | None = ComparableField(weight=1.0)


# ---------------------------------------------------------------------------
# typing.Optional models (regression guard)
# ---------------------------------------------------------------------------


class InnerTyping(StructuredModel):
    match_threshold = 0.7

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class OuterTyping(StructuredModel):
    match_threshold = 0.7

    key: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    items: Optional[List[InnerTyping]] = ComparableField(weight=1.0)


class RootTyping(StructuredModel):
    match_threshold = 0.7

    outer: Optional[List[OuterTyping]] = ComparableField(weight=1.0)


# ---------------------------------------------------------------------------
# Unit tests: type detection methods
# ---------------------------------------------------------------------------


class TestTypeDetectionPep604:
    """Verify type-detection methods recognize PEP 604 union syntax."""

    def test_is_list_field_pep604_structured(self):
        """_is_list_field returns True for list[StructuredModel] | None."""
        instance = OuterPep604(key="A", items=[])
        assert instance._is_list_field("items") is True

    def test_is_list_field_pep604_primitive(self):
        """_is_list_field returns True for list[str] | None."""
        instance = OuterPrimitivePep604(key="A", items=[])
        assert instance._is_list_field("items") is True

    def test_is_list_field_pep604_root(self):
        """_is_list_field returns True for list[Outer] | None at root level."""
        instance = RootPep604(outer=[])
        assert instance._is_list_field("outer") is True

    def test_is_structured_field_type_pep604(self):
        """is_structured_field_type returns True for list[StructuredModel] | None."""
        field_info = OuterPep604.model_fields["items"]
        assert ConfigurationHelper.is_structured_field_type(field_info) is True

    def test_is_structured_field_type_pep604_root(self):
        """is_structured_field_type returns True for list[Outer] | None."""
        field_info = RootPep604.model_fields["outer"]
        assert ConfigurationHelper.is_structured_field_type(field_info) is True

    def test_is_structured_field_type_pep604_primitive_not_structured(self):
        """is_structured_field_type returns False for list[str] | None (not a StructuredModel)."""
        field_info = OuterPrimitivePep604.model_fields["items"]
        assert ConfigurationHelper.is_structured_field_type(field_info) is False

    def test_is_list_of_structured_model_type_pep604(self):
        """_is_list_of_structured_model_type handles list[StructuredModel] | None."""
        field_info = OuterPep604.model_fields["items"]
        assert (
            OuterPep604._is_list_of_structured_model_type(field_info.annotation) is True
        )

    def test_is_list_structured_model_pep604(self):
        """_is_list_structured_model handles list[StructuredModel] | None."""
        annotation = OuterPep604.model_fields["items"].annotation
        assert ConfigurationHelper._is_list_structured_model(annotation) is True

    def test_extract_structured_class_from_list_pep604(self):
        """_extract_structured_class_from_list returns the class for PEP 604."""
        annotation = OuterPep604.model_fields["items"].annotation
        result = ConfigurationHelper._extract_structured_class_from_list(annotation)
        assert result is InnerPep604

    def test_typing_optional_still_works(self):
        """Regression guard: typing.Optional[List[X]] still detected correctly."""
        instance = OuterTyping(key="A", items=[])
        assert instance._is_list_field("items") is True

        field_info = OuterTyping.model_fields["items"]
        assert ConfigurationHelper.is_structured_field_type(field_info) is True
        assert (
            OuterTyping._is_list_of_structured_model_type(field_info.annotation) is True
        )


# ---------------------------------------------------------------------------
# Integration tests: comparison behavior
# ---------------------------------------------------------------------------


class TestEmptyNestedListPep604:
    """Verify empty list comparison produces correct results with PEP 604 syntax."""

    def test_empty_nested_list_pep604_true_negative(self):
        """[] vs [] inside a nested structured list produces score=1.0 with PEP 604."""
        gt = RootPep604(outer=[OuterPep604(key="A", items=[])])
        pred = RootPep604(outer=[OuterPep604(key="A", items=[])])

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"] is True

    def test_empty_nested_list_pep604_false_addition(self):
        """[] vs [Inner(...)] produces a score penalty with PEP 604."""
        gt = RootPep604(outer=[OuterPep604(key="A", items=[])])
        pred = RootPep604(
            outer=[OuterPep604(key="A", items=[InnerPep604(name="extra")])]
        )

        result = gt.compare_with(pred)
        assert result["overall_score"] < 1.0, (
            "Expected score penalty for pred-only items"
        )

    def test_non_empty_nested_list_pep604_dispatches_correctly(self):
        """Non-empty list[StructuredModel] | None dispatches through structured comparison."""
        gt = RootPep604(
            outer=[OuterPep604(key="A", items=[InnerPep604(name="hello")])]
        )
        pred = RootPep604(
            outer=[OuterPep604(key="A", items=[InnerPep604(name="hello")])]
        )

        result = gt.compare_with(pred, include_confusion_matrix=True)
        assert result["overall_score"] == 1.0

        cm = result["confusion_matrix"]
        items_field = cm["fields"]["outer"]["fields"]["items"]
        assert items_field["overall"]["tp"] >= 1, (
            "Expected TP for matching structured items"
        )

    def test_primitive_list_pep604_empty_true_negative(self):
        """list[str] | None with [] vs [] produces score=1.0."""
        gt = RootPrimitivePep604(outer=[OuterPrimitivePep604(key="A", items=[])])
        pred = RootPrimitivePep604(outer=[OuterPrimitivePep604(key="A", items=[])])

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"] is True

    def test_typing_optional_list_unchanged(self):
        """Regression guard: Optional[List[X]] empty list still produces score=1.0."""
        gt = RootTyping(outer=[OuterTyping(key="A", items=[])])
        pred = RootTyping(outer=[OuterTyping(key="A", items=[])])

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"] is True

    def test_non_empty_structured_list_match_with_confusion_matrix(self):
        """Structured list comparison produces correct confusion matrix entries."""
        gt = RootPep604(
            outer=[OuterPep604(key="A", items=[InnerPep604(name="X")])]
        )
        pred = RootPep604(
            outer=[OuterPep604(key="A", items=[InnerPep604(name="X")])]
        )

        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]

        # items should appear as a sub-field when non-empty
        items_field = cm["fields"]["outer"]["fields"]["items"]
        assert "overall" in items_field
        assert items_field["overall"]["tp"] == 1


# ---------------------------------------------------------------------------
# Automated validation: parametrized proof the fix works
# ---------------------------------------------------------------------------

_MODELS = {
    "pep604": (RootPep604, OuterPep604, InnerPep604),
    "typing_optional": (RootTyping, OuterTyping, InnerTyping),
}


@pytest.mark.parametrize("annotation_style", ["pep604", "typing_optional"])
class TestIssue139Validation:
    """Automated validation for issue #139 fix.

    Run: pytest tests/structured_object_evaluator/test_pep604_union_type_handling.py::TestIssue139Validation -v

    Before fix: pep604 tests FAIL, typing_optional tests PASS
    After fix: ALL tests PASS
    """

    def test_empty_list_produces_correct_score(self, annotation_style):
        """The core assertion: [] vs [] inside a nested structured list -> score=1.0."""
        Root, Outer, _ = _MODELS[annotation_style]
        gt = Root(outer=[Outer(key="A", items=[])])
        pred = Root(outer=[Outer(key="A", items=[])])

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0, (
            f"[{annotation_style}] Expected score=1.0 for matching empty lists, "
            f"got {result['overall_score']}"
        )
        assert result["all_fields_matched"] is True, (
            f"[{annotation_style}] Expected all_fields_matched=True"
        )

    def test_type_detection_recognizes_list(self, annotation_style):
        """_is_list_field returns True for the items field."""
        _, Outer, _ = _MODELS[annotation_style]
        outer_instance = Outer(key="A", items=[])
        assert outer_instance._is_list_field("items") is True, (
            f"[{annotation_style}] _is_list_field('items') returned False"
        )

    def test_non_empty_list_matches(self, annotation_style):
        """Non-empty matching lists produce score=1.0 regardless of annotation style."""
        Root, Outer, Inner = _MODELS[annotation_style]
        gt = Root(outer=[Outer(key="A", items=[Inner(name="hello")])])
        pred = Root(outer=[Outer(key="A", items=[Inner(name="hello")])])

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0, (
            f"[{annotation_style}] Expected score=1.0 for matching non-empty lists, "
            f"got {result['overall_score']}"
        )

    def test_mismatched_list_penalizes(self, annotation_style):
        """Mismatched lists produce score < 1.0 regardless of annotation style."""
        Root, Outer, Inner = _MODELS[annotation_style]
        gt = Root(outer=[Outer(key="A", items=[])])
        pred = Root(outer=[Outer(key="A", items=[Inner(name="extra")])])

        result = gt.compare_with(pred)
        assert result["overall_score"] < 1.0, (
            f"[{annotation_style}] Expected score < 1.0 for mismatched lists, "
            f"got {result['overall_score']}"
        )
