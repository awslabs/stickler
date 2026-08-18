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
# Direct (non-list) optional-model spellings — parity guard for the
# is_structured_field_type default-comparator decision.
# ---------------------------------------------------------------------------


class DirectHolderPep604(StructuredModel):
    match_threshold = 0.7

    child: InnerPep604 | None = ComparableField()


class DirectHolderTyping(StructuredModel):
    match_threshold = 0.7

    child: Optional[InnerTyping] = ComparableField()


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

    def test_direct_model_pep604_matches_optional_parity(self):
        """`Model | None` and `Optional[Model]` must resolve identically.

        Guards issue #146 review: a divergent PEP 604 arm previously routed a
        direct `Model | None` field to StructuredModelComparator@0.9 while
        `Optional[Model]` fell through to LevenshteinComparator@0.5. Both must
        return False from is_structured_field_type so match decisions on
        existing user models cannot silently flip.
        """
        pep604 = DirectHolderPep604.model_fields["child"]
        typing_opt = DirectHolderTyping.model_fields["child"]
        assert ConfigurationHelper.is_structured_field_type(pep604) is False
        assert ConfigurationHelper.is_structured_field_type(typing_opt) is False
        assert (
            ConfigurationHelper.is_structured_field_type(pep604)
            == ConfigurationHelper.is_structured_field_type(typing_opt)
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

    def test_none_vs_empty_list_pep604(self):
        """None vs [] is treated as equivalent (issue #139 None-case contract).

        NullHelper.is_effectively_null_for_lists treats None and [] the same,
        so a None list on one side and an empty list on the other must not
        penalize.
        """
        gt = RootPep604(outer=[OuterPep604(key="A", items=None)])
        pred = RootPep604(outer=[OuterPep604(key="A", items=[])])

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"] is True

    def test_none_vs_none_list_pep604(self):
        """None vs None for a PEP 604 list field produces score=1.0."""
        gt = RootPep604(outer=[OuterPep604(key="A", items=None)])
        pred = RootPep604(outer=[OuterPep604(key="A", items=None)])

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"] is True

    def test_empty_list_pep604_items_field_matches(self):
        """Per-field assertion pinning issue #139's exact symptom.

        The reported bug was `items` ([] vs []) scoring 0.0 and routing
        through the primitive comparator as a mismatch. Assert the field-level
        score directly (not just the aggregate), and confirm the confusion
        matrix records it as a true negative rather than a false positive.
        """
        gt = OuterPep604(key="A", items=[])
        pred = OuterPep604(key="A", items=[])

        result = gt.compare_with(pred, include_confusion_matrix=True)
        assert result["field_scores"]["items"] == 1.0

        items_cm = result["confusion_matrix"]["fields"]["items"]["overall"]
        assert items_cm["tn"] == 1, "empty vs empty list must be a true negative"
        assert items_cm["fp"] == 0
        assert items_cm["fn"] == 0


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


# ---------------------------------------------------------------------------
# from_json round-trip: exercises _is_list_structured_model and
# _extract_structured_class_from_list, which are only reachable via from_json
# (not the direct-construction integration tests above).
# ---------------------------------------------------------------------------


class TestFromJsonRoundTripPep604:
    """Cover the from_json code paths for PEP 604 nested list fields."""

    def test_from_json_builds_nested_models(self):
        """from_json recursively constructs nested StructuredModel instances."""
        root = RootPep604.from_json(
            {"outer": [{"key": "A", "items": [{"name": "hello"}]}]}
        )
        assert isinstance(root.outer[0], OuterPep604)
        assert isinstance(root.outer[0].items[0], InnerPep604)
        assert root.outer[0].items[0].name == "hello"

    def test_from_json_empty_nested_list_round_trips_to_true_negative(self):
        """A from_json-built empty nested list still scores as a true negative."""
        gt = RootPep604.from_json({"outer": [{"key": "A", "items": []}]})
        pred = RootPep604.from_json({"outer": [{"key": "A", "items": []}]})

        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"] is True

    def test_from_json_matches_direct_construction(self):
        """from_json and direct construction produce equivalent comparisons."""
        json_built = RootPep604.from_json(
            {"outer": [{"key": "A", "items": [{"name": "X"}]}]}
        )
        direct = RootPep604(outer=[OuterPep604(key="A", items=[InnerPep604(name="X")])])

        result = json_built.compare_with(direct)
        assert result["overall_score"] == 1.0


# ---------------------------------------------------------------------------
# __init_subclass__ validation: list[Model] | None with an explicit threshold
# now raises at class-definition (import) time, matching the long-standing
# Optional[List[Model]] behavior. This is a behavior change on upgrade for
# users who spelled the field with PEP 604 syntax.
# ---------------------------------------------------------------------------


class TestInitSubclassValidationPep604:
    """The Hungarian-matching threshold guard fires for the PEP 604 spelling."""

    def test_threshold_on_pep604_list_of_models_raises(self):
        with pytest.raises(ValueError, match="threshold"):

            class BadPep604(StructuredModel):  # noqa: F811
                match_threshold = 0.7
                items: list[InnerPep604] | None = ComparableField(threshold=0.8)

    def test_threshold_on_typing_optional_list_of_models_raises(self):
        """Parity guard: the typing.Optional spelling raises identically."""
        with pytest.raises(ValueError, match="threshold"):

            class BadTyping(StructuredModel):  # noqa: F811
                match_threshold = 0.7
                items: Optional[List[InnerTyping]] = ComparableField(threshold=0.8)
