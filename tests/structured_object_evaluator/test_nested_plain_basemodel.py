"""A nested plain pydantic `BaseModel` is scored, not treated as a type mismatch.

`ComparisonDispatcher` routed on `isinstance(value, StructuredModel)`. A plain
`BaseModel` is not one, so a nested one fell through to the mismatched-types
branch and scored `0.0` against an **identical** object, reported as a false
discovery: a perfect match that is also a failure, which is the contradiction
#287 removed elsewhere.

The asymmetry is what made it clearly a defect rather than an unsupported shape:

    Optional[Plain]        identical -> 0.0   and fd=1
    Optional[List[Plain]]  identical -> 1.0

because the list branch sends any non-`StructuredModel` element to the primitive
list comparator. The singular and list forms of one shape disagreed.

Both forms now compare the model's canonical string through the field's declared
comparator. Scoring a plain `BaseModel` field by field is what `StructuredModel`
is for, and what `stickler.evaluate()` does by wrapping one; doing it in the
dispatcher instead would make a nested plain model behave differently from the
same model inside a list, trading this inconsistency for another. See #135.

See https://github.com/awslabs/stickler/issues/318
"""

from typing import List, Optional

import pytest
from pydantic import BaseModel

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)


class Plain(BaseModel):
    """A vanilla pydantic model, the shape a user brings from elsewhere."""

    sku: Optional[str] = None
    qty: Optional[int] = None


class Structured(StructuredModel):
    """The same shape, configured, so the two paths can be compared."""

    sku: Optional[str] = ComparableField(default=None)
    qty: Optional[int] = ComparableField(default=None)


class Nested(StructuredModel):
    kid: Optional[Plain] = ComparableField(default=None)


class NestedList(StructuredModel):
    rows: Optional[List[Plain]] = ComparableField(default=None)


class TestIdenticalObjectsAreNotAFailure:
    """The reported bug: `0.0` and `fd=1` on two equal objects."""

    def test_an_identical_nested_plain_model_scores_one(self):
        result = Nested(kid=Plain(sku="x", qty=1)).compare_with(
            Nested(kid=Plain(sku="x", qty=1))
        )
        assert result["field_scores"]["kid"] == pytest.approx(1.0)
        assert result["overall_score"] == pytest.approx(1.0)

    def test_it_is_classified_as_a_true_positive(self):
        """Score and classification have to agree, which is the #287 lesson."""
        matrix = Nested(kid=Plain(sku="x")).compare_with(
            Nested(kid=Plain(sku="x")), include_confusion_matrix=True
        )["confusion_matrix"]

        assert matrix["overall"]["tp"] == 1
        assert matrix["overall"]["fd"] == 0
        assert matrix["overall"]["fp"] == 0

    def test_a_different_nested_plain_model_still_fails(self):
        """The fix must not make everything match."""
        result = Nested(kid=Plain(sku="completely-different", qty=99)).compare_with(
            Nested(kid=Plain(sku="x", qty=1))
        )
        assert result["field_scores"]["kid"] < 1.0


class TestTheSingularAndListFormsAgree:
    """The asymmetry that identified the bug must not survive the fix."""

    @pytest.mark.parametrize(
        "ground_truth, prediction",
        (
            (Plain(sku="x", qty=1), Plain(sku="x", qty=1)),
            (Plain(sku="a", qty=1), Plain(sku="b", qty=1)),
            (Plain(sku="a"), Plain(sku="a", qty=7)),
        ),
    )
    def test_one_element_scores_the_same_either_way(self, ground_truth, prediction):
        singular = Nested(kid=ground_truth).compare_with(Nested(kid=prediction))[
            "field_scores"
        ]["kid"]
        listed = NestedList(rows=[ground_truth]).compare_with(
            NestedList(rows=[prediction])
        )["field_scores"]["rows"]

        assert singular == pytest.approx(listed)

    def test_the_list_form_was_already_correct_and_is_unchanged(self):
        assert NestedList(rows=[Plain(sku="x")]).compare_with(
            NestedList(rows=[Plain(sku="x")])
        )["field_scores"]["rows"] == pytest.approx(1.0)


class TestTheDeclaredComparatorIsUsed:
    """It routes through the field's comparator, not a hardcoded equality.

    This is the same reasoning as the dict branch added for #297: the "primitive"
    path is only primitive in name, and calls whatever the field declares.
    """

    def test_an_exact_comparator_makes_it_all_or_nothing(self):
        class Strict(StructuredModel):
            kid: Optional[Plain] = ComparableField(
                comparator=ExactComparator(), threshold=1.0, default=None
            )

        assert Strict(kid=Plain(sku="x")).compare_with(Strict(kid=Plain(sku="x")))[
            "field_scores"
        ]["kid"] == pytest.approx(1.0)

        # One character apart would earn partial credit on edit distance.
        assert Strict(kid=Plain(sku="a")).compare_with(Strict(kid=Plain(sku="b")))[
            "field_scores"
        ]["kid"] == pytest.approx(0.0)

    def test_the_default_comparator_gives_partial_credit(self):
        """Levenshtein over the canonical string, so a near miss is not zero."""
        score = Nested(kid=Plain(sku="a", qty=1)).compare_with(
            Nested(kid=Plain(sku="b", qty=1))
        )["field_scores"]["kid"]
        assert 0.0 < score < 1.0


class TestTheCanonicalFormIsStable:
    """Comparison is on the rendered model, so its rendering must be order-free.

    If it were keyword-order dependent, two equal objects built differently would
    score below 1.0, which is the class of bug #276 was about for dicts.
    """

    def test_keyword_order_at_construction_does_not_matter(self):
        result = Nested(kid=Plain(sku="x", qty=1)).compare_with(
            Nested(kid=Plain(qty=1, sku="x"))
        )
        assert result["field_scores"]["kid"] == pytest.approx(1.0)

    def test_an_absent_optional_field_is_rendered_consistently(self):
        result = Nested(kid=Plain(sku="x")).compare_with(Nested(kid=Plain(sku="x")))
        assert result["field_scores"]["kid"] == pytest.approx(1.0)


class TestAStructuredModelIsUnaffected:
    """The new branch sits after the StructuredModel one and must not shadow it.

    `StructuredModel` subclasses `BaseModel`, so an `isinstance` check on the
    latter would catch both. Order in the dispatch chain is what keeps them
    apart, and a nested `StructuredModel` must keep its per-field detail.
    """

    def test_a_nested_structured_model_still_reports_per_field_detail(self):
        class Holder(StructuredModel):
            kid: Optional[Structured] = ComparableField(default=None)

        matrix = Holder(kid=Structured(sku="a", qty=1)).compare_with(
            Holder(kid=Structured(sku="b", qty=1)), include_confusion_matrix=True
        )["confusion_matrix"]

        assert matrix["fields"]["kid"]["fields"], "per-field detail was lost"
        assert "sku" in matrix["fields"]["kid"]["fields"]

    def test_a_nested_plain_model_reports_no_per_field_detail(self):
        """Recorded as the deliberate consequence of comparing the whole model.

        A plain `BaseModel` carries no per-field comparison configuration, so
        there is nothing to report per field. Anyone wanting that detail declares
        the nested model as a `StructuredModel`.
        """
        matrix = Nested(kid=Plain(sku="a")).compare_with(
            Nested(kid=Plain(sku="b")), include_confusion_matrix=True
        )["confusion_matrix"]

        assert not matrix["fields"]["kid"].get("fields")


class TestNullHandlingIsUnchanged:
    """The new branch is only reached when both sides are present."""

    def test_both_absent_is_a_true_negative(self):
        matrix = Nested(kid=None).compare_with(
            Nested(kid=None), include_confusion_matrix=True
        )["confusion_matrix"]
        assert matrix["overall"]["tn"] == 1

    def test_prediction_absent_is_a_false_negative(self):
        matrix = Nested(kid=Plain(sku="x")).compare_with(
            Nested(kid=None), include_confusion_matrix=True
        )["confusion_matrix"]
        assert matrix["overall"]["fn"] == 1

    def test_ground_truth_absent_is_a_false_alarm(self):
        matrix = Nested(kid=None).compare_with(
            Nested(kid=Plain(sku="x")), include_confusion_matrix=True
        )["confusion_matrix"]
        assert matrix["overall"]["fa"] == 1
