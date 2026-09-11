"""A field absent on both sides is not evidence that two objects match (#345).

`compare()` has skipped absent-on-both fields since #233, with the reasoning
written down at the call site: a true negative is absence of evidence, not
evidence of agreement. `compare_recursive()`, which backs `compare_with()` and
therefore `stickler.evaluate()`, folded them in at a perfect score and full
weight. Empty fields paid out as matches.

The headline shape, on a five-field model where the ONE populated field disagrees
completely:

    overall_score  0.8      <- four empty fields carried it
    matched        True     <- the only extracted value is wrong
    f1             0.0

The two defects this file pins are one bug seen from two directions:

* the mean is inflated by fields that contributed nothing, and
* the two aggregation paths therefore disagree about the same pair.

The second is the more dangerous one, because `compare()` is the Hungarian cost
function. List pairing and per-item tp/fd classification were decided on one
number while the score reported beside them was a different number.

Why 2739 tests missed it, recorded so the tests below are the right ones: no test
scored a multi-field model with fields absent on BOTH sides -- fixtures populate
what they assert on -- and no test asserted the two paths agree. #301 unified the
*gate* the two paths consult and left the arithmetic alone, so `TestBothPathsAgree`
is deliberately structural rather than a list of remembered numbers.
"""

from typing import Optional

import pytest
from pydantic import BaseModel

import stickler
from stickler import ComparableField, StructuredModel


class FiveOptional(StructuredModel):
    a: Optional[str] = ComparableField(default=None)
    b: Optional[str] = ComparableField(default=None)
    c: Optional[str] = ComparableField(default=None)
    d: Optional[str] = ComparableField(default=None)
    e: Optional[str] = ComparableField(default=None)


class TwoOptional(StructuredModel):
    present: Optional[str] = ComparableField(default=None)
    absent: Optional[str] = ComparableField(default=None)


class ZeroWeight(StructuredModel):
    x: str = ComparableField(weight=0.0)


class TestEmptyFieldsDoNotPayOut:
    """The defect as a user meets it, through the public entry point."""

    def test_one_wrong_value_among_empty_fields_is_not_a_match(self):
        """The regression in its original form (#345).

        Four fields absent on both sides, one populated field that disagrees
        completely. Before the fix this reported 0.8 and `matched=True` at an F1
        of 0.0 -- the pair the library exists to catch, reported as passing.
        """

        class Invoice(BaseModel):
            a: Optional[str] = None
            b: Optional[str] = None
            c: Optional[str] = None
            d: Optional[str] = None
            e: Optional[str] = None

        result = stickler.evaluate(
            Invoice(a="ACME Corporation"), Invoice(a="totally different value")
        )

        assert result.matched is False
        assert result.overall_score < 0.5
        # The point of the test: `matched` cannot be True while F1 is 0.0. That
        # combination is what made the old behaviour indefensible rather than
        # merely generous.
        assert not (result.matched and result.f1 == 0.0)

    def test_the_score_ignores_how_many_fields_were_empty(self):
        """Adding empty fields to a schema must not raise its score.

        This is the property that makes the defect scale: the sparser the
        document, the more the mean was inflated, and sparse documents are the
        common case in extraction.
        """
        one_wrong = FiveOptional(a="hello"), FiveOptional(a="wrong entirely")
        narrow = TwoOptional(present="hello"), TwoOptional(present="wrong entirely")

        wide_score = one_wrong[0].compare_with(one_wrong[1])["overall_score"]
        narrow_score = narrow[0].compare_with(narrow[1])["overall_score"]

        assert wide_score == pytest.approx(narrow_score), (
            "a five-field model with four empty fields scored differently from a "
            "two-field model with one, on the same populated pair"
        )

    def test_absent_on_both_is_still_a_true_negative(self):
        """Kept OUT of the mean, kept IN the matrix.

        The fix must not overreach: an empty-on-both field is a genuine true
        negative and still belongs in the confusion matrix. Only its contribution
        to the weighted average is wrong. Asserting this stops a future
        simplification from dropping the field entirely.
        """
        result = FiveOptional(a="hello").compare_with(
            FiveOptional(a="hello"), include_confusion_matrix=True
        )

        assert result["confusion_matrix"]["overall"]["tn"] == 4
        assert set(result["field_scores"]) == set(FiveOptional.model_fields) - {
            "extra_fields"
        }

    def test_identical_empty_objects_are_still_a_perfect_match(self):
        """#233, which the fix must not regress.

        Nothing disagreed, so there is nothing to report as wrong. This is the
        case that makes `total_weight == 0` mean two different things, and it is
        why the guard needs a compared-field count rather than a weight test.
        """
        empty = FiveOptional()

        assert empty.compare(FiveOptional()) == pytest.approx(1.0)
        assert empty.compare_with(FiveOptional())["overall_score"] == pytest.approx(1.0)


class TestZeroWeightIsNotAPerfectMatch:
    def test_weights_summing_to_zero_do_not_score_one(self):
        """`total_weight == 0` had one answer for two different causes.

        `return 1.0` is right for "nothing was compared" and catastrophic for
        "fields were compared at zero weight": two completely different objects
        scored a perfect match. Because this is the Hungarian cost function, it
        also made every pairing in a list of such models free.
        """
        assert ZeroWeight(x="aaa").compare(ZeroWeight(x="zzz")) == pytest.approx(0.0)

    def test_zero_weight_agrees_across_both_paths(self):
        different = ZeroWeight(x="aaa"), ZeroWeight(x="zzz")

        assert different[0].compare(different[1]) == pytest.approx(
            different[0].compare_with(different[1])["overall_score"]
        )

    def test_zero_weight_is_not_rescued_by_being_identical(self):
        """Deliberately pinned: identical values at zero weight score 0.0 too.

        Not obviously the only defensible answer -- an argument exists for 1.0 --
        but it is what `compare_with()` returned before this change, so the two
        paths agree and the behaviour is recorded rather than accidental. Revisit
        with a decision, not with a patch.
        """
        same = ZeroWeight(x="aaa"), ZeroWeight(x="aaa")

        assert same[0].compare(same[1]) == pytest.approx(0.0)
        assert same[0].compare_with(same[1])["overall_score"] == pytest.approx(0.0)


class TestBothPathsAgree:
    """The test that would have caught this, and the one #301 stopped short of.

    #301 unified the GATE the two paths consult. It never asserted they produce
    the same number, so `compare()` carried the #233 skip and
    `compare_recursive()` did not, and the same pair scored 0.174 and 0.8.

    Parametrized over shapes rather than asserting remembered constants: the claim
    is that the two agree, whatever the right value turns out to be.
    """

    @pytest.mark.parametrize(
        "gt_kwargs,pred_kwargs,label",
        [
            ({}, {}, "all fields empty on both sides"),
            ({"a": "v"}, {"a": "v"}, "one identical, four empty"),
            ({"a": "hello"}, {"a": "hellp"}, "one near match, four empty"),
            ({"a": "hello", "b": "world"}, {"a": "hellp", "b": "world"}, "two present"),
            (
                {k: "hello" for k in "abcde"},
                {k: "hellp" for k in "abcde"},
                "all five present and near",
            ),
            ({"a": "only gt"}, {}, "present on gt, absent on prediction"),
            ({}, {"a": "only pred"}, "absent on gt, present on prediction"),
        ],
    )
    def test_compare_matches_compare_with(self, gt_kwargs, pred_kwargs, label):
        gt, pred = FiveOptional(**gt_kwargs), FiveOptional(**pred_kwargs)

        direct = gt.compare(pred)
        aggregated = gt.compare_with(pred)["overall_score"]

        assert direct == pytest.approx(aggregated), (
            f"{label}: compare() gave {direct:.4f} and compare_with() gave "
            f"{aggregated:.4f}. These feed Hungarian pairing and the reported "
            f"score respectively, so a gap means items are paired on one number "
            f"and reported on another."
        )

    def test_the_two_paths_share_one_absence_predicate(self):
        """Structural, because the numeric tests above cannot prove intent.

        Two copies of the absence rule is what let them drift for two releases.
        This asserts there is one, so a future edit to either path has to go
        through the shared helper or fail here.
        """
        import inspect

        from stickler.structured_object_evaluator.models import (
            comparison_engine,
            structured_model,
        )

        for module in (comparison_engine, structured_model):
            source = inspect.getsource(module)
            assert "absent_on_both(" in source, (
                f"{module.__name__} no longer routes through the shared "
                f"absent-on-both predicate"
            )
            assert "is_effectively_null_for_lists" not in source, (
                f"{module.__name__} re-implements the absence rule instead of "
                f"calling `absent_on_both`, which is how the two paths drifted "
                f"apart in the first place"
            )


class TestTheRemainingDivergenceIsDeliberate:
    """`compare()` is raw, `compare_with()` is threshold-applied. That is by design.

    Recorded because the two DO still differ when clipping fires, and a future
    reader comparing them will find it. `compare()` deliberately skips thresholds:
    it is a cost function, and clipping every weak pair to 0.0 would make them
    indistinguishable to the assignment algorithm, which needs graded costs to
    pick the best pairing rather than an arbitrary one.
    """

    def test_clipping_is_the_only_remaining_gap(self):
        gt = FiveOptional(a="ACME Corporation")
        pred = FiveOptional(a="totally different value")

        # Clipped: below threshold, so the reported score zeroes while the cost
        # function keeps the graded value.
        assert gt.compare(pred) > 0.0
        assert gt.compare_with(pred)["overall_score"] == pytest.approx(0.0)

    def test_with_clipping_disabled_the_two_agree_exactly(self):
        """Isolates the cause: disable clipping and the gap closes."""

        class NoClip(StructuredModel):
            a: Optional[str] = ComparableField(
                default=None, clip_under_threshold=False
            )
            b: Optional[str] = ComparableField(
                default=None, clip_under_threshold=False
            )

        gt = NoClip(a="ACME Corporation")
        pred = NoClip(a="totally different value")

        assert gt.compare(pred) == pytest.approx(
            gt.compare_with(pred)["overall_score"]
        )
