"""A field neither side populated does not pay out as a match.

`compare_field_raw` scores an absent-on-both field `1.0`, which is right for the
question it answers -- nothing disagreed -- but wrong to fold into a weighted
average as though evidence of a match had been observed. `StructuredModel.compare`
already omitted those fields; the aggregation path in `ComparisonEngine` did not.

The consequence scaled with how optional a schema is. A five-field model whose one
populated field disagreed completely reported `overall_score=0.8`, and because
`EvalResult.matched` is `overall_score >= match_threshold`, that pair reported
`matched=True` alongside an F1 of zero. Sparse real documents -- the common case --
inflated most.

`compare` is also the Hungarian cost function, so the two readers disagreeing meant
list pairing and per-item true-positive/false-discovery classification were decided
on a different number from the one reported beside them. That is the divergence #301
set out to close, and it stayed open on the aggregation path. Both readers now ask
`absent_on_both` and reduce through `resolve_weighted_mean`, so the drift cannot
reappear in one place only.

Two things deliberately did NOT change:

* Absent-on-both still counts as a **true negative** in the confusion matrix.
  "Neither side had this" is a real observation and the matrix is where it belongs;
  only the weighted mean rejects it. `TestTheConfusionMatrixStillCountsThem` pins
  this, because suppressing the score without keeping the count would trade one
  inaccuracy for another.
* Identical empty objects still score `1.0` (#233). Nothing disagreed, so there is
  nothing to penalise -- and `resolve_weighted_mean` reaches that answer only when
  no field was compared at all, which is what separates it from the zero-weight
  case below.

`total_weight == 0` is ambiguous and the two meanings need different answers:
nothing was compared (a perfect match, above) versus fields were compared but the
caller declared them weightless (not a match -- the values were never examined).
Returning `1.0` for the second made completely different values free to pair.
"""

from typing import Optional

import pytest
from pydantic import BaseModel

import stickler
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class Sparse(StructuredModel):
    """Four of five fields go unpopulated, as a real extraction target does."""

    a: Optional[str] = ComparableField(default=None)
    b: Optional[str] = ComparableField(default=None)
    c: Optional[str] = ComparableField(default=None)
    d: Optional[str] = ComparableField(default=None)
    e: Optional[str] = ComparableField(default=None)


class Pair(StructuredModel):
    present: Optional[str] = ComparableField(default=None)
    absent: Optional[str] = ComparableField(default=None)


class TestAbsentFieldsDoNotInflateTheScore:
    """The only populated field disagrees, so the pair is not a match."""

    def test_the_empty_fields_are_left_out_of_the_mean(self):
        gt = Sparse(a="ACME Corporation")
        pred = Sparse(a="totally different value")

        # Four absent fields scoring 1.0 each used to carry this to 0.8.
        assert gt.compare_with(pred)["overall_score"] == pytest.approx(0.0)

    def test_a_disagreeing_pair_is_not_reported_as_matched(self):
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
        # `matched` and `f1` disagreeing is the symptom that made this visible.
        assert result.f1 == pytest.approx(0.0)

    def test_a_populated_field_still_carries_its_own_score(self):
        """Guard against fixing inflation by zeroing everything instead."""
        gt = Sparse(a="hello")
        pred = Sparse(a="hello")

        assert gt.compare_with(pred)["overall_score"] == pytest.approx(1.0)


class TestTheConfusionMatrixStillCountsThem:
    """Absence is a true negative; only the weighted mean declines to score it."""

    def test_absent_on_both_is_still_a_true_negative(self):
        gt = Sparse(a="ACME Corporation")
        pred = Sparse(a="totally different value")

        overall = gt.compare_with(pred, include_confusion_matrix=True)[
            "confusion_matrix"
        ]["overall"]

        assert overall["tn"] == 4
        assert overall["fd"] == 1
        assert overall["tp"] == 0


class TestBothScoreReadersAgree:
    """`compare` is the Hungarian cost; it must match the reported score."""

    def test_the_two_readers_return_the_same_number(self):
        gt, pred = Pair(present="hello"), Pair(present="hellp")

        # Previously 0.8 and 0.9: the absent field was omitted by one reader and
        # counted as a match by the other.
        assert gt.compare(pred) == pytest.approx(
            gt.compare_with(pred)["overall_score"]
        )

    def test_they_agree_when_every_field_is_absent(self):
        gt, pred = Pair(), Pair()

        assert gt.compare(pred) == pytest.approx(1.0)
        assert gt.compare_with(pred)["overall_score"] == pytest.approx(1.0)


class TestIdenticalEmptyObjectsRemainAPerfectMatch:
    """#233. Nothing disagreed, so there is nothing to penalise."""

    def test_no_field_compared_scores_one(self):
        assert Sparse().compare(Sparse()) == pytest.approx(1.0)
        assert Sparse().compare_with(Sparse())["overall_score"] == pytest.approx(1.0)


class TestZeroWeightIsNotAPerfectMatch:
    """Weightless fields were never examined, so they cannot vouch for a match."""

    def test_different_values_at_zero_weight_do_not_score_one(self):
        class ZeroWeight(StructuredModel):
            x: str = ComparableField(weight=0.0)

        gt, pred = ZeroWeight(x="aaa"), ZeroWeight(x="zzz")

        # Returning 1.0 here made every pairing in a list of these models free.
        assert gt.compare(pred) == pytest.approx(0.0)
        assert gt.compare_with(pred)["overall_score"] == pytest.approx(0.0)

    def test_it_is_distinguished_from_nothing_being_compared(self):
        """The two routes to `total_weight == 0` get different answers."""

        class ZeroWeight(StructuredModel):
            x: Optional[str] = ComparableField(default=None, weight=0.0)

        # Compared at zero weight -> not a match.
        assert ZeroWeight(x="aaa").compare(ZeroWeight(x="zzz")) == pytest.approx(0.0)
        # Never compared at all -> unchanged by #233.
        assert ZeroWeight().compare(ZeroWeight()) == pytest.approx(1.0)
