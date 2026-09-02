"""ANLS* leaf comparison respects the type of what it is comparing.

Edit distance is a string metric. Applying it to every leaf meant character
overlap between two numbers read as partial correctness: `1000` against `9000`
shares three of four characters, scored 0.75, cleared the 0.7 field threshold and
reported a nine-fold error as a TRUE POSITIVE. That is worse than the ranking
failure ANLS* was introduced to fix, because it corrupts the confusion matrix
rather than merely flattening it.

Upstream `anls_star` stringifies everything and has the same flaw. Diverging is
deliberate, and narrows the metric to where it means something.
"""

import datetime
from typing import Any, Dict

import pytest
from pydantic import BaseModel

import stickler
from stickler import ANLSStarComparator


class P(BaseModel):
    meta: Dict[str, Any] = {}


def score(gt: Dict[str, Any], pred: Dict[str, Any]) -> float:
    return stickler.evaluate(P(meta=gt), P(meta=pred)).field_scores["meta"]


class TestNumericLeaves:
    def test_a_wrong_number_scores_zero_not_character_overlap(self):
        assert score({"amount": 1000}, {"amount": 9000}) == pytest.approx(0.0)

    def test_and_is_classified_as_a_miss(self):
        """The part that mattered: the confusion matrix, not just the score."""
        result = stickler.evaluate(P(meta={"amount": 1000}), P(meta={"amount": 9000}))
        assert result.f1 == pytest.approx(0.0)
        assert result.precision == pytest.approx(0.0)

    def test_an_identical_number_still_scores_one(self):
        assert score({"amount": 1000}, {"amount": 1000}) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "gt,pred",
        [(1000, 1001), (1.5, 1.6), (True, False), (10, 100)],
    )
    def test_any_numeric_difference_is_a_miss(self, gt, pred):
        assert score({"v": gt}, {"v": pred}) == pytest.approx(0.0)

    def test_a_number_matches_its_own_json_string_form(self):
        """Keeps model_dump() and model_dump(mode="json") equivalent."""
        assert score({"v": 1000}, {"v": "1000"}) == pytest.approx(1.0)


class TestDateLeaves:
    """A `date` reaches the leaf already serialised to ISO, so the type signal
    was destroyed by this library before the comparison happened."""

    def test_a_wrong_date_object_scores_zero(self):
        assert score(
            {"d": datetime.date(2024, 1, 1)}, {"d": datetime.date(2024, 11, 11)}
        ) == pytest.approx(0.0)

    def test_a_wrong_iso_date_string_scores_zero(self):
        assert score({"d": "2024-01-01"}, {"d": "2024-11-11"}) == pytest.approx(0.0)

    def test_the_same_date_matches_across_object_and_string_form(self):
        assert score(
            {"d": datetime.date(2024, 1, 1)}, {"d": "2024-01-01"}
        ) == pytest.approx(1.0)

    def test_datetimes_too(self):
        assert score(
            {"d": "2024-01-01T09:00:00"}, {"d": "2024-11-11T09:00:00"}
        ) == pytest.approx(0.0)


class TestTextLeavesKeepPartialCredit:
    """The property ANLS* exists for must survive all of the above."""

    def test_a_near_miss_still_earns_partial_credit(self):
        s = score({"vendor": "Acme Corporation"}, {"vendor": "Acme Corp"})
        assert 0.0 < s < 1.0

    def test_identical_text_is_one(self):
        assert score({"vendor": "Acme"}, {"vendor": "Acme"}) == pytest.approx(1.0)

    def test_extractors_are_still_ordered(self):
        GT = {"vendor": "Acme Corporation", "terms": "Net 30", "po": "PO-88231"}
        near = score(GT, {**GT, "vendor": "Acme Corp"})
        dropped = score(GT, {"vendor": "Acme Corporation", "terms": "Net 30"})
        wrong = score(GT, {"vendor": "Zeta Ltd", "terms": "x", "po": "y"})
        assert wrong < dropped < near < 1.0

    def test_a_numeric_string_is_still_text(self):
        """Deliberate asymmetry: the annotation is the signal.

        A caller who put the *string* "1000" in a dict wrote text, and ANLS*
        scores text by edit distance. A caller who wants numeric semantics should
        declare the field. Pinned so the inconsistency is a decision, not a
        surprise.
        """
        assert score({"a": "1000"}, {"a": "9000"}) > 0.0


class TestTheComparatorDirectly:
    def test_leaf_threshold_still_gates_text(self):
        GT = {"vendor": "Acme Corporation"}
        near = {"vendor": "Acme Corp"}
        lenient = ANLSStarComparator(leaf_threshold=0.5).compare(GT, near)
        strict = ANLSStarComparator(leaf_threshold=0.85).compare(GT, near)
        assert lenient > strict

    def test_leaf_threshold_cannot_rescue_a_wrong_number(self):
        """No cutoff setting makes 9000 partially correct."""
        for tau in (0.0, 0.5, 0.99):
            assert ANLSStarComparator(leaf_threshold=tau).compare(
                {"a": 1000}, {"a": 9000}
            ) == pytest.approx(0.0)
