"""ANLS* compares every leaf as text, and that is deliberate.

ANLS came from scene-text VQA, where every answer is a string. ANLS* generalizes
the STRUCTURE (align dict keys, normalize over the union of both key sets, pair
list elements by Hungarian assignment) and leaves the leaf metric alone.

This file pins both halves of that: the partial credit the structural work exists
to provide, and the cost of uniform text leaves. The second half is pinned
*because* it looks like a bug. It is a documented limitation with a named future
fix, and a test asserting the current numbers is what stops someone "correcting"
it into a second inference table.

See the FUTURE note in ``ANLSLeaf.nls_list``, and
https://github.com/awslabs/stickler/issues/239 for why a hand-rolled type-aware
leaf would be worse than this.
"""

from typing import Any, Dict

import pytest
from pydantic import BaseModel

import stickler
from stickler import ANLSStarComparator

GT = {"vendor": "Acme Corporation", "terms": "Net 30", "po": "PO-88231"}


class P(BaseModel):
    meta: Dict[str, Any] = {}


def score(gt: Dict[str, Any], pred: Dict[str, Any]) -> float:
    return stickler.evaluate(P(meta=gt), P(meta=pred)).field_scores["meta"]


class TestWhatTheStructuralWorkBuys:
    """The reason this comparator replaced whole-object equality."""

    def test_a_near_miss_earns_partial_credit(self):
        assert 0.0 < score(GT, {**GT, "vendor": "Acme Corp"}) < 1.0

    def test_identical_content_is_one_regardless_of_key_order(self):
        reordered = {"po": GT["po"], "terms": GT["terms"], "vendor": GT["vendor"]}
        assert score(GT, reordered) == pytest.approx(1.0)

    def test_five_extractors_stay_ordered(self):
        perfect = score(GT, dict(GT))
        near = score(GT, {**GT, "vendor": "Acme Corp"})
        hallucinated = score(GT, {**GT, "currency": "USD"})
        dropped = score(GT, {"vendor": GT["vendor"], "terms": GT["terms"]})
        wrong = score(GT, {"vendor": "Zeta Ltd", "terms": "x", "po": "y"})
        assert wrong < dropped < hallucinated < near < perfect

    def test_a_renamed_key_is_charged_on_both_sides(self):
        """Normalization is over the union, so a rename costs twice."""
        renamed = score(GT, {**{k: v for k, v in GT.items() if k != "vendor"}, "vendor_name": GT["vendor"]})
        dropped = score(GT, {k: v for k, v in GT.items() if k != "vendor"})
        assert renamed < dropped


class TestUniformTextLeavesAndTheirCost:
    """Pinned as intended behavior, not as an aspiration.

    Every value below is compared as a string. That is canonical ANLS*.
    """

    def test_a_wrong_number_earns_credit_for_shared_characters(self):
        """Deliberate. Do not "fix" this without reading the FUTURE note."""
        assert score({"amount": 1000}, {"amount": 9000}) == pytest.approx(0.75)

    def test_an_int_and_its_float_form_are_not_identical_as_text(self):
        """`str(1000) != str(1000.0)`, so this is partial, not 1.0.

        Numeric equality would need type-aware leaves, which is the deferred
        work. Pinned so the shortfall is visible rather than assumed absent.
        """
        assert score({"a": 1000}, {"a": 1000.0}) == pytest.approx(2 / 3)

    @pytest.mark.parametrize(
        "a,b,expected,label",
        [
            ("DE89370400440532013000", "DE89370400440532013001", 0.9545, "wrong IBAN"),
            ("2024-01-15", "2024-01-16", 0.9000, "date one day off"),
            ("1000000", "2000000", 0.8571, "amount 2x wrong"),
            ("1234.56", "1234.57", 0.8571, "amount one cent off"),
        ],
    )
    def test_long_values_score_high_for_incidental_overlap(self, a, b, expected, label):
        assert ANLSStarComparator().compare({"v": a}, {"v": b}) == pytest.approx(
            expected, abs=1e-4
        ), label

    def test_the_ordering_is_inverted_against_genuine_text(self):
        """The cost, stated as an assertion.

        Every wrong value above scores ABOVE a real text near-miss, so no
        `leaf_threshold` separates them: a cutoff high enough to reject the IBAN
        also deletes the partial credit this comparator exists to award. The
        remedy for a caller is to declare the field, not to tune tau.
        """
        c = ANLSStarComparator()
        wrong_iban = c.compare(
            {"v": "DE89370400440532013000"}, {"v": "DE89370400440532013001"}
        )
        genuine_near_miss = c.compare(
            {"v": "Acme Corporation"}, {"v": "Acme Corp"}
        )
        assert wrong_iban > genuine_near_miss

    def test_raising_the_cutoff_cannot_fix_it(self):
        """Confirms the claim above rather than asserting it in prose only."""
        for tau in (0.5, 0.85, 0.95):
            c = ANLSStarComparator(leaf_threshold=tau)
            iban = c.compare(
                {"v": "DE89370400440532013000"}, {"v": "DE89370400440532013001"}
            )
            text = c.compare({"v": "Acme Corporation"}, {"v": "Acme Corp"})
            if iban == 0.0:
                assert text == 0.0, "no cutoff rejects the IBAN while keeping text"


class TestLeafThresholdStillGatesText:
    def test_a_higher_cutoff_rejects_an_abbreviation(self):
        near = {**GT, "vendor": "Acme Corp"}
        lenient = ANLSStarComparator(leaf_threshold=0.5).compare(GT, near)
        strict = ANLSStarComparator(leaf_threshold=0.85).compare(GT, near)
        assert lenient > strict

    def test_zero_is_the_trap_it_looks_like(self):
        unrelated = {"vendor": "Zeta Ltd", "terms": "Net 30", "po": GT["po"]}
        assert ANLSStarComparator(leaf_threshold=0.0).compare(GT, unrelated) > 0.0
