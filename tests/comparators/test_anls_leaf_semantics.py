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

import datetime
import pathlib
import uuid
import warnings
from decimal import Decimal
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
            # a 22-character bank account number, one character wrong
            ("DE89370400440532013000", "DE89370400440532013001", 0.9545, "account no."),
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
        `leaf_threshold` separates them: a cutoff high enough to reject the account number
        also deletes the partial credit this comparator exists to award. The
        remedy for a caller is to declare the field, not to tune tau.
        """
        c = ANLSStarComparator()
        wrong_account_number = c.compare(
            {"v": "DE89370400440532013000"}, {"v": "DE89370400440532013001"}
        )
        genuine_near_miss = c.compare(
            {"v": "Acme Corporation"}, {"v": "Acme Corp"}
        )
        assert wrong_account_number > genuine_near_miss

    def test_raising_the_cutoff_cannot_fix_it(self):
        """Confirms the claim above rather than asserting it in prose only."""
        for tau in (0.5, 0.85, 0.95):
            c = ANLSStarComparator(leaf_threshold=tau)
            account_number = c.compare(
                {"v": "DE89370400440532013000"}, {"v": "DE89370400440532013001"}
            )
            text = c.compare({"v": "Acme Corporation"}, {"v": "Acme Corp"})
            if account_number == 0.0:
                assert text == 0.0, "no cutoff rejects the account number while keeping text"


class TestLeafThresholdStillGatesText:
    def test_a_higher_cutoff_rejects_an_abbreviation(self):
        near = {**GT, "vendor": "Acme Corp"}
        lenient = ANLSStarComparator(leaf_threshold=0.5).compare(GT, near)
        strict = ANLSStarComparator(leaf_threshold=0.85).compare(GT, near)
        assert lenient > strict

    def test_zero_is_the_trap_it_looks_like(self):
        unrelated = {"vendor": "Zeta Ltd", "terms": "Net 30", "po": GT["po"]}
        assert ANLSStarComparator(leaf_threshold=0.0).compare(GT, unrelated) > 0.0


class TestValuesWithNoJsonForm:
    """ANLS* scores JSON values. An arbitrary Python object is not one.

    Such a value used to be handed to `str()`, which for an object without a
    value-based `__repr__` yields `<module.Class object at 0xADDRESS>`. Two of
    those share a long prefix, so edit distance was comparing memory addresses.
    """

    def test_two_unrelated_objects_do_not_match(self):
        class Plain:
            pass

        assert ANLSStarComparator().compare(
            {"o": Plain()}, {"o": Plain()}
        ) == pytest.approx(0.0), "was 0.8684, from shared address text"

    def test_a_longer_class_name_does_not_raise_the_score(self):
        """The old failure got WORSE the more specific your code was: a longer
        module path meant a longer shared prefix meant a higher false match."""

        class A:
            pass

        class AVeryDeeplyNamespacedConfigurationValueObject:
            pass

        c = ANLSStarComparator()
        short = c.compare({"o": A()}, {"o": A()})
        long = c.compare(
            {"o": AVeryDeeplyNamespacedConfigurationValueObject()},
            {"o": AVeryDeeplyNamespacedConfigurationValueObject()},
        )
        assert short == long == pytest.approx(0.0), "was 0.8684 vs 0.9315"

    def test_equal_objects_are_also_refused_not_scored(self):
        """Deliberate. Out of domain is out of domain; we do not claim to score
        it, so equality is not consulted. Under-crediting here is the safe
        direction for a metric."""

        class Eq:
            def __init__(self, v):
                self.v = v

            def __eq__(self, other):
                return isinstance(other, Eq) and self.v == other.v

        assert Eq(1) == Eq(1)
        assert ANLSStarComparator().compare({"o": Eq(1)}, {"o": Eq(1)}) == pytest.approx(
            0.0
        )

    def test_only_the_unscoreable_key_is_refused(self):
        """A neighbouring key that IS scoreable keeps its credit."""

        class Plain:
            pass

        score = ANLSStarComparator().compare(
            {"good": "x", "weird": Plain()}, {"good": "x", "weird": Plain()}
        )
        assert score == pytest.approx(0.5), "one of two keys scoreable and equal"

    @pytest.mark.parametrize(
        "value",
        [
            datetime.date(2024, 1, 1),
            datetime.datetime(2024, 1, 1, 9, 30),
            datetime.time(12, 30),
            Decimal("10.50"),
            uuid.UUID(int=1),
            pathlib.Path("/tmp/x"),
            b"bytes",
            {1, 2},
        ],
    )
    def test_types_with_a_json_form_are_still_scored(self, value):
        """The refusal must not widen. Every type here is serialised natively by
        pydantic and never reaches the fallback, so identity must hold."""
        assert ANLSStarComparator().compare({"v": value}, {"v": value}) == pytest.approx(
            1.0
        )

    def test_even_an_object_with_a_good_repr_is_refused(self):
        """The refusal keys on JSON-representability, not on repr quality.

        `WithRepr(1)` has a perfectly value-based `__repr__`, and two of them
        would have scored 1.0 as text. It is still refused, because it still has
        no JSON form, and that is what puts it outside this comparator's domain.

        Uniform on purpose. Sniffing whether a repr "looks value-based" would be
        a heuristic about someone else's code, and it would make the domain
        depend on how a class happens to be written."""

        class WithRepr:
            def __init__(self, v):
                self.v = v

            def __repr__(self):
                return f"WithRepr({self.v})"

        c = ANLSStarComparator()
        assert c.compare({"o": WithRepr(1)}, {"o": WithRepr(1)}) == pytest.approx(0.0)


class TestTheRefusalIsAnnounced:
    def test_an_unscoreable_value_warns(self):
        """A silent 0.0 is indistinguishable from a wrong extraction, so the
        caller is told which one happened."""

        class Announced:
            pass

        with pytest.warns(UserWarning, match="no JSON representation"):
            ANLSStarComparator().compare({"o": Announced()}, {"o": Announced()})

    def test_it_warns_once_per_type_not_once_per_document(self):
        """A bulk run over a corpus must not emit one warning per document."""

        class OnlyOnce:
            pass

        c = ANLSStarComparator()
        with pytest.warns(UserWarning):
            c.compare({"o": OnlyOnce()}, {"o": OnlyOnce()})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c.compare({"o": OnlyOnce()}, {"o": OnlyOnce()})

    def test_a_json_type_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ANLSStarComparator().compare(
                {"v": Decimal("1.5")}, {"v": Decimal("1.5")}
            ) == pytest.approx(1.0)


class TestNumericallyEqualValuesCanScoreBelowOne:
    """The other direction of uniform text leaves, and the likelier one to bite.

    A ground truth loaded from a database as an integer against a prediction
    parsed from JSON as a float is a perfect extraction, and text comparison
    scores it as a miss. Pinned as documented behaviour so nobody "fixes" it into
    the type-aware leaves that were deliberately reverted; see the FUTURE note in
    `ANLSLeaf.nls_list`.
    """

    @pytest.mark.parametrize(
        "gt,pred,expected",
        [
            (5, 5.0, 0.0),
            (0, 0.0, 0.0),
            (1000, 1000.0, 2 / 3),
            (Decimal("10.50"), Decimal("10.5"), 0.8),
        ],
    )
    def test_equal_numbers_of_different_spelling(self, gt, pred, expected):
        assert ANLSStarComparator().compare({"v": gt}, {"v": pred}) == pytest.approx(
            expected, abs=1e-4
        )

    def test_the_documented_remedy_actually_works(self):
        """The docs point at NumericComparator, so check that it does the job."""
        from stickler.comparators.numeric import NumericComparator

        assert NumericComparator().compare(5, 5.0) == pytest.approx(1.0)
        assert NumericComparator().compare(1000, 1000.0) == pytest.approx(1.0)
