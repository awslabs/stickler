"""A ``None`` element in a list is absent, wherever it sits.

The rule is "``None`` elements in a list" in
``docs/docs/Advanced/classification-logic.md``: a ``None`` element (or a float
NaN) is left out before matching, so it is never paired, counted or scored, and
a list of only ``None`` is an empty list. Present elements keep their original
indices in reports.

Before it, what a list held was read from its first element, so a leading
``None`` sent ``[None, Model]`` down the primitive path: the class gate and
field-by-field scoring were skipped, and a ``Cat`` scored against a ``Dog`` got
0.961 with no warning (#386). The Hungarian ``match_threshold`` came from the
parent rather than the element class (#322). And a ``None`` element counted as
an element, so two of them were a true positive and a missed row was a false
discovery.

Each order-dependence case runs in both orders and asserts the two agree,
because order independence is the requirement: pinning two constants
separately lets them drift apart together.
"""

import math
import warnings
from typing import Any, List, Optional, Union

import pytest
from pydantic import BaseModel

from stickler.comparators import StructuredModelComparator
from stickler.comparators.anls import ANLSStarComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.hungarian_helper import (
    HungarianHelper,
)
from stickler.structured_object_evaluator.models.null_helper import NullHelper
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)
from stickler.utils import deprecation as _deprecation


class Cat(BaseModel):
    name: Optional[str] = None


class Dog(BaseModel):
    name: Optional[str] = None


class Holder(StructuredModel):
    pet: Optional[Union[Cat, Dog]] = ComparableField(
        comparator=ANLSStarComparator(), default=None
    )


class Pets(StructuredModel):
    rows: List[Optional[Holder]] = ComparableField()


class Line(StructuredModel):
    sku: Optional[str] = ComparableField(default=None)
    tags: Optional[List[str]] = ComparableField(default=None)


class Lines(StructuredModel):
    rows: List[Optional[Line]] = ComparableField()


class Tags(StructuredModel):
    rows: List[Optional[str]] = ComparableField()


class Floats(StructuredModel):
    rows: List[Optional[float]] = ComparableField()


def _counts(result, field="rows"):
    counts = result["confusion_matrix"]["fields"][field]["overall"]
    return {k: counts[k] for k in ("tp", "fa", "fd", "fn", "tn")}


def _outcome(gt, pred):
    # One warning per field per process: clear it so each order can warn,
    # as the conftest fixture does between tests.
    _deprecation._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gt.compare_with(
            pred, include_confusion_matrix=True, document_non_matches=True
        )
        raw = gt.compare(pred)
    return {
        "score": pytest.approx(result["overall_score"]),
        "compare": pytest.approx(raw),
        "counts": _counts(result),
        "non_matches": len(result["non_matches"]),
        "class_warnings": sum(
            "Different classes are scored" in str(w.message) for w in caught
        ),
    }


def _both_orders(model, item_gt, item_pred):
    """The same pair with the None first and with it last."""
    return [
        (model(rows=[None, item_gt]), model(rows=[None, item_pred])),
        (model(rows=[item_gt, None]), model(rows=[item_pred, None])),
    ]


A, B, C = Line(sku="aaaa"), Line(sku="bbbb"), Line(sku="cccc")
_NONE = {"tp": 0, "fa": 0, "fd": 0, "fn": 0, "tn": 0}


class TestNoneElementsAreAbsent:
    """The table in the docs, for a list of models and a list of strings."""

    @pytest.mark.parametrize(
        "model, gt, pred, counts, score",
        [
            (Lines, [A, None], [A, None], {"tp": 1}, 1.0),
            (Lines, [A, B, C], [A, None, C], {"tp": 2, "fn": 1}, 2 / 3),
            (Lines, [A], [None], {"fn": 1}, 0.0),
            (Lines, [None], [A], {"fa": 1}, 0.0),
            (Lines, [None], [None], {"tn": 1}, 1.0),
            (Lines, [None, None], [None], {"tn": 1}, 1.0),
            (Tags, ["a", None], ["a", None], {"tp": 1}, 1.0),
            (Tags, ["a", "b"], ["a", None], {"tp": 1, "fn": 1}, 0.5),
            (Tags, ["a"], [None], {"fn": 1}, 0.0),
            (Tags, [None], ["a"], {"fa": 1}, 0.0),
            (Tags, [None], [None], {"tn": 1}, 1.0),
        ],
    )
    def test_the_documented_table(self, model, gt, pred, counts, score):
        outcome = _outcome(model(rows=gt), model(rows=pred))
        assert outcome["counts"] == {**_NONE, **counts}
        assert outcome["score"] == score
        assert outcome["compare"] == score

    def test_a_missed_row_is_a_false_negative_and_costs_recall(self):
        """As an FD, the missed row left recall at 1.0 under the default formula."""
        result = Lines(rows=[A, B, C]).compare_with(
            Lines(rows=[A, None, C]), include_confusion_matrix=True
        )
        assert result["confusion_matrix"]["overall"]["derived"][
            "cm_recall"
        ] == pytest.approx(2 / 3)

    def test_a_nan_element_is_absent_like_none(self):
        outcome = _outcome(Floats(rows=[1.0, math.nan]), Floats(rows=[1.0]))
        assert outcome["counts"] == {**_NONE, "tp": 1}

    def test_an_empty_string_element_is_still_a_value(self):
        outcome = _outcome(Tags(rows=["a", ""]), Tags(rows=["a"]))
        assert outcome["counts"] == {**_NONE, "tp": 1, "fn": 1}

    def test_an_any_sub_field_holding_a_list_drops_none(self):
        """A sub-field of a list element that is not a declared list reaches its
        comparator through the per-field metrics, not the list comparators."""

        class Loose(StructuredModel):
            sku: str = ComparableField()
            extra: Any = ComparableField(default=None)

        class Doc(StructuredModel):
            rows: List[Optional[Loose]] = ComparableField()

        result = Doc(rows=[Loose(sku="s", extra=[None, "a"])]).compare_with(
            Doc(rows=[Loose(sku="s", extra=["a"])]), include_confusion_matrix=True
        )
        extra = result["confusion_matrix"]["fields"]["rows"]["fields"]["extra"]
        assert {k: extra["overall"][k] for k in ("tp", "fd")} == {"tp": 1, "fd": 0}

    def test_a_list_of_none_in_an_any_field_is_empty(self):
        """`compare()` reaches an `Any` field's list without the dispatcher's
        null handling, so the rule has to hold there too."""

        class Loose(StructuredModel):
            rows: Any = ComparableField(comparator=StructuredModelComparator())

        assert Loose(rows=[None]).compare(Loose(rows=[])) == pytest.approx(1.0)


class TestReportsKeepOriginalIndices:
    def test_non_matches_and_field_comparisons_use_the_callers_indices(self):
        gt = Lines(rows=[None, A, B])
        pred = Lines(rows=[A, None])
        result = gt.compare_with(
            pred, document_non_matches=True, document_field_comparisons=True
        )
        assert [nm["field_path"] for nm in result["non_matches"]] == ["rows[2]"]
        # No row for either None element (gt rows[0], pred rows[1]).
        assert [
            (fc["expected_key"], fc["actual_key"]) for fc in result["field_comparisons"]
        ] == [
            ("rows[1].sku", "rows[0].sku"),
            ("rows[1].tags", "rows[0].tags"),
            ("rows[2]", None),
        ]

    @pytest.mark.parametrize(
        "gt, pred", [(None, [None]), ([None], None)], ids=["gt-none", "pred-none"]
    )
    def test_a_list_of_none_against_none_has_no_field_comparison(self, gt, pred):
        """`[None]` against `None` is a TN, as `[]` is, so it reports no row."""

        class Maybe(StructuredModel):
            rows: Optional[List[Optional[str]]] = ComparableField(default=None)

        result = Maybe(rows=gt).compare_with(
            Maybe(rows=pred),
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )
        assert result["field_comparisons"] == []
        assert _counts(result)["tn"] == 1

    def test_the_matcher_maps_back_to_original_indices(self):
        info = HungarianHelper().get_complete_matching_info(
            [None, Line(sku="a")], [Line(sku="a"), None, Line(sku="z")]
        )
        assert [(i, j) for i, j, _ in info["matched_pairs"]] == [(1, 0)]
        assert info["unmatched_gt_indices"] == []
        assert info["unmatched_pred_indices"] == [2]

    def test_evaluator_format_matches_the_list_without_nones(self):
        with_nones = Lines(rows=[None, A, B]).compare_with(
            Lines(rows=[A, None]), evaluator_format=True
        )
        without = Lines(rows=[A, B]).compare_with(
            Lines(rows=[A]), evaluator_format=True
        )
        assert with_nones["overall"] == pytest.approx(without["overall"])


class TestTheClassGateRunsWhateverTheOrder:
    def test_a_cat_against_a_dog_is_a_false_discovery_in_both_orders(self):
        first, last = (
            _outcome(gt, pred)
            for gt, pred in _both_orders(
                Pets, Holder(pet=Cat(name="x")), Holder(pet=Dog(name="x"))
            )
        )
        assert first == last
        assert first["counts"] == {**_NONE, "fd": 1}
        assert first["score"] == 0.0
        assert first["class_warnings"] == 1

    @pytest.mark.filterwarnings("ignore:Field 'pet' compared a Cat against a Dog")
    def test_compare_gates_the_class_in_both_orders(self):
        """compare() is the cost function a parent list matches on. With the
        None first it scored the Cat against the Dog 0.95."""
        for gt, pred in _both_orders(
            Pets, Holder(pet=Cat(name="x")), Holder(pet=Dog(name="x"))
        ):
            assert gt.compare(pred) == pytest.approx(0.0)


class TestOrderDoesNotChangeTheResult:
    def test_identical_lists_score_one_in_both_orders(self):
        """With the None first, an identical list scored 0.974 rather than 1.0."""
        for gt, pred in _both_orders(
            Lines, Line(sku="a", tags=["x", "y"]), Line(sku="a", tags=["y", "x"])
        ):
            outcome = _outcome(gt, pred)
            assert outcome["score"] == 1.0
            assert outcome["counts"] == {**_NONE, "tp": 1}
            assert outcome["non_matches"] == 0

    def test_a_nested_list_field_differs_the_same_in_both_orders(self):
        """The simple-list check on a sub-field read `gt_list[0]` as well."""
        first, last = (
            _outcome(gt, pred)
            for gt, pred in _both_orders(
                Lines, Line(sku="a", tags=["x"]), Line(sku="a", tags=["z"])
            )
        )
        assert first == last
        assert first["counts"] == {**_NONE, "fd": 1}

    def test_evaluator_format_accepts_a_leading_none(self):
        gt, pred = Lines(rows=[None, Line(sku="a")]), Lines(rows=[None, Line(sku="a")])
        assert isinstance(gt.compare_with(pred, evaluator_format=True), dict)


class TestTheElementThresholdIsReadPastANone:
    """#322: the gate came from the parent when the list started with None.

    Only ``none-first`` exercises #322. ``none-last`` also failed on dev, but
    because the two ``None``s counted as a TP, not because of the threshold."""

    class Item(StructuredModel):
        match_threshold = 0.95
        sku: Optional[str] = ComparableField(threshold=0.1, default=None)

    class Parent(StructuredModel):
        match_threshold = 0.1
        items: Optional[
            List[Optional["TestTheElementThresholdIsReadPastANone.Item"]]
        ] = ComparableField(default=None)

    @pytest.mark.parametrize(
        "gt, pred",
        [
            ([None, "aaaa"], [None, "azzz"]),
            (["aaaa", None], ["azzz", None]),
        ],
        ids=["none-first", "none-last"],
    )
    def test_the_element_class_threshold_applies(self, gt, pred):
        def build(skus):
            return self.Parent(
                items=[None if s is None else self.Item(sku=s) for s in skus]
            )

        result = build(gt).compare_with(build(pred), include_confusion_matrix=True)
        # "aaaa" vs "azzz" is well under the element class's 0.95, so it is a
        # false discovery; under the parent's 0.1 it was a true positive.
        assert _counts(result, "items") == {**_NONE, "fd": 1}


class TestUnchangedShapes:
    def test_a_mixed_list_still_takes_its_first_element(self):
        """Only absent elements are skipped; a list led by a non-model value is
        unchanged."""

        class Mixed(StructuredModel):
            rows: Any = ComparableField(comparator=StructuredModelComparator())

        result = Mixed(rows=["s", Line(sku="a")]).compare_with(
            Mixed(rows=["s", Line(sku="a")])
        )
        assert result["overall_score"] == pytest.approx(1.0)


class TestTheHelpers:
    def test_first_present_skips_absent_and_falls_back_to_later_sequences(self):
        assert NullHelper.first_present([None, "a"], ["b"]) == "a"
        assert NullHelper.first_present([None], [math.nan, "b"]) == "b"
        assert NullHelper.first_present([], None, [None]) is None
        assert NullHelper.first_present([0, None]) == 0

    def test_present_and_present_indices(self):
        items = [None, "a", math.nan, "", 0]
        assert NullHelper.present(items) == ["a", "", 0]
        assert NullHelper.present_indices(items) == [1, 3, 4]
        assert NullHelper.present(None) == []

    def test_a_list_of_only_absent_elements_is_null(self):
        assert NullHelper.is_effectively_null_for_lists([None, math.nan])
        assert not NullHelper.is_effectively_null_for_lists([None, ""])
