"""One threshold test at every classification gate (#417).

Similarity scores are floats, so a score can print as its threshold and still
sit just under it: ``0.7 + 0.1`` is ``0.7999999999999999``. A primitive list
element was classified with ``ThresholdHelper.is_above_threshold``, which
accepts a score up to ``1e-10`` under the threshold, while a field, a nested
model and a ``List[StructuredModel]`` pair were classified with a bare ``>=``.
The same score was then a TP in one place and an FD in another.

Each gate below is checked at four scores: one float rounding error under the
threshold, exactly at it, just outside the tolerance, and clearly under it.
"""

from typing import List, Optional

import pytest

import stickler
from stickler import (
    ComparableField,
    ExactComparator,
    LevenshteinComparator,
    StructuredModel,
)
from stickler.algorithms.hungarian import HungarianMatcher
from stickler.comparators.base import BaseComparator
from stickler.structured_object_evaluator.models.comparison_info import (
    ComparableFieldConfig,
    ComparisonInfo,
)
from stickler.structured_object_evaluator.models.metrics_helper import MetricsHelper
from stickler.structured_object_evaluator.models.threshold_helper import (
    ThresholdHelper,
)

THRESHOLD = 0.8

# 0.7 + 0.1 is 0.7999999999999999 in IEEE 754 doubles, 1.1e-16 under 0.8.
NEAR = 0.7 + 0.1

# Just outside the 1e-10 tolerance, so still under the threshold.
OUTSIDE = THRESHOLD - 1e-9

# (score, whether it meets THRESHOLD)
CASES = [
    pytest.param(NEAR, True, id="float-error-under"),
    pytest.param(THRESHOLD, True, id="at-threshold"),
    pytest.param(OUTSIDE, False, id="outside-tolerance"),
    pytest.param(0.79, False, id="clearly-under"),
]


class _FixedScore(BaseComparator):
    """Scores equal values 1.0 and every other pair at ``score``."""

    def __init__(self, score: float, threshold: Optional[float] = None):
        super().__init__(threshold=threshold)
        self.score = score

    def _compare(self, str1, str2) -> float:
        return 1.0 if str1 == str2 else self.score


def _counts(node: dict) -> tuple:
    overall = node["overall"]
    return overall["tp"], overall["fd"]


def _expected_counts(matches: bool) -> tuple:
    return (1, 0) if matches else (0, 1)


def _full_result(gt: StructuredModel, pred: StructuredModel) -> dict:
    return gt.compare_with(
        pred,
        include_confusion_matrix=True,
        document_non_matches=True,
        document_field_comparisons=True,
    )


def _non_match_paths(result: dict) -> list:
    return [entry["field_path"] for entry in result["non_matches"]]


def _comparison_rows(result: dict) -> dict:
    return {row["expected_key"]: row["match"] for row in result["field_comparisons"]}


def _flat_model(score: float):
    class Flat(StructuredModel):
        name: str = ComparableField(comparator=_FixedScore(score), threshold=THRESHOLD)

    return Flat


def _list_item_models(score: float):
    """A list item whose ``name`` sub-field scores ``score``.

    ``key`` matches exactly and ``match_threshold`` is low, so the pair always
    matches as an object and only the sub-field sits at the boundary.
    """

    class Item(StructuredModel):
        key: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
        name: str = ComparableField(comparator=_FixedScore(score), threshold=THRESHOLD)
        match_threshold = 0.5

    class Holder(StructuredModel):
        items: List[Item] = ComparableField()

    return Item, Holder


def test_the_premise():
    """Guard the inputs, so the tests below mean what they say."""
    assert NEAR < THRESHOLD
    assert ThresholdHelper.is_above_threshold(NEAR, THRESHOLD)
    assert not ThresholdHelper.is_above_threshold(OUTSIDE, THRESHOLD)


class TestFieldGate:
    """A top-level primitive field, scored by ``FieldComparator``."""

    @pytest.mark.parametrize("score,matches", CASES)
    def test_counts(self, score, matches):
        Flat = _flat_model(score)
        result = _full_result(Flat(name="a"), Flat(name="b"))
        assert _counts(result["confusion_matrix"]["fields"]["name"]) == (
            _expected_counts(matches)
        )

    @pytest.mark.parametrize("score,matches", CASES)
    def test_only_a_miss_is_clipped(self, score, matches):
        Flat = _flat_model(score)
        result = _full_result(Flat(name="a"), Flat(name="b"))
        assert result["field_scores"]["name"] == (score if matches else 0.0)

    @pytest.mark.parametrize("score,matches", CASES)
    def test_reports_agree_with_the_counts(self, score, matches):
        Flat = _flat_model(score)
        result = _full_result(Flat(name="a"), Flat(name="b"))
        assert _non_match_paths(result) == ([] if matches else ["name"])
        assert _comparison_rows(result) == {"name": matches}


class TestNestedModelGate:
    """A nested ``StructuredModel`` field, classified on its object score."""

    @pytest.mark.parametrize("score,matches", CASES)
    def test_counts_and_score(self, score, matches):
        Inner = _flat_model(score)

        class Outer(StructuredModel):
            inner: Inner = ComparableField(threshold=THRESHOLD)

        gt, pred = Outer(inner=Inner(name="a")), Outer(inner=Inner(name="b"))
        assert gt.inner.compare(pred.inner) == score

        result = _full_result(gt, pred)
        assert _counts(result["confusion_matrix"]["fields"]["inner"]) == (
            _expected_counts(matches)
        )
        assert result["field_scores"]["inner"] == (score if matches else 0.0)


class TestListObjectGate:
    """A ``List[StructuredModel]`` pair against the element's ``match_threshold``."""

    @staticmethod
    def _models(score: float):
        class Item(StructuredModel):
            name: str = ComparableField(comparator=_FixedScore(score), threshold=0.5)
            match_threshold = THRESHOLD

        class Holder(StructuredModel):
            items: List[Item] = ComparableField()

        return Item, Holder

    @pytest.mark.parametrize("score,matches", CASES)
    def test_counts_and_recursion(self, score, matches):
        Item, Holder = self._models(score)
        result = _full_result(
            Holder(items=[Item(name="a")]), Holder(items=[Item(name="b")])
        )
        items = result["confusion_matrix"]["fields"]["items"]
        assert _counts(items) == _expected_counts(matches)
        # A matched pair is descended into; a pair below the gate is not.
        assert ("name" in items["fields"]) is matches

    @pytest.mark.parametrize("score,matches", CASES)
    def test_reports_agree_with_the_counts(self, score, matches):
        Item, Holder = self._models(score)
        result = _full_result(
            Holder(items=[Item(name="a")]), Holder(items=[Item(name="b")])
        )
        assert _non_match_paths(result) == ([] if matches else ["items[0]"])
        if matches:
            assert _comparison_rows(result) == {"items[0].name": True}
        else:
            assert _comparison_rows(result) == {"items[0]": False}

    def test_the_documented_worked_example(self):
        """The scenarios and result block in ``threshold-gated-evaluation.md``."""

        class Product(StructuredModel):
            product_id: str = ComparableField(
                comparator=ExactComparator(), threshold=1.0, weight=3.0
            )
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
            )
            price: float = ComparableField(threshold=0.9, weight=1.0)
            match_threshold = 0.8

        class Order(StructuredModel):
            order_id: str = ComparableField(
                comparator=ExactComparator(), threshold=1.0, weight=2.0
            )
            products: List[Product] = ComparableField(weight=3.0)

        gt = [
            Product(product_id="PROD-001", name="Laptop", price=999.99),
            Product(product_id="PROD-002", name="Mouse", price=29.99),
            Product(product_id="PROD-003", name="Cable", price=14.99),
        ]
        pred = [
            Product(product_id="PROD-001", name="Laptop Computer", price=999.99),
            Product(product_id="PROD-002", name="Different Product", price=99.99),
            Product(product_id="PROD-004", name="New Product", price=19.99),
        ]
        assert gt[0].compare(pred[0]) == NEAR

        def run(gt_items, pred_items):
            return _full_result(
                Order(order_id="ORD-1", products=gt_items),
                Order(order_id="ORD-1", products=pred_items),
            )

        result = run(gt, pred)
        products = result["confusion_matrix"]["fields"]["products"]
        overall = products["overall"]
        assert [overall[key] for key in ("tp", "fd", "fn", "fa")] == [1, 2, 0, 0]
        assert overall["derived"]["cm_precision"] == pytest.approx(1 / 3)
        assert overall["derived"]["cm_recall"] == 1.0
        assert overall["derived"]["cm_f1"] == pytest.approx(0.5)
        assert _counts(products["fields"]["product_id"]) == (1, 0)
        assert _counts(products["fields"]["name"]) == (0, 1)
        assert _counts(products["fields"]["price"]) == (1, 0)
        assert [
            (entry["field_path"], entry["non_match_type"], round(entry["similarity_score"], 4))
            for entry in result["non_matches"]
        ] == [
            ("products[0].name", "false_discovery", 0.4),
            ("products[1]", "false_discovery", 0.6725),
            ("products[2]", "false_discovery", 0.1333),
        ]

        # Only lists of different lengths leave an item unmatched.
        for gt_items, pred_items, kind, counts in (
            (gt, pred[:2], "false_negative", [1, 1, 1, 0]),
            (gt[:2], pred, "false_alarm", [1, 1, 0, 1]),
        ):
            result = run(gt_items, pred_items)
            overall = result["confusion_matrix"]["fields"]["products"]["overall"]
            assert [overall[key] for key in ("tp", "fd", "fn", "fa")] == counts
            last = result["non_matches"][-1]
            assert (last["field_path"], last["non_match_type"]) == ("products[2]", kind)


class TestListSubFieldGate:
    """A primitive sub-field of a matched ``List[StructuredModel]`` pair."""

    @pytest.mark.parametrize("score,matches", CASES)
    def test_counts(self, score, matches):
        Item, Holder = _list_item_models(score)
        result = _full_result(
            Holder(items=[Item(key="k", name="a")]),
            Holder(items=[Item(key="k", name="b")]),
        )
        name = result["confusion_matrix"]["fields"]["items"]["fields"]["name"]
        assert _counts(name) == _expected_counts(matches)

    @pytest.mark.parametrize("score,matches", CASES)
    def test_reports_agree_with_the_counts(self, score, matches):
        Item, Holder = _list_item_models(score)
        result = _full_result(
            Holder(items=[Item(key="k", name="a")]),
            Holder(items=[Item(key="k", name="b")]),
        )
        assert _non_match_paths(result) == ([] if matches else ["items[0].name"])
        assert _comparison_rows(result) == {
            "items[0].key": True,
            "items[0].name": matches,
        }

    @pytest.mark.parametrize("score,matches", CASES)
    def test_agrees_with_a_field_and_a_list_element(self, score, matches):
        """The same score and threshold give one verdict wherever they meet."""
        Item, Holder = _list_item_models(score)
        sub_field = _full_result(
            Holder(items=[Item(key="k", name="a")]),
            Holder(items=[Item(key="k", name="b")]),
        )["confusion_matrix"]["fields"]["items"]["fields"]["name"]

        Flat = _flat_model(score)
        field = _full_result(Flat(name="a"), Flat(name="b"))["confusion_matrix"][
            "fields"
        ]["name"]

        class Names(StructuredModel):
            names: List[str] = ComparableField(
                comparator=_FixedScore(score), threshold=THRESHOLD
            )

        element = _full_result(Names(names=["a"]), Names(names=["b"]))[
            "confusion_matrix"
        ]["fields"]["names"]

        expected = _expected_counts(matches)
        assert _counts(sub_field) == _counts(field) == _counts(element) == expected


class TestObjectVerdict:
    """``EvalResult.matched`` reads the same ``match_threshold`` as list pairing."""

    @pytest.mark.parametrize("score,matches", CASES)
    def test_matched(self, score, matches):
        class Flat(StructuredModel):
            name: str = ComparableField(comparator=_FixedScore(score), threshold=0.5)

        result = stickler.evaluate(
            Flat(name="a"), Flat(name="b"), match_threshold=THRESHOLD
        )
        assert result.overall_score == score
        assert result.matched is matches


class TestStandaloneGates:
    @pytest.mark.parametrize("score,matches", CASES)
    def test_hungarian_matcher_counts(self, score, matches):
        matcher = HungarianMatcher(_FixedScore(score), match_threshold=THRESHOLD)
        metrics = matcher.calculate_metrics(["a"], ["b"])
        assert (metrics["tp"], metrics["fd"]) == _expected_counts(matches)

    @pytest.mark.parametrize("score,matches", CASES)
    def test_binary_metrics(self, score, matches):
        """The fallback ``evaluator_format=True`` uses without a confusion matrix."""
        metrics = MetricsHelper().convert_score_to_binary_metrics(score, THRESHOLD)
        assert metrics["recall"] == (1.0 if matches else 0.0)

    @pytest.mark.parametrize("score,matches", CASES)
    def test_comparator_and_field_config(self, score, matches):
        """``binary_compare`` and the field configs' clipping ``compare``."""
        comparator = _FixedScore(score, threshold=THRESHOLD)
        assert comparator.binary_compare("a", "b") == ((1, 0) if matches else (0, 1))

        kept = score if matches else 0.0
        for config_cls in (ComparisonInfo, ComparableFieldConfig):
            config = config_cls(_FixedScore(score), threshold=THRESHOLD)
            assert config.compare("a", "b") == kept
