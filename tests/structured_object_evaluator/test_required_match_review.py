"""Review follow-ups for ``required_match``: reporting paths, symmetry and the matcher.

``test_required_match.py`` pins the counts through ``compare_with``. These
tests cover the other consumers of a vetoed pair (``non_matches``, the
evaluator format, the bulk evaluator), a veto across two classes, the nested
refusal, and ``HungarianMatcher`` with exclusions against a brute force.
"""

import itertools
import random
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import List, Literal, Optional, Union

import pytest

from stickler.algorithms.hungarian import HungarianMatcher
from stickler.comparators import NumericComparator
from stickler.structured_object_evaluator import ComparableField, StructuredModel
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

from .test_required_match import (
    Detection,
    Frame,
    LooseDetection,
    detection,
    exact,
)

TWO_OBJECTS_ONE_VETOED = (
    [detection(), detection(label="car", x=5.0, y=5.0)],
    [detection(label="bike"), detection(label="car", x=5.0, y=5.0)],
)


def frames(frame_cls=Frame):
    gt, pred = TWO_OBJECTS_ONE_VETOED
    return (
        frame_cls.from_json({"detections": gt}),
        frame_cls.from_json({"detections": pred}),
    )


# one veto and one match, through every reporting path


def test_one_veto_and_one_match_score_half_and_agree_across_entry_points():
    gt, pred = frames()

    result = gt.compare_with(pred)

    assert result["overall_score"] == 0.5
    assert gt.compare(pred) == result["overall_score"]


def test_one_veto_is_a_false_negative_and_a_false_alarm_in_non_matches():
    gt, pred = frames()

    result = gt.compare_with(pred, document_non_matches=True)

    found = {
        (record["field_path"], record["non_match_type"].value)
        for record in result["non_matches"]
    }
    assert found == {
        ("detections[0]", "false_negative"),
        ("detections[0]", "false_alarm"),
    }


def test_vetoed_objects_are_absent_from_the_aggregate_but_counted_in_the_list():
    gt, pred = frames()

    matrix = gt.compare_with(pred, include_confusion_matrix=True)["confusion_matrix"]

    list_overall = matrix["fields"]["detections"]["overall"]
    assert (list_overall["tp"], list_overall["fn"], list_overall["fa"]) == (1, 1, 1)
    assert (matrix["aggregate"]["fn"], matrix["aggregate"]["fa"]) == (0, 0)


def test_evaluator_format_reports_the_list_precision_and_recall_of_one_half():
    gt, pred = frames()

    result = gt.compare_with(pred, evaluator_format=True, document_non_matches=True)

    overall = result["fields"]["detections"]["overall"]
    assert overall["precision"] == overall["recall"] == 0.5
    for value in overall.values():
        assert 0.0 <= value <= 1.0


def test_bulk_evaluator_counts_a_veto_as_one_fn_and_one_fa_and_no_fd():
    gt, pred = frames()
    evaluator = BulkStructuredModelEvaluator(target_schema=Frame)

    evaluator.update(gt, pred)
    metrics = evaluator.compute()

    detections = metrics.field_metrics["detections"]
    assert (detections["fn"], detections["fa"], detections["fd"]) == (1, 1, 0)


def test_every_pair_vetoed_in_a_three_by_two_list_keeps_the_count_identities():
    gt = [detection(label=name) for name in ("a", "b", "c")]
    pred = [detection(label="x"), detection(label="y")]

    matrix = Frame.from_json({"detections": gt}).compare_with(
        Frame.from_json({"detections": pred}), include_confusion_matrix=True
    )["confusion_matrix"]["fields"]["detections"]["overall"]

    assert (matrix["tp"], matrix["fd"], matrix["fn"], matrix["fa"]) == (0, 0, 3, 2)
    assert matrix["tp"] + matrix["fn"] + matrix["fd"] == 3
    assert matrix["tp"] + matrix["fp"] == 2


# null handling on a required field


@pytest.mark.parametrize(
    ("gt_label", "pred_label", "vetoed"),
    [
        ("", "person", True),
        (None, "person", True),
        ("person", "", True),
        ("", None, False),
    ],
)
def test_null_on_one_side_vetoes_and_null_on_both_passes(gt_label, pred_label, vetoed):
    class Row(StructuredModel):
        label: Optional[str] = exact(required_match=True)
        note: str = exact()

    gt = Row(label=gt_label, note="n")
    pred = Row(label=pred_label, note="n")

    assert gt.fails_required_match(pred) is vetoed


def test_nan_on_one_side_vetoes():
    class Row(StructuredModel):
        value: Optional[float] = exact(required_match=True)

    assert Row(value=float("nan")).fails_required_match(Row(value=1.0)) is True


# two classes in one list


class LooseOther(StructuredModel):
    label: str = exact()
    x: Optional[float] = exact()
    y: float = exact()
    note: str = exact()


class NoLabel(StructuredModel):
    x: Optional[float] = exact()
    y: float = exact()
    note: str = exact()


@pytest.mark.parametrize("other_cls", [LooseDetection, LooseOther])
def test_a_veto_holds_whichever_class_is_ground_truth(other_cls):
    required = Detection(**detection(label="person"))
    loose = other_cls(**detection(label="car"))

    assert required.fails_required_match(loose) is True
    assert loose.fails_required_match(required) is True


def test_a_field_required_on_one_class_and_absent_on_the_other_vetoes_both_ways():
    required = Detection(**detection())
    absent = NoLabel(x=1.0, y=1.0, note="n")

    assert required.fails_required_match(absent) is True
    assert absent.fails_required_match(required) is True


@pytest.mark.parametrize("other", [None, {}, "text", 3, []])
def test_a_non_model_partner_is_a_veto_and_does_not_raise(other):
    assert Detection(**detection()).fails_required_match(other) is True


# the direction a required field is scored in


class Reading(StructuredModel):
    match_threshold = 0.5
    x: float = ComparableField(
        comparator=NumericComparator(relative_tolerance=0.1),
        threshold=1.0,
        required_match=True,
    )
    name: str = exact()


class Readings(StructuredModel):
    items: List[Reading] = ComparableField()


def test_a_required_field_is_scored_ground_truth_first_as_the_field_is():
    # 90 is within 10% of 100, but 100 is not within 10% of 90. The field
    # itself is a TP, so the pair must not be vetoed by the reverse direction.
    gt, pred = Reading(x=100.0, name="a"), Reading(x=90.0, name="a")

    field = gt.compare_with(pred, include_confusion_matrix=True)["confusion_matrix"]
    items = Readings(items=[gt]).compare_with(
        Readings(items=[pred]), include_confusion_matrix=True
    )["confusion_matrix"]["fields"]["items"]["overall"]

    assert field["fields"]["x"]["overall"]["tp"] == 1
    assert gt.fails_required_match(pred) is False
    assert (items["tp"], items["fn"], items["fa"]) == (1, 0, 0)


# the nested refusal and the rejected types


class Position(StructuredModel):
    x: float = exact(required_match=True)


def test_a_nested_model_with_a_required_field_is_refused():
    with pytest.raises(ValueError, match="nested Position"):

        class Holder(StructuredModel):
            position: Position = ComparableField()


def test_an_optional_nested_model_with_a_required_field_is_refused():
    with pytest.raises(ValueError, match="nested Position"):

        class Holder(StructuredModel):
            position: Optional[Position] = ComparableField()


def test_a_list_of_models_with_a_required_field_is_allowed():
    class Holder(StructuredModel):
        positions: List[Position] = ComparableField()

    assert Holder._required_match_fields == ()


class Color(Enum):
    RED = "red"


@pytest.mark.parametrize(
    "annotation",
    [Literal["a", "b"], Color, date, Decimal, Union[int, str], List[str]],
)
def test_a_non_primitive_required_annotation_is_refused(annotation):
    with pytest.raises(ValueError, match="required_match"):

        class Row(StructuredModel):
            value: annotation = ComparableField(required_match=True)


@pytest.mark.filterwarnings("ignore:.*threshold=0.0:UserWarning")
def test_a_threshold_of_zero_never_vetoes_two_present_values():
    class Row(StructuredModel):
        label: str = ComparableField(threshold=0.0, required_match=True)

    assert Row(label="a").fails_required_match(Row(label="zzzzzz")) is False


# the matcher on its own


def best_assignment(similarity, excluded):
    """Most allowed pairs first, then the highest total similarity."""
    m, n = len(similarity), len(similarity[0])
    best = (-1, float("-inf"))
    for k in range(min(m, n), -1, -1):
        for rows in itertools.combinations(range(m), k):
            for cols in itertools.permutations(range(n), k):
                pairs = list(zip(rows, cols))
                if any(pair in excluded for pair in pairs):
                    continue
                best = max(best, (k, sum(similarity[i][j] for i, j in pairs)))
    return best


def matcher_for(similarity, excluded):
    # calculate_metrics prepares the lists and may turn an int into text, so
    # the items are read back with int()
    matcher = HungarianMatcher(
        comparator=None, is_pair_excluded=lambda i, j: (int(i), int(j)) in excluded
    )
    assert hasattr(HungarianMatcher, "_score"), "the helper patches _score"
    matcher._score = lambda i, j: similarity[int(i)][int(j)]
    return matcher


@pytest.mark.parametrize("low, high", [(0.0, 1.0), (-3.0, 5.0)])
def test_matcher_keeps_the_most_allowed_pairs_then_the_best_total(low, high):
    rng = random.Random(539)
    for _ in range(1500):
        m, n = rng.randint(3, 5), rng.randint(3, 5)
        similarity = [[rng.uniform(low, high) for _ in range(n)] for _ in range(m)]
        excluded = {(i, j) for i in range(m) for j in range(n) if rng.random() < 0.4}

        pairs, _ = matcher_for(similarity, excluded).match(
            list(range(m)), list(range(n))
        )

        expected_count, expected_total = best_assignment(similarity, excluded)
        assert len(pairs) == expected_count
        assert sum(similarity[i][j] for i, j in pairs) == pytest.approx(expected_total)


@pytest.mark.parametrize("m, n", [(2, 2), (2, 3), (3, 2), (1, 1)])
def test_matcher_with_every_pair_excluded_pairs_nothing(m, n):
    similarity = [[0.9] * n for _ in range(m)]
    matcher = matcher_for(similarity, {(i, j) for i in range(m) for j in range(n)})

    metrics = matcher.calculate_metrics(list(range(m)), list(range(n)))

    assert metrics["matched_pairs"] == []
    assert (metrics["tp"], metrics["fd"], metrics["fn"], metrics["fa"]) == (0, 0, m, n)
