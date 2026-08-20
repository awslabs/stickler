"""
Regression tests for simple list fields within structured lists.

Validates that List[str] (and similar primitive list types) inside a
List[StructuredModel] are compared element-by-element using Hungarian matching,
not treated as atomic primitive values.

See: https://github.com/awslabs/stickler/issues/33
"""

from typing import Any, List, Optional

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.comparison_helper import (
    ComparisonHelper,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

# ---------------------------------------------------------------------------
# Models — match_threshold=1.0 (strict, used in the original issue)
# ---------------------------------------------------------------------------

class LineItemsInfo(StructuredModel):
    LineItemDays: Optional[List[str]] | Any = ComparableField(weight=1.0)
    match_threshold = 1.0


class Invoice(StructuredModel):
    LineItems: Optional[List[LineItemsInfo]] | Any = ComparableField(weight=1.0)


class Pep604LineItemsInfo(StructuredModel):
    LineItemDays: list[str] | None = ComparableField(weight=1.0)
    match_threshold = 1.0


class Pep604Invoice(StructuredModel):
    LineItems: list[Pep604LineItemsInfo] = ComparableField(weight=1.0)


# ---------------------------------------------------------------------------
# Models — lower threshold so partial-match tests get field recursion
# ---------------------------------------------------------------------------

class TaggedItem(StructuredModel):
    tags: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    match_threshold = 0.3  # Low threshold so most pairs get field recursion


class TaggedContainer(StructuredModel):
    items: List[TaggedItem] = ComparableField(weight=1.0)


class TaskItem(StructuredModel):
    tags: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    priority: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    match_threshold = 0.7


class TaskList(StructuredModel):
    tasks: List[TaskItem] = ComparableField(weight=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overall(cm, *field_path):
    """Navigate into cm['fields'][f1]['fields'][f2]...['overall']."""
    node = cm
    for f in field_path:
        node = node["fields"][f]
    return {k: v for k, v in node["overall"].items()
            if k in ("tp", "fa", "fd", "fp", "tn", "fn")}


# ---------------------------------------------------------------------------
# Tests — exact reproduction of issue #33
# ---------------------------------------------------------------------------

def test_issue_33_exact_repro():
    """Exact scenario from the GitHub issue — comparing identical data."""
    gt_data = {
        "LineItems": [
            {"LineItemDays": ["M", "T", "W", "Th", "F"]},
            {"LineItemDays": ["Su"]},
        ]
    }
    gt = Invoice(**gt_data)
    pred = Invoice(**gt_data)

    result = gt.compare_with(pred, include_confusion_matrix=True)
    agg = result["confusion_matrix"]["aggregate"]

    # 5 elements + 1 element = 6 TPs total
    assert agg["tp"] == 6
    assert agg["fa"] == 0
    assert agg["fd"] == 0
    assert agg["fp"] == 0
    assert agg["fn"] == 0


def test_issue_33_field_level_metrics():
    """Verify field-level metrics show element counts, not object counts."""
    gt_data = {
        "LineItems": [
            {"LineItemDays": ["M", "T", "W"]},
        ]
    }
    gt = Invoice(**gt_data)
    pred = Invoice(**gt_data)

    result = gt.compare_with(pred, include_confusion_matrix=True)
    metrics = _overall(result["confusion_matrix"], "LineItems", "LineItemDays")

    assert metrics["tp"] == 3
    assert metrics["fn"] == 0
    assert metrics["fa"] == 0


# ---------------------------------------------------------------------------
# Tests — partial matches / mismatches in simple lists
# Uses TaggedItem with low match_threshold so field recursion happens
# ---------------------------------------------------------------------------

def test_simple_list_missing_elements():
    """Prediction list shorter than GT → FN for missing elements."""
    gt = TaggedContainer(items=[TaggedItem(tags=["M", "T", "W"])])
    pred = TaggedContainer(items=[TaggedItem(tags=["M"])])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    metrics = _overall(result["confusion_matrix"], "items", "tags")

    assert metrics["tp"] == 1
    assert metrics["fn"] == 2


def test_simple_list_extra_elements():
    """Prediction list longer than GT → FA for extra elements."""
    gt = TaggedContainer(items=[TaggedItem(tags=["M"])])
    pred = TaggedContainer(items=[TaggedItem(tags=["M", "T", "W"])])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    metrics = _overall(result["confusion_matrix"], "items", "tags")

    assert metrics["tp"] == 1
    assert metrics["fa"] == 2


def test_simple_list_no_match():
    """Completely different elements — a below-threshold matched pair is FD.

    The objects share no aligning tags, so similarity is ~0, below the 0.3
    match_threshold. Hungarian still assigns the pair (it is the only possible
    assignment), so the threshold splits it into FD rather than un-matching it
    into FN + FA. Similarity magnitude does not change that; see issue #224.
    """
    gt = TaggedContainer(items=[TaggedItem(tags=["X", "Y"])])
    pred = TaggedContainer(items=[TaggedItem(tags=["A", "B"])])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    obj_metrics = _overall(result["confusion_matrix"], "items")

    # One assigned pair, below threshold: one FD, no FN/FA.
    assert obj_metrics["fd"] == 1
    assert obj_metrics["fn"] == 0
    assert obj_metrics["fa"] == 0


# ---------------------------------------------------------------------------
# Tests — multiple structured list items with simple lists
# ---------------------------------------------------------------------------

def test_multiple_items_aggregate_correctly():
    """Element counts from multiple structured list items should sum."""
    gt = Invoice(
        LineItems=[
            LineItemsInfo(LineItemDays=["A", "B", "C"]),
            LineItemsInfo(LineItemDays=["X", "Y"]),
        ]
    )
    pred = Invoice(
        LineItems=[
            LineItemsInfo(LineItemDays=["A", "B", "C"]),
            LineItemsInfo(LineItemDays=["X", "Y"]),
        ]
    )

    result = gt.compare_with(pred, include_confusion_matrix=True)
    agg = result["confusion_matrix"]["aggregate"]

    assert agg["tp"] == 5  # 3 + 2
    assert agg["fn"] == 0
    assert agg["fa"] == 0


def test_simple_list_alongside_primitive_field():
    """Simple list and primitive field coexist correctly in the same structured item."""
    gt = TaskList(
        tasks=[
            TaskItem(tags=["urgent", "backend"], priority="high"),
            TaskItem(tags=["frontend"], priority="low"),
        ]
    )
    pred = TaskList(
        tasks=[
            TaskItem(tags=["urgent", "backend"], priority="high"),
            TaskItem(tags=["frontend"], priority="low"),
        ]
    )

    result = gt.compare_with(pred, include_confusion_matrix=True)
    cm = result["confusion_matrix"]

    tags_metrics = _overall(cm, "tasks", "tags")
    priority_metrics = _overall(cm, "tasks", "priority")

    # tags: 2 + 1 = 3 element-level TPs
    assert tags_metrics["tp"] == 3

    # priority: 2 field-level TPs (one per matched task item)
    assert priority_metrics["tp"] == 2


def test_empty_simple_list_within_structured_list():
    """Empty simple lists on both sides — the pair is a TP, not a false discovery.

    Regression test for GitHub issue #233.

    The two objects are identical, so ``overall_score`` is 1.0. The confusion
    matrix used to read a different score for the same pair: an object whose
    only field was an empty list scored 0.0 on the list path, landing below
    ``match_threshold`` and classifying as FD. The same pair was therefore a
    perfect match and a false discovery at once.

    Also retains the #224 property: an assigned pair is not reported as FN + FA.
    """
    gt = Invoice(LineItems=[LineItemsInfo(LineItemDays=[])])
    pred = Invoice(LineItems=[LineItemsInfo(LineItemDays=[])])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    obj_metrics = _overall(result["confusion_matrix"], "LineItems")

    # Identical objects are a perfect match.
    assert result["overall_score"] == 1.0

    # The pair is a true positive, and specifically not a false discovery.
    assert obj_metrics["tp"] == 1
    assert obj_metrics["fd"] == 0

    # The #224 property: the pair is assigned, so neither side is orphaned.
    assert obj_metrics["fn"] == 0
    assert obj_metrics["fa"] == 0

    # The pair is counted exactly once.
    assert obj_metrics["tp"] + obj_metrics["fd"] + obj_metrics["tn"] == 1


def test_raw_empty_lists_score_as_perfect_match():
    """The raw unordered-list path agrees that two empty lists match."""
    result = ComparisonHelper.compare_unordered_lists(
        [], [], ExactComparator(), threshold=1.0
    )

    assert result == {
        "tp": 0,
        "fd": 0,
        "fa": 0,
        "fn": 0,
        "fp": 0,
        "overall_score": 1.0,
    }


@pytest.mark.parametrize("n_items", [1, 2, 3])
def test_empty_simple_lists_are_true_positives_at_any_length(n_items):
    """Identical empty-list objects are TPs at n=1 and n>1 alike.

    Regression test for GitHub issue #233, which recorded the contradiction at
    both lengths: ``overall_score=1.0`` with ``fd=1`` at n=1 and ``fd=2`` at
    n=2. Parameterized because the 1-vs-1 case reaches Hungarian matching
    differently from the multi-item case.
    """
    items = [LineItemsInfo(LineItemDays=[]) for _ in range(n_items)]
    gt = Invoice(LineItems=list(items))
    pred = Invoice(LineItems=list(items))

    result = gt.compare_with(pred, include_confusion_matrix=True)
    obj_metrics = _overall(result["confusion_matrix"], "LineItems")

    assert result["overall_score"] == 1.0
    assert obj_metrics["tp"] == n_items
    assert obj_metrics["fd"] == 0
    assert obj_metrics["fn"] == 0
    assert obj_metrics["fa"] == 0


@pytest.mark.parametrize(
    "gt_days,pred_days",
    [
        ([], []),
        (None, None),
        ([], None),
        (None, []),
    ],
)
def test_absent_simple_list_agrees_across_score_readers(gt_days, pred_days):
    """``overall_score`` and the confusion matrix agree on an absent list field.

    Regression test for GitHub issue #233.

    For a list field, ``None`` and ``[]`` both mean "no items", so every
    combination of the two is a perfect match. ``overall_score`` read that from
    the threshold-corrected score while the confusion matrix read the raw
    similarity, and the raw path scored ``[]`` against ``None`` as 0.0 — so the
    pair reported 1.0 and FD together. Both readers must now agree.
    """
    gt = Invoice(LineItems=[LineItemsInfo(LineItemDays=gt_days)])
    pred = Invoice(LineItems=[LineItemsInfo(LineItemDays=pred_days)])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    obj_metrics = _overall(result["confusion_matrix"], "LineItems")

    # A perfect score cannot coexist with a false discovery.
    assert result["overall_score"] == 1.0
    assert obj_metrics["fd"] == 0
    assert obj_metrics["tp"] == 1


@pytest.mark.parametrize("gt_days,pred_days", [([], None), (None, [])])
def test_pep604_optional_absent_list_agrees_across_score_readers(gt_days, pred_days):
    """PEP 604 optional lists use the same absent-list semantics."""
    gt = Pep604Invoice(LineItems=[Pep604LineItemsInfo(LineItemDays=gt_days)])
    pred = Pep604Invoice(LineItems=[Pep604LineItemsInfo(LineItemDays=pred_days)])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    cm = result["confusion_matrix"]
    object_metrics = _overall(cm, "LineItems")
    field_metrics = _overall(cm, "LineItems", "LineItemDays")

    assert result["overall_score"] == 1.0
    assert object_metrics["tp"] == 1
    assert object_metrics["fd"] == 0
    assert field_metrics["tn"] == 1
    assert field_metrics["fn"] == 0
    assert field_metrics["fa"] == 0


# ---------------------------------------------------------------------------
# Tests — unmatched structured objects with simple list fields
# Exercises the unmatched-object path for simple lists (PR #83 feedback)
# ---------------------------------------------------------------------------

def test_unmatched_gt_object_simple_list_contributes_fn():
    """When GT has more structured items than pred, the unmatched GT item's
    simple list elements should contribute FN at the field level.

    GT: 2 TaskItems, Pred: 1 TaskItem (matching the first).
    The unmatched GT item has tags=["X", "Y", "Z"] → should add 3 FN.
    """
    gt = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
        TaskItem(tags=["X", "Y", "Z"], priority="low"),
    ])
    pred = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
    ])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    tags_metrics = _overall(result["confusion_matrix"], "tasks", "tags")
    priority_metrics = _overall(result["confusion_matrix"], "tasks", "priority")

    # Matched pair: tags TP=2, priority TP=1
    # Unmatched GT: priority gets FN=1 (primitive path handles this)
    #               tags should get FN=3 (one per list element)
    assert tags_metrics["tp"] == 2
    assert tags_metrics["fn"] == 3
    assert priority_metrics["tp"] == 1
    assert priority_metrics["fn"] == 1


def test_unmatched_pred_object_simple_list_contributes_fa():
    """When pred has more structured items than GT, the unmatched pred item's
    simple list elements should contribute FA/FP at the field level.

    GT: 1 TaskItem, Pred: 2 TaskItems (first matches GT).
    The unmatched pred item has tags=["X", "Y", "Z"] → should add 3 FA.
    """
    gt = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
    ])
    pred = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
        TaskItem(tags=["X", "Y", "Z"], priority="low"),
    ])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    tags_metrics = _overall(result["confusion_matrix"], "tasks", "tags")
    priority_metrics = _overall(result["confusion_matrix"], "tasks", "priority")

    # Matched pair: tags TP=2, priority TP=1
    # Unmatched pred: priority gets FA=1, FP=1 (primitive path handles this)
    #                 tags should get FA=3, FP=3 (one per list element)
    assert tags_metrics["tp"] == 2
    assert tags_metrics["fa"] == 3
    assert priority_metrics["tp"] == 1
    assert priority_metrics["fa"] == 1
