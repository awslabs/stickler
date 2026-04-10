"""
Regression tests for simple list fields within structured lists.

Validates that List[str] (and similar primitive list types) inside a
List[StructuredModel] are compared element-by-element using Hungarian matching,
not treated as atomic primitive values.

See: https://github.com/awslabs/stickler/issues/33
"""

from typing import Any, List, Optional

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
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
    """Completely different elements — objects don't match, classified at object level.

    With match_threshold=0.3 and completely different single-character tags,
    the object similarity is ~0 which is below even 0.3. The objects get
    paired by Hungarian but classified as unmatched (FN + FA) because the
    similarity is effectively zero.
    """
    gt = TaggedContainer(items=[TaggedItem(tags=["X", "Y"])])
    pred = TaggedContainer(items=[TaggedItem(tags=["A", "B"])])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    obj_metrics = _overall(result["confusion_matrix"], "items")

    # Objects are so different they end up unmatched: 1 FN (GT) + 1 FA (pred)
    assert obj_metrics["fn"] == 1
    assert obj_metrics["fa"] == 1


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
    """Empty simple lists on both sides — objects with only empty fields
    get similarity 0 and don't match, so field-level metrics are all zeros.
    This is pre-existing behavior for objects whose only field is an empty list."""
    gt = Invoice(LineItems=[LineItemsInfo(LineItemDays=[])])
    pred = Invoice(LineItems=[LineItemsInfo(LineItemDays=[])])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    obj_metrics = _overall(result["confusion_matrix"], "LineItems")

    # Objects with only empty-list fields get similarity 0 → unmatched
    assert obj_metrics["fn"] + obj_metrics["fa"] >= 1


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
