"""`evaluator_format=True` precision and recall come from the confusion counts.

Without `include_confusion_matrix=True`, `overall` was a conversion of the score
that charged every lost point to precision once the score reached 0.5, and each
list field's `overall` always was. One TP and one FN (`[A, B]` vs `[A]`) then
read precision 0.5 and recall 1.0, the wrong way round, while the same call
with the confusion matrix said precision 1.0 and recall 0.5.
"""

from typing import List, Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import ExactComparator


class Item(StructuredModel):
    a: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)


class Items(StructuredModel):
    items: Optional[List[Item]] = ComparableField(default=None)


class Tags(StructuredModel):
    tags: Optional[List[str]] = ComparableField(
        comparator=ExactComparator(), default=None
    )


class Flat(StructuredModel):
    a: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)
    b: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)


def _pr(metrics):
    return metrics["precision"], metrics["recall"]


@pytest.mark.parametrize(
    "gt,pred",
    [
        (Items(items=[Item(a="A"), Item(a="B")]), Items(items=[Item(a="A")])),
        (Tags(tags=["A", "B"]), Tags(tags=["A"])),
        (Flat(a="x", b="y"), Flat(a="x")),
    ],
    ids=["struct-list", "str-list", "flat"],
)
def test_a_missed_value_lowers_recall_not_precision(gt, pred):
    result = gt.compare_with(pred, evaluator_format=True)

    assert _pr(result["overall"]) == (1.0, 0.5)
    assert result["overall"]["anls_score"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "gt,pred,field",
    [
        (Items(items=[Item(a="A"), Item(a="B")]), Items(items=[Item(a="A")]), "items"),
        (Tags(tags=["A", "B"]), Tags(tags=["A"]), "tags"),
    ],
    ids=["struct-list", "str-list"],
)
def test_list_field_overall_reads_its_own_counts(gt, pred, field):
    for include_cm in (False, True):
        result = gt.compare_with(
            pred, evaluator_format=True, include_confusion_matrix=include_cm
        )
        assert _pr(result["fields"][field]["overall"]) == (1.0, 0.5)


def test_a_spurious_value_lowers_precision():
    result = Tags(tags=["A"]).compare_with(Tags(tags=["A", "B"]), evaluator_format=True)

    assert _pr(result["overall"]) == (0.5, 1.0)
    assert _pr(result["fields"]["tags"]["overall"]) == (0.5, 1.0)


@pytest.mark.parametrize("recall_with_fd", [False, True])
def test_overall_does_not_depend_on_include_confusion_matrix(recall_with_fd):
    gt = Flat(a="x", b="y")
    pred = Flat(a="x")
    without_cm = gt.compare_with(
        pred, evaluator_format=True, recall_with_fd=recall_with_fd
    )
    with_cm = gt.compare_with(
        pred,
        evaluator_format=True,
        include_confusion_matrix=True,
        recall_with_fd=recall_with_fd,
    )

    assert without_cm["overall"] == with_cm["overall"]


def test_list_field_matches_its_confusion_matrix_node():
    gt = Tags(tags=["A", "B", "C"])
    pred = Tags(tags=["A", "X"])
    result = gt.compare_with(pred, evaluator_format=True, include_confusion_matrix=True)
    derived = gt.compare_with(pred, include_confusion_matrix=True)["confusion_matrix"][
        "fields"
    ]["tags"]["overall"]["derived"]

    list_overall = result["fields"]["tags"]["overall"]
    assert list_overall["precision"] == derived["cm_precision"]
    assert list_overall["recall"] == derived["cm_recall"]
    assert list_overall["f1"] == derived["cm_f1"]
    assert list_overall["accuracy"] == derived["cm_accuracy"]


class Pair(StructuredModel):
    a: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)
    b: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)


class Pairs(StructuredModel):
    items: Optional[List[Pair]] = ComparableField(default=None)


def test_per_item_overall_reads_counts():
    """Each `items[i]` is that pair's own evaluator-format result, so a missed
    `b` lowers its recall, not its precision."""
    result = Pairs(items=[Pair(a="x", b="y")]).compare_with(
        Pairs(items=[Pair(a="x")]), evaluator_format=True
    )

    assert _pr(result["fields"]["items"]["items"][0]["overall"]) == (1.0, 0.5)
