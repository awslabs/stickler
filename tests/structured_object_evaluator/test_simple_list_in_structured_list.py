"""
Regression tests for simple list fields within structured lists.

Validates that List[str] (and similar primitive list types) inside a
List[StructuredModel] are compared element-by-element using Hungarian matching,
not treated as atomic primitive values.

See: https://github.com/awslabs/stickler/issues/33
"""

from typing import Annotated, Any, Dict, List, Optional

import pytest
from pydantic import Field

from stickler.comparators.base import BaseComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.comparison_helper import (
    ComparisonHelper,
    _maybe_absent,
)
from stickler.structured_object_evaluator.models.hungarian_helper import HungarianHelper
from stickler.structured_object_evaluator.models.non_match_field import NonMatchType
from stickler.structured_object_evaluator.models.null_helper import NullHelper
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


class EveryListSpelling(StructuredModel):
    """Every spelling of a list annotation. All must be recognized alike.

    Two families used to read as non-list:

    - The unparameterized ones, because ``get_origin`` returns ``list`` only for
      a *parameterized* spelling -- including ``list | None``, a PEP 604
      optional list.
    - The ``Annotated`` ones, because pydantic strips ``Annotated`` only when it
      wraps the whole annotation, and ``get_origin`` reports ``Annotated``
      rather than the type inside. ``annotated_whole`` is the spelling pydantic
      does strip, so it always worked; it is here to hold that contrast in
      place.

    See ``_annotation_is_list`` and ``optional_annotation.unwrap_annotated``.
    """

    bare: list = ComparableField(weight=1.0)
    bare_pep604_optional: list | None = ComparableField(weight=1.0)
    bare_optional: Optional[list] = ComparableField(weight=1.0)
    bare_typing: List = ComparableField(weight=1.0)
    param_pep604_optional: list[str] | None = ComparableField(weight=1.0)
    param_optional: Optional[List[str]] = ComparableField(weight=1.0)
    annotated_whole: Annotated[Optional[List[str]], "meta"] = ComparableField(weight=1.0)
    annotated_optional: Optional[Annotated[List[str], "meta"]] = ComparableField(
        weight=1.0
    )
    annotated_pep604: Annotated[List[str], "meta"] | None = ComparableField(weight=1.0)
    annotated_bare: Annotated[list, "meta"] | None = ComparableField(weight=1.0)
    annotated_field: Optional[Annotated[List[str], Field(description="d")]] = (
        ComparableField(weight=1.0)
    )


LIST_SPELLINGS = (
    "bare",
    "bare_pep604_optional",
    "bare_optional",
    "bare_typing",
    "param_pep604_optional",
    "param_optional",
    "annotated_whole",
    "annotated_optional",
    "annotated_pep604",
    "annotated_bare",
    "annotated_field",
)

# The spellings that carry an `Annotated` wrapper on the *union arm*, which is
# where pydantic leaves it in place. `annotated_whole` is excluded: pydantic
# strips a wrapper around the whole annotation, so it never had the problem.
ANNOTATED_ARM_SPELLINGS = (
    "annotated_optional",
    "annotated_pep604",
    "annotated_bare",
    "annotated_field",
)


class UntypedListHolder(StructuredModel):
    """An ``Any``-annotated field, which is deliberately *not* a list field."""

    vals: Any = ComparableField(weight=1.0)


class NoteLeaf(StructuredModel):
    """A non-list field, for the primitive half of #233's parity rule."""

    note: Optional[str] = ComparableField(weight=1.0)
    match_threshold = 1.0


class NoteContainer(StructuredModel):
    items: List[NoteLeaf] = ComparableField(weight=1.0)


class DictLeaf(StructuredModel):
    """A dict field, the third kind of "empty" the documented rule names."""

    meta: Optional[Dict[str, str]] = ComparableField(weight=1.0)
    match_threshold = 1.0


class DictContainer(StructuredModel):
    items: List[DictLeaf] = ComparableField(weight=1.0)


class PartiallyWrongItem(StructuredModel):
    """Two absent-on-both list fields and one populated field."""

    a: Optional[List[str]] = ComparableField(weight=1.0)
    b: Optional[List[str]] = ComparableField(weight=1.0)
    c: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    match_threshold = 0.5


class PartiallyWrongContainer(StructuredModel):
    items: List[PartiallyWrongItem] = ComparableField(weight=1.0)


class MissedValueItem(StructuredModel):
    """One real value and five fields that may be true negatives."""

    a: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    b: list = ComparableField(weight=1.0)
    c: list = ComparableField(weight=1.0)
    d: list = ComparableField(weight=1.0)
    e: list = ComparableField(weight=1.0)
    f: list = ComparableField(weight=1.0)
    match_threshold = 0.8


class MissedValueContainer(StructuredModel):
    items: List[MissedValueItem] = ComparableField(weight=1.0)


class CompetingCandidateItem(StructuredModel):
    """Two fields that distinguish an empty from a useful candidate."""

    a: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    b: list = ComparableField(weight=1.0)
    match_threshold = 0.7


class CompetingCandidateContainer(StructuredModel):
    items: List[CompetingCandidateItem] = ComparableField(weight=1.0)


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


class _NeverCalledComparator(BaseComparator):
    """A comparator that fails the test if it is ever invoked.

    Lets a test assert that a code path does not consult its comparator, rather
    than leaving an unused argument that reads as significant.
    """

    def _compare(self, str1, str2) -> float:
        raise AssertionError(
            f"comparator must not be invoked on this path, got {str1!r}, {str2!r}"
        )


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
    """The raw unordered-list path agrees that two empty lists match.

    Two empty lists take the *StructuredModel* branch of
    ``compare_unordered_lists``, vacuously: that branch is guarded by
    ``all(isinstance(item, StructuredModel) for item in gt_list[:1])`` on both
    sides, and ``[][:1]`` is empty, so both ``all()`` calls are trivially true
    however the lists are annotated. That branch hardcodes its own
    ``classification_threshold`` of ``0.01`` and never reads ``comparator``.

    So the comparator and ``threshold`` arguments below are inert, and the test
    says so rather than implying they matter: the comparator is one that raises
    if it is ever called, which turns the claim into an assertion instead of a
    comment. ``threshold`` cannot be shown inert the same way -- with no matched
    pairs the threshold loop never runs, so no value of it changes the result.
    """
    result = ComparisonHelper.compare_unordered_lists(
        [], [], _NeverCalledComparator(), threshold=1.0
    )

    assert result == {
        "tp": 0,
        "fd": 0,
        "fa": 0,
        "fn": 0,
        "fp": 0,
        "overall_score": 1.0,
    }


@pytest.mark.parametrize(
    "gt_list,pred_list,expected",
    [
        ([], ["x"], {"fa": 1, "fn": 0}),
        (["x"], [], {"fa": 0, "fn": 1}),
    ],
    ids=["gt-empty", "pred-empty"],
)
def test_raw_one_sided_empty_list_is_not_a_match(gt_list, pred_list, expected):
    """One empty side scores 0.0, and reaches ``unordered_list_metrics`` the
    other way -- through the comparator branch.

    This is the companion route to ``test_raw_empty_lists_score_as_perfect_match``
    and the guard on the ``else 0.0`` half of the both-empty conditional. Because
    only one side is empty, exactly one of the two ``all()`` guards is false, so
    this takes the comparator branch and the ``threshold`` argument *is* honored
    as ``classification_threshold`` -- both of which the both-empty case cannot
    reach, since two empty lists always satisfy both guards vacuously.

    Pinning it keeps the #233 fix from over-reaching: scoring two empty lists
    1.0 must not leak into scoring an empty list against a populated one, which
    is a genuine miss (FN) or a genuine spurious find (FA), not agreement.
    """
    result = ComparisonHelper.compare_unordered_lists(
        gt_list, pred_list, ExactComparator(), threshold=1.0
    )

    assert result["overall_score"] == 0.0
    assert result["tp"] == 0
    assert result["fd"] == 0
    assert result["fa"] == expected["fa"]
    assert result["fn"] == expected["fn"]


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
# Tests — every spelling of a list annotation is recognized as a list field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name", LIST_SPELLINGS)
def test_every_list_spelling_is_recognized_as_a_list_field(field_name):
    """``_is_list_field`` is true for parameterized and bare spellings alike.

    ``get_origin`` returns ``list`` only for a parameterized annotation, so
    ``list``, ``list | None`` and ``Optional[list]`` used to answer ``False``
    while ``List``, ``list[str] | None`` and ``Optional[List[str]]`` answered
    ``True``. ``list | None`` answering differently from ``list[str] | None`` is
    the odd one: both are PEP 604 optional lists.
    """
    instance = EveryListSpelling(**{name: [] for name in LIST_SPELLINGS})

    assert instance._is_list_field(field_name) is True


def test_every_list_spelling_records_a_true_negative_when_empty_on_both_sides():
    """All six spellings produce the same classification evidence.

    The spelling of an annotation must not change the metrics. A field that
    reads as non-list skips the list null handling in
    ``ComparisonDispatcher.dispatch_field_comparison`` STEP 3 and falls into the
    primitive null check, where ``[]`` is not "effectively null" -- so it was
    routed to ``PrimitiveListComparator``, scored ``1.0``, and recorded *no*
    classification evidence at all: no TN, no TP, nothing. The score looked
    right, so the missing row was easy to miss.

    A TN is what the documented rule calls for: both sides agree there are no
    items. See ``docs/docs/Advanced/classification-logic.md``.
    """
    values = {name: [] for name in LIST_SPELLINGS}
    gt = EveryListSpelling(**values)
    pred = EveryListSpelling(**values)

    result = gt.compare_with(pred, include_confusion_matrix=True)
    cm = result["confusion_matrix"]

    assert result["overall_score"] == 1.0
    for name in LIST_SPELLINGS:
        assert _overall(cm, name) == {
            "tp": 0,
            "fa": 0,
            "fd": 0,
            "fp": 0,
            "tn": 1,
            "fn": 0,
        }, f"{name} disagrees with the other spellings"

    # One TN per field, and no spelling silently contributing nothing.
    assert cm["aggregate"]["tn"] == len(LIST_SPELLINGS)


@pytest.mark.parametrize("field_name", ANNOTATED_ARM_SPELLINGS)
@pytest.mark.parametrize(
    "gt_val,pred_val",
    [([], []), ([], None), (None, [])],
)
def test_annotated_arm_matches_the_plain_optional_list(field_name, gt_val, pred_val):
    """An ``Annotated`` wrapper on a union arm must not change the metrics.

    Pydantic strips ``Annotated`` when it wraps the whole annotation but leaves
    it on a union arm, and ``get_origin`` reports ``Annotated`` rather than the
    wrapped type. So ``Optional[Annotated[List[str], ...]]`` read as non-list
    while its sibling ``Optional[List[str]]`` read as a list -- and
    ``Annotated[List[str], ...] | None`` normalises to exactly that spelling.

    The cost showed up in three different ways on the same field, all of them
    silent: ``[]`` against ``[]`` recorded *no counter at all* (the field
    vanished from the confusion matrix), ``[]`` against ``None`` recorded an FN,
    and ``None`` against ``[]`` recorded an FA. The plain spelling records a TN
    for all three.

    ``Field(description=...)`` produces this annotation, so it is the common
    spelling in an extraction schema rather than an exotic one.
    """
    gt = EveryListSpelling(**{field_name: gt_val})
    pred = EveryListSpelling(**{field_name: pred_val})
    plain_gt = EveryListSpelling(**{"param_optional": gt_val})
    plain_pred = EveryListSpelling(**{"param_optional": pred_val})

    assert gt._is_list_field(field_name) is True

    annotated = _overall(
        gt.compare_with(pred, include_confusion_matrix=True)["confusion_matrix"],
        field_name,
    )
    plain = _overall(
        plain_gt.compare_with(plain_pred, include_confusion_matrix=True)[
            "confusion_matrix"
        ],
        "param_optional",
    )

    assert annotated == plain, f"{field_name} disagrees with Optional[List[str]]"
    assert annotated["tn"] == 1


def test_an_any_annotated_field_is_not_a_list_field():
    """``Any`` holding a list is still not a list *field*, by design.

    This pins the boundary of the spelling fix. ``Any`` is not a list
    annotation, so it keeps primitive null semantics and routes an empty list to
    ``PrimitiveListComparator`` -- the one remaining way two empty lists reach
    ``compare_unordered_lists`` from a model comparison, and what keeps the
    empty-list fall-through in that function live rather than dead code.

    The pair still scores ``1.0``, and still records no classification evidence.
    Recording a TN here would mean inferring list-ness from a runtime value
    rather than an annotation, which is a larger decision than #233 -- so this
    documents the current answer rather than asserting it is the desired one.
    """
    gt = UntypedListHolder(vals=[])
    pred = UntypedListHolder(vals=[])

    assert gt._is_list_field("vals") is False
    assert gt.compare(pred) == 1.0

    result = gt.compare_with(pred, include_confusion_matrix=True)
    assert result["overall_score"] == 1.0
    assert _overall(result["confusion_matrix"], "vals") == {
        "tp": 0,
        "fa": 0,
        "fd": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }


@pytest.mark.parametrize(
    "gt_note,pred_note",
    [
        ("", ""),
        (None, None),
        ("", None),
        (None, ""),
    ],
)
def test_absent_string_field_agrees_across_score_readers(gt_note, pred_note):
    """The parity rule covers non-list fields too, not just lists.

    Regression test for GitHub issue #233.

    The list half of the parity fix left the primitive half open. The dispatcher
    reads absence for a non-list field with
    ``NullHelper.is_effectively_null_for_primitives``, which treats ``""`` and
    ``None`` as the same thing, while the raw path tested bare ``is None``. So
    the same contradiction reproduced for a string field that is ``""`` on one
    side and ``None`` on the other: ``overall_score == 1.0`` reported alongside
    ``fd == 1``. Both readers must agree here for the same reason they must
    agree for lists.
    """
    gt = NoteContainer(items=[NoteLeaf(note=gt_note)])
    pred = NoteContainer(items=[NoteLeaf(note=pred_note)])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    cm = result["confusion_matrix"]
    object_metrics = _overall(cm, "items")
    field_metrics = _overall(cm, "items", "note")

    # A perfect score cannot coexist with a false discovery.
    assert result["overall_score"] == 1.0
    assert object_metrics["tp"] == 1
    assert object_metrics["fd"] == 0

    # Both sides agree there is no value, which is a true negative.
    assert field_metrics["tn"] == 1
    assert field_metrics["fn"] == 0
    assert field_metrics["fa"] == 0


# ---------------------------------------------------------------------------
# Tests — an empty dict is absent, the third case the documented rule names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gt_val,pred_val",
    [({}, {}), ({}, None), (None, {}), (None, None)],
)
def test_absent_dict_field_records_a_true_negative(gt_val, pred_val):
    """``{}`` is absent, exactly as ``""`` and ``[]`` are.

    ``docs/docs/Advanced/classification-logic.md`` states the rule for all
    three: "Empty strings, empty lists, and empty objects are treated as null.
    Comparing any of these with null yields TN."
    ``is_effectively_null_for_primitives`` covered ``None`` and ``""`` but not
    ``{}``, so the #233 contradiction survived for a dict field in the opposite
    direction: two **identical** objects each holding ``{}`` classified as a
    false discovery rather than a match.
    """
    gt = DictContainer(items=[DictLeaf(meta=gt_val)])
    pred = DictContainer(items=[DictLeaf(meta=pred_val)])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    cm = result["confusion_matrix"]

    assert result["overall_score"] == 1.0
    assert _overall(cm, "items")["tp"] == 1
    assert _overall(cm, "items")["fd"] == 0
    assert _overall(cm, "items", "meta")["tn"] == 1


@pytest.mark.parametrize(
    "gt_val,pred_val,expected",
    [({}, {"k": "v"}, "fa"), ({"k": "v"}, {}, "fn"), (None, {"k": "v"}, "fa")],
)
def test_dict_field_absent_on_one_side_is_reported_not_raised(
    gt_val, pred_val, expected
):
    """A populated dict against an absent one is FA/FN, and does not raise.

    No comparator accepts a dict -- ``LevenshteinComparator`` raises
    ``TypeError`` explaining that a ``StructuredModel`` should be used instead.
    Because ``{}`` did not read as absent, ``{}`` against a populated dict fell
    through to that comparator, so the pair was *uncomparable* rather than
    merely mismatched: ``compare()`` raised, which took out Hungarian matching
    for any ``List[StructuredModel]`` whose element model had a populated dict
    field. Reading ``{}`` as absent short-circuits before any comparator is
    consulted.

    Two populated dicts still reach the comparator and still raise; that is a
    separate gap, and this pins only the absent-on-one-side half.

    Asserted on the leaf directly rather than through ``DictContainer``: the
    pair scores ``0.0``, so inside a list it is a below-threshold FD and gets no
    field breakdown at all -- that is the documented threshold-gated recursion,
    not a second bug. The object-level half of that is pinned below.
    """
    gt = DictLeaf(meta=gt_val)
    pred = DictLeaf(meta=pred_val)

    # The regression: this used to raise TypeError out of LevenshteinComparator.
    assert gt.compare(pred) == 0.0

    cm = gt.compare_with(pred, include_confusion_matrix=True)["confusion_matrix"]
    assert cm["fields"]["meta"]["overall"][expected] == 1


def test_dict_field_absent_on_one_side_is_an_object_level_false_discovery():
    """Inside a list, the same pair is an FD with no field breakdown.

    The companion to the test above, pinning the gating rather than the field
    classification. ``DictLeaf.match_threshold`` is ``1.0`` and the pair scores
    ``0.0``, so the objects are "not really the same" and
    ``docs/docs/Advanced/threshold-gated-evaluation.md`` calls for treating the
    pair as atomic. What matters for this fix is that it is reached by scoring
    rather than by raising.
    """
    gt = DictContainer(items=[DictLeaf(meta={})])
    pred = DictContainer(items=[DictLeaf(meta={"k": "v"})])

    cm = gt.compare_with(pred, include_confusion_matrix=True)["confusion_matrix"]

    assert _overall(cm, "items")["fd"] == 1
    assert _overall(cm, "items")["tp"] == 0
    assert cm["fields"]["items"].get("fields", {}) == {}


def test_maybe_absent_is_a_superset_of_both_null_rules():
    """The perf guard must never hide a value either rule calls absent.

    ``compare_field_raw`` consults a field's annotation only when
    ``_maybe_absent`` says one side could be absent, because that lookup runs
    for every field of every cell in a Hungarian cost matrix. The guard is
    therefore an over-approximation of both ``NullHelper`` predicates, and it is
    only sound while it stays a superset: a value either predicate would call
    absent but the guard rejects would silently skip true-negative and
    false-negative handling.

    Adding a case to either predicate without widening the guard fails here.
    """
    candidates = [
        None,
        "",
        [],
        {},
        "x",
        ["x"],
        {"k": "v"},
        0,
        0.0,
        False,
        set(),
        (),
        NoteLeaf(note=None),
    ]

    for value in candidates:
        either_rule_calls_it_absent = NullHelper.is_effectively_null_for_lists(
            value
        ) or NullHelper.is_effectively_null_for_primitives(value)
        if either_rule_calls_it_absent:
            assert _maybe_absent(value) is True, (
                f"{value!r} is absent under a NullHelper rule but the guard "
                f"in compare_field_raw would skip the check"
            )


@pytest.mark.parametrize(
    "gt_note,pred_note",
    [
        ("", "n"),
        ("n", ""),
        (None, "n"),
        ("n", None),
    ],
)
def test_absent_string_against_populated_is_not_leniently_matched(gt_note, pred_note):
    """Absent-vs-populated is still a total mismatch, on both readers.

    The #233 rule is equivalence between the two spellings of *absent*, not
    leniency: ``""`` does not become a wildcard that matches any string. This
    pins the other side of the branch, so widening the null rule cannot quietly
    turn a missed value into a match.
    """
    leaf_score = NoteLeaf(note=gt_note).compare(NoteLeaf(note=pred_note))
    gt = NoteContainer(items=[NoteLeaf(note=gt_note)])
    pred = NoteContainer(items=[NoteLeaf(note=pred_note)])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    object_metrics = _overall(result["confusion_matrix"], "items")

    # Both readers agree the pair is wrong, so there is no contradiction here
    # either -- it is just a false discovery rather than a true positive.
    assert leaf_score == 0.0
    assert result["overall_score"] == 0.0
    assert object_metrics["tp"] == 0
    assert object_metrics["fd"] == 1


# ---------------------------------------------------------------------------
# Tests — true negatives are not evidence for object matching
# ---------------------------------------------------------------------------

def test_all_absent_fields_define_empty_denominator_as_a_match():
    """An empty denominator is 1.0, preserving the original #233 fix."""
    gt = LineItemsInfo(LineItemDays=[])
    pred = LineItemsInfo(LineItemDays=None)

    assert gt.compare(pred) == 1.0


@pytest.mark.parametrize("absent", [[], None], ids=["empty-list", "none"])
def test_absent_list_fields_do_not_lift_a_disagreeing_pair(absent):
    """True negatives are excluded from object similarity in either spelling."""
    gt_item = PartiallyWrongItem(a=absent, b=absent, c="alpha")
    pred_item = PartiallyWrongItem(a=absent, b=absent, c="zzzzz")

    assert gt_item.compare(pred_item) == 0.0

    result = PartiallyWrongContainer(items=[gt_item]).compare_with(
        PartiallyWrongContainer(items=[pred_item]), include_confusion_matrix=True
    )
    cm = result["confusion_matrix"]

    assert _overall(cm, "items")["tp"] == 0
    assert _overall(cm, "items")["fd"] == 1
    assert cm["fields"]["items"]["overall"]["derived"]["cm_f1"] == 0.0
    assert cm["fields"]["items"].get("fields", {}) == {}


def test_prediction_that_extracts_nothing_is_not_a_perfect_match():
    """Five true negatives cannot hide the only missed value."""
    gt_item = MissedValueItem(
        a="VALUE", b=[], c=[], d=[], e=[], f=[]
    )
    pred_item = MissedValueItem(
        a=None, b=[], c=[], d=[], e=[], f=[]
    )

    assert gt_item.compare(pred_item) == 0.0

    result = MissedValueContainer(items=[gt_item]).compare_with(
        MissedValueContainer(items=[pred_item]), include_confusion_matrix=True
    )
    object_metrics = _overall(result["confusion_matrix"], "items")

    assert object_metrics["tp"] == 0
    assert object_metrics["fd"] == 1
    assert (
        result["confusion_matrix"]["fields"]["items"]["overall"]["derived"]["cm_f1"]
        == 0.0
    )


def test_hungarian_prefers_content_agreement_over_an_empty_candidate():
    """Assignment prefers content, while PET still rejects a weak pair."""
    gt_item = CompetingCandidateItem(a="x", b=[])
    empty_candidate = CompetingCandidateItem(a=None, b=[])
    correct_candidate = CompetingCandidateItem(a="x", b=["z"])

    assert gt_item.compare(empty_candidate) == 0.0
    assert gt_item.compare(correct_candidate) == pytest.approx(0.5)

    matching = HungarianHelper().get_complete_matching_info(
        [gt_item], [empty_candidate, correct_candidate]
    )
    assert matching["assignments"] == [(0, 1)]
    assert matching["matched_pairs"][0][2] == pytest.approx(0.5)
    assert matching["unmatched_pred_indices"] == [0]

    result = CompetingCandidateContainer(items=[gt_item]).compare_with(
        CompetingCandidateContainer(items=[empty_candidate, correct_candidate]),
        include_confusion_matrix=True,
        document_non_matches=True,
        document_field_comparisons=True,
    )
    object_metrics = _overall(result["confusion_matrix"], "items")
    fd_entry = next(
        non_match
        for non_match in result["non_matches"]
        if non_match["non_match_type"] == NonMatchType.FALSE_DISCOVERY
    )
    fa_entry = next(
        non_match
        for non_match in result["non_matches"]
        if non_match["non_match_type"] == NonMatchType.FALSE_ALARM
    )
    item_comparisons = [
        comparison
        for comparison in result["field_comparisons"]
        if comparison["expected_key"].startswith("items[")
    ]

    assert result["overall_score"] == pytest.approx(0.25)
    assert object_metrics["tp"] == 0
    assert object_metrics["fd"] == 1
    assert object_metrics["fa"] == 1
    assert result["confusion_matrix"]["fields"]["items"]["fields"] == {}

    assert fd_entry["field_path"] == "items[0]"
    assert fd_entry["ground_truth_value"] == gt_item.model_dump()
    assert fd_entry["prediction_value"] == correct_candidate.model_dump()
    assert fd_entry["similarity"] == pytest.approx(0.5)
    assert fa_entry["field_path"] == "items[0]"
    assert fa_entry["ground_truth_value"] is None
    assert fa_entry["prediction_value"] == empty_candidate.model_dump()

    assert len(item_comparisons) == 2
    assert all("." not in comparison["expected_key"] for comparison in item_comparisons)
    assigned_comparison = next(
        comparison
        for comparison in item_comparisons
        if comparison["actual_value"] == correct_candidate.model_dump()
    )
    assert assigned_comparison["expected_key"] == "items[0]"
    assert assigned_comparison["actual_key"] == "items[1]"
    assert assigned_comparison["match"] is False
    assert assigned_comparison["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests — unmatched structured objects remain atomic
# ---------------------------------------------------------------------------

def test_unmatched_gt_object_does_not_contribute_leaf_fn():
    """An unmatched GT object is one object-level FN, not leaf-level FNs."""
    gt = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
        TaskItem(tags=["X", "Y", "Z"], priority="low"),
    ])
    pred = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
    ])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    task_metrics = _overall(result["confusion_matrix"], "tasks")
    tags_metrics = _overall(result["confusion_matrix"], "tasks", "tags")
    priority_metrics = _overall(result["confusion_matrix"], "tasks", "priority")

    assert task_metrics["fn"] == 1
    assert tags_metrics["tp"] == 2
    assert tags_metrics["fn"] == 0
    assert priority_metrics["tp"] == 1
    assert priority_metrics["fn"] == 0


def test_unmatched_pred_object_does_not_contribute_leaf_fa():
    """An unmatched prediction is one object-level FA, not leaf-level FAs."""
    gt = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
    ])
    pred = TaskList(tasks=[
        TaskItem(tags=["A", "B"], priority="high"),
        TaskItem(tags=["X", "Y", "Z"], priority="low"),
    ])

    result = gt.compare_with(pred, include_confusion_matrix=True)
    task_metrics = _overall(result["confusion_matrix"], "tasks")
    tags_metrics = _overall(result["confusion_matrix"], "tasks", "tags")
    priority_metrics = _overall(result["confusion_matrix"], "tasks", "priority")

    assert task_metrics["fa"] == 1
    assert tags_metrics["tp"] == 2
    assert tags_metrics["fa"] == 0
    assert priority_metrics["tp"] == 1
    assert priority_metrics["fa"] == 0
