"""
Regression tests for simple list fields within structured lists.

Validates that List[str] (and similar primitive list types) inside a
List[StructuredModel] are compared element-by-element using Hungarian matching,
not treated as atomic primitive values.

See: https://github.com/awslabs/stickler/issues/33
"""

from typing import Any, List, Optional

import pytest

from stickler.comparators.base import BaseComparator
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


class EveryListSpelling(StructuredModel):
    """Every spelling of a list annotation. All must be recognized alike.

    ``get_origin`` returns ``list`` only for a *parameterized* spelling, so the
    three unparameterized ones here used to read as non-list -- including
    ``list | None``, a PEP 604 optional list. See ``_annotation_is_list``.
    """

    bare: list = ComparableField(weight=1.0)
    bare_pep604_optional: list | None = ComparableField(weight=1.0)
    bare_optional: Optional[list] = ComparableField(weight=1.0)
    bare_typing: List = ComparableField(weight=1.0)
    param_pep604_optional: list[str] | None = ComparableField(weight=1.0)
    param_optional: Optional[List[str]] = ComparableField(weight=1.0)


LIST_SPELLINGS = (
    "bare",
    "bare_pep604_optional",
    "bare_optional",
    "bare_typing",
    "param_pep604_optional",
    "param_optional",
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


class PartiallyWrongItem(StructuredModel):
    """Two absent-on-both list fields and one field the prediction gets wrong.

    ``match_threshold`` is low enough that the two agreeing empty fields alone
    carry the pair over it — see
    ``test_absent_list_fields_lift_a_disagreeing_pair_to_a_true_positive``.
    """

    a: Optional[List[str]] = ComparableField(weight=1.0)
    b: Optional[List[str]] = ComparableField(weight=1.0)
    c: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    match_threshold = 0.5


class PartiallyWrongContainer(StructuredModel):
    items: List[PartiallyWrongItem] = ComparableField(weight=1.0)


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
# Tests — object matching counts absent-on-both fields toward the threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("absent", [[], None], ids=["empty-list", "none"])
def test_absent_list_fields_lift_a_disagreeing_pair_to_a_true_positive(absent):
    """Absent-on-both fields count toward ``match_threshold``. Deliberate.

    This is the metric consequence of the #233 fix, pinned because it is not
    self-evidently desirable. ``HungarianHelper`` classifies an object pair by
    its raw similarity, and an absent-on-both field now contributes a full
    ``1.0`` to that average. So a pair that disagrees on *every* value a user
    actually supplied can still clear ``match_threshold`` on the strength of its
    absent fields: here two of three fields are absent on both sides, the third
    disagrees outright, and the pair scores ``2/3`` and classifies as TP. Before
    the fix the ``[]`` spelling scored ``0.0`` and classified as FD.

    Parameterized over both spellings of absent because that is the argument for
    the flip: ``None`` already behaved this way, so the fix extends existing
    semantics to ``[]`` rather than inventing a rule. The two ids must stay
    equal — if they ever diverge, the parity #233 is about has reopened.

    The inflation this permits is real and worth stating: precision rises with
    the number of absent optional fields a schema declares. The field-level
    counts are what keep the report honest — ``c`` is still reported as a false
    discovery, and the aggregate still carries ``fd == 1``. Only the
    *object-level* classification moves.
    """
    gt_item = PartiallyWrongItem(a=absent, b=absent, c="alpha")
    pred_item = PartiallyWrongItem(a=absent, b=absent, c="zzzzz")

    # Two of three fields agree (vacuously), the third does not.
    assert gt_item.compare(pred_item) == pytest.approx(2 / 3)

    result = PartiallyWrongContainer(items=[gt_item]).compare_with(
        PartiallyWrongContainer(items=[pred_item]), include_confusion_matrix=True
    )
    cm = result["confusion_matrix"]

    # 2/3 clears match_threshold=0.5, so the pair is an object-level TP.
    assert _overall(cm, "items")["tp"] == 1
    assert _overall(cm, "items")["fd"] == 0

    # The disagreement is not swallowed: the field that is wrong is still wrong,
    # the vacuously-agreeing fields are true negatives rather than true
    # positives, and the aggregate still reports the false discovery.
    assert _overall(cm, "items", "c")["fd"] == 1
    assert _overall(cm, "items", "a")["tn"] == 1
    assert _overall(cm, "items", "b")["tn"] == 1
    assert cm["aggregate"]["fd"] == 1
    assert cm["aggregate"]["tp"] == 0


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
