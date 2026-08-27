"""The two confusion-matrix rollup nodes answer two different questions.

`confusion_matrix.overall` gives object verdicts at its own level: was this
pairing genuine, or spurious. `confusion_matrix.aggregate` gives leaf detail for
the objects that were comparable.

`match_threshold` is the line between them, and the gating is deliberate. An
object scoring below it is classified as a single FD, a spurious non-match, and
is not descended into: reporting the leaves of an object already rejected as a
whole would be scoring something declared not comparable. A caller who wants
those leaves counted lowers `match_threshold` so the object qualifies, which
`test_lowering_match_threshold_exposes_the_leaves` pins.

The two nodes coincide in exactly two situations, both of which have no accepted
subtree to expand: a model with no nesting, and a subtree rejected outright.

These tests pin the numbers the documentation publishes, so neither can go stale:

    docs/docs/Advanced/aggregate-metrics.md#which-node-answers-which-question
    docs/docs/Guides/Evaluation/understanding-results.md

Whether aggregates should also count the leaves of below-threshold objects is a
deliberately deferred question, not a defect. See
https://github.com/awslabs/stickler/issues/288
"""

from typing import List, Optional

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

FIELDS = ("sku", "desc", "qty", "unit", "tax", "total")


class Line(StructuredModel):
    """Six exact-match leaves, so every leaf comparison is unambiguous."""

    match_threshold = 0.7

    sku: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    desc: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    qty: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    unit: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    tax: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    total: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )


class Invoice(StructuredModel):
    lines: List[Line] = []


def _line(index: int, *, wrong: bool = False) -> Line:
    values = {name: f"{name}{index}" for name in FIELDS}
    if wrong:
        values["total"] = "WRONG"
    return Line(**values)


@pytest.fixture
def one_bad_leaf_of_thirty():
    """Five line items of six leaves each; one leaf of one item is wrong.

    Every item still pairs above `match_threshold`, which is the whole point:
    the failure is invisible to anything counting pairings.
    """
    ground_truth = Invoice(lines=[_line(i) for i in range(5)])
    prediction = Invoice(lines=[_line(i, wrong=(i == 2)) for i in range(5)])
    return ground_truth.compare_with(
        prediction, include_confusion_matrix=True, add_derived_metrics=True
    )


class TestRejectedObjectsReportNoLeafDetail:
    """An object below `match_threshold` is one FD and is not descended into.

    This is the intended semantics, not an oversight: the leaves of an object
    rejected as a whole are not reported. `match_threshold` is therefore the
    knob controlling how much leaf detail a caller gets.
    """

    def test_leaf_rows_stop_when_the_object_is_not_comparable(self):
        good = {name: f"{name}0" for name in FIELDS}
        ground_truth = Invoice(lines=[Line(**good), Line(**good)])

        rows = []
        for n_wrong in range(4):
            bad = {**good}
            for name in list(FIELDS)[:n_wrong]:
                bad[name] = "WRONG"
            cm = ground_truth.compare_with(
                Invoice(lines=[Line(**good), Line(**bad)]),
                include_confusion_matrix=True,
                add_derived_metrics=True,
            )["confusion_matrix"]
            aggregate = cm["aggregate"]
            rows.append((n_wrong, aggregate["tp"] + aggregate["fd"] + aggregate["fn"]))

        # 12 leaves exist in every case, but only comparable objects report
        # them. The second item drops below match_threshold (0.7) at 2 of 6
        # wrong, from which point only the first item's 6 leaves are scored.
        assert rows == [(0, 12), (1, 12), (2, 6), (3, 6)]

    def test_a_rejected_object_reports_itself_and_not_its_leaves(self):
        good = {name: f"{name}0" for name in FIELDS}
        bad = {**good, "sku": "WRONG", "desc": "WRONG"}  # 4/6 = 0.6667

        cm = Invoice(lines=[Line(**good)]).compare_with(
            Invoice(lines=[Line(**bad)]),
            include_confusion_matrix=True,
            add_derived_metrics=True,
        )["confusion_matrix"]

        # The object is the unit of report here, not its six leaves.
        assert cm["aggregate"]["fd"] == 1
        assert cm["aggregate"]["tp"] == 0
        assert cm["overall"]["fd"] == 1

    def test_lowering_match_threshold_exposes_the_leaves(self):
        """The documented remedy: make the object comparable and leaves follow.

        `match_threshold` decides what counts as the same object, and leaf
        reporting follows from that. This is the answer for a caller who wants
        below-threshold detail, rather than a change to what `aggregate` counts.
        """
        good = {name: f"{name}0" for name in FIELDS}
        bad = {**good, "sku": "WRONG", "desc": "WRONG"}  # scores 4/6 = 0.6667

        def counts(match_threshold):
            line_type = type(
                "Line",
                (StructuredModel,),
                {
                    "__annotations__": {name: Optional[str] for name in FIELDS},
                    "match_threshold": match_threshold,
                    **{
                        name: ComparableField(
                            comparator=ExactComparator(), threshold=1.0, default=None
                        )
                        for name in FIELDS
                    },
                },
            )
            doc_type = type(
                "Doc",
                (StructuredModel,),
                {"__annotations__": {"lines": List[line_type]}, "lines": []},
            )
            cm = doc_type(lines=[line_type(**good)]).compare_with(
                doc_type(lines=[line_type(**bad)]),
                include_confusion_matrix=True,
                add_derived_metrics=True,
            )["confusion_matrix"]
            return cm["aggregate"]["tp"], cm["aggregate"]["fd"]

        # Above the object's score: rejected, so one FD and no leaf rows.
        assert counts(0.70) == (0, 1)

        # Below it: comparable, so all six leaves are scored and the two bad
        # ones are reported individually.
        assert counts(0.66) == (4, 2)


class TestRollupNodesCountDifferentThings:
    def test_overall_counts_item_pairings(self, one_bad_leaf_of_thirty):
        """Five items paired, so five true positives and nothing wrong."""
        overall = one_bad_leaf_of_thirty["confusion_matrix"]["overall"]

        assert overall["tp"] == 5
        assert overall["fd"] == 0
        assert overall["fn"] == 0
        assert overall["fp"] == 0

    def test_aggregate_counts_leaf_comparisons(self, one_bad_leaf_of_thirty):
        """Thirty leaves, one of which failed."""
        aggregate = one_bad_leaf_of_thirty["confusion_matrix"]["aggregate"]

        assert aggregate["tp"] == 29
        assert aggregate["fd"] == 1
        assert aggregate["fn"] == 0
        assert aggregate["fp"] == 1  # fp == fa + fd

    def test_the_documented_derived_metrics(self, one_bad_leaf_of_thirty):
        """The exact numbers both doc pages print, side by side."""
        cm = one_bad_leaf_of_thirty["confusion_matrix"]
        overall = cm["overall"]["derived"]
        aggregate = cm["aggregate"]["derived"]

        # `overall` reports a flawless document.
        assert overall["cm_precision"] == pytest.approx(1.0)
        assert overall["cm_recall"] == pytest.approx(1.0)
        assert overall["cm_f1"] == pytest.approx(1.0)

        # `aggregate` sees the bad leaf.
        assert aggregate["cm_precision"] == pytest.approx(0.9667, abs=1e-4)
        assert aggregate["cm_recall"] == pytest.approx(1.0)
        assert aggregate["cm_f1"] == pytest.approx(0.9831, abs=1e-4)

    def test_overall_score_agrees_with_aggregate_and_not_with_overall(
        self, one_bad_leaf_of_thirty
    ):
        """The claim the docs rest on.

        `overall_score` is a weighted mean over the whole tree, so it tracks the
        leaf view. Reporting `overall.derived` precision beside `overall_score`
        puts two numbers in the same report that disagree by construction.
        """
        cm = one_bad_leaf_of_thirty["confusion_matrix"]
        score = one_bad_leaf_of_thirty["overall_score"]

        assert score == pytest.approx(0.9667, abs=1e-4)
        assert score == pytest.approx(cm["aggregate"]["derived"]["cm_precision"])
        assert score != pytest.approx(cm["overall"]["derived"]["cm_precision"])

    def test_a_complete_failure_check_reads_both_nodes(self):
        """The two nodes scope different things, so a full check needs both.

        `aggregate.fd + fn == 0` alone answers only "did every leaf of every
        comparable object land". It says nothing about objects that were not
        comparable, nor about invented fields, both of which land on `overall`.
        """
        good = {name: f"{name}0" for name in FIELDS}
        two_wrong = {**good, "sku": "WRONG", "desc": "WRONG"}  # 4/6, does not pair

        ground_truth = Invoice(lines=[Line(**good), Line(**good)])
        prediction = Invoice(lines=[Line(**good), Line(**two_wrong)])
        cm = ground_truth.compare_with(
            prediction, include_confusion_matrix=True, add_derived_metrics=True
        )["confusion_matrix"]

        aggregate, overall = cm["aggregate"], cm["overall"]

        # Every leaf of the one comparable object landed, so the leaf view is
        # clean. The rejected object contributes no leaf rows.
        assert aggregate["fd"] + aggregate["fn"] == 0

        # Its rejection is recorded here, which is why a full check reads both.
        assert overall["fd"] == 1
        clean = (
            aggregate["fd"] + aggregate["fn"] == 0
            and overall["fd"] + overall["fn"] + overall["fa"] == 0
        )
        assert clean is False

    def test_the_nodes_converge_when_there_is_no_list(self):
        """Without a list field there are no pairings, so both count leaves.

        This is why the divergence is easy to miss: it needs a list field to
        appear at all, and the flat case gives no warning that the two nodes can
        ever disagree.
        """
        ground_truth = _line(0)
        prediction = _line(0, wrong=True)
        result = ground_truth.compare_with(
            prediction, include_confusion_matrix=True, add_derived_metrics=True
        )
        cm = result["confusion_matrix"]

        assert cm["overall"]["tp"] == cm["aggregate"]["tp"] == 5
        assert cm["overall"]["fd"] == cm["aggregate"]["fd"] == 1
        assert (
            cm["overall"]["derived"]["cm_precision"]
            == cm["aggregate"]["derived"]["cm_precision"]
        )
