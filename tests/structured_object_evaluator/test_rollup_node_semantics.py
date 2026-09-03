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

The two nodes coincide only where there is no accepted subtree to expand at all: a
model with no nesting, or a document in which *every* subtree was rejected. One
rejected subtree among several makes them diverge further rather than converge,
since `aggregate` then reports a flawless precision over the accepted items only,
which `test_a_rejected_subtree_separates_the_two_numbers` pins.

These tests pin the numbers and the snippet the documentation publishes, so
neither can go stale:

    docs/docs/Advanced/aggregate-metrics.md#which-node-answers-which-question
    docs/docs/Advanced/threshold-gated-evaluation.md
    docs/docs/Guides/Evaluation/understanding-results.md

Whether aggregates should also count the leaves of below-threshold objects is a
deliberately deferred question, not a defect. See
https://github.com/awslabs/stickler/issues/288
"""

from pathlib import Path
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


def _line_type(match_threshold: float) -> type:
    """A six-leaf line-item model with the given `match_threshold`."""
    return type(
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


def _doc_type(line_type: type) -> type:
    """A document holding a list of the given line-item model."""
    return type(
        "Doc",
        (StructuredModel,),
        {"__annotations__": {"lines": List[line_type]}, "lines": []},
    )


_PAGES_PUBLISHING_THE_CLEAN_CHECK = (
    "docs/docs/Advanced/aggregate-metrics.md",
    "docs/docs/Advanced/threshold-gated-evaluation.md",
    "docs/docs/Guides/Evaluation/understanding-results.md",
)


def _documented_clean_check(cm: dict) -> bool:
    """The "did anything fail" check the doc pages publish, verbatim.

    Defined once so the pages and the tests cannot drift apart. It sums `fp`
    rather than `fa + fd` because `FP = FA + FD` by construction, so the two are
    equivalent today and `fp` cannot go stale if a class is ever added.

    Both nodes are read because each is blind to the other's failures: a
    below-threshold item is one `fd` on `overall` and contributes no leaf rows,
    while a value invented on a null leaf is `fa` under `aggregate` and leaves
    `overall` untouched.

    Published in three places, all of which must stay in step with this:

        docs/docs/Advanced/aggregate-metrics.md#asking-whether-anything-failed
        docs/docs/Advanced/threshold-gated-evaluation.md
        docs/docs/Guides/Evaluation/understanding-results.md  (twice)
    """
    return (
        cm["aggregate"]["fp"] + cm["aggregate"]["fn"] == 0
        and cm["overall"]["fp"] + cm["overall"]["fn"] == 0
    )


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
            line_type = _line_type(match_threshold)
            doc_type = _doc_type(line_type)
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

    def test_the_two_documented_threshold_tables(self):
        """Pins the `match_threshold` tables the two pages print.

        These were the only published figures no test covered, and both had
        drifted to the numbers of a one-item document while their prose described
        a two-item and a five-item one. The counts depend on the document, so the
        document has to be part of the assertion.

            understanding-results.md   two items, second at 4/6
            aggregate-metrics.md       five items, third at 4/6
        """

        def counts(match_threshold, items, bad_index):
            line_type = _line_type(match_threshold)
            doc_type = _doc_type(line_type)
            good = {name: f"{name}0" for name in FIELDS}
            bad = {**good, "sku": "WRONG", "desc": "WRONG"}  # 4/6 = 0.6667
            cm = doc_type(lines=[line_type(**good) for _ in range(items)]).compare_with(
                doc_type(
                    lines=[
                        line_type(**(bad if i == bad_index else good))
                        for i in range(items)
                    ]
                ),
                include_confusion_matrix=True,
                add_derived_metrics=True,
            )["confusion_matrix"]
            return (
                cm["overall"]["tp"],
                cm["overall"]["fd"],
                cm["aggregate"]["tp"],
                cm["aggregate"]["fd"],
            )

        # understanding-results.md: two items, the second at 4/6.
        assert counts(0.70, items=2, bad_index=1) == (1, 1, 6, 0)
        assert counts(0.66, items=2, bad_index=1) == (2, 0, 10, 2)

        # aggregate-metrics.md: five items, the third at 4/6.
        assert counts(0.70, items=5, bad_index=2) == (4, 1, 24, 0)
        assert counts(0.66, items=5, bad_index=2) == (5, 0, 28, 2)


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
        puts two numbers in the same report that disagree by construction. That
        inequality is the load-bearing half.

        The equality with `aggregate` precision is narrower than it looks, and is
        not a general property. A weighted mean of leaf scores equals
        `tp / (tp + fp)` only while all four of this fixture's conditions hold:
        every leaf scores exactly 0.0 or 1.0, weights are uniform, there are no
        FN, and no subtree was rejected. Break the last one and they part
        company, which `test_a_rejected_subtree_separates_the_two_numbers`
        pins.
        """
        cm = one_bad_leaf_of_thirty["confusion_matrix"]
        score = one_bad_leaf_of_thirty["overall_score"]

        assert score == pytest.approx(0.9667, abs=1e-4)
        assert score == pytest.approx(cm["aggregate"]["derived"]["cm_precision"])
        assert score != pytest.approx(cm["overall"]["derived"]["cm_precision"])

    def test_a_rejected_subtree_separates_the_two_numbers(self):
        """The equality above is a property of the fixture, not of the engine.

        With one item of five rejected, `aggregate` sees only the 24 leaves of
        the four accepted items, all correct, so its precision is a flawless
        1.0 while `overall_score` carries the rejection.
        """

        def rejected(index: int) -> Line:
            """Two of six leaves wrong, so 4/6, below `match_threshold`."""
            values = {name: f"{name}{index}" for name in FIELDS}
            values["total"] = "WRONG"
            values["tax"] = "WRONG"
            return Line(**values)

        ground_truth = Invoice(lines=[_line(i) for i in range(5)])
        prediction = Invoice(
            lines=[rejected(i) if i == 2 else _line(i) for i in range(5)]
        )
        result = ground_truth.compare_with(
            prediction, include_confusion_matrix=True, add_derived_metrics=True
        )
        cm = result["confusion_matrix"]

        assert result["overall_score"] == pytest.approx(0.9333, abs=1e-4)
        assert cm["aggregate"]["derived"]["cm_precision"] == pytest.approx(1.0)
        assert result["overall_score"] != pytest.approx(
            cm["aggregate"]["derived"]["cm_precision"]
        )

    def test_a_complete_failure_check_reads_both_nodes(self):
        """The two nodes scope different things, so a full check needs both.

        `aggregate.fp + fn == 0` alone answers only "did every leaf of every
        comparable object land". It says nothing about objects that were not
        comparable, which land on `overall`.
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
        assert aggregate["fp"] + aggregate["fn"] == 0

        # Its rejection is recorded here, which is why a full check reads both.
        assert overall["fd"] == 1
        assert _documented_clean_check(cm) is False

    def test_the_clean_check_catches_a_value_invented_on_a_null_leaf(self):
        """The case that makes the published check read `fp`, not `fa + fd`.

        A leaf whose ground truth is null and whose prediction supplies a value
        is `fa` at that leaf, and it rolls up into `aggregate`. The item still
        pairs, so the root `overall` stays completely clean. Any check that reads
        `fa` on `overall` alone therefore reports a hallucinated value as clean,
        which is the same shape of miss as reading `overall` alone.
        """
        good = {name: f"{name}0" for name in FIELDS}
        ground_truth = Invoice(lines=[Line(**{**good, "total": None})])
        prediction = Invoice(lines=[Line(**{**good, "total": "INVENTED"})])
        cm = ground_truth.compare_with(
            prediction, include_confusion_matrix=True, add_derived_metrics=True
        )["confusion_matrix"]

        # The hallucination lands on `aggregate`, and nowhere on `overall`.
        assert cm["aggregate"]["fa"] == 1
        assert cm["aggregate"]["fp"] == 1
        assert cm["overall"]["fa"] == 0
        assert cm["overall"]["fp"] == 0

        assert _documented_clean_check(cm) is False

        # The superseded form, kept here to pin exactly why it was replaced: it
        # reads `fa` only on `overall`, where this failure never appears.
        superseded = (
            cm["aggregate"]["fd"] + cm["aggregate"]["fn"] == 0
            and cm["overall"]["fd"] + cm["overall"]["fn"] + cm["overall"]["fa"] == 0
        )
        assert superseded is True

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


class TestTheDocsAndTheEngineCannotDrift:
    """The published snippet exists in four places; none may drift from this one.

    Both errors this file now guards against reached review because a number or a
    snippet was written into a page that nothing executed. `_documented_clean_check`
    is the executable copy, and this walks the pages to confirm they still match it.
    """

    def test_every_page_publishes_the_current_clean_check(self):
        repo_root = Path(__file__).resolve().parents[2]
        expected_lines = (
            "cm['aggregate']['fp'] + cm['aggregate']['fn'] == 0",
            "and cm['overall']['fp'] + cm['overall']['fn'] == 0",
        )
        superseded = "cm['overall']['fa']"

        for relative in _PAGES_PUBLISHING_THE_CLEAN_CHECK:
            page = repo_root / relative
            assert page.exists(), f"{relative} moved; update this list"
            text = page.read_text()

            assert "clean = (" in text, f"{relative} no longer publishes the check"
            for line in expected_lines:
                assert line in text, f"{relative} is missing: {line}"

            # The form that reported a hallucinated value as clean.
            assert superseded not in text, (
                f"{relative} still reads `fa` on `overall` alone, which is clean "
                f"on a value invented against a null ground-truth leaf"
            )
