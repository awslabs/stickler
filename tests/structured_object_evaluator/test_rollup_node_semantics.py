"""The two confusion-matrix rollup nodes answer two different questions.

`confusion_matrix.overall` classifies a node's direct children. On a list field
those children are the item pairings, so it reads as an object verdict: was this
pairing genuine, or spurious. At the root they are the root's own fields, so the
two units can mix in one count, which `TestTheRootMixesHeaderLeavesWithItemPairings`
pins. `confusion_matrix.aggregate` gives leaf detail for the objects that were
comparable.

`match_threshold` is the line between them, and the gating is deliberate. An
object scoring below it is classified as a single FD, a spurious non-match, and
is not descended into: reporting the leaves of an object already rejected as a
whole would be scoring something declared not comparable. A caller who wants
those leaves counted lowers `match_threshold` so the object qualifies, which
`test_lowering_match_threshold_exposes_the_leaves` pins.

The two nodes coincide whenever every child contributes the same number of rows to
each: a model with no nesting, a list whose items were all rejected, and also a
nested object holding exactly one leaf, where the object is one row and its leaf is
one row. That last case is an ACCEPTED, EXPANDED subtree, so "they coincide only
where there is nothing left to expand" is not the rule --
`test_an_accepted_single_leaf_subtree_also_coincides` pins it. One rejected subtree
among several usually makes them diverge further rather than converge, since
`aggregate` then reports a flawless precision over the accepted items only, which
`test_a_rejected_subtree_separates_the_two_numbers` pins.

Coinciding does not mean nothing was hidden, which is the separate and more useful
point. With header fields beside a list whose items were all rejected, both root
nodes read the same numbers while leaves exist that neither counted --
`TestCoincidingNodesAreNotEvidenceOfAgreement` pins that shape. Note what it does
NOT show: that shape has no accepted subtree either (the header fields are leaves
and the list is childless), so it is evidence about hiding, not a counter-example to
the coincidence rule. Conflating the two is what put a false claim on the page.

These tests pin the numbers and the snippet the documentation publishes, so
neither can go stale:

    docs/docs/Advanced/aggregate-metrics.md#which-node-answers-which-question
    docs/docs/Advanced/threshold-gated-evaluation.md
    docs/docs/Guides/Evaluation/understanding-results.md

Whether aggregates should also count the leaves of below-threshold objects is a
deliberately deferred question, not a defect. See
https://github.com/awslabs/stickler/issues/288
"""

import re
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
    # Correct today, and nothing was stopping it drifting: the page walk named
    # only the three doc pages and the repo walk covers `*.py` plus
    # `docs/**/*.md`, so the release note the snippet ships in was unguarded.
    "CHANGELOG.md",
)

# Third-party code is not ours to lint, and a contributor's environment is not
# always `.venv`. Restricting the walk to the trees we author is both the fix for
# that and a large speed-up over globbing the repo.
_TREES_WE_AUTHOR = ("src", "tests", "docs", "examples")

# `overall` and `fa` in ONE subscript chain, in either quoting style, with an
# optional node variable in front (`cm['overall']['fa']`, `overall["fa"]`).
# Deliberately not "both words appear in the block": reading `fa` off `aggregate`
# alongside `fp` off `overall` is correct and must not be flagged.
_READS_FA_OFF_OVERALL = re.compile(
    r"""\[\s*['"]overall['"]\s*\]\s*\[\s*['"]fa['"]\s*\]"""
    r"""|\boverall\s*\[\s*['"]fa['"]\s*\]"""
)


def _authored_prose(repo_root: Path):
    """Every line this repo authors about the rollup nodes: (path, lineno, text).

    `src/**/*.py` and `docs/**/*.md` are the surfaces a reader sees, and
    `tests/**/*.py` because a worked example in a test is the next person's copy
    source. `CHANGELOG.md` is included too, but only its NEWEST notes: older shipped
    release notes record what was said at the time and are not rewritten, while the
    newest section is authored by the same change as the docs and drifted from them
    twice. "Newest" is the first heading through the end of the first section with
    content, not the literal `## [Unreleased]`, which a release empties. The walk
    this replaces never reached the changelog at all, and carried an
    `exempt = {Path("CHANGELOG.md")}` that therefore could not fire -- dead code
    reading as a deliberate exemption.

    This file is skipped of necessity: the guards below spell the banned phrases out.
    """
    for path in (
        sorted(repo_root.glob("src/**/*.py"))
        + sorted(repo_root.glob("docs/**/*.md"))
        + sorted(repo_root.glob("tests/**/*.py"))
    ):
        if path == Path(__file__).resolve() or ".venv" in path.parts:
            continue
        relative = path.relative_to(repo_root)
        for number, line in enumerate(
            path.read_text(errors="ignore").splitlines(), start=1
        ):
            yield relative, number, line

    lines = (repo_root / "CHANGELOG.md").read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("## [")]
    assert starts, "CHANGELOG.md has no `## [...]` heading to scan"
    # The NEWEST notes: `## [Unreleased]` between releases, and the just-cut version
    # immediately after one. Keyed on "first section with content" rather than on the
    # literal `[Unreleased]`, because cutting a release empties that heading and moves
    # the notes under a version number, which broke every guard here the moment 1.0
    # was cut. Older shipped sections stay out, which is the original point.
    window_end = None
    for index, start in enumerate(starts):
        section_end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        if any(lines[j].strip() for j in range(start + 1, section_end)):
            window_end = section_end
            break
    # Asserted, not defaulted. `window_end = len(lines)` as a fallback would scan
    # every shipped release note -- the exact inverse of the intent above -- and do
    # it silently. The paired test derives the window the same way and asserts here
    # too, so a default would also make the two disagree in precisely the case this
    # pair exists to keep in agreement.
    assert window_end is not None, (
        "every `## [...]` section in CHANGELOG.md is empty, so there are no newest "
        "notes to scan"
    )
    for number in range(starts[0], window_end):
        yield Path("CHANGELOG.md"), number + 1, lines[number]


def _phrase_hits(repo_root: Path, banned) -> list:
    """Every authored line containing a banned phrase, wraps included.

    Matching per line let a phrase hide in a hard wrap. `CHANGELOG.md` is wrapped
    at ~80 columns, and a banned phrase was already split across two lines there,
    so the guards below passed while the claim was present. This joins each file's
    authored lines, collapses runs of whitespace, and reports the line the match
    STARTS on, so a wrapped phrase is caught and still located.
    """
    per_file = {}
    for path, number, line in _authored_prose(repo_root):
        per_file.setdefault(path, []).append((number, line))

    offenders = []
    for path, numbered in per_file.items():
        # Character offset -> source line, so a hit can be attributed.
        joined_parts, starts = [], []
        cursor = 0
        for number, line in numbered:
            text = line.strip()
            starts.append((cursor, number))
            joined_parts.append(text)
            cursor += len(text) + 1
        joined = " ".join(joined_parts).lower()
        joined = re.sub(r"\s+", " ", joined)
        for phrase in banned:
            start = joined.find(phrase)
            while start != -1:
                number = next(
                    (n for off, n in reversed(starts) if off <= start), numbered[0][0]
                )
                offenders.append(f"{path}:{number}")
                start = joined.find(phrase, start + 1)
    return sorted(set(offenders))


def _documented_unit_label(node: dict, *, is_object_list: bool) -> str:
    """Is this node's `aggregate` a count of leaves? The check the pages publish.

    The unit CANNOT be derived from the confusion matrix. Measured:

        prims      (List[str], one element wrong)   fields={}  aggregate tp=1 fd=1
        items_bad  (List[Model], item rejected)     fields={}  aggregate tp=0 fd=1

    The first is two element comparisons, which are leaves; the second is one
    rejected object. Both are childless structured nodes with non-zero counts, so
    `"fields" in node and not node["fields"]` -- the form this replaces -- called the
    primitive list's leaf counts "object rows". It also mislabelled a scalar that was
    null on both sides, which carries `fields == {}` as well.

    The earlier form before that, `node["overall"]["tp"] == 0`, was wrong in the
    other direction: true of a primitive field that simply failed.

    So the caller has to supply what only the schema knows -- whether the field is a
    `List[StructuredModel]` -- and `overall["tp"] == 0` then means every item was
    rejected.

    Published in:

        docs/docs/Advanced/aggregate-metrics.md#aggregate-counts-objects-for-an-all-rejected-list
        docs/docs/Guides/Evaluation/understanding-results.md  (prose and snippet)
    """
    counts_objects = is_object_list and node["overall"]["tp"] == 0
    return "object rows" if counts_objects else "leaves"


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
            # `CHANGELOG.md` is read through the same newest-notes window as
            # `_authored_prose`, not whole. Reading all of it forbade the retired
            # form anywhere in the file, including a future shipped note quoting it
            # as history -- which contradicts the policy that shipped notes record
            # what was said at the time and are not rewritten. It would also break
            # on archiving old releases into a separate file.
            if relative == "CHANGELOG.md":
                text = "\n".join(
                    line
                    for path, _, line in _authored_prose(repo_root)
                    if path == Path("CHANGELOG.md")
                )
                # PRESENCE is not required here, unlike the doc pages. A release
                # note publishes the snippet once and is then frozen, so the
                # snippet leaves this window as soon as any entry is added after
                # that release. Requiring it would fail on the first post-release
                # PR -- which is what happened when 1.0 was cut and the window
                # became the shipped `[1.0.0]` section.
                #
                # What must hold is the other two halves: if the newest notes DO
                # publish it, it is the current form, and they never publish the
                # retired form. Both still run below.
                publishes = "clean = (" in text
            else:
                text = page.read_text()
                publishes = True
                assert "clean = (" in text, f"{relative} no longer publishes the check"

            if publishes:
                for line in expected_lines:
                    assert line in text, f"{relative} is missing: {line}"

            # The form that reported a hallucinated value as clean.
            assert superseded not in text, (
                f"{relative} still reads `fa` on `overall` alone, which is clean "
                f"on a value invented against a null ground-truth leaf"
            )

    def test_no_file_describes_overall_as_an_object_verdict(self):
        """`overall` classifies a node's DIRECT CHILDREN, not the node itself.

        This phrasing survived three review rounds because each round corrected the
        sites that were cited and not the claim. It is only true where the node's
        children happen to be objects: on a list field. At the root the children are
        the root's own fields, so the count mixes leaves with pairings, which
        `TestTheRootMixesHeaderLeavesWithItemPairings` pins.

        Prose is not usually worth a guard, but this one is the whole subject of the
        pages, it is published to the API reference through two docstrings, and every
        recurrence has been a fresh review finding. The fourth recurrence was in
        `CHANGELOG.md` under `## [Unreleased]`, which the walk did not reach, so
        `_authored_prose` now scans the changelog's newest notes; older shipped
        release notes stay exempt because they record what was said at the time.

        This file is exempt of necessity, since `banned` below spells the phrases
        out. That exemption is doing real work rather than being a formality: the
        module docstring says a list field's `overall` "reads as an object verdict",
        which is the one context where the phrase is accurate. Read that as the
        boundary this guard cannot police, not as an oversight.
        """
        repo_root = Path(__file__).resolve().parents[2]
        # "the unit is the object" is here because it is the form that actually
        # recurred: the phrase-based guard caught `object verdict` and friends
        # while `aggregate-metrics.md` went on saying "The unit is the object" for
        # another review round, on the page every cross-link targets. Banning the
        # concept in one spelling and not the other is how a guard passes while
        # the claim survives.
        banned = (
            "object verdict",
            "verdicts at",
            "own direct classification",
            "unit is the object",
        )
        offenders = _phrase_hits(repo_root, banned)

        assert not offenders, (
            "these describe `overall` as a verdict on the node rather than a "
            "classification of its direct children, which is false at the root "
            f"and in both docstrings published to the API reference: {offenders}"
        )

    def test_only_the_newest_changelog_notes_are_scanned(self):
        """Both directions on the scanner, because the last exemption was dead code.

        `exempt = {Path("CHANGELOG.md")}` read as a policy and was unreachable: the
        walk covered `src`, `docs` and `tests` only. So this asserts the newest notes
        really are reached, and that an older shipped release really is not.

        Derives the window the same way `_authored_prose` does rather than pinning
        the literal `## [Unreleased]`. A release empties that heading and moves the
        notes under a version number, so the literal made this test and the walker
        disagree exactly when a release was cut.
        """
        repo_root = Path(__file__).resolve().parents[2]
        scanned = {(path, number) for path, number, _ in _authored_prose(repo_root)}
        changelog = (repo_root / "CHANGELOG.md").read_text().splitlines()
        # `startswith`, matching `_authored_prose`. Exact equality meant the two
        # disagreed the moment a heading gained a suffix (`## [Unreleased] - TBD`).
        starts = [
            i for i, line in enumerate(changelog, start=1) if line.startswith("## [")
        ]
        assert starts, "CHANGELOG.md has no `## [...]` heading for either side to find"

        window_end = None
        for index, start in enumerate(starts):
            section_end = starts[index + 1] if index + 1 < len(starts) else None
            body_end = (section_end - 1) if section_end else len(changelog)
            if any(line.strip() for line in changelog[start:body_end]):
                window_end = section_end
                break
        assert window_end is not None, (
            "every `## [...]` section in CHANGELOG.md is empty, or the newest notes "
            "run to end of file, so the not-scanned half cannot be checked"
        )

        assert (Path("CHANGELOG.md"), starts[0]) in scanned
        assert (Path("CHANGELOG.md"), window_end - 1) in scanned
        assert (Path("CHANGELOG.md"), window_end) not in scanned
        assert (Path("CHANGELOG.md"), len(changelog)) not in scanned

    def test_no_file_states_the_retired_all_zero_fallback_mechanism(self):
        """`aggregate` does not fall back to summing children's `overall`.

        Round four wrote that story into the warning, the `Calculation Logic` step
        beneath it and the release note, and it is not the mechanism. A rejected item
        is not descended into, so an all-rejected list has *no* child fields;
        `AggregateMetricsCalculator` then classes the node as a leaf and copies its
        own `overall`. Read literally, the retired rule sums over an empty set and
        predicts `tp=0 fd=0` where the engine reports `fd=2`, so it contradicted the
        worked output printed beside it.

        `TestAllRejectedAggregateCountsObjects` pins the real behaviour. This guard
        exists because the observable claim was right, which is what let the wrong
        causal story survive review.
        """
        repo_root = Path(__file__).resolve().parents[2]
        banned = (
            "falls back to summing",
            "recursive leaf sum",
            "leaf sum is all-zero",
            "leaf sum comes out all-zero",
            "unless every one of them is zero",
            "values are summed instead",
        )
        offenders = _phrase_hits(repo_root, banned)

        assert not offenders, (
            "these state that `aggregate` sums its children's `overall` when the "
            "leaf sum is all-zero. The engine instead treats a node with no child "
            f"fields as a leaf and copies its own `overall`: {offenders}"
        )

    def test_no_page_reads_an_evalresult_attribute_off_a_compare_with_result(self):
        """`compare_with()` returns a `dict`; only `evaluate()` returns an `EvalResult`.

        The `EvalResult.precision` note was written into a page whose every example
        is `compare_with()`, and told the reader that "on the first example above
        `result.precision` is 1.0". Copied, that line raises `AttributeError`.

        Scoped to the identifier `result`, which is what every page binds, and driven
        by the nearest preceding `result = ` in the file, so a page is judged on the
        binding actually in scope rather than on whether the entry point is mentioned
        somewhere nearby -- which is exactly what made the wrong line look fine.
        """
        repo_root = Path(__file__).resolve().parents[2]
        attributes = (
            "precision",
            "recall",
            "f1",
            "accuracy",
            "overall_score",
            "field_scores",
            "matched",
            "confusion_matrix",
        )
        reads = re.compile(r"(?<![\w.])result\.(" + "|".join(attributes) + r")\b")
        binds = re.compile(r"(?<![\w.])result\s*=\s*(?P<rhs>.+)")
        offenders = []

        for page in sorted(repo_root.glob("docs/**/*.md")):
            binding = None
            for number, line in enumerate(
                page.read_text(errors="ignore").splitlines(), start=1
            ):
                bound = binds.search(line)
                if bound:
                    binding = (number, bound.group("rhs"))
                if reads.search(line) and binding and "evaluate(" not in binding[1]:
                    offenders.append(
                        f"{page.relative_to(repo_root)}:{number} reads an EvalResult "
                        f"attribute, but `result` was bound at :{binding[0]} by "
                        f"`{binding[1].strip()}`"
                    )

        assert not offenders, (
            "these read an `EvalResult` attribute off something that is not an "
            f"`EvalResult`, so the published line raises: {offenders}"
        )

    def test_no_file_anywhere_publishes_the_superseded_form(self):
        """The page walk above cannot see a copy that is not a page.

        One survived in `tests/auto/test_risk_surface.py`, under a comment
        calling it "the reliable question", so the form this file retires was
        still in the repo as a worked example for the next person to copy. The
        page walk missed it because it matches on the docs' quoting style and
        that copy used different variable names. This one is quoting-agnostic:
        it finds every `clean = (` in the tree and checks the block under it.

        Scoped to the trees we author rather than the whole repo. Globbing
        `**/*.py` and skipping only `.venv` failed with third-party paths for
        anyone whose environment is `venv/`, `env/` or `.tox/`, reporting them
        under a message about stickler's clean check.
        """
        repo_root = Path(__file__).resolve().parents[2]
        offenders = []

        candidates = [
            path
            for tree in _TREES_WE_AUTHOR
            for pattern in ("**/*.py", "**/*.md")
            for path in repo_root.glob(f"{tree}/{pattern}")
        ]
        for path in sorted(set(candidates)):
            if path == Path(__file__).resolve():
                continue
            lines = path.read_text(errors="ignore").splitlines()
            for number, line in enumerate(lines):
                if "clean = (" not in line:
                    continue
                # Only the assignment itself. A fixed window read past the
                # closing paren, so an ordinary `assert overall["fa"] == 0`
                # written after a corrected block would fail this test, in
                # another file, for a reason unrelated to the change being
                # made. Stop when the parentheses balance.
                block_lines = []
                depth = 0
                for candidate in lines[number:]:
                    block_lines.append(candidate)
                    depth += candidate.count("(") - candidate.count(")")
                    if depth <= 0:
                        break
                block = "\n".join(block_lines)
                # The retired form reads `fa` OFF `overall`, in one subscript
                # chain: a value invented on a null ground-truth leaf is `fa` on
                # `aggregate` and leaves `overall` clean, so such a check passes
                # on a hallucination. Sum `fp` on both nodes instead.
                #
                # Matched as one chain rather than as "both words appear
                # somewhere in the block", which condemned a correct check that
                # happens to read `fa` off the OTHER node -- for example
                # `cm['overall']['fp'] == 0 and cm['aggregate']['fa'] == 0`.
                if _READS_FA_OFF_OVERALL.search(block):
                    offenders.append(f"{path.relative_to(repo_root)}:{number + 1}")

        assert not offenders, (
            "these publish a clean check reading `fa` on `overall`, which "
            f"reports clean on a hallucinated value: {offenders}"
        )


class _Contact(StructuredModel):
    """A nested object, deliberately not a list element."""

    match_threshold = 0.7

    a: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    b: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    c: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )


class TestANestedObjectIsNotThresholdGated:
    """The gating is a property of list pairing, not of objects.

    The pages said "an object below `match_threshold` is one FD and is not
    descended into" without qualification. That is true for a
    `List[StructuredModel]` item, which `StructuredListComparator` pairs and then
    accepts or rejects. A single nested `StructuredModel` field goes through
    `FieldComparator`, which has no such stage.

    Two consequences a reader with a nested-object schema needs, and neither was
    stated: the leaves are always reported, and `match_threshold` is not the knob.
    The published remedy -- "lower `match_threshold` to get leaf detail for a
    marginal object" -- was a no-op for this shape.
    """

    @staticmethod
    def _compare(field_threshold=None):
        extra = {} if field_threshold is None else {"threshold": field_threshold}

        class Doc(StructuredModel):
            contact: Optional[_Contact] = ComparableField(default=None, **extra)
            name: Optional[str] = ComparableField(
                comparator=ExactComparator(), threshold=1.0, default=None
            )

        return Doc(contact=_Contact(a="1", b="2", c="3"), name="n").compare_with(
            Doc(contact=_Contact(a="1", b="2", c="WRONG"), name="n"),
            include_confusion_matrix=True,
        )["confusion_matrix"]

    def test_the_failing_leaf_is_reported_even_when_the_object_is_rejected(self):
        """`aggregate` shows these leaves; the docs said it hid them."""
        cm = self._compare(field_threshold=0.9)
        assert (cm["overall"]["tp"], cm["overall"]["fd"]) == (1, 1)
        assert (cm["aggregate"]["tp"], cm["aggregate"]["fd"]) == (3, 1)

    def test_the_leaves_are_reported_when_it_is_accepted_too(self):
        """Same `aggregate` either way, which is the point: nothing is excluded."""
        cm = self._compare(field_threshold=0.5)
        assert (cm["overall"]["tp"], cm["overall"]["fd"]) == (2, 0)
        assert (cm["aggregate"]["tp"], cm["aggregate"]["fd"]) == (3, 1)

    def test_the_field_threshold_is_what_decides_the_verdict(self):
        boundary = {t: self._compare(field_threshold=t)["overall"] for t in (0.9, 0.5)}
        assert boundary[0.9]["fd"] == 1
        assert boundary[0.5]["fd"] == 0

    @pytest.mark.parametrize("match_threshold", (0.9, 0.7, 0.5, 0.1))
    def test_match_threshold_is_inert_for_this_shape(self, match_threshold):
        """The retired remedy, pinned as a no-op so it cannot be re-published."""
        original = _Contact.match_threshold
        try:
            _Contact.match_threshold = match_threshold
            cm = self._compare()
        finally:
            _Contact.match_threshold = original
        assert (cm["overall"]["tp"], cm["overall"]["fd"]) == (2, 0)
        assert (cm["aggregate"]["tp"], cm["aggregate"]["fd"]) == (3, 1)


class TestAllRejectedAggregateCountsObjects:
    """`aggregate` stops being a leaf count when every item of a list is rejected.

    The mechanism, which an earlier round of this file got wrong: a rejected item is
    not descended into, so a list with every item rejected has *no child fields at
    all*. `AggregateMetricsCalculator` splits leaf from parent on exactly that test
    (`aggregate_metrics_calculator.py`, `is_leaf_node`), so the node is treated as a
    leaf and its `aggregate` is a copy of its own `overall` -- one row per rejected
    item. Nothing sums the children's `overall`; there are no children to sum, and
    `test_the_rejected_list_node_has_no_children_to_sum` pins that.

    So the unit of the count changes with the data. The page promised "the unit is
    the leaf" and "primitive field metrics", so dividing an `aggregate` count by a
    leaf total, or calling its `derived` block leaf-level precision, is wrong here.
    """

    @staticmethod
    def _two_items(rejected: int):
        def item(index: int, wrong: bool) -> Line:
            values = {name: f"{name}{index}" for name in FIELDS}
            if wrong:
                values["tax"] = "WRONG"
                values["total"] = "WRONG"
            return Line(**values)

        class Doc(StructuredModel):
            items: Optional[List[Line]] = ComparableField(default=None)

        gt = [item(i, False) for i in range(2)]
        pred = [item(i, i < rejected) for i in range(2)]
        return Doc(items=gt).compare_with(
            Doc(items=pred), include_confusion_matrix=True
        )["confusion_matrix"]

    def test_one_rejected_still_counts_leaves(self):
        aggregate = self._two_items(rejected=1)["aggregate"]
        assert (aggregate["tp"], aggregate["fd"]) == (6, 0)

    def test_both_rejected_counts_objects_instead(self):
        """Twelve leaves exist; `aggregate` reports two rows."""
        cm = self._two_items(rejected=2)
        assert (cm["aggregate"]["tp"], cm["aggregate"]["fd"]) == (0, 2)
        assert (cm["overall"]["tp"], cm["overall"]["fd"]) == (0, 2)

    def test_so_the_two_nodes_agree_only_because_the_unit_changed(self):
        cm = self._two_items(rejected=2)
        assert cm["aggregate"]["fd"] == cm["overall"]["fd"] == 2

    def test_the_rejected_list_node_has_no_children_to_sum(self):
        """The actual mechanism, in both directions.

        A rejected item is not descended into, so the `fields` dict of the list node
        empties out entirely. That, and not any fallback that sums children's
        `overall`, is why the node reports object rows: with no children it is a leaf,
        and a leaf's `aggregate` is a copy of its `overall`. Read the retired rule
        literally and it sums over an empty set, predicting `tp=0 fd=0` where the
        engine reports `fd=2`.
        """
        partly = self._two_items(rejected=1)["fields"]["items"]
        wholly = self._two_items(rejected=2)["fields"]["items"]

        assert set(partly["fields"]) == set(FIELDS)  # six leaf children to sum
        assert wholly["fields"] == {}  # nothing to sum

        # A leaf's `aggregate` is its own `overall`, which is where `fd=2` comes from.
        metrics = ("tp", "fa", "fd", "fp", "tn", "fn")
        assert [wholly["aggregate"][m] for m in metrics] == [
            wholly["overall"][m] for m in metrics
        ]

        # What the retired rule predicted instead, for the record.
        summed_child_overall = sum(
            child["overall"]["fd"] for child in wholly["fields"].values()
        )
        assert summed_child_overall == 0
        assert wholly["aggregate"]["fd"] == 2


class _Header(StructuredModel):
    """A document with fields BESIDE the list, which is the ordinary shape.

    Every other fixture in this file is a model whose only field is the list, and
    that shape hides two things: `overall` at the root sums the node's direct
    children, so header leaves and item pairings land in one count; and the list's
    `aggregate` can switch to object rows while the document as a whole is plainly
    not all-rejected.
    """

    invoice_id: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    vendor: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    date: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, default=None
    )
    lines: Optional[List[Line]] = ComparableField(default=None)


def _header_doc(item_count: int, rejected: int, vendor_wrong: bool = False):
    """Three header leaves beside a list of `item_count` items, `rejected` of them bad.

    `vendor_wrong` breaks one HEADER field. Needed because a correct header leaf has
    `overall['tp'] == 1`, so a document built without it cannot tell the published
    unit condition from the retired `overall['tp'] == 0` one -- the failing primitive
    is the only node where those two disagree.
    """

    def item(index: int, wrong: bool) -> Line:
        values = {name: f"{name}{index}" for name in FIELDS}
        if wrong:
            values["tax"] = "WRONG"
            values["total"] = "WRONG"
        return Line(**values)

    common = {"invoice_id": "i", "date": "d"}
    gt = _Header(
        **common, vendor="v", lines=[item(i, False) for i in range(item_count)]
    )
    pred = _Header(
        **common,
        vendor="WRONG" if vendor_wrong else "v",
        lines=[item(i, i < rejected) for i in range(item_count)],
    )
    return gt.compare_with(pred, include_confusion_matrix=True)["confusion_matrix"]


def _primitive_list_doc():
    """A `List[str]` with one of two elements wrong.

    Childless with non-zero `aggregate` counts, exactly like an all-rejected object
    list, and yet its rows are element comparisons, which are leaves. The pair of
    shapes the published unit condition has to tell apart without help from the
    matrix, because the matrix cannot tell them apart.
    """

    class Doc(StructuredModel):
        tags: Optional[List[str]] = ComparableField(default=None)

    return Doc(tags=["a", "b"]).compare_with(
        Doc(tags=["a", "Z"]), include_confusion_matrix=True
    )["confusion_matrix"]


class TestTheRootMixesHeaderLeavesWithItemPairings:
    """`overall` at the root is not a count of list items.

    The published lookup row said "How many list items did the model find? ->
    `overall`", which is true only of a model whose sole field is the list. Add the
    header fields a real invoice has and the root count silently becomes the sum of
    two different units.
    """

    def test_the_root_count_is_leaves_plus_pairings(self):
        cm = _header_doc(item_count=5, rejected=0)
        assert cm["overall"]["tp"] == 8  # 3 header leaves + 5 item pairings

    def test_the_list_field_is_the_node_that_counts_items(self):
        cm = _header_doc(item_count=5, rejected=0)
        assert cm["fields"]["lines"]["overall"]["tp"] == 5

    def test_so_the_two_disagree_and_only_one_answers_the_question(self):
        cm = _header_doc(item_count=5, rejected=0)
        assert cm["overall"]["tp"] != cm["fields"]["lines"]["overall"]["tp"]


class TestCoincidingNodesAreNotEvidenceOfAgreement:
    """The two nodes reading alike does not mean nothing was hidden.

    With correct header fields beside a list whose items were all rejected, both
    root nodes read the same numbers while leaves exist that neither counted.

    What this shape does NOT show is that an accepted subtree can be present while
    the nodes coincide. It has no accepted subtree: the header fields are primitive
    leaves with no `fields` key, and the list is childless because every item was
    rejected. An earlier revision claimed otherwise, and named a test
    `test_and_an_accepted_subtree_is_present` whose only assertion was
    `overall['tp'] == 3` -- three matched LEAVES, which is not what the name says.
    That case is `test_an_accepted_single_leaf_subtree_also_coincides` below.
    """

    def test_both_root_nodes_read_alike(self):
        cm = _header_doc(item_count=2, rejected=2)
        assert (cm["overall"]["tp"], cm["overall"]["fd"]) == (3, 2)
        assert (cm["aggregate"]["tp"], cm["aggregate"]["fd"]) == (3, 2)

    def test_no_accepted_subtree_is_present_here(self):
        """States the shape honestly, rather than asserting its own name away.

        Three primitive leaves (no `fields` key at all) and one childless list. So
        this is evidence about hiding, not a counter-example to the coincidence rule.
        """
        cm = _header_doc(item_count=2, rejected=2)
        assert cm["overall"]["tp"] == 3  # the three header leaves matched
        headers = ["invoice_id", "vendor", "date"]
        assert all("fields" not in cm["fields"][name] for name in headers)
        assert cm["fields"]["lines"]["fields"] == {}

    def test_while_leaves_neither_node_counted_exist(self):
        """Two items of six fields is twelve leaves; `aggregate` reports five rows."""
        cm = _header_doc(item_count=2, rejected=2)
        assert cm["aggregate"]["tp"] + cm["aggregate"]["fd"] == 5

    def test_the_document_is_not_all_rejected(self):
        """So a reader checking the document-level condition is not protected."""
        cm = _header_doc(item_count=2, rejected=2)
        assert cm["overall"]["tp"] > 0

    def test_the_list_field_is_where_the_condition_shows(self):
        """It is the list node, not the root, that lost its children.

        `overall['tp'] == 0` also holds here, and was published as the check for a
        while, but it is not the condition: a primitive field that merely failed has
        `tp == 0` too. `_documented_unit_label` is the discriminator, and it needs
        the schema's answer for the field, which is why `is_object_list` is passed
        explicitly on both calls -- `lines` is a `List[StructuredModel]`, the root
        document is not a list at all.
        """
        cm = _header_doc(item_count=2, rejected=2)
        lines = cm["fields"]["lines"]

        assert lines["fields"] == {}
        assert _documented_unit_label(lines, is_object_list=True) == "object rows"
        assert _documented_unit_label(cm, is_object_list=False) == "leaves"

    def test_root_precision_reads_as_a_leaf_rate_and_is_not_one(self):
        cm = _header_doc(item_count=2, rejected=2)
        assert cm["aggregate"]["derived"]["cm_precision"] == pytest.approx(0.6)


class TestTheUnitLabelPublishedForRankingSections:
    """The ranking snippet annotates each section with the unit of its counts.

    Two earlier forms of that annotation were wrong in opposite directions:

        node["overall"]["tp"] == 0            also true of a FAILING PRIMITIVE
        "fields" in node and not node["fields"]   also true of a PRIMITIVE LIST

    The second is the subtler one and is why the annotation now needs the schema. A
    `List[str]` with one element wrong and a `List[Model]` with its only item
    rejected are indistinguishable in the confusion matrix -- both `fields == {}`
    with non-zero counts -- yet the first is element comparisons, which are leaves,
    and the second is one row per rejected object. A scalar null on both sides also
    carries `fields == {}`.

    So `_documented_unit_label` takes `is_object_list` from the caller, which is what
    only the model knows, and these pin it in both directions.
    """

    @staticmethod
    def _sections(**kwargs):
        return _header_doc(**kwargs)["fields"]

    def test_a_failing_primitive_is_still_one_leaf(self):
        """The row the first retired check mislabelled."""

        def item(index):
            return Line(**{name: f"{name}{index}" for name in FIELDS})

        common = {"invoice_id": "i", "date": "d"}
        gt = _Header(**common, vendor="v", lines=[item(i) for i in range(5)])
        pred = _Header(**common, vendor="WRONG", lines=[item(i) for i in range(5)])
        sections = gt.compare_with(pred, include_confusion_matrix=True)[
            "confusion_matrix"
        ]["fields"]

        assert sections["vendor"]["overall"]["tp"] == 0  # what the first check saw
        assert "fields" not in sections["vendor"]  # a leaf has no `fields` key
        assert (
            _documented_unit_label(sections["vendor"], is_object_list=False) == "leaves"
        )

    def test_a_primitive_list_is_a_leaf_count(self):
        """The row the second retired check mislabelled.

        Two elements, one wrong: `aggregate tp=1 fd=1` really is two ELEMENT
        comparisons, and an element of a primitive list is a leaf. The node is
        childless all the same, so a check keyed on that called it object rows.
        """

        class Doc(StructuredModel):
            tags: Optional[List[str]] = ComparableField(default=None)

        cm = Doc(tags=["a", "b"]).compare_with(
            Doc(tags=["a", "Z"]), include_confusion_matrix=True
        )["confusion_matrix"]
        tags = cm["fields"]["tags"]

        assert tags["fields"] == {}  # childless, like an all-rejected object list
        assert (tags["aggregate"]["tp"], tags["aggregate"]["fd"]) == (1, 1)
        assert _documented_unit_label(tags, is_object_list=False) == "leaves"

    def test_the_two_shapes_are_indistinguishable_in_the_matrix(self):
        """Why the schema has to supply the answer, stated as an assertion.

        If these two ever became distinguishable, the annotation could be derived
        again and this whole parameter could go.
        """

        class Doc(StructuredModel):
            tags: Optional[List[str]] = ComparableField(default=None)
            rows: Optional[List[Line]] = ComparableField(default=None)

        wrong = Line(**{**{n: f"{n}0" for n in FIELDS}, "tax": "X", "total": "X"})
        cm = Doc(
            tags=["a", "b"], rows=[Line(**{n: f"{n}0" for n in FIELDS})]
        ).compare_with(
            Doc(tags=["a", "Z"], rows=[wrong]), include_confusion_matrix=True
        )["confusion_matrix"]

        tags, rows = cm["fields"]["tags"], cm["fields"]["rows"]
        assert tags["fields"] == rows["fields"] == {}
        assert tags["aggregate"]["tp"] + tags["aggregate"]["fd"] > 0
        assert rows["aggregate"]["tp"] + rows["aggregate"]["fd"] > 0
        # Same shape, different unit. Only the annotation tells them apart.
        assert _documented_unit_label(tags, is_object_list=False) == "leaves"
        assert _documented_unit_label(rows, is_object_list=True) == "object rows"

    def test_an_all_rejected_list_is_not_a_leaf_count(self):
        sections = self._sections(item_count=2, rejected=2)
        assert _documented_unit_label(sections["lines"], is_object_list=True) == (
            "object rows"
        )

    def test_a_partly_rejected_list_is_still_a_leaf_count(self):
        sections = self._sections(item_count=2, rejected=1)
        assert (
            _documented_unit_label(sections["lines"], is_object_list=True) == "leaves"
        )

    def test_a_clean_list_is_a_leaf_count(self):
        sections = self._sections(item_count=5, rejected=0)
        assert (
            _documented_unit_label(sections["lines"], is_object_list=True) == "leaves"
        )

    def test_a_list_null_on_both_sides_reports_no_rows_at_all(self):
        """`tn=1` and no children. Calling it either unit is vacuous, but it must
        not be reported as a leaf rate, since there are no leaves."""
        common = {"invoice_id": "i", "vendor": "v", "date": "d"}
        cm = _Header(**common, lines=None).compare_with(
            _Header(**common, lines=None), include_confusion_matrix=True
        )["confusion_matrix"]
        lines = cm["fields"]["lines"]

        assert (lines["overall"]["tn"], lines["overall"]["tp"]) == (1, 0)
        assert lines["fields"] == {}
        assert lines["aggregate"]["tp"] + lines["aggregate"]["fd"] == 0
        assert _documented_unit_label(lines, is_object_list=True) == "object rows"

    def test_a_nested_object_is_a_leaf_count(self):
        """It is never gated, so it always still has its children."""

        class Doc(StructuredModel):
            contact: Optional[_Contact] = ComparableField(default=None, threshold=0.9)

        cm = Doc(contact=_Contact(a="1", b="2", c="3")).compare_with(
            Doc(contact=_Contact(a="1", b="2", c="WRONG")),
            include_confusion_matrix=True,
        )["confusion_matrix"]
        contact = cm["fields"]["contact"]

        assert contact["overall"]["fd"] == 1  # rejected at the field's own threshold
        assert set(contact["fields"]) == {"a", "b", "c"}  # and still expanded
        assert _documented_unit_label(contact, is_object_list=False) == "leaves"

    def test_both_pages_say_the_unit_cannot_be_derived_from_the_matrix(self):
        """Both claims, matched on their full phrases rather than on substrings.

        An earlier version of this guard tested for `"cannot be"`, which
        `aggregate-metrics.md` also contains in an unrelated sentence about the
        all-zero fallback, so deleting the actual claim left the guard passing.
        Whitespace is collapsed first so a reflow across a line break still matches.
        """
        repo_root = Path(__file__).resolve().parents[2]
        required = (
            "cannot be derived from the confusion matrix",
            "list of primitives",
        )
        for relative in (
            "docs/docs/Advanced/aggregate-metrics.md",
            "docs/docs/Guides/Evaluation/understanding-results.md",
        ):
            page = repo_root / relative
            assert page.exists(), page
            text = " ".join(page.read_text().split()).lower()
            for phrase in required:
                assert phrase in text, (
                    f"{relative} no longer says '{phrase}'. The unit cannot be read "
                    f"off the matrix, and both derived conditions published before "
                    f"this were wrong, in opposite directions."
                )

    def test_the_published_snippet_agrees_with_this_helper(self):
        """Execute the page's own lines rather than matching on their text.

        Both wrong annotations reached review because they were written into a page
        that nothing ran. This lifts the published condition out of the snippet and
        checks it against `_documented_unit_label` on every node shape.
        """
        repo_root = Path(__file__).resolve().parents[2]
        page = (
            repo_root / "docs/docs/Guides/Evaluation/understanding-results.md"
        ).read_text()
        cond = re.search(r"^\s*counts_objects = (.+)$", page, re.M)
        unit = re.search(r"^\s*unit = (.+)$", page, re.M)
        assert cond and unit, "the ranking snippet lost its unit label"

        def published_label(node: dict, section: str, object_lists: set) -> str:
            counts_objects = eval(  # noqa: S307
                cond.group(1),
                {},
                {"data": node, "section": section, "OBJECT_LISTS": object_lists},
            )
            return eval(unit.group(1), {}, {"counts_objects": counts_objects})  # noqa: S307

        header = _header_doc(item_count=2, rejected=2)
        clean = _header_doc(item_count=2, rejected=0)
        partial = _header_doc(item_count=2, rejected=1)
        # A header field that FAILED. This is the discriminating case: it is the only
        # node where the published condition and the retired `overall['tp'] == 0`
        # form disagree, so without it the snippet could drop `section in
        # OBJECT_LISTS` and every assertion here would still pass. Verified: with
        # that guard removed, this case is what fails.
        failed_leaf = _header_doc(item_count=2, rejected=0, vendor_wrong=True)
        # A list of PRIMITIVES that is childless with non-zero counts, which is the
        # shape the other retired form (`'fields' in data and not data['fields']`)
        # mislabelled.
        primitive_list = _primitive_list_doc()
        cases = [
            (header["fields"]["lines"], "lines", {"lines"}),
            (clean["fields"]["lines"], "lines", {"lines"}),
            (partial["fields"]["lines"], "lines", {"lines"}),
            (header["fields"]["vendor"], "vendor", {"lines"}),
            (failed_leaf["fields"]["vendor"], "vendor", {"lines"}),
            (primitive_list["fields"]["tags"], "tags", {"lines"}),
            (header, "root", {"lines"}),
        ]
        assert failed_leaf["fields"]["vendor"]["overall"]["tp"] == 0, (
            "the discriminating case must actually have tp == 0, or it discriminates "
            "nothing"
        )
        assert primitive_list["fields"]["tags"]["fields"] == {}, (
            "the primitive-list case must actually be childless"
        )
        for node, section, object_lists in cases:
            assert published_label(node, section, object_lists) == (
                _documented_unit_label(node, is_object_list=section in object_lists)
            ), section

        labels = {published_label(n, s, o) for n, s, o in cases}
        assert labels == {"leaves", "object rows"}


class TestAnAcceptedSubtreeCanCoincideToo:
    """The case that actually falsifies "coincide only where nothing is left to expand".

    A nested object holding exactly ONE leaf: the object is one row on `overall` and
    its single leaf is one row on `aggregate`, so the two agree while the subtree is
    accepted AND descended into. The published rule characterised coincidence by the
    absence of an expandable subtree, and this shape has one.
    """

    @staticmethod
    def _one_leaf_doc():
        Kid = type(
            "Kid",
            (StructuredModel,),
            {
                "__annotations__": {"a": Optional[str]},
                "match_threshold": 0.5,
                "a": ComparableField(
                    comparator=ExactComparator(), threshold=1.0, default=None
                ),
            },
        )
        Doc = type(
            "Doc",
            (StructuredModel,),
            {
                "__annotations__": {"kid": Optional[Kid]},
                "kid": ComparableField(
                    comparator=ExactComparator(), threshold=1.0, default=None
                ),
            },
        )
        return Doc, Kid

    def _cm(self):
        Doc, Kid = self._one_leaf_doc()
        return Doc(kid=Kid(a="x")).compare_with(
            Doc(kid=Kid(a="x")), include_confusion_matrix=True
        )["confusion_matrix"]

    def test_an_accepted_single_leaf_subtree_also_coincides(self):
        cm = self._cm()
        assert (cm["overall"]["tp"], cm["overall"]["fd"]) == (1, 0)
        assert (cm["aggregate"]["tp"], cm["aggregate"]["fd"]) == (1, 0)

    def test_and_that_subtree_really_was_expanded(self):
        """Without this the case would be indistinguishable from a childless node."""
        cm = self._cm()
        assert sorted(cm["fields"]["kid"]["fields"].keys()) == ["a"]

    def test_and_it_really_was_accepted(self):
        """A rejected subtree would coincide for the uninteresting reason."""
        cm = self._cm()
        assert cm["fields"]["kid"]["overall"]["fd"] == 0
        assert cm["fields"]["kid"]["overall"]["tp"] == 1
