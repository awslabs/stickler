"""`explain()` must name what actually ran, on a configured StructuredModel.

`explain()` is the justification surface for a score, so a row that names a
comparator the engine never called is worse than no row at all: it invites a
reader to trust a number for the wrong reason.

Two structural field kinds are not compared by their field's comparator:

    nested StructuredModel      compared by recursing into its fields
    List[StructuredModel]       elements paired by Hungarian, then recursed

For both, the configured path read the comparator straight off the
`ComparableField`, which is an inert `LevenshteinComparator` default the engine
never consults. Fallback labels existed but were unreachable, because
`ComparableField` always installs a comparator so the `if comparator` branch
always won.

The list row's threshold was worse than inert. `ConfigurationHelper` installs a
hardcoded `0.9` there, which gates nothing: the field score is identical at any
value. The real gate is the ELEMENT class's `match_threshold`, which is what
decides whether a pair is a TP or an FD.

See https://github.com/awslabs/stickler/pull/250
"""

from typing import List, Optional

import pytest
from pydantic import BaseModel

import stickler
from stickler.comparators.base import BaseComparator
from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

NESTED_LABEL = "StructuredModelComparator"
LIST_LABEL = "Hungarian (per-element StructuredModel)"


class Always37(BaseComparator):
    """Returns a value no real comparison would produce, so its use is visible."""

    def _compare(self, a, b):  # noqa: D102
        return 0.37


class Line(StructuredModel):
    match_threshold = 0.8

    sku: Optional[str] = ComparableField(default=None)


class Addr(StructuredModel):
    city: Optional[str] = ComparableField(default=None)


class PlainLine(BaseModel):
    sku: Optional[str] = None


class PlainAddr(BaseModel):
    city: Optional[str] = None


class PlainDoc(BaseModel):
    """The same shape with no configuration, so it routes through inference."""

    addr: Optional[PlainAddr] = None
    items: List[PlainLine] = []
    name: Optional[str] = None


class Doc(StructuredModel):
    addr: Optional[Addr] = ComparableField(default=None)
    items: List[Line] = []
    name: Optional[str] = ComparableField(default=None)


class TestTheReportedComparatorIsTheOneThatRuns:
    def test_a_nested_model_reports_recursive_comparison(self):
        assert stickler.eval_for(Doc).explain()["addr"]["comparator"] == NESTED_LABEL

    def test_a_model_list_reports_hungarian_matching(self):
        assert stickler.eval_for(Doc).explain()["items"]["comparator"] == LIST_LABEL

    def test_a_plain_field_still_reports_its_real_comparator(self):
        """Only the two structural kinds are relabelled."""
        assert (
            stickler.eval_for(Doc).explain()["name"]["comparator"]
            == "LevenshteinComparator"
        )

    def test_the_inert_default_is_never_reported_for_a_structural_field(self):
        """The specific wrong answer this fixes.

        `ComparableField` installs `LevenshteinComparator` when the caller names
        nothing, and edit distance is never what compares two nested models.
        """
        explained = stickler.eval_for(Doc).explain()
        assert explained["addr"]["comparator"] != "LevenshteinComparator"
        assert explained["items"]["comparator"] != "LevenshteinComparator"


class TestBothExplainPathsAgree:
    """A configured model and an inferred one describe one engine.

    Anyone diffing or aggregating `explain()` across the two paths saw two names
    for one behaviour, which is the divergence this closes.
    """

    @pytest.mark.parametrize("field_name", ("addr", "items"))
    def test_the_structural_labels_match_across_paths(self, field_name):
        configured = stickler.eval_for(Doc).explain()
        inferred = stickler.eval_for(PlainDoc).explain()

        assert (
            configured[field_name]["comparator"] == inferred[field_name]["comparator"]
        )

    def test_clip_is_not_claimed_on_a_row_where_it_is_not_applied(self):
        """Reporting `True` told a reader a low score had been zeroed.

        Asserted on both paths so the two cannot drift apart again.
        """
        assert (
            stickler.eval_for(PlainDoc).explain()["items"]["clip_under_threshold"]
            is False
        )
        assert (
            stickler.eval_for(Doc).explain()["items"]["clip_under_threshold"] is False
        )


class TestTheReportedThresholdIsTheRealGate:
    def test_a_model_list_reports_the_element_match_threshold(self):
        """Not the hardcoded `0.9` that ConfigurationHelper installs."""
        assert stickler.eval_for(Doc).explain()["items"]["threshold"] == 0.8
        assert Line.match_threshold == 0.8

    @pytest.mark.parametrize("declared", (0.1, 0.5, 0.99))
    def test_the_reported_threshold_tracks_the_element_class(self, declared):
        element = type(
            "El",
            (StructuredModel,),
            {
                "match_threshold": declared,
                "__annotations__": {"a": Optional[str]},
                "a": ComparableField(default=None),
            },
        )
        holder = type(
            "D",
            (StructuredModel,),
            {"__annotations__": {"rows": List[element]}, "rows": []},
        )

        assert stickler.eval_for(holder).explain()["rows"]["threshold"] == declared

    def test_the_element_match_threshold_is_what_classifies_a_pair(self):
        """Why that number is the honest one to publish.

        The field score is identical whatever the threshold, so the old `0.9`
        described nothing. `match_threshold` is what moves a pairing between TP
        and FD, which is the decision a reader is auditing.
        """
        outcomes = {}
        for declared in (0.1, 0.9):
            element = type(
                "El",
                (StructuredModel,),
                {
                    "match_threshold": declared,
                    "__annotations__": {"a": Optional[str]},
                    "a": ComparableField(threshold=0.0, default=None),
                },
            )
            holder = type(
                "D",
                (StructuredModel,),
                {"__annotations__": {"rows": List[element]}, "rows": []},
            )
            matrix = holder(rows=[element(a="aaaa")]).compare_with(
                holder(rows=[element(a="azzz")]), include_confusion_matrix=True
            )["confusion_matrix"]
            outcomes[declared] = (matrix["overall"]["tp"], matrix["overall"]["fd"])

        assert outcomes[0.1] == (1, 0)
        assert outcomes[0.9] == (0, 1)

    def test_a_nested_model_keeps_its_own_threshold_and_clip(self):
        """These are NOT inert for a nested model, unlike the list case.

        The field's threshold and clip apply to the subtree mean, so reporting
        them as configured is correct and they must not be rewritten.
        """

        class Parent(StructuredModel):
            kid: Optional[Addr] = ComparableField(
                threshold=0.66, clip_under_threshold=True, default=None
            )

        explained = stickler.eval_for(Parent).explain()["kid"]
        assert explained["threshold"] == 0.66
        assert explained["clip_under_threshold"] is True

    def test_the_nested_threshold_it_reports_really_does_clip(self):
        """Pins the asymmetry with the list row, so neither is "fixed" wrongly."""

        class Child(StructuredModel):
            a: Optional[str] = ComparableField(threshold=0.0, default=None)
            b: Optional[str] = ComparableField(threshold=0.0, default=None)

        def score(threshold):
            parent = type(
                "P",
                (StructuredModel,),
                {
                    "__annotations__": {"kid": Optional[Child]},
                    "kid": ComparableField(
                        threshold=threshold, clip_under_threshold=True, default=None
                    ),
                },
            )
            return parent(kid=Child(a="aaaa", b="aaaa")).compare_with(
                parent(kid=Child(a="aaaa", b="zzzz"))
            )["field_scores"]["kid"]

        assert score(0.0) == pytest.approx(0.5)
        assert score(0.6) == pytest.approx(0.0)


class TestAnIgnoredComparatorStaysVisible:
    """Relabelling must not hide a setting the user wrote and the engine drops.

    Before, `explain()` at least printed `Always37`, which revealed the dead
    configuration. Simply renaming the row to `StructuredModelComparator` would
    have removed the only signal, so the provenance trail carries it instead.
    """

    class Parent(StructuredModel):
        child: Optional[Addr] = ComparableField(
            comparator=Always37(), threshold=0.1, default=None
        )

    def test_the_row_reports_what_actually_runs(self):
        explained = stickler.eval_for(self.Parent).explain()["child"]
        assert explained["comparator"] == NESTED_LABEL

    def test_the_provenance_names_the_ignored_comparator(self):
        why = " ".join(stickler.eval_for(self.Parent).explain()["child"]["why"])
        assert "ignored" in why
        assert "Always37" in why

    def test_the_ignored_comparator_really_is_ignored(self):
        """The measurement behind the wording: 0.37 would mean it ran."""
        scores = self.Parent(child=Addr(city="a")).compare_with(
            self.Parent(child=Addr(city="b"))
        )["field_scores"]
        assert scores["child"] != pytest.approx(0.37)

    def test_no_ignored_note_when_the_user_configured_nothing(self):
        """The note is a warning, so it must not fire on ordinary models."""
        why = " ".join(stickler.eval_for(Doc).explain()["addr"]["why"])
        assert "ignored" not in why


class TestNestedPathsStillResolve:
    """#255 added the dotted rows; relabelling must not drop them."""

    @pytest.mark.parametrize("path", ("addr.city", "items.sku"))
    def test_the_nested_rows_are_present(self, path):
        assert path in stickler.eval_for(Doc).explain()

    def test_every_reported_comparator_is_a_real_class_or_a_known_label(self):
        """The two labels are not class names, so they need explicit allowance.

        Without this, a reader (or a tool) cannot tell a typo'd class name from a
        deliberate description of engine behaviour.
        """
        known_labels = {NESTED_LABEL, LIST_LABEL}
        for name, spec in stickler.eval_for(Doc).explain().items():
            reported = spec["comparator"]
            if reported in known_labels:
                continue
            assert hasattr(stickler, reported), (
                f"{name} reports '{reported}', which is neither an exported "
                f"comparator nor one of {sorted(known_labels)}"
            )


class TestPlainBaseModelFieldsKeepTheirRealComparator:
    """The relabelling must key on `StructuredModel`, not on field shape.

    `_field_kind` answers "model" / "model_list" for a plain `BaseModel` too, but
    only a `StructuredModel` gets recursive and Hungarian handling. On a plain
    `BaseModel` the field's own comparator genuinely runs, over the stringified
    objects, so labelling by shape alone hid a live comparator: the same defect
    this module exists to prevent, one shape over.
    """

    def test_a_list_of_plain_models_reports_the_comparator_that_runs(self):
        class Doc(StructuredModel):
            rows: Optional[List[PlainLine]] = ComparableField(default=None)

        assert (
            stickler.eval_for(Doc).explain()["rows"]["comparator"]
            == "LevenshteinComparator"
        )

    def test_that_comparator_really_does_run_on_a_list_of_plain_models(self):
        """The measurement behind the previous test.

        `0.857` is edit distance over the stringified elements. If Hungarian
        per-element StructuredModel matching were happening, this would not be
        the score, and the label would be the honest one.
        """

        class Doc(StructuredModel):
            rows: Optional[List[PlainLine]] = ComparableField(default=None)

        score = Doc(rows=[PlainLine(sku="a")]).compare_with(
            Doc(rows=[PlainLine(sku="b")])
        )["field_scores"]["rows"]
        assert score == pytest.approx(0.857, abs=1e-3)

    def test_a_nested_plain_model_reports_the_comparator_that_runs(self):
        class Doc(StructuredModel):
            kid: Optional[PlainAddr] = ComparableField(default=None)

        assert (
            stickler.eval_for(Doc).explain()["kid"]["comparator"]
            == "LevenshteinComparator"
        )

    def test_no_list_row_claims_to_clip(self):
        """This test previously asserted the opposite, and was wrong.

        No list clips, of any element type. The list comparators threshold per
        element and return the mean, so a mean below the field threshold survives
        unzeroed: measured 0.5 at a declared threshold of 0.7 with
        `clip_under_threshold=True`. Correcting only the `List[StructuredModel]`
        row left the other list kinds making the same false claim.
        """

        class Doc(StructuredModel):
            rows: Optional[List[PlainLine]] = ComparableField(default=None)
            tags: Optional[List[str]] = ComparableField(default=None)

        explained = stickler.eval_for(Doc).explain()
        assert explained["rows"]["clip_under_threshold"] is False
        assert explained["tags"]["clip_under_threshold"] is False

    def test_the_engine_really_does_not_clip_a_list(self):
        """The measurement behind the previous test."""

        class Doc(StructuredModel):
            tags: Optional[List[str]] = ComparableField(
                threshold=0.7, clip_under_threshold=True, default=None
            )

        score = Doc(tags=["same", "aaaa"]).compare_with(Doc(tags=["same", "zzzz"]))[
            "field_scores"
        ]["tags"]
        assert score == pytest.approx(0.5)
        assert score < 0.7


class TestTheIgnoredNoteUsesTheExplicitMarker:
    """`ComparableField` records whether the caller named a comparator.

    `json_schema_extra._comparator_explicit` is the same marker
    `ConfigurationHelper` reads before substituting a mapping comparator. Using it
    makes the note exact, where an `isinstance` carve-out could not tell an
    explicit `LevenshteinComparator` from the installed default and so stayed
    silent on a real misconfiguration.
    """

    @staticmethod
    def _explain(field):
        model = type(
            "P",
            (StructuredModel,),
            {"__annotations__": {"kid": Optional[Addr]}, "kid": field},
        )
        return " | ".join(stickler.eval_for(model).explain()["kid"]["why"])

    def test_no_note_for_a_bare_comparable_field(self):
        assert "ignored" not in self._explain(ComparableField(default=None))

    def test_a_note_for_an_explicitly_named_default_comparator(self):
        """The case an isinstance check could not reach.

        Naming `LevenshteinComparator` on a nested model field is a real dead
        setting, and it is indistinguishable from the default by class alone.
        """
        from stickler.comparators.levenshtein import LevenshteinComparator

        why = self._explain(
            ComparableField(comparator=LevenshteinComparator(), default=None)
        )
        assert "ignored" in why
        assert "LevenshteinComparator" in why

    def test_a_note_for_any_other_explicit_comparator(self):
        why = self._explain(ComparableField(comparator=Always37(), default=None))
        assert "ignored" in why
        assert "Always37" in why


class TestExplainDescendsOnlyWhereTheEngineDoes:
    """A dotted row promises a per-field comparison, so it must be real.

    A nested plain `BaseModel` is compared as a whole through the field's own
    comparator, and the elements of a `List[plain BaseModel]` likewise, so
    emitting `kid.sku` there advertised a comparison that never happens. The
    engine's own behaviour is pinned in
    `tests/structured_object_evaluator/test_nested_plain_basemodel.py`.
    """

    def test_a_nested_plain_model_emits_no_child_rows(self):
        class Doc(StructuredModel):
            kid: Optional[PlainAddr] = ComparableField(default=None)

        explained = stickler.eval_for(Doc).explain()
        assert "kid" in explained
        assert [key for key in explained if key.startswith("kid.")] == []

    def test_a_list_of_plain_models_emits_no_child_rows(self):
        class Doc(StructuredModel):
            rows: Optional[List[PlainLine]] = ComparableField(default=None)

        explained = stickler.eval_for(Doc).explain()
        assert "rows" in explained
        assert [key for key in explained if key.startswith("rows.")] == []

    def test_structured_nesting_still_emits_child_rows(self):
        """The useful case must survive: this is not a blanket removal."""
        explained = stickler.eval_for(Doc).explain()

        assert "addr.city" in explained
        assert "items.sku" in explained


class TestTheHungarianGateAgreesOnBothPaths:
    """A `List[StructuredModel]` element is passed through, not wrapped.

    `_shadow_for` leaves an element that is already a `StructuredModel` alone, so
    `StructuredListComparator` gates on that class's own `match_threshold`. The
    inference path reported the `match_threshold` argument instead, so one element
    class read `0.8` through a configured parent and `0.7` through a plain one:
    the divergence this module is supposed to close, surviving in the other
    direction.
    """

    class PlainHolder(BaseModel):
        items: List[Line] = []

    def test_a_structured_element_reports_its_own_match_threshold(self):
        assert Line.match_threshold == 0.8
        explained = stickler.eval_for(self.PlainHolder).explain()
        assert explained["items"]["threshold"] == 0.8

    def test_both_parents_report_the_same_gate_for_one_element_class(self):
        through_structured = stickler.eval_for(Doc).explain()["items"]["threshold"]
        through_plain = stickler.eval_for(self.PlainHolder).explain()["items"][
            "threshold"
        ]
        assert through_structured == through_plain == 0.8

    def test_the_reported_gate_is_the_one_that_classifies(self):
        """Why 0.8 is the honest number to publish, not 0.7.

        A pairing scoring between the two would read as a true positive against
        the reported 0.7 and is actually a false discovery.
        """
        holder = type(
            "H",
            (StructuredModel,),
            {"__annotations__": {"items": List[Line]}, "items": []},
        )
        matrix = holder(items=[Line(sku="aaaa")]).compare_with(
            holder(items=[Line(sku="aaab")]), include_confusion_matrix=True
        )["confusion_matrix"]
        assert (matrix["overall"]["tp"], matrix["overall"]["fd"]) == (0, 1)

    def test_a_plain_element_still_reports_the_argument(self):
        """Nothing to read off a plain BaseModel, so the caller's value stands."""

        class PlainElementHolder(BaseModel):
            rows: List[PlainLine] = []

        assert (
            stickler.eval_for(PlainElementHolder).explain()["rows"]["threshold"] == 0.7
        )


class TestExplicitnessSurvivesARoundTrip:
    """`from_json_schema` supplies an inferred comparator when the schema names
    none, and marking that explicit made every rebuilt field claim to be
    deliberately configured.

    Fixed in the importer rather than by excluding a comparator class here, so an
    explicitly named `LevenshteinComparator` is still reported as the dead setting
    it is on a nested-model field.
    """

    @staticmethod
    def _has_ignored_note(model, field="addr") -> bool:
        return "ignored" in " ".join(stickler.eval_for(model).explain()[field]["why"])

    def test_a_round_tripped_bare_field_gains_no_note(self):
        class Original(StructuredModel):
            addr: Optional[Addr] = ComparableField(default=None)

        rebuilt = StructuredModel.from_json_schema(Original.to_json_schema())

        assert self._has_ignored_note(Original) is False
        assert self._has_ignored_note(rebuilt) is False

    def test_an_explicitly_named_default_comparator_is_still_reported(self):
        """The case a class-based exclusion would have silenced."""
        from stickler.comparators.levenshtein import LevenshteinComparator

        class Original(StructuredModel):
            addr: Optional[Addr] = ComparableField(
                comparator=LevenshteinComparator(), default=None
            )

        assert self._has_ignored_note(Original) is True

    def test_a_round_tripped_explicit_comparator_loses_its_note(self):
        """Pins a real gap, so the reason is recorded rather than rediscovered.

        `to_json_schema()` does not write `x-aws-stickler-comparator` for an
        object-typed property, so an explicit comparator on a nested-model field
        is absent from the exported schema. The importer is therefore right to
        mark the rebuilt field as not explicit: the information is genuinely not
        there. Reporting no note is honest about the data available.

        The export gap is the same family as
        https://github.com/awslabs/stickler/issues/317, where the threshold on
        the same node is dropped, and is not fixed here.
        """

        class Original(StructuredModel):
            addr: Optional[Addr] = ComparableField(
                comparator=ExactComparator(), default=None
            )

        exported = Original.to_json_schema()["properties"]["addr"]
        assert "x-aws-stickler-comparator" not in exported

        rebuilt = StructuredModel.from_json_schema(Original.to_json_schema())
        assert self._has_ignored_note(Original) is True
        assert self._has_ignored_note(rebuilt) is False
