"""ANLS* comparison for dict fields whose keys are not declared.

Covers the three things that make this usable rather than merely present:

1. The identity invariant. A mapping compared against itself scores 1.0, on
   every path a dict field can be declared. Before this comparator existed the
   explicit path scored 0.0 here, because the dispatcher had no dict branch and
   the declared comparator was never consulted (#297).
2. Partial credit, which is the point. Whole-object equality scored a near miss
   and a wholly wrong extraction identically, so neither ranking two extractors
   nor detecting a regression was possible (#277).
3. Tau is configurable. The per-leaf cutoff is a parameter rather than a fixed
   constant, because whether an abbreviation counts depends on the data.

See https://github.com/awslabs/stickler/issues/277 and
https://github.com/awslabs/stickler/issues/297
"""

from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel, Field

import stickler
from stickler import ANLSStarComparator, ComparableField, StructuredModel
from stickler.comparators.anls import DEFAULT_LEAF_THRESHOLD
from stickler.comparators.levenshtein import LevenshteinComparator

GT = {"vendor": "Acme Corporation", "terms": "Net 30", "po": "PO-88231"}
REORDERED = {"po": "PO-88231", "terms": "Net 30", "vendor": "Acme Corporation"}


class TestTheComparatorItself:
    def test_identity_and_key_order(self):
        """A mapping matches itself, and key order is not signal."""
        c = ANLSStarComparator()
        assert c.compare(GT, dict(GT)) == pytest.approx(1.0)
        assert c.compare(GT, REORDERED) == pytest.approx(1.0)

    def test_partial_credit_ranks_extractors(self):
        """The property whole-object equality could not provide.

        Under `==` every row below except the last scored 0.0, so a near miss
        and a wholly wrong extraction were indistinguishable.
        """
        c = ANLSStarComparator(threshold=0.5)
        near_miss = c.compare(GT, {**GT, "vendor": "Acme Corp"})
        dropped_key = c.compare(GT, {"vendor": "Acme Corporation", "terms": "Net 30"})
        hallucinated = c.compare(GT, {**GT, "currency": "USD"})
        all_wrong = c.compare(GT, {"vendor": "Zeta", "terms": "x", "po": "y"})

        assert all_wrong == pytest.approx(0.0)
        assert 0.0 < dropped_key < near_miss < 1.0
        assert 0.0 < hallucinated < 1.0

    def test_a_renamed_key_is_charged_on_both_sides(self):
        """Normalization is over the union of key sets.

        A rename counts once as missing from the prediction and once as
        unexpected in it, which is why per-key FN/FA rows would be itemizing
        information the score already carries rather than adding any.
        """
        c = ANLSStarComparator(threshold=0.5)
        renamed = c.compare(
            GT, {"vendor_name": "Acme Corporation", "terms": "Net 30", "po": "PO-88231"}
        )
        dropped = c.compare(GT, {"terms": "Net 30", "po": "PO-88231"})

        # Two of three keys still match in both cases, but the rename also adds
        # an unexpected key, so it scores strictly worse than simply dropping.
        assert renamed < dropped

    def test_tau_is_configurable_and_changes_the_verdict(self):
        """The leaf cutoff is a parameter, not a fixed constant."""
        abbreviated = {**GT, "vendor": "Acme Corp"}
        lenient = ANLSStarComparator(threshold=0.5).compare(GT, abbreviated)
        strict = ANLSStarComparator(threshold=0.85).compare(GT, abbreviated)

        assert lenient > strict, "a higher tau must reject the abbreviation"
        assert ANLSStarComparator().threshold == DEFAULT_LEAF_THRESHOLD

    def test_tau_zero_is_the_trap_it_looks_like(self):
        """Why tau cannot simply be removed.

        With no cutoff, an unrelated string earns credit for incidental
        character overlap, so a wholly wrong value scores above zero.
        """
        unrelated = {"vendor": "Zeta Ltd", "terms": "Net 30", "po": "PO-88231"}
        assert ANLSStarComparator(threshold=0.0).compare(GT, unrelated) > 0.0
        assert ANLSStarComparator(threshold=0.5).compare(GT, unrelated) < (
            ANLSStarComparator(threshold=0.0).compare(GT, unrelated)
        )

    def test_arbitrary_depth(self):
        """Nesting is handled structurally, not flattened."""
        c = ANLSStarComparator()

        def nest(depth: int, leaf: str = "value") -> Any:
            obj: Any = leaf
            for i in range(depth):
                obj = {f"k{i}": obj}
            return obj

        for depth in (1, 5, 25):
            assert c.compare(nest(depth), nest(depth)) == pytest.approx(1.0)
            assert c.compare(nest(depth), nest(depth, "WRONG")) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "other",
        ["a string", 42, ["a", "list"], None, object()],
    )
    def test_a_non_mapping_degrades_rather_than_raising(self, other):
        """An unexpected shape scores 0.0 instead of failing the evaluation.

        A comparator that raises mid-corpus costs the whole run; a shape
        mismatch is a scoring outcome, not a program error.
        """
        assert ANLSStarComparator().compare(GT, other) == pytest.approx(0.0)

    def test_config_round_trips_only_when_non_default(self):
        assert ANLSStarComparator().config is None
        assert ANLSStarComparator(threshold=0.85).config == {"threshold": 0.85}


class TestEveryWayToDeclareADictField:
    """Whichever way a dict field is declared, identity must hold (#297)."""

    def test_bare_annotation(self):
        class M(StructuredModel):
            v: Dict[str, Any] = Field(default_factory=dict)

        assert M._get_comparison_info("v").comparator.name == "ANLSStarComparator"
        assert M(v=GT).compare_with(M(v=REORDERED))["field_scores"]["v"] == 1.0

    def test_comparable_field_with_no_comparator(self):
        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField()

        assert M._get_comparison_info("v").comparator.name == "ANLSStarComparator"
        assert M(v=GT).compare_with(M(v=REORDERED))["field_scores"]["v"] == 1.0

    def test_optional_dict_annotation(self):
        class M(StructuredModel):
            v: Optional[Dict[str, Any]] = ComparableField(default=None)

        assert M._get_comparison_info("v").comparator.name == "ANLSStarComparator"
        assert M(v=GT).compare_with(M(v=REORDERED))["field_scores"]["v"] == 1.0

    def test_an_explicit_choice_is_never_overridden(self):
        """The substitution only fills a default; a named comparator survives.

        Declaring Levenshtein on a dict is a user error, but it is reported by
        warning and a false discovery rather than by raising. The shape of a
        value can be data-dependent, so raising fails partway through a corpus
        on document N after succeeding on N-1; the warning says the same thing
        without ending the run.
        """

        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField(comparator=LevenshteinComparator())

        assert M._get_comparison_info("v").comparator.name == "levenshtein"

        with pytest.warns(UserWarning, match="ANLSStarComparator"):
            result = M(v=GT).compare_with(M(v=REORDERED), include_confusion_matrix=True)

        # Identical mappings, but the declared comparator cannot see that.
        assert result["field_scores"]["v"] == 0.0
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_clip_is_off_so_partial_credit_survives(self):
        """A container keeps its partial score rather than being zeroed.

        With clip_under_threshold=True a below-threshold dict would contribute
        0.0, deleting exactly the partial credit this comparator produces.
        """

        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField()

        info = M._get_comparison_info("v")
        assert info.clip_under_threshold is False

        partial = M(v=GT).compare_with(M(v={**GT, "vendor": "Acme Corp"}))
        assert 0.0 < partial["field_scores"]["v"] < 1.0


class TestZeroConfigPath:
    def test_a_dict_field_is_inferred_structurally(self):
        class P(BaseModel):
            metadata: Dict[str, Any] = {}

        spec = stickler.eval_for(P)
        entry = spec.explain()["metadata"]
        assert entry["comparator"] == "ANLSStarComparator"
        assert entry["clip_under_threshold"] is False
        assert any("ANLSStarComparator" in why for why in entry["why"])

    def test_tau_is_the_standard_anls_value(self):
        """The zero-config default must rank the motivating cases apart.

        There is deliberately no per-call knob for tau on this path: it is a
        property of the comparator, set per field with
        ``ComparableField(comparator=ANLSStarComparator(threshold=...))``. A
        general per-field override for zero-config is #263.

        That makes the default load-bearing, which is why it is 0.5 rather than
        something stricter. At 0.85 an abbreviated value and a missing key both
        score 0.6667 on a three-key mapping, so the two are indistinguishable --
        the exact ranking failure this comparator exists to remove.
        """

        class P(BaseModel):
            metadata: Dict[str, Any] = {}

        spec = stickler.eval_for(P)
        installed = spec.eval_model._get_comparison_info("metadata").comparator
        assert installed.threshold == DEFAULT_LEAF_THRESHOLD == 0.5

        truth = P(metadata=GT)
        abbreviated = stickler.evaluate(
            truth, P(metadata={**GT, "vendor": "Acme Corp"})
        ).field_scores["metadata"]
        dropped = stickler.evaluate(
            truth, P(metadata={"vendor": "Acme Corporation", "terms": "Net 30"})
        ).field_scores["metadata"]
        assert abbreviated > dropped, (
            "at the shipped default these must be distinguishable"
        )

    def test_dump_mode_equivalence_survives_non_string_keys(self):
        """A dict is normalized to JSON form without being stringified.

        `model_dump()` gives native date keys and `model_dump(mode="json")`
        gives ISO strings. Both must score 1.0 against each other, which is
        what the wire layer's canonicalization used to guarantee by turning the
        whole mapping into a string.
        """
        import datetime

        class P(BaseModel):
            by_day: Dict[datetime.date, str] = {}

        instance = P(by_day={datetime.date(2024, 1, 1): "open"})
        spec = stickler.eval_for(P)
        native = spec.eval_model.from_json(instance.model_dump())
        json_form = spec.eval_model.from_json(instance.model_dump(mode="json"))

        assert native.compare_with(json_form)["overall_score"] == pytest.approx(1.0)


class TestRankingAcrossExtractors:
    """The end-to-end property this work exists for."""

    def test_five_extractors_are_ordered(self):
        class Invoice(BaseModel):
            invoice_id: str
            metadata: Dict[str, Any] = {}

        truth = Invoice(invoice_id="INV-1042", metadata=GT)

        def score(metadata: Dict[str, Any]) -> float:
            prediction = Invoice(invoice_id="INV-1042", metadata=metadata)
            return stickler.evaluate(truth, prediction).field_scores["metadata"]

        perfect = score(dict(GT))
        near_miss = score({**GT, "vendor": "Acme Corp"})
        hallucinated = score({**GT, "currency": "USD"})
        dropped = score({"vendor": "Acme Corporation", "terms": "Net 30"})
        all_wrong = score({"vendor": "Zeta Ltd", "terms": "x", "po": "y"})

        assert perfect == pytest.approx(1.0)
        assert all_wrong == pytest.approx(0.0)
        # Every intermediate outcome is distinguishable, which is what
        # whole-object equality could not do: it scored all four as 0.0.
        assert len({near_miss, hallucinated, dropped}) == 3
        assert all_wrong < dropped < hallucinated < near_miss < perfect
