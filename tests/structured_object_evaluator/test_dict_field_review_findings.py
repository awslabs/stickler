"""Regressions found reviewing the ANLS* dict work, each pinned separately.

Four defects, all in the seam between "what the annotation says" and "what the
field metadata records". They are grouped here because they share that cause,
not because they share a symptom.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, MutableMapping

import pytest
from pydantic import BaseModel

import stickler
from stickler import ComparableField, StructuredModel
from stickler.comparators.exact import ExactComparator
from stickler.comparators.structured import StructuredModelComparator


class TestTheMappingFamilyIsRouted:
    """`Mapping[...]` fell through to the exotic branch and was canonicalised to
    a JSON string, so the auto path disagreed with the explicit path about the
    same annotation, and with three docs surfaces."""

    @pytest.mark.parametrize(
        "annotation", [Dict[str, str], Mapping[str, str], MutableMapping[str, str]]
    )
    def test_every_mapping_spelling_gets_the_structural_comparator(self, annotation):
        model = type("M", (BaseModel,), {"__annotations__": {"m": annotation}, "m": {}})
        entry = stickler.eval_for(model).explain()["m"]
        assert entry["comparator"] == "ANLSStarComparator"

    def test_and_therefore_earns_partial_credit(self):
        class M(BaseModel):
            m: Mapping[str, str] = {}

        score = stickler.evaluate(
            M(m={"vendor": "Acme Corporation"}), M(m={"vendor": "Acme Corp"})
        ).field_scores["m"]
        assert 0.0 < score < 1.0, "was 0.0 via ExactComparator over canonical JSON"


class TestPep563:
    """This module has `from __future__ import annotations`, so every annotation
    above is a STRING at class-creation time. The substitution used to read the
    raw `__annotations__`, see "Dict[str, Any]", decide it was not a mapping, and
    silently skip every field."""

    def test_the_engine_and_the_exported_schema_agree(self):
        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField()

        runtime = type(M._get_comparison_info("v").comparator).__name__
        exported = M.to_json_schema()["properties"]["v"]["x-aws-stickler-comparator"]
        assert runtime == "ANLSStarComparator"
        assert exported == runtime, "export must not report a comparator the engine does not use"

    def test_a_round_tripped_schema_keeps_the_comparator(self):
        """Before, re-importing installed Levenshtein EXPLICITLY, which suppressed
        the substitution and made the rebuilt model raise on a dict the original
        scored fine. Now the comparator survives the trip.

        The ANNOTATION does not: a dict field exports as `type: "string"` and
        re-imports as `Optional[str]`, so the rebuilt model cannot hold a mapping.
        That is pre-existing (identical on dev, which also exports `"string"`) and
        is tracked separately; asserting it here so the gap is recorded rather
        than mistaken for this work.
        """
        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField()

        exported = M.to_json_schema()["properties"]["v"]
        assert exported["x-aws-stickler-comparator"] == "ANLSStarComparator"

        rebuilt = StructuredModel.from_json_schema(M.to_json_schema())
        info = rebuilt._get_comparison_info("v")
        assert type(info.comparator).__name__ == "ANLSStarComparator"
        assert info.clip_under_threshold is False
        # the known gap
        assert rebuilt.model_fields["v"].annotation is not Dict[str, Any]

    def test_an_explicit_clip_choice_survives(self):
        class Stated(StructuredModel):
            v: Dict[str, Any] = ComparableField(clip_under_threshold=True)

        class Unstated(StructuredModel):
            v: Dict[str, Any] = ComparableField()

        assert Stated._get_comparison_info("v").clip_under_threshold is True
        assert Unstated._get_comparison_info("v").clip_under_threshold is False


class TestComparatorsThatCanScoreAMapping:
    """The `handles_mappings` gate zeroed comparators that work fine on a dict."""

    @pytest.mark.parametrize("comparator", [ExactComparator, StructuredModelComparator])
    def test_an_explicitly_declared_capable_comparator_is_used(self, comparator):
        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField(comparator=comparator())

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning should fire
            score = M(v={"a": 1}).compare_with(M(v={"a": 1}))["field_scores"]["v"]
        assert score == pytest.approx(1.0)

    @pytest.mark.parametrize("comparator", [ExactComparator, StructuredModelComparator])
    def test_and_key_order_does_not_matter(self, comparator):
        """ExactComparator used str(dict), so identical content scored 0.0
        whenever key order differed, and 1.0 when it happened to agree."""
        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField(comparator=comparator())

        score = M(v={"a": 1, "b": 2}).compare_with(M(v={"b": 2, "a": 1}))["field_scores"]["v"]
        assert score == pytest.approx(1.0)

    def test_a_comparator_that_genuinely_cannot_still_degrades(self):
        """Levenshtein raises on a dict, and Fuzzy scores a changed value higher
        than a mere reordering. Those stay excluded."""
        from stickler.comparators.levenshtein import LevenshteinComparator

        class M(StructuredModel):
            v: Dict[str, Any] = ComparableField(comparator=LevenshteinComparator())

        with pytest.warns(UserWarning, match="ANLSStarComparator"):
            result = M(v={"a": 1}).compare_with(M(v={"a": 1}), include_confusion_matrix=True)
        assert result["field_scores"]["v"] == 0.0
        assert result["confusion_matrix"]["overall"]["fd"] == 1


class TestSharedComparableFieldDoesNotLeak:
    def test_a_shared_field_descriptor_is_not_rewritten_by_another_model(self):
        """Pydantic does not clone the json_schema_extra closure, so mutating it
        in place retroactively changed the other field."""
        shared = ComparableField(threshold=0.8)

        class WithString(StructuredModel):
            v: str = shared

        class WithDict(StructuredModel):
            v: Dict[str, Any] = shared

        assert type(WithString._get_comparison_info("v").comparator).__name__ == "LevenshteinComparator"
        assert type(WithDict._get_comparison_info("v").comparator).__name__ == "ANLSStarComparator"
        assert WithString._get_comparison_info("v").clip_under_threshold is True
        assert WithDict._get_comparison_info("v").clip_under_threshold is False
