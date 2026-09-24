"""Regression coverage for reporting the list comparisons scored by #332."""

from typing import Any, Optional

import pytest
from pydantic import BaseModel

from stickler import (
    ANLSStarComparator,
    ComparableField,
    ExactComparator,
    StructuredModel,
)
from stickler.comparators.base import BaseComparator
from stickler.structured_object_evaluator.models.non_matches_helper import (
    NonMatchesHelper,
)


class Plain(BaseModel):
    sku: str


class Other(BaseModel):
    sku: str


class Document(StructuredModel):
    items: Optional[list[Any]] = ComparableField(
        default=None, comparator=ANLSStarComparator()
    )


def report(gt, pred, model=Document, **options):
    return model(items=gt).compare_with(
        model(items=pred),
        include_confusion_matrix=True,
        document_non_matches=True,
        document_field_comparisons=True,
        **options,
    )


def test_refused_pair_is_reported_in_both_formats():
    result = report([Plain(sku="a")], [Other(sku="a")])
    assert result["field_scores"]["items"] == 0.0
    assert result["confusion_matrix"]["fields"]["items"]["overall"]["fd"] == 1
    assert len(result["non_matches"]) == 1
    entry = result["non_matches"][0]
    assert entry["field_path"] == "items[0]"
    assert entry["non_match_type"] == "false_discovery"
    assert entry["ground_truth_value"] == {"sku": "a"}
    assert entry["prediction_value"] == {"sku": "a"}
    assert entry["similarity_score"] == 0.0
    (row,) = result["field_comparisons"]
    assert (row["expected_key"], row["actual_key"]) == ("items[0]", "items[0]")
    assert row["match"] is False
    assert row["score"] == 0.0


def test_reordered_mixed_classes_do_not_report_discarded_cross_pairs():
    result = report(
        [Plain(sku="same"), Other(sku="same")],
        [Other(sku="same"), Plain(sku="same")],
    )
    assert result["field_scores"]["items"] == 1.0
    assert result["non_matches"] == []
    rows = result["field_comparisons"]
    assert {(row["expected_key"], row["actual_key"]) for row in rows} == {
        ("items[0]", "items[1]"),
        ("items[1]", "items[0]"),
    }
    assert all(row["match"] for row in rows)


def test_duplicate_refusals_keep_distinct_indices():
    result = report([Plain(sku="a"), Plain(sku="a")], [Other(sku="a")])
    counts = result["confusion_matrix"]["fields"]["items"]["overall"]
    assert (counts["fd"], counts["fn"]) == (1, 1)
    assert {entry["field_path"] for entry in result["non_matches"]} == {
        "items[0]",
        "items[1]",
    }
    assert {entry["non_match_type"] for entry in result["non_matches"]} == {
        "false_discovery",
        "false_negative",
    }


class ConstantComparator(BaseComparator):
    def _compare(self, left, right):
        return 0.6


@pytest.mark.parametrize("threshold, matches", [(0.5, True), (0.6, True), (0.8, False)])
@pytest.mark.parametrize("clip", [True, False])
def test_custom_comparator_and_threshold_are_used_by_both_reports(
    threshold, matches, clip
):
    class CustomDocument(StructuredModel):
        items: list[str] = ComparableField(
            comparator=ConstantComparator(),
            threshold=threshold,
            clip_under_threshold=clip,
        )

    result = report(["a"], ["a"], model=CustomDocument)
    assert result["field_scores"]["items"] == pytest.approx(
        0.6 if matches or not clip else 0.0
    )
    (row,) = result["field_comparisons"]
    assert row["score"] == pytest.approx(0.6)
    assert row["match"] is matches
    if matches:
        assert result["non_matches"] == []
    else:
        (entry,) = result["non_matches"]
        assert entry["similarity_score"] == pytest.approx(0.6)
        assert entry["reason"] == "below threshold (0.600 < 0.8)"


def test_exact_comparator_is_not_overridden_by_report_normalization():
    class ExactDocument(StructuredModel):
        items: list[str] = ComparableField(
            comparator=ExactComparator(case_sensitive=True)
        )

    result = report(["A"], ["a"], model=ExactDocument)
    assert result["field_scores"]["items"] == 0.0
    assert len(result["non_matches"]) == 1
    assert result["field_comparisons"][0]["match"] is False


@pytest.mark.parametrize("gt,pred,count", [([], [], 0), ([], ["a"], 1), (["a"], [], 1)])
def test_empty_lists_preserve_report_counts(gt, pred, count):
    result = report(gt, pred)
    assert len(result["non_matches"]) == count
    assert len(result["field_comparisons"]) == count


@pytest.mark.parametrize(
    "gt,pred", [(["aaa"], ["zzz"]), ([Plain(sku="a")], [Plain(sku="a")])]
)
def test_controls_and_reporting_flags_do_not_change_scores(gt, pred):
    result = report(gt, pred)
    without_reports = Document(items=gt).compare_with(
        Document(items=pred), include_confusion_matrix=True
    )
    assert "non_matches" not in without_reports
    assert "field_comparisons" not in without_reports
    assert {
        key: value
        for key, value in result.items()
        if key not in {"non_matches", "field_comparisons"}
    } == without_reports
    assert len(result["non_matches"]) == (1 if gt == ["aaa"] else 0)


@pytest.mark.parametrize(
    "non_matches,comparisons",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_refusal_reporting_flags_preserve_metrics(non_matches, comparisons):
    gt = Document(items=[Plain(sku="a")])
    pred = Document(items=[Other(sku="a")])
    baseline = gt.compare_with(pred, include_confusion_matrix=True)
    result = gt.compare_with(
        pred,
        include_confusion_matrix=True,
        document_non_matches=non_matches,
        document_field_comparisons=comparisons,
    )
    assert ("non_matches" in result) is non_matches
    assert ("field_comparisons" in result) is comparisons
    assert {key: result[key] for key in baseline} == baseline
    if non_matches:
        assert len(result["non_matches"]) == 1
    if comparisons:
        assert result["field_comparisons"][0]["match"] is False


def test_nested_report_paths_reach_the_same_class_gate():
    class Envelope(StructuredModel):
        document: Document

    result = Envelope(document=Document(items=[Plain(sku="a")])).compare_with(
        Envelope(document=Document(items=[Other(sku="a")])),
        document_non_matches=True,
        document_field_comparisons=True,
    )
    (entry,) = result["non_matches"]
    assert entry["field_path"] == "document.items[0]"
    (row,) = result["field_comparisons"]
    assert row["expected_key"] == "document.items[0]"
    assert row["match"] is False


def test_unrelated_dict_pair_in_mixed_list_is_not_refused():
    result = report(
        [{"code": "a"}, Plain(sku="b")],
        [{"code": "a"}, Other(sku="b")],
    )
    assert result["field_scores"]["items"] == pytest.approx(0.5)
    (entry,) = result["non_matches"]
    assert entry["field_path"] == "items[1]"
    assert result["field_comparisons"][0]["match"] is True


def test_structured_list_keeps_element_threshold_not_parent_comparator():
    class Item(StructuredModel):
        match_threshold = 0.9
        code: str = ComparableField(comparator=ExactComparator())

    class Parent(StructuredModel):
        # Runtime StructuredModel elements use recursive matching even when
        # the annotation also permits primitives with a field comparator.
        items: list[Any] = ComparableField(comparator=ConstantComparator(), threshold=0.9)

    result = report([Item(code="a")], [Item(code="a")], model=Parent)
    assert result["field_scores"]["items"] == 1.0
    assert result["non_matches"] == []
    (row,) = result["field_comparisons"]
    assert row["expected_key"] == "items[0].code"
    assert row["score"] == 1.0


def test_report_matching_does_not_emit_warnings_for_discarded_pairs():
    import warnings

    class Note(BaseModel):
        sku: str

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = report(
            [Plain(sku="a"), Note(sku="a")], [Note(sku="a"), Plain(sku="a")]
        )
    assert result["non_matches"] == []
    assert not [
        warning for warning in caught if "Different classes" in str(warning.message)
    ]


def test_reports_reuse_the_scoring_cost_matrix():
    class CountingComparator(BaseComparator):
        calls = 0

        def _compare(self, left, right):
            type(self).calls += 1
            return 1.0 if left == right else 0.0

    class Tags(StructuredModel):
        items: list[str] = ComparableField(comparator=CountingComparator())

    values = ["a", "b", "c", "d"]
    result = report(values, values, model=Tags)

    assert result["confusion_matrix"]["fields"]["items"]["overall"]["tp"] == 4
    assert result["non_matches"] == []
    assert len(result["field_comparisons"]) == 4
    assert CountingComparator.calls == 16


def test_reports_share_scoring_threshold_tolerance():
    class NearThresholdComparator(BaseComparator):
        def _compare(self, left, right):
            return 0.7 - 1e-12

    class Tags(StructuredModel):
        items: list[str] = ComparableField(
            comparator=NearThresholdComparator(), threshold=0.7
        )

    result = report(["a"], ["b"], model=Tags)

    assert result["confusion_matrix"]["fields"]["items"]["overall"]["tp"] == 1
    assert result["non_matches"] == []
    assert result["field_comparisons"][0]["match"] is True
    assert "within threshold tolerance" in result["field_comparisons"][0]["reason"]


def test_reports_use_the_score_from_a_stateful_comparator_once():
    class ChangingComparator(BaseComparator):
        calls = 0

        def _compare(self, left, right):
            type(self).calls += 1
            return 1.0 if type(self).calls == 1 else 0.0

    class Tags(StructuredModel):
        items: list[str] = ComparableField(comparator=ChangingComparator())

    result = report(["a"], ["a"], model=Tags)

    assert result["confusion_matrix"]["fields"]["items"]["overall"]["tp"] == 1
    assert result["non_matches"] == []
    assert result["field_comparisons"][0]["match"] is True
    assert result["field_comparisons"][0]["score"] == 1.0
    assert ChangingComparator.calls == 1


def test_direct_non_match_reason_uses_the_supplied_threshold():
    entry = NonMatchesHelper().create_non_match_entry(
        "items", "a", "b", "FD", 0, 0.6, match_threshold=0.8
    )
    assert entry["reason"] == "below threshold (0.600 < 0.8)"


def test_private_pairing_details_do_not_appear_in_public_results():
    ground_truth = Document(items=["a"])
    prediction = Document(items=["a"])
    recursive_result = ground_truth.compare_recursive(prediction)
    reported_result = report(["a"], ["a"])

    assert set(recursive_result["fields"]["items"]) == {
        "overall", "fields", "raw_similarity_score", "similarity_score",
        "threshold_applied_score", "weight",
    }
    assert set(reported_result["confusion_matrix"]["fields"]["items"]) >= {
        "overall", "fields", "raw_similarity_score", "similarity_score",
        "threshold_applied_score", "weight",
    }
    assert "_list_matching" not in reported_result["confusion_matrix"]["fields"]["items"]


@pytest.mark.parametrize("override", ["comparator", "match_threshold"])
def test_direct_structured_list_helper_rejects_ignored_field_overrides(override):
    class Item(StructuredModel):
        code: str = ComparableField(comparator=ExactComparator())

    kwargs = {override: ExactComparator() if override == "comparator" else 0.9}
    with pytest.raises(ValueError, match="StructuredModel list"):
        NonMatchesHelper().collect_list_non_matches(
            "items", [Item(code="a")], [Item(code="a")], **kwargs
        )
