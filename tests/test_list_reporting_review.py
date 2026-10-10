"""Contracts from the October review of list reporting (#364)."""

import inspect
import json
import weakref
from typing import Any

import pytest

from stickler import ComparableField, ExactComparator, StructuredModel
from stickler.comparators.base import BaseComparator
from stickler.structured_object_evaluator.models.comparison_dispatcher import (
    ComparisonDispatcher,
)
from stickler.structured_object_evaluator.models.comparison_engine import (
    ComparisonEngine,
)
from stickler.structured_object_evaluator.models.comparison_helper_base import (
    ComparisonHelperBase,
)
from stickler.structured_object_evaluator.models.field_comparison_collector import (
    FieldComparisonCollector,
)
from stickler.structured_object_evaluator.models.non_match_collector import (
    NonMatchCollector,
)
from stickler.structured_object_evaluator.models.structured_list_comparator import (
    StructuredListComparator,
)


class Item(StructuredModel):
    code: str = ComparableField(comparator=ExactComparator())


class Document(StructuredModel):
    items: list[Item]


@pytest.mark.parametrize("structured", [False, True])
def test_direct_dispatch_result_is_json_serializable_without_private_state(structured):
    class AnyDocument(StructuredModel):
        items: list[Any] = ComparableField(comparator=ExactComparator())

    model = Document if structured else AnyDocument
    items = [Item(code="a")] if structured else ["a"]
    gt, pred = model(items=items), model(items=items)
    result = gt._dispatch_field_comparison("items", gt.items, pred.items)
    serialized = json.dumps(result, default=str)
    assert "_list_matching" not in serialized
    assert "pair_results" not in serialized
    assert result["overall"]["tp"] == 1


@pytest.mark.parametrize("accepts_kwargs", [False, True])
@pytest.mark.parametrize("report", [False, True])
def test_list_scoring_overrides_keep_their_existing_call_contract(
    accepts_kwargs, report
):
    class LegacyDocument(StructuredModel):
        items: list[str] = ComparableField(comparator=ExactComparator())

        def _compare_unordered_lists(
            self,
            gt_list,
            pred_list,
            comparator,
            threshold,
            clip_under_threshold=True,
            field_name="",
        ):
            return super()._compare_unordered_lists(
                gt_list,
                pred_list,
                comparator,
                threshold,
                clip_under_threshold,
                field_name=field_name,
            )

    class KwargsDocument(LegacyDocument):
        def _compare_unordered_lists(self, *args, **kwargs):
            # Existing extensions may consume/ignore unknown options.
            kwargs.pop("pair_sink", None)
            return super()._compare_unordered_lists(*args, **kwargs)

    model = KwargsDocument if accepts_kwargs else LegacyDocument
    result = model(items=["a"]).compare_with(
        model(items=["a"]),
        include_confusion_matrix=True,
        document_non_matches=report,
        document_field_comparisons=report,
    )
    assert result["confusion_matrix"]["fields"]["items"]["overall"]["tp"] == 1
    if report:
        assert result["non_matches"] == []
        (row,) = result["field_comparisons"]
        assert row["match"] is True
        assert row["expected_key"] == row["actual_key"] == "items[0]"


@pytest.mark.parametrize(
    "non_matches,comparisons,bulk",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_only_reported_accepted_pairs_retain_child_traversals(
    monkeypatch,
    tmp_path,
    non_matches,
    comparisons,
    bulk,
):
    class TrackedResult(dict):
        pass

    references = []
    retained_counts = []
    dispatch = ComparisonDispatcher.dispatch_field_comparison
    score = StructuredListComparator._calculate_struct_list_similarity

    def tracked_dispatch(self, *args, **kwargs):
        result = dispatch(self, *args, **kwargs)
        if isinstance(self.model, Item):
            result = TrackedResult(result)
            references.append(weakref.ref(result))
        return result

    def observed_score(self, *args, **kwargs):
        result = score(self, *args, **kwargs)
        if isinstance(self.parent_model, Document):
            retained_counts.append(sum(ref() is not None for ref in references))
        return result

    monkeypatch.setattr(
        ComparisonDispatcher, "dispatch_field_comparison", tracked_dispatch
    )
    monkeypatch.setattr(
        StructuredListComparator, "_calculate_struct_list_similarity", observed_score
    )
    gt = Document(items=[Item(code="a"), Item(code="b")])
    pred = Document(items=[Item(code="a"), Item(code="x")])
    if bulk:
        from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
            BulkStructuredModelEvaluator,
        )

        output = tmp_path / "results.jsonl"
        evaluator = BulkStructuredModelEvaluator(
            target_schema=Document,
            document_non_matches=non_matches,
            individual_results_jsonl=str(output),
        )
        evaluator.update(gt, pred)
        evaluator.compute()
        result = json.loads(output.read_text(encoding="utf-8"))["comparison_result"]
    else:
        result = gt.compare_with(
            pred,
            document_non_matches=non_matches,
            document_field_comparisons=comparisons,
        )
    assert references, "The probe must observe actual child scoring."
    assert retained_counts == [1 if non_matches or comparisons else 0]
    assert all(ref() is None for ref in references)
    assert result["overall_score"] == 0.5


def test_nested_list_verdict_comes_from_element_metrics_not_aggregate_threshold():
    class Child(StructuredModel):
        match_threshold = 0.4
        values: list[str] = ComparableField(comparator=ExactComparator(), threshold=0.3)

    class Parent(StructuredModel):
        children: list[Child]

    result = Parent(children=[Child(values=["a", "b"])]).compare_with(
        Parent(children=[Child(values=["a", "x"])]),
        include_confusion_matrix=True,
        document_non_matches=True,
        document_field_comparisons=True,
    )
    counts = result["confusion_matrix"]["fields"]["children"]["fields"]["values"][
        "overall"
    ]
    assert (counts["tp"], counts["fd"]) == (1, 1)
    assert len(result["non_matches"]) == 1
    (row,) = result["field_comparisons"]
    assert row["score"] == 0.5
    assert row["match"] is False
    assert "non-matching" in row["reason"]


def test_extraction_contract_documents_cached_pair_data():
    signature = inspect.signature(ComparisonHelperBase._extract_entries_from_objects)
    assert signature.parameters["pair_result"].default is None


def test_legacy_extraction_override_works_when_a_pair_result_is_available():
    class LegacyHelper(ComparisonHelperBase):
        def create_entry(self, **kwargs):
            return kwargs

        def _extract_entries_from_objects(
            self,
            field_name,
            gt_object,
            pred_object,
            gt_index,
            pred_index,
            is_match,
            similarity_score,
            reason,
        ):
            return [{"match": is_match, "reason": reason}]

    rows = LegacyHelper().collect_list_entries(
        "items",
        [Item(code="a")],
        [Item(code="a")],
        matching={
            "pairs": [(0, 0, 1.0)],
            "threshold": 0.7,
            "verdicts": [True],
            "pair_results": {(0, 0): {"field_scores": {}}},
        },
    )
    assert rows == [{"match": True, "reason": "exact match"}]


def test_direct_collectors_preserve_primitive_threshold_tolerance():
    class NearThreshold(BaseComparator):
        def _compare(self, left, right):
            return 0.7 - 1e-12

    class Values(StructuredModel):
        values: list[str] = ComparableField(comparator=NearThreshold(), threshold=0.7)

    gt, pred = Values(values=["a"]), Values(values=["b"])
    recursive = gt.compare_recursive(pred)
    assert recursive["fields"]["values"]["overall"]["tp"] == 1
    assert NonMatchCollector(gt).collect_enhanced_non_matches(recursive, pred) == []
    (row,) = FieldComparisonCollector(gt).collect_field_comparisons(recursive, pred)
    assert row["match"] is True
    assert "within threshold tolerance" in row["reason"]


@pytest.mark.parametrize(
    "extra_options",
    [
        {},
        {"include_confusion_matrix": True},
        {"evaluator_format": True},
        {"add_confidence_metrics": True},
        {"add_bbox_metrics": True},
    ],
)
def test_reused_engine_reports_are_serializable_and_do_not_retain_prior_pairs(
    extra_options,
):
    gt = Document(items=[Item(code="a")])
    engine = ComparisonEngine(gt)
    engine.compare_with(
        Document(items=[Item(code="a")]), document_field_comparisons=True
    )
    result = engine.compare_with(
        Document(items=[Item(code="x")]),
        document_non_matches=True,
        document_field_comparisons=True,
        **extra_options,
    )
    fresh = gt.compare_with(
        Document(items=[Item(code="x")]),
        document_non_matches=True,
        document_field_comparisons=True,
        **extra_options,
    )
    assert result == fresh
    assert "_list_matching" not in json.dumps(result, default=str)
    assert "pair_results" not in json.dumps(result, default=str)


def test_engine_recovery_after_scoring_error_does_not_reuse_partial_report_state():
    class Failable(BaseComparator):
        def _compare(self, left, right):
            if right == "error":
                raise ValueError("comparison failed")
            return float(left == right)

    class Values(StructuredModel):
        first: list[str] = ComparableField(comparator=ExactComparator())
        second: str = ComparableField(comparator=Failable())

    gt = Values(first=["a"], second="a")
    engine = ComparisonEngine(gt)
    with pytest.raises(ValueError, match="comparison failed"):
        engine.compare_with(
            Values(first=["a"], second="error"), document_non_matches=True
        )
    pred = Values(first=["x"], second="a")
    assert engine.compare_with(pred, document_non_matches=True) == gt.compare_with(
        pred,
        document_non_matches=True,
    )


@pytest.mark.parametrize("option", ["add_confidence_metrics", "add_bbox_metrics"])
def test_implicit_field_reports_reuse_accepted_child_scoring(option):
    class Changing(BaseComparator):
        calls = 0

        def _compare(self, left, right):
            type(self).calls += 1
            return 1.0 if type(self).calls <= 3 else 0.0

    class Child(StructuredModel):
        value: str = ComparableField(comparator=Changing())

    class Parent(StructuredModel):
        children: list[Child]

    with pytest.warns(UserWarning, match="Single-document"):
        result = Parent(children=[Child(value="a")]).compare_with(
            Parent(children=[Child(value="a")]), **{option: True}
        )
    assert Changing.calls == 3
    (row,) = result["field_comparisons"]
    assert row["match"] is True
    assert row["score"] == 1.0


@pytest.mark.parametrize("none_first", [False, True])
def test_direct_collectors_share_nullable_list_dispatch(none_first):
    class Nullable(StructuredModel):
        items: list[Item | None]

    items = [None, Item(code="a")] if none_first else [Item(code="a"), None]
    gt, pred = Nullable(items=items), Nullable(items=list(reversed(items)))
    recursive = gt.compare_recursive(pred)
    result = gt.compare_with(
        pred, document_non_matches=True, document_field_comparisons=True
    )
    assert (
        NonMatchCollector(gt).collect_enhanced_non_matches(recursive, pred)
        == result["non_matches"]
        == []
    )
    rows = FieldComparisonCollector(gt).collect_field_comparisons(recursive, pred)
    assert rows == result["field_comparisons"]
    assert len(rows) == 2
    assert all(row["match"] for row in rows)
