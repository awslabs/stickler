"""Preserve pre-report-context helper extension contracts (#364)."""

import pytest

from stickler import ComparableField, ExactComparator, StructuredModel
from stickler.structured_object_evaluator.models.field_comparison_collector import (
    FieldComparisonCollector,
)
from stickler.structured_object_evaluator.models.field_comparison_helper import (
    FieldComparisonHelper,
)
from stickler.structured_object_evaluator.models.non_match_collector import (
    NonMatchCollector,
)
from stickler.structured_object_evaluator.models.non_matches_helper import (
    NonMatchesHelper,
)


class Item(StructuredModel):
    code: str = ComparableField(comparator=ExactComparator())


@pytest.mark.parametrize("helper_type", [FieldComparisonHelper, NonMatchesHelper])
@pytest.mark.parametrize("hook", ["get_optimal_assignments", "process_matched_pairs"])
@pytest.mark.parametrize("forwarding_wrapper", [False, True])
def test_original_list_hooks_remain_callable(helper_type, hook, forwarding_wrapper):
    calls = []

    class Assignments(helper_type):
        def get_optimal_assignments(self, gt_list, pred_list):
            calls.append("assignments")
            return super().get_optimal_assignments(gt_list, pred_list)

    class Pairs(helper_type):
        def process_matched_pairs(self, field, gt, pred, pairs, threshold):
            calls.append("pairs")
            return super().process_matched_pairs(field, gt, pred, pairs, threshold)

    parent = Assignments if hook == "get_optimal_assignments" else Pairs
    if forwarding_wrapper:
        # A generic forwarding wrapper is not evidence that the wrapped old
        # implementation accepts newly introduced options.
        def forward(self, *args, **kwargs):
            return getattr(super(type(self), self), hook)(*args, **kwargs)

        parent = type("ForwardingHelper", (parent,), {hook: forward})

    rows = parent().collect_list_entries(
        "items", [Item(code="a"), Item(code="b")], [Item(code="b"), Item(code="a")]
    )
    assert len(calls) == 1
    if helper_type is NonMatchesHelper:
        assert rows == []
    else:
        assert [
            (row["expected_key"], row["actual_key"], row["match"]) for row in rows
        ] == [
            ("items[0].code", "items[1].code", True),
            ("items[1].code", "items[0].code", True),
        ]


def test_non_match_wrapper_preserves_original_list_entry_override():
    class Legacy(NonMatchesHelper):
        def collect_list_entries(self, field, gt, pred):
            return [{"field_path": field, "match": False, "custom": (gt, pred)}]

    assert Legacy().collect_list_non_matches("items", ["a"], ["b"]) == [
        {"field_path": "items", "match": False, "custom": (["a"], ["b"])}
    ]


@pytest.mark.parametrize("non_matches", [False, True])
@pytest.mark.parametrize("with_context", [False, True], ids=["direct", "scored"])
def test_collectors_preserve_original_helper_entry_override(non_matches, with_context):
    class Values(StructuredModel):
        items: list[str] = ComparableField(comparator=ExactComparator())

    class Fields(FieldComparisonHelper):
        def collect_list_entries(self, field, gt, pred):
            return [{"field": field, "custom": (gt, pred)}]

    class NonMatches(NonMatchesHelper):
        def collect_list_non_matches(self, field, gt, pred):
            return [{"field": field, "custom": (gt, pred)}]

    gt, pred = Values(items=["a"]), Values(items=["b"])
    recursive = gt.compare_recursive(pred)
    kwargs = {}
    if with_context:
        from stickler.structured_object_evaluator.models.reporting_context import (
            ReportingContext,
        )

        kwargs["report_context"] = ReportingContext(lists={"items": matching()})
    if non_matches:
        collector = NonMatchCollector(gt)
        collector.helper = NonMatches()
        rows = collector.collect_enhanced_non_matches(recursive, pred, **kwargs)
    else:
        collector = FieldComparisonCollector(gt)
        collector.helper = Fields()
        rows = collector.collect_field_comparisons(recursive, pred, **kwargs)
    assert rows == [{"field": "items", "custom": (["a"], ["b"])}]


@pytest.mark.parametrize("with_context", [False, True], ids=["direct", "scored"])
def test_original_nested_non_match_extraction_override(with_context):
    class Legacy(NonMatchesHelper):
        def _extract_field_level_non_matches(self, field, gt, pred, index):
            return [{"field_path": f"{field}[{index}].custom", "match": False}]

    kwargs = {"matching": matching(score=1.0)} if with_context else {}
    rows = Legacy().collect_list_non_matches(
        "items", [Item(code="a")], [Item(code="a")], **kwargs
    )
    assert rows == [{"field_path": "items[0].custom", "match": False}]


@pytest.mark.parametrize("entry_point", ["create_entry", "collect_list_non_matches"])
def test_original_non_match_formatter_override_is_used(entry_point):
    class Legacy(NonMatchesHelper):
        def create_non_match_entry(self, field, gt, pred, kind, index=None, score=None):
            row = super().create_non_match_entry(field, gt, pred, kind, index, score)
            row["custom"] = True
            return row

    helper = Legacy()
    if entry_point == "create_entry":
        rows = [helper.create_entry("items", "a", "b", "FD", 0, 0.0)]
    else:
        rows = helper.collect_list_non_matches(
            "items", [Item(code="a")], [Item(code="b")]
        )
    assert len(rows) == 1
    assert rows[0]["custom"] is True
    assert rows[0]["field_path"] == "items[0]"


def matching(score=0.7 - 1e-12):
    from stickler.structured_object_evaluator.models.reporting_context import (
        ReportingContext,
    )

    return {
        "pairs": [(0, 0, score)],
        "threshold": 0.7,
        "verdicts": [True],
        "pair_results": {(0, 0): ReportingContext()},
    }


def test_explicit_optional_hook_parameters_receive_scoring_options():
    observed = {}
    comparator = ExactComparator()

    class OptIn(FieldComparisonHelper):
        def get_optimal_assignments(self, gt, pred, *, comparator=None):
            observed["comparator"] = comparator
            return super().get_optimal_assignments(gt, pred, comparator)

        def process_matched_pairs(
            self, field, gt, pred, pairs, threshold, *, verdicts=None, pair_results=None
        ):
            observed["verdicts"] = verdicts
            observed["pair_results"] = pair_results
            return super().process_matched_pairs(
                field, gt, pred, pairs, threshold, verdicts, pair_results
            )

    helper = OptIn()
    helper.collect_list_entries("items", ["a"], ["a"], comparator=comparator)
    assert observed["comparator"] is comparator
    cached = matching()
    (row,) = helper.collect_list_entries("items", ["a"], ["b"], matching=cached)
    assert row["match"] is True
    assert "within threshold tolerance" in row["reason"]
    assert observed["verdicts"] is cached["verdicts"]
    assert observed["pair_results"] is cached["pair_results"]
    assert cached["pairs"] == [(0, 0, 0.7 - 1e-12)]


@pytest.mark.parametrize("hook", ["get_optimal_assignments", "process_matched_pairs"])
def test_hook_type_errors_propagate_without_retry(hook):
    calls = []

    def fail(self, *args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("extension failed inside its body")

    helper_type = type("FailingHelper", (FieldComparisonHelper,), {hook: fail})
    with pytest.raises(TypeError, match="extension failed inside its body"):
        helper_type().collect_list_entries("items", ["a"], ["b"])
    assert len(calls) == 1
