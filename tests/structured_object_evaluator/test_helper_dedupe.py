"""Pins for the shared missing-side, unmatched-item and null-case logic in the comparison helpers."""

import pytest

from stickler.structured_object_evaluator.models.comparison_helper_base import (
    ComparisonHelperBase,
)
from stickler.structured_object_evaluator.models.field_comparison_helper import (
    FieldComparisonHelper,
    _missing_side_reason,
)

FN = "false negative (unmatched ground truth)"
FA = "false alarm (unmatched prediction)"


class RecordingHelper(ComparisonHelperBase):
    """Records every extraction call so tests can compare call sequences."""

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
        pair_result=None,
    ):
        return [
            {
                "field": field_name,
                "gt": gt_object,
                "pred": pred_object,
                "gt_index": gt_index,
                "pred_index": pred_index,
                "match": is_match,
                "score": similarity_score,
                "reason": reason,
            }
        ]


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        ("a", None, FN),
        ("a", float("nan"), FN),
        (None, "a", FA),
        (float("nan"), "a", FA),
        (None, None, None),
        (float("nan"), None, None),
        ("a", "b", None),
        (0, 0, None),
    ],
)
def test_missing_side_reason(gt, pred, expected):
    assert _missing_side_reason(gt, pred) == expected


@pytest.mark.parametrize(
    "gt_list, pred_list",
    [
        ([], []),
        (["a", "b"], []),
        ([], ["x", "y"]),
        ([None, "a", None, "b"], [None]),
        ([None], [None, "x", None, "y"]),
        ([None, None], [None]),
        (["a"], ["x"]),
    ],
)
def test_null_cases_are_the_unmatched_cases_with_no_assignments(gt_list, pred_list):
    helper = RecordingHelper()
    rows = helper.process_null_cases("items", gt_list, pred_list)
    gt_empty = not [v for v in gt_list if v is not None]
    pred_empty = not [v for v in pred_list if v is not None]
    if gt_empty == pred_empty:
        # Both empty: nothing. Both present: not a null case.
        assert rows == []
    elif pred_empty:
        assert rows == helper.process_unmatched_gt_items("items", gt_list, [])
        assert all(row["reason"] == FN for row in rows)
    else:
        assert rows == helper.process_unmatched_pred_items("items", pred_list, [])
        assert all(row["reason"] == FA for row in rows)


def test_unmatched_items_skip_assigned_and_none_elements_and_keep_indices():
    helper = RecordingHelper()
    gt_rows = helper.process_unmatched_gt_items(
        "items", ["a", None, "c", "d"], [(0, 1), (3, 0)]
    )
    assert [(r["gt"], r["gt_index"], r["pred"], r["pred_index"]) for r in gt_rows] == [
        ("c", 2, None, None)
    ]
    pred_rows = helper.process_unmatched_pred_items(
        "items", ["x", "y", None, "z"], [(0, 0), (1, 3)]
    )
    assert [(r["pred"], r["pred_index"], r["gt"], r["gt_index"]) for r in pred_rows] == [
        ("y", 1, None, None)
    ]
    assert all(r["match"] is False and r["score"] == 0.0 for r in gt_rows + pred_rows)


def test_overriding_a_public_unmatched_method_does_not_change_null_cases():
    class Overriding(RecordingHelper):
        def process_unmatched_gt_items(self, field_name, gt_list, assignments):
            return [{"overridden": "gt"}]

        def process_unmatched_pred_items(self, field_name, pred_list, assignments):
            return [{"overridden": "pred"}]

    helper = Overriding()
    assert helper.process_null_cases("items", [], ["x"]) == RecordingHelper().process_null_cases("items", [], ["x"])
    assert helper.process_null_cases("items", ["a"], []) == RecordingHelper().process_null_cases("items", ["a"], [])
    assert helper.process_unmatched_gt_items("items", ["a"], []) == [{"overridden": "gt"}]


def test_create_entry_is_not_part_of_the_base_contract():
    # Only _extract_entries_from_objects is abstract now.
    assert ComparisonHelperBase.__abstractmethods__ == frozenset(
        {"_extract_entries_from_objects"}
    )
    FieldComparisonHelper()  # instantiable without a create_entry alias
    assert not hasattr(FieldComparisonHelper, "create_entry")

    class LegacyCreateEntry(RecordingHelper):
        def create_entry(self, *args, **kwargs):
            return {"args": args}

    assert LegacyCreateEntry().create_entry(1) == {"args": (1,)}
