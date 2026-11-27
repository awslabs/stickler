"""One box held in a list field is scored as one box (#635).

A list field pairs its elements, so ``BBoxIoUComparator`` on ``List[float]``
used to receive single coordinates, score ``0.0`` against an identical box and
count one decision per coordinate. A comparator now declares, per value, that
it scores a list whole through ``scores_as_one_value()``. The annotation
decides when it names the element type, so a prediction's shape cannot collapse
a list of boxes; a bare ``list`` is decided by the ground truth.
"""

import typing
from typing import Any, List, NewType, Optional, Sequence, Tuple, Union

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import BBoxIoUComparator, NumericComparator
from stickler.comparators.base import BaseComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

BOX = [0.0, 0.0, 10.0, 10.0]
HALF_BOX = [0.0, 0.0, 5.0, 10.0]


class FloatList(StructuredModel):
    box: List[float] = ComparableField(comparator=BBoxIoUComparator(), threshold=0.5)


class BareList(StructuredModel):
    box: list = ComparableField(comparator=BBoxIoUComparator(), threshold=0.5)


class OptionalFloatList(StructuredModel):
    box: Optional[List[float]] = ComparableField(
        comparator=BBoxIoUComparator(), threshold=0.5
    )


class OptionalElementList(StructuredModel):
    box: List[Optional[float]] = ComparableField(
        comparator=BBoxIoUComparator(), threshold=0.5
    )


class BoxList(StructuredModel):
    boxes: List[List[float]] = ComparableField(
        comparator=BBoxIoUComparator(), threshold=0.5
    )


class SequenceList(StructuredModel):
    box: Sequence[float] = ComparableField(
        comparator=BBoxIoUComparator(), threshold=0.5
    )


class AnyList(StructuredModel):
    box: List[Any] = ComparableField(comparator=BBoxIoUComparator(), threshold=0.5)


ONE_BOX_MODELS = [FloatList, BareList, OptionalFloatList, SequenceList, AnyList]
THREE_FLAT_BOXES = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
TWO_POINT_SHAPE = [[0.0, 0.0], [10.0, 10.0]]


def _box_counts(result, field="box"):
    overall = result["confusion_matrix"]["fields"][field]["overall"]
    return {k: overall[k] for k in ("tp", "fd", "fa", "fn")}


@pytest.mark.parametrize("model", ONE_BOX_MODELS, ids=lambda m: m.__name__)
def test_identical_box_scores_one_and_counts_one_tp(model):
    gt, pred = model(box=BOX), model(box=list(BOX))
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert gt.compare(pred) == 1.0
    assert result["overall_score"] == 1.0
    assert _box_counts(result) == {"tp": 1, "fd": 0, "fa": 0, "fn": 0}
    assert gt.compare_with(pred, evaluator_format=True, document_non_matches=True)[
        "overall"
    ]["anls_score"] == pytest.approx(1.0)


def test_bulk_evaluator_counts_one_box():
    evaluator = BulkStructuredModelEvaluator(target_schema=FloatList)
    evaluator.update(FloatList(box=BOX), FloatList(box=BOX))

    assert evaluator.compute().field_metrics["box"]["tp"] == 1


@pytest.mark.parametrize(
    "gt_box,pred_box",
    [(BOX, HALF_BOX), ([[0, 0], [10, 10]], [[0, 0], [5, 10]])],
    ids=["flat", "two-point"],
)
def test_partial_overlap_scores_the_direct_iou(gt_box, pred_box):
    direct = BBoxIoUComparator().compare(gt_box, pred_box)
    gt, pred = BareList(box=gt_box), BareList(box=pred_box)

    assert direct == pytest.approx(0.5)
    assert gt.compare(pred) == pytest.approx(direct)
    assert gt.compare_with(pred)["overall_score"] == pytest.approx(direct)


@pytest.mark.parametrize(
    "model,pred_box",
    [(FloatList, [0.0, 0.0, 10.0]), (BareList, ["a", "b", "c", "d"])],
    ids=["short", "strings"],
)
def test_malformed_prediction_is_one_false_discovery(model, pred_box):
    gt, pred = model(box=BOX), model(box=pred_box)
    result = gt.compare_with(
        pred, include_confusion_matrix=True, document_non_matches=True
    )

    assert gt.compare(pred) == result["overall_score"] == 0.0
    assert _box_counts(result) == {"tp": 0, "fd": 1, "fa": 0, "fn": 0}
    assert [nm["field_path"] for nm in result["non_matches"]] == ["box"]


def test_reports_have_one_row_for_the_box():
    gt, pred = FloatList(box=BOX), FloatList(box=[20.0, 20.0, 30.0, 30.0])
    result = gt.compare_with(
        pred, document_non_matches=True, document_field_comparisons=True
    )

    assert [nm["field_path"] for nm in result["non_matches"]] == ["box"]
    rows = result["field_comparisons"]
    assert [(r["expected_key"], r["expected_value"]) for r in rows] == [("box", BOX)]


def test_list_of_boxes_still_pairs_box_by_box():
    other = [20.0, 20.0, 30.0, 30.0]
    gt, pred = BoxList(boxes=[BOX, other]), BoxList(boxes=[other, HALF_BOX])
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert result["overall_score"] == pytest.approx(0.75)
    assert _box_counts(result, "boxes") == {"tp": 2, "fd": 0, "fa": 0, "fn": 0}


def test_non_declaring_comparator_still_pairs_elements():
    class Numbers(StructuredModel):
        values: List[float] = ComparableField(
            comparator=NumericComparator(), threshold=0.5
        )

    gt, pred = Numbers(values=BOX), Numbers(values=list(reversed(BOX)))
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert NumericComparator().scores_as_one_value(BOX) is False
    assert result["overall_score"] == 1.0
    assert _box_counts(result, "values")["tp"] == 4


def test_json_schema_array_field_scores_one_box():
    model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Schema",
            "properties": {
                "box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "x-aws-stickler-comparator": "BBoxIoUComparator",
                }
            },
        }
    )
    gt, pred = model(box=BOX), model(box=HALF_BOX)

    assert gt.compare(pred) == pytest.approx(0.5)
    assert gt.compare_with(pred)["overall_score"] == pytest.approx(0.5)


def test_config_list_field_scores_one_box():
    model = StructuredModel.model_from_json(
        {
            "model_name": "Config",
            "fields": {"box": {"type": "list", "comparator": "BBoxIoUComparator"}},
        }
    )
    gt, pred = model(box=BOX), model(box=HALF_BOX)

    assert gt.compare(pred) == pytest.approx(0.5)
    assert gt.compare_with(pred)["overall_score"] == pytest.approx(0.5)


FA = {"tp": 0, "fd": 0, "fa": 1, "fn": 0}
FN = {"tp": 0, "fd": 0, "fa": 0, "fn": 1}


@pytest.mark.parametrize(
    "model,gt_box,pred_box,expected",
    [
        (OptionalFloatList, [], BOX, FA),
        (OptionalFloatList, None, BOX, FA),
        (OptionalFloatList, BOX, None, FN),
        (OptionalElementList, [None] * 4, BOX, FA),
        (OptionalElementList, BOX, [None] * 4, FN),
        (FloatList, BOX, [float("nan")] * 4, FN),
        (BareList, [None], BOX, FA),
        (BareList, BOX, [None], FN),
    ],
    ids=[
        "empty-gt",
        "none-gt",
        "none-pred",
        "all-none-gt",
        "all-none-pred",
        "all-nan-pred",
        "bare-none-element-gt",
        "bare-none-element-pred",
    ],
)
def test_box_against_absent_side_is_one_fn_or_fa(model, gt_box, pred_box, expected):
    gt, pred = model(box=gt_box), model(box=pred_box)
    result = gt.compare_with(
        pred,
        include_confusion_matrix=True,
        document_non_matches=True,
        document_field_comparisons=True,
    )

    assert _box_counts(result) == expected
    assert len(result["non_matches"]) == 1
    [row] = result["field_comparisons"]
    assert row["reason"].startswith(("false alarm", "false negative"))


@pytest.mark.parametrize("model", [BoxList, BareList], ids=lambda m: m.__name__)
def test_prediction_shape_does_not_collapse_a_list_of_boxes(model):
    field = "boxes" if model is BoxList else "box"
    gt, pred = model(**{field: THREE_FLAT_BOXES}), model(**{field: TWO_POINT_SHAPE})
    result = gt.compare_with(
        pred, include_confusion_matrix=True, document_non_matches=True
    )

    assert _box_counts(result, field) == {"tp": 0, "fd": 2, "fa": 0, "fn": 1}
    assert len(result["non_matches"]) == 3


_ALIAS_TYPE = getattr(typing, "TypeAliasType", None)
SEQUENCE_ELEMENT_ANNOTATIONS = [
    pytest.param(List[Union[List[float], Tuple[float, float]]], id="union"),
    pytest.param(List[Union[List[float], str]], id="union-with-str"),
    pytest.param(List[NewType("Box", list)], id="newtype"),
    pytest.param(
        List[_ALIAS_TYPE("BoxAlias", List[float])] if _ALIAS_TYPE else None,
        marks=pytest.mark.skipif(_ALIAS_TYPE is None, reason="Python 3.12+"),
        id="type-alias",
    ),
]


@pytest.mark.parametrize("annotation", SEQUENCE_ELEMENT_ANNOTATIONS)
def test_sequence_element_through_union_or_alias_stays_a_list_of_boxes(annotation):
    model = type(
        "Boxes",
        (StructuredModel,),
        {
            "__annotations__": {"box": annotation},
            "box": ComparableField(comparator=BBoxIoUComparator(), threshold=0.5),
        },
    )
    gt, pred = model(box=THREE_FLAT_BOXES), model(box=TWO_POINT_SHAPE)
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert _box_counts(result) == {"tp": 0, "fd": 2, "fa": 0, "fn": 1}


def test_json_schema_any_of_array_items_stays_a_list_of_boxes():
    model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Schema",
            "properties": {
                "box": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "number"}},
                            {"type": "array", "items": {"type": "integer"}},
                        ]
                    },
                    "x-aws-stickler-comparator": "BBoxIoUComparator",
                    "x-aws-stickler-threshold": 0.5,
                }
            },
        }
    )
    gt, pred = model(box=THREE_FLAT_BOXES), model(box=TWO_POINT_SHAPE)
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert _box_counts(result) == {"tp": 0, "fd": 2, "fa": 0, "fn": 1}


def test_scalar_union_element_is_one_box():
    class UnionElements(StructuredModel):
        box: List[Union[int, float]] = ComparableField(
            comparator=BBoxIoUComparator(), threshold=0.5
        )

    gt, pred = UnionElements(box=BOX), UnionElements(box=list(BOX))
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert _box_counts(result) == {"tp": 1, "fd": 0, "fa": 0, "fn": 0}


def test_list_of_points_annotation_is_not_read_as_one_box():
    gt, pred = BoxList(boxes=TWO_POINT_SHAPE), BoxList(boxes=TWO_POINT_SHAPE)
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert _box_counts(result, "boxes") == {"tp": 0, "fd": 2, "fa": 0, "fn": 0}


def test_comparator_without_the_hook_still_scores_lists():
    class DuckTypedExact:
        threshold = 0.5

        def compare(self, a, b):
            return 1.0 if a == b else 0.0

        def binary_compare(self, a, b):
            return self.compare(a, b) >= self.threshold

    class Tags(StructuredModel):
        tags: List[str] = ComparableField(comparator=DuckTypedExact(), threshold=0.5)

    gt, pred = Tags(tags=["a", "b"]), Tags(tags=["b", "a"])

    assert gt.compare(pred) == 1.0
    assert gt.compare_with(pred)["overall_score"] == 1.0


def test_non_callable_scores_as_one_value_attribute_is_ignored():
    class FlagOnClass(BaseComparator):
        scores_as_one_value = True

        def _compare(self, a, b):
            return 1.0 if a == b else 0.0

    class Values(StructuredModel):
        values: List[float] = ComparableField(comparator=FlagOnClass(), threshold=0.5)

    gt, pred = Values(values=BOX), Values(values=list(BOX))
    result = gt.compare_with(pred, include_confusion_matrix=True)

    assert gt.compare(pred) == 1.0
    assert _box_counts(result, "values")["tp"] == 4
