"""Variable-length, listed and nested tuples keep their names in a config.

`to_stickler_config()` writes `Tuple[float, ...]`, `List[Tuple[...]]` and
nested tuples by name, so the model `model_from_json()` rebuilds keeps their
length and element checks. A tuple with an element that has no config name
is still written as `tuple`. Every score-parity test declares its comparator:
an undeclared tuple field hits #391.
"""

import json
import warnings
from typing import Any, List, Optional, Tuple

import pytest
from pydantic import ValidationError

from stickler import BBoxIoUComparator, ComparableField, StructuredModel
from stickler.comparators import ExactComparator
from stickler.structured_object_evaluator.models.type_resolver import (
    resolve_type_string,
)

Box = Tuple[float, float, float, float]
GT_BOX = (0.0, 0.0, 10.0, 10.0)
# IoU 0.5 against GT_BOX.
PRED_BOX = (0.0, 0.0, 10.0, 5.0)


def _export_and_rebuild(model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = model.to_stickler_config()
    return config, StructuredModel.model_from_json(json.loads(json.dumps(config)))


def _scores(model, gt, pred):
    result = model(**gt).compare_with(model(**pred))
    return result["field_scores"], result["overall_score"]


@pytest.mark.parametrize("annotation", [Tuple[float, ...], Optional[Tuple[float, ...]]])
def test_a_variable_length_tuple_keeps_its_element_type(annotation):
    class Holder(StructuredModel):
        box: annotation = ComparableField(comparator=BBoxIoUComparator(), threshold=0.5)

    config, Rebuilt = _export_and_rebuild(Holder)
    assert config["fields"]["box"]["type"] == "Tuple[float, ...]"
    with pytest.raises(ValidationError):
        Rebuilt(box=("a", "b"))
    gt, pred = {"box": GT_BOX}, {"box": PRED_BOX}
    assert _scores(Rebuilt, gt, pred) == _scores(Holder, gt, pred)


@pytest.mark.parametrize("annotation", [List[Box], Optional[List[Box]]])
def test_a_list_of_boxes_keeps_the_box_type(annotation):
    class Holder(StructuredModel):
        boxes: annotation = ComparableField(
            comparator=BBoxIoUComparator(), threshold=0.5
        )

    config, Rebuilt = _export_and_rebuild(Holder)
    assert (
        config["fields"]["boxes"]["type"] == "List[Tuple[float, float, float, float]]"
    )
    with pytest.raises(ValidationError):
        Rebuilt(boxes=[(0.0, 0.0, 1.0)])
    gt = {"boxes": [GT_BOX, (20.0, 20.0, 30.0, 30.0)]}
    pred = {"boxes": [(20.0, 20.0, 30.0, 30.0), PRED_BOX]}
    assert _scores(Rebuilt, gt, pred) == _scores(Holder, gt, pred)


def test_a_list_of_optional_boxes_keeps_none_elements():
    class Holder(StructuredModel):
        boxes: List[Optional[Box]] = ComparableField(
            comparator=BBoxIoUComparator(), threshold=0.5
        )

    config, Rebuilt = _export_and_rebuild(Holder)
    assert (
        config["fields"]["boxes"]["type"]
        == "List[Optional[Tuple[float, float, float, float]]]"
    )
    assert Rebuilt(boxes=[GT_BOX, None]).boxes == [GT_BOX, None]


def test_corner_pairs_export_by_nested_name():
    Corners = Tuple[Tuple[float, float], Tuple[float, float]]

    class Holder(StructuredModel):
        corners: Corners = ComparableField(comparator=ExactComparator())

    config, Rebuilt = _export_and_rebuild(Holder)
    assert (
        config["fields"]["corners"]["type"]
        == "Tuple[Tuple[float, float], Tuple[float, float]]"
    )
    value = ((0.0, 0.0), (10.0, 10.0))
    assert Rebuilt(corners=value).corners == value
    with pytest.raises(ValidationError):
        Rebuilt(corners=((0.0, 0.0), (10.0,)))
    for pred in (value, ((0.0, 0.0), (10.0, 5.0))):
        gt, pred = {"corners": value}, {"corners": pred}
        assert _scores(Rebuilt, gt, pred) == _scores(Holder, gt, pred)


@pytest.mark.parametrize(
    "annotation",
    [
        Tuple[Optional[float], float],
        tuple,
        # One unnamed leaf keeps the whole tuple unnamed.
        Tuple[Tuple[Optional[float], float], float],
    ],
)
def test_a_tuple_with_an_unnamed_element_stays_tuple_without_a_warning(annotation):
    class Holder(StructuredModel):
        value: annotation = ComparableField(comparator=ExactComparator())

    config, _ = _export_and_rebuild(Holder)
    assert config["fields"]["value"]["type"] == "tuple"


@pytest.mark.parametrize(
    "type_string, expected",
    [
        ("Tuple[int, ...]", Tuple[int, ...]),
        ("Optional[Tuple[float, ...]]", Optional[Tuple[float, ...]]),
        ("List[Tuple[float, ...]]", List[Tuple[float, ...]]),
        ("Tuple[Any, ...]", Tuple[Any, ...]),
    ],
)
def test_an_ellipsis_is_read_as_a_variable_length_tuple(type_string, expected):
    assert resolve_type_string(type_string) == expected


@pytest.mark.parametrize(
    "type_string",
    [
        "Tuple[..., int]",
        "List[...]",
        "Tuple[int, ..., int]",
        "Tuple[...]",
        "Tuple[..., ...]",
        "Dict[str, ...]",
    ],
)
def test_an_ellipsis_anywhere_else_is_refused(type_string):
    with pytest.raises(ValueError) as excinfo:
        resolve_type_string(type_string)
    assert f"Invalid type '{type_string}'" in str(excinfo.value)
    assert "second of two Tuple arguments" in str(excinfo.value)


def test_a_variable_length_tuple_default_reimports_as_a_tuple():
    class Holder(StructuredModel):
        value: Tuple[float, ...] = ComparableField(
            comparator=ExactComparator(), default=(1.0, 2.0, 3.0)
        )

    config = json.loads(json.dumps(Holder.to_stickler_config()))
    assert config["fields"]["value"]["default"] == [1.0, 2.0, 3.0]
    assert StructuredModel.model_from_json(config)().value == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "annotation, default",
    [
        (Tuple[Tuple[float, float], Tuple[float, float]], ((0.0, 0.0), (1.0, 1.0))),
        (Tuple[Tuple[int, ...], ...], ((1, 2), (3,))),
        # No argument list to follow: only the outer level becomes a tuple.
        (tuple, (1, 2)),
    ],
)
def test_a_nested_tuple_default_reimports_as_tuples(annotation, default):
    class Holder(StructuredModel):
        value: annotation = ComparableField(
            comparator=ExactComparator(), default=default
        )

    config = json.loads(json.dumps(Holder.to_stickler_config()))
    Rebuilt = StructuredModel.model_from_json(config)
    assert Rebuilt().value == default
    assert all(
        isinstance(level, tuple)
        for level in (Rebuilt().value, *Rebuilt().value)
        if isinstance(level, (list, tuple))
    )
    score = Rebuilt().compare_with(Rebuilt(value=default))["overall_score"]
    assert score == 1.0
