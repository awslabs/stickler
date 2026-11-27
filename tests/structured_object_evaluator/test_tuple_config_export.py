"""A tuple field survives to_stickler_config() -> model_from_json() (#636)."""

import json
from typing import Any, Optional, Tuple

import pytest

from stickler import BBoxIoUComparator, ComparableField, StructuredModel


@pytest.mark.parametrize(
    "annotation, exported, value",
    [
        (
            Tuple[float, float, float, float],
            "Tuple[float, float, float, float]",
            (0.0, 0.0, 10.0, 10.0),
        ),
        (tuple[str, int], "Tuple[str, int]", ("a", 1)),
        (Tuple[Any, bool], "Tuple[Any, bool]", ([1], True)),
        (Tuple[float, ...], "Tuple[float, ...]", (1.0, 2.0)),
        # No config spelling for these argument lists: stays a tuple.
        (Tuple[Optional[int], int], "tuple", (None, 1)),
        (tuple, "tuple", (1, "a")),
    ],
)
def test_tuple_field_round_trips_through_config(annotation, exported, value):
    class Model(StructuredModel):
        item: annotation = ComparableField()

    config = Model.to_stickler_config()
    assert config["fields"]["item"]["type"] == exported

    rebuilt = StructuredModel.model_from_json(json.loads(json.dumps(config)))
    assert rebuilt(item=value).item == value


def test_optional_tuple_box_keeps_its_comparator():
    class Model(StructuredModel):
        box: Optional[Tuple[float, float, float, float]] = ComparableField(
            default=None, comparator=BBoxIoUComparator(), threshold=0.5
        )

    rebuilt = StructuredModel.model_from_json(Model.to_stickler_config())
    box = (0.0, 0.0, 10.0, 10.0)

    assert rebuilt().box is None
    assert rebuilt(box=box).compare_with(rebuilt(box=box))["overall_score"] == 1.0
