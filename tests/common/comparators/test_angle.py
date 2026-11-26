"""Tests for AngleComparator: wraparound, tolerance, config and field use."""

import math
from decimal import Decimal
from typing import List

import pytest

from stickler import AngleComparator, ComparableField, StructuredModel


@pytest.mark.parametrize(
    "angle1, angle2, expected",
    [
        (359, 1, 1.0),
        (1, 359, 1.0),
        (0, 180, 0.0),
        (-3, 2, 1.0),
        (-90, 270, 1.0),
        (725, 3, 1.0),
        (0, 6, 0.0),
    ],
)
def test_degrees_use_the_shortest_difference(angle1, angle2, expected):
    assert AngleComparator(max_angular_difference=5).compare(angle1, angle2) == (
        expected
    )


def test_tolerance_is_inclusive():
    heading = AngleComparator(max_angular_difference=10)
    assert heading.compare(355, 5) == 1.0
    assert heading.compare(355, 5.5) == 0.0
    # 359.9 - 360 + 0.1 is 0.2 plus float error, which still meets 0.2.
    assert AngleComparator(max_angular_difference=0.2).compare(359.9, 0.1) == 1.0


def test_large_angles_reduce_without_overflow():
    heading = AngleComparator()
    assert heading.compare(1e308, 1e308) == 1.0
    assert heading.compare(1e308, -1e308) == 0.0
    # 1e17 is 280 degrees plus a whole number of turns.
    assert heading.compare(1e17, 280.0) == 1.0


def test_default_is_exact():
    assert AngleComparator().compare(-90, 270) == 1.0
    assert AngleComparator().compare(0, 0.5) == 0.0


def test_radians():
    yaw = AngleComparator(max_angular_difference=0.1, angle_unit="radians")
    assert yaw.compare(math.pi - 0.01, -math.pi + 0.01) == 1.0
    assert yaw.compare(0.05, 0.05 + 4 * math.pi) == 1.0
    assert yaw.compare(0, math.pi) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angle_unit": "deg"},
        {"max_angular_difference": -1},
        {"max_angular_difference": 180},
        {"max_angular_difference": 5, "angle_unit": "radians"},
        {"max_angular_difference": math.nan},
        {"max_angular_difference": True},
        {"max_angular_difference": "5"},
        {"max_angular_difference": 10**400},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        AngleComparator(**kwargs)


@pytest.mark.parametrize("bad", ["10", True, math.inf, Decimal("10"), [10], 10**400])
def test_malformed_input_scores_zero(bad):
    assert AngleComparator(max_angular_difference=5).compare(bad, 10) == 0.0


def test_missing_values_follow_the_shared_policy():
    assert AngleComparator().compare(None, None) == 1.0
    assert AngleComparator().compare(math.nan, 10) == 0.0


def test_config_round_trips():
    assert AngleComparator().config is None
    assert (
        type(AngleComparator(max_angular_difference=5).max_angular_difference) is float
    )
    yaw = AngleComparator(max_angular_difference=0.1, angle_unit="radians")
    assert AngleComparator(**yaw.config).config == yaw.config


def test_structured_model_field_and_list_field():
    class Pose(StructuredModel):
        yaw: float = ComparableField(
            comparator=AngleComparator(max_angular_difference=5)
        )
        waypoint_headings: List[float] = ComparableField(
            comparator=AngleComparator(max_angular_difference=5)
        )

    # A list field pairs its elements, so each element is scored as one angle
    # and a reordered prediction still matches across the wrap.
    gt = Pose(yaw=359.0, waypoint_headings=[358.0, 90.0])
    result = gt.compare_with(Pose(yaw=1.0, waypoint_headings=[92.0, 2.0]))
    assert result["field_scores"] == {"yaw": 1.0, "waypoint_headings": 1.0}

    result = gt.compare_with(Pose(yaw=180.0, waypoint_headings=[180.0, 90.0]))
    assert result["field_scores"]["yaw"] == 0.0
    assert result["field_scores"]["waypoint_headings"] == 0.5


def test_json_schema_and_config_import():
    schema = {
        "type": "object",
        "properties": {
            "yaw": {
                "type": "number",
                "x-aws-stickler-comparator": "AngleComparator",
                "x-aws-stickler-comparator-config": {"max_angular_difference": 5},
            }
        },
    }
    from_schema = StructuredModel.from_json_schema(schema)
    from_config = StructuredModel.model_from_json(
        {
            "model_name": "Pose",
            "fields": {
                "yaw": {
                    "type": "float",
                    "comparator": "AngleComparator",
                    "comparator_config": {"max_angular_difference": 5},
                }
            },
        }
    )
    for model in (from_schema, from_config):
        result = model(yaw=359.0).compare_with(model(yaw=1.0))
        assert result["field_scores"]["yaw"] == 1.0
