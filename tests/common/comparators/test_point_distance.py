"""Tests for PointDistanceComparator."""

import math
from decimal import Decimal
from fractions import Fraction
from typing import List, Tuple

import numpy as np
import pytest

from stickler import ComparableField, PointDistanceComparator, StructuredModel


@pytest.fixture
def comparator():
    return PointDistanceComparator(max_distance=10.0)


class TestScore:
    def test_identical_points_score_one(self, comparator):
        assert comparator.compare([1, 2], [1, 2]) == 1.0
        assert comparator.compare((1, 2, 3), (1, 2, 3)) == 1.0

    def test_2d_score_at_known_distance(self, comparator):
        # A 3-4-5 triangle: distance 5 of max_distance 10.
        assert comparator.compare([0, 0], [3, 4]) == 0.5

    def test_3d_score_at_known_distance(self, comparator):
        # Distance sqrt(4 + 16 + 16) = 6 of max_distance 10.
        assert comparator.compare([0, 0, 0], [2, 4, 4]) == pytest.approx(0.4)

    def test_score_is_zero_at_and_beyond_max_distance(self, comparator):
        assert comparator.compare([0, 0], [10, 0]) == 0.0
        assert comparator.compare([0, 0], [100, 0]) == 0.0

    def test_gate_is_a_radius_not_a_box(self, comparator):
        """A corner of the per-axis box lies outside the radius."""
        assert comparator.binary_compare([0, 0], [5, 0]) == (1, 0)
        assert comparator.binary_compare([0, 0], [4, 4]) == (0, 1)


class TestInvalidConfig:
    @pytest.mark.parametrize("max_distance", [0, -1.0, math.nan, math.inf, True, "10"])
    def test_invalid_max_distance_raises(self, max_distance):
        with pytest.raises(ValueError, match="max_distance"):
            PointDistanceComparator(max_distance=max_distance)

    def test_max_distance_is_required(self):
        with pytest.raises(TypeError):
            PointDistanceComparator()


class TestMalformedInput:
    @pytest.mark.parametrize(
        "point",
        [
            [1],
            [1, 2, 3, 4],
            ["1", "2"],
            [1, math.nan],
            [1, math.inf],
            [1, 10**400],
            [Decimal(1), Decimal(2)],
            np.array([[1, 2]]),
            "1, 2",
            {"x": 1, "y": 2},
        ],
    )
    def test_malformed_point_scores_zero(self, comparator, point):
        assert comparator.compare(point, [1, 2]) == 0.0

    def test_mismatched_dimensions_score_zero(self, comparator):
        assert comparator.compare([0, 0], [0, 0, 0]) == 0.0

    def test_numpy_and_fraction_coordinates_are_accepted(self, comparator):
        assert comparator.compare([np.float32(0), np.int64(0)], [0, 0]) == 1.0
        assert comparator.compare(np.array([0, 0]), np.array([3, 4])) == 0.5
        assert comparator.compare([Fraction(3), 4], [0, 0]) == 0.5

    def test_bool_coordinates_are_accepted(self, comparator):
        assert comparator.compare([True, False], [1, 0]) == 1.0


class TestStructuredModel:
    def test_point_field(self):
        class Detection(StructuredModel):
            position: Tuple[float, float, float] = ComparableField(
                comparator=PointDistanceComparator(max_distance=10.0),
                threshold=0.5,
            )

        result = Detection(position=(0, 0, 0)).compare_with(
            Detection(position=(0, 3, 4))
        )
        assert result["field_scores"]["position"] == 0.5

    def test_list_of_points_field_matches_by_distance(self):
        class Scene(StructuredModel):
            points: List[List[float]] = ComparableField(
                comparator=PointDistanceComparator(max_distance=10.0),
                threshold=0.5,
            )

        gt = Scene(points=[[0, 0], [50, 50]])
        pred = Scene(points=[[50, 51], [3, 4]])
        # Pairs are matched by distance, not by position: scores 0.9 and 0.5.
        assert gt.compare_with(pred)["field_scores"]["points"] == pytest.approx(0.7)


class TestImport:
    def test_json_schema_import_and_round_trip(self):
        schema = {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "x-aws-stickler-comparator": "PointDistanceComparator",
                    "x-aws-stickler-comparator-config": {"max_distance": 10},
                    "x-aws-stickler-threshold": 0.5,
                }
            },
        }
        model = StructuredModel.from_json_schema(schema)
        result = model(points=[[0, 0]]).compare_with(model(points=[[3, 4]]))
        assert result["field_scores"]["points"] == 0.5

        reimported = StructuredModel.from_json_schema(model.to_json_schema())
        comparator = reimported._get_comparison_info("points").comparator
        assert comparator.max_distance == 10.0

    def test_stickler_config_import(self):
        config = {
            "model_name": "Scene",
            "fields": {
                "points": {
                    "type": "list",
                    "comparator": "PointDistanceComparator",
                    "comparator_config": {"max_distance": 10},
                    "threshold": 0.5,
                }
            },
        }
        model = StructuredModel.model_from_json(config)
        result = model(points=[[0, 0]]).compare_with(model(points=[[3, 4]]))
        assert result["field_scores"]["points"] == 0.5

    def test_stickler_config_tuple_point(self):
        config = {
            "model_name": "Detection",
            "fields": {
                "position": {
                    "type": "tuple",
                    "comparator": "PointDistanceComparator",
                    "comparator_config": {"max_distance": 10},
                    "threshold": 0.5,
                }
            },
        }
        model = StructuredModel.model_from_json(config)
        result = model(position=(0, 0)).compare_with(model(position=(3, 4)))
        assert result["field_scores"]["position"] == 0.5

    def test_missing_max_distance_fails_at_import(self):
        schema = {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "x-aws-stickler-comparator": "PointDistanceComparator",
                }
            },
        }
        with pytest.raises(ValueError, match="max_distance"):
            StructuredModel.from_json_schema(schema)
