"""A ``List[Dict[...]]`` field round-trips through ``to_json_schema()`` (#333).

The element used to export as ``{"type": "string"}``, so the re-imported field
was ``List[str]`` and rejected the data the original model accepted. These tests
export, re-import, load the ORIGINAL data, and require the same scores.
"""

from typing import Any, Dict, List, Mapping, Optional

import pytest

from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.configuration_helper import (
    ConfigurationHelper,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class Rows(StructuredModel):
    default_clip: List[Dict[str, str]] = ComparableField(threshold=0.9)
    clipped: List[Dict[str, str]] = ComparableField(
        threshold=0.9, clip_under_threshold=True
    )
    any_values: List[Dict[str, Any]] = ComparableField()
    bare_dict: List[dict] = ComparableField()
    mapping: List[Mapping[str, str]] = ComparableField()
    optional_list: Optional[List[Dict[str, str]]] = ComparableField()
    optional_items: List[Optional[Dict[str, str]]] = ComparableField()


FIELDS = [name for name in Rows.model_fields if name != "extra_fields"]

# One key of two wrong, so each element scores 0.5: under the 0.9 threshold,
# which makes the clip setting visible in the score.
GT = {name: [{"a": "x", "b": "y"}] for name in FIELDS}
PRED = {name: [{"a": "x", "b": "z"}] for name in FIELDS}
GT["optional_items"] = [{"a": "x", "b": "y"}, None]


@pytest.mark.parametrize("name", FIELDS)
def test_mapping_items_export_as_object(name):
    items = Rows.to_json_schema()["properties"][name]["items"]
    expected = ["object", "null"] if name == "optional_items" else "object"
    assert items == {"type": expected}


@pytest.mark.parametrize("name", FIELDS)
def test_exported_clip_is_the_resolved_clip(name):
    prop = Rows.to_json_schema()["properties"][name]
    resolved = ConfigurationHelper.get_comparison_info(Rows, name)
    assert prop["x-aws-stickler-clip-under-threshold"] is resolved.clip_under_threshold


@pytest.mark.parametrize("name", FIELDS)
def test_reimported_field_resolves_like_the_original(name):
    Reimported = StructuredModel.from_json_schema(Rows.to_json_schema())
    original = ConfigurationHelper.get_comparison_info(Rows, name)
    rebuilt = ConfigurationHelper.get_comparison_info(Reimported, name)
    assert type(rebuilt.comparator) is type(original.comparator)
    assert rebuilt.threshold == original.threshold
    assert rebuilt.clip_under_threshold is original.clip_under_threshold


def test_json_schema_roundtrip_loads_original_data_and_scores_the_same():
    Reimported = StructuredModel.from_json_schema(Rows.to_json_schema())

    original = Rows(**GT).compare_with(Rows(**PRED))
    rebuilt = Reimported(**GT).compare_with(Reimported(**PRED))

    assert rebuilt["field_scores"] == pytest.approx(original["field_scores"])
    assert rebuilt["overall_score"] == pytest.approx(original["overall_score"])
    # The clip setting is doing work in both: off keeps the partial credit,
    # an explicit True zeroes it.
    assert rebuilt["field_scores"]["default_clip"] == pytest.approx(0.5)
    assert rebuilt["field_scores"]["clipped"] == 0.0


def test_stickler_config_roundtrip_loads_original_data_and_scores_the_same():
    class Config(StructuredModel):
        default_clip: List[Dict[str, str]] = ComparableField(threshold=0.9)
        clipped: List[Dict[str, str]] = ComparableField(
            threshold=0.9, clip_under_threshold=True
        )

    Reimported = StructuredModel.model_from_json(Config.to_stickler_config())
    gt = {"default_clip": GT["default_clip"], "clipped": GT["clipped"]}
    pred = {"default_clip": PRED["default_clip"], "clipped": PRED["clipped"]}

    original = Config(**gt).compare_with(Config(**pred))
    rebuilt = Reimported(**gt).compare_with(Reimported(**pred))

    assert rebuilt["field_scores"] == pytest.approx(original["field_scores"])
