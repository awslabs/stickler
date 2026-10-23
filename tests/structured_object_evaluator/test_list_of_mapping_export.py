"""A ``List[Dict[...]]`` field round-trips through ``to_json_schema()`` (#333).

The element used to export as ``{"type": "string"}``, so the re-imported field
was ``List[str]`` and rejected the data the original model accepted. These tests
export, re-import, load the ORIGINAL data, and require the same scores.
"""

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, OrderedDict

import pytest
from pydantic import Field

from stickler.comparators.levenshtein import LevenshteinComparator
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
    any_values: List[Dict[str, Any]] = ComparableField(weight=2.0)
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
    assert rebuilt.weight == original.weight
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


def test_reimported_field_does_not_coerce_values():
    """Pins the documented limitation: the value type is not exported."""

    class Typed(StructuredModel):
        rows: List[Dict[str, int]] = ComparableField()

    Reimported = StructuredModel.from_json_schema(Typed.to_json_schema())
    assert Typed(rows=[{"n": "5"}]).rows == [{"n": 5}]
    assert Reimported(rows=[{"n": "5"}]).rows == [{"n": "5"}]


def test_stickler_config_keeps_an_optional_element_nullable():
    class WithNone(StructuredModel):
        rows: List[Optional[Dict[str, str]]] = ComparableField(threshold=0.9)

    config = WithNone.to_stickler_config()
    assert config["fields"]["rows"]["type"] == "List[Optional[Dict]]"

    Reimported = StructuredModel.model_from_json(config)
    gt = {"rows": [{"a": "x", "b": "y"}, None]}
    pred = {"rows": [{"a": "x", "b": "z"}, None]}
    original = WithNone(**gt).compare_with(WithNone(**pred))
    rebuilt = Reimported(**gt).compare_with(Reimported(**pred))
    assert rebuilt["field_scores"] == pytest.approx(original["field_scores"])


def test_stickler_config_keeps_a_declared_comparator_config_in_place():
    class Tuned(StructuredModel):
        rows: List[Dict[str, str]] = ComparableField(
            comparator=LevenshteinComparator(normalize=False), threshold=0.8
        )

    field = Tuned.to_stickler_config()["fields"]["rows"]
    assert field["comparator_config"] == {"normalize": False}
    assert list(field)[:3] == ["type", "comparator", "comparator_config"]


class LateRows(StructuredModel):
    """Declared, but the element only resolves at ``model_rebuild()``.

    The class-definition substitution cannot see a mapping yet, so the field
    still holds Levenshtein; the engine swaps in ANLS* when it reads the field.
    The export must carry what the engine scores with.
    """

    rows: "List[_LateElement]" = ComparableField(threshold=0.9)


_LateElement = Dict[str, str]
LateRows.model_rebuild()


@pytest.mark.parametrize(
    "roundtrip",
    [
        lambda m: StructuredModel.from_json_schema(m.to_json_schema()),
        lambda m: StructuredModel.model_from_json(m.to_stickler_config()),
    ],
    ids=["json_schema", "stickler_config"],
)
def test_late_resolved_field_exports_the_engine_comparator(roundtrip):
    assert "comparator_config" not in LateRows.to_stickler_config()["fields"]["rows"]
    Reimported = roundtrip(LateRows)

    assert _resolved(Reimported, "rows") == _resolved(LateRows, "rows")
    gt, pred = {"rows": GT["default_clip"]}, {"rows": PRED["default_clip"]}
    original = LateRows(**gt).compare_with(LateRows(**pred))
    rebuilt = Reimported(**gt).compare_with(Reimported(**pred))
    assert rebuilt["field_scores"]["rows"] == pytest.approx(0.5)
    assert original["field_scores"]["rows"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "element",
    [Mapping[str, str], MutableMapping[str, str], OrderedDict[str, str], dict],
)
def test_stickler_config_writes_every_mapping_element_as_dict(element):
    class WithMapping(StructuredModel):
        rows: List[element] = ComparableField(threshold=0.9)

    config = WithMapping.to_stickler_config()
    assert config["fields"]["rows"]["type"] == "List[Dict]"

    Reimported = StructuredModel.model_from_json(config)
    gt, pred = {"rows": GT["default_clip"]}, {"rows": PRED["default_clip"]}
    original = WithMapping(**gt).compare_with(WithMapping(**pred))
    rebuilt = Reimported(**gt).compare_with(Reimported(**pred))
    assert rebuilt["field_scores"] == pytest.approx(original["field_scores"])


class BareRows(StructuredModel):
    """No ``ComparableField``, so the engine's defaults are what must survive."""

    match_threshold = 0.8
    rows: List[Dict[str, str]] = Field(default_factory=list)
    optional_rows: Optional[List[Mapping[str, str]]] = None


BARE_FIELDS = ["rows", "optional_rows"]
BARE_GT = {name: GT["default_clip"] for name in BARE_FIELDS}
BARE_PRED = {name: PRED["default_clip"] for name in BARE_FIELDS}


def _resolved(model, name):
    info = ConfigurationHelper.get_comparison_info(model, name)
    return (
        type(info.comparator),
        info.threshold,
        info.weight,
        info.clip_under_threshold,
    )


@pytest.mark.parametrize(
    "roundtrip",
    [
        lambda m: StructuredModel.from_json_schema(m.to_json_schema()),
        lambda m: StructuredModel.model_from_json(m.to_stickler_config()),
    ],
    ids=["json_schema", "stickler_config"],
)
def test_undeclared_field_roundtrips_its_resolved_comparison(roundtrip):
    Reimported = roundtrip(BareRows)

    for name in BARE_FIELDS:
        assert _resolved(Reimported, name) == _resolved(BareRows, name)
    original = BareRows(**BARE_GT).compare_with(BareRows(**BARE_PRED))
    rebuilt = Reimported(**BARE_GT).compare_with(Reimported(**BARE_PRED))
    assert rebuilt["field_scores"] == pytest.approx(original["field_scores"])
    assert rebuilt["field_scores"]["rows"] == pytest.approx(0.5)


def test_undeclared_field_reexports_the_same_schema():
    schema = BareRows.to_json_schema()
    Reimported = StructuredModel.from_json_schema(schema)
    assert Reimported.to_json_schema()["properties"] == schema["properties"]
