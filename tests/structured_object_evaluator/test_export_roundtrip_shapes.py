"""Plain models, lists of lists and model unions are not exported as scalars (#349).

`to_json_schema()` and `to_stickler_config()` fell back to `"string"` / `"str"`
for any annotation missing from their type tables. A plain `BaseModel`, a list
of them, a list of lists and a union of models all took that fallback, so the
re-imported model could not load the data the original model accepted.

A plain model is scored as one object, as a mapping is, so it now exports as a
free-form object and comes back as `dict`, and a list of lists exports as nested
arrays. `to_stickler_config()` has no type for a union with a model arm or for a
StructuredModel inside a nested list. 1.0.1 ships no breaking change, so both
exporters write those as 1.0.0 did and warn; 1.0.2 raises. `Optional` and
`Annotated` are stripped first, so a nested model under `Field(...)` on an
optional field exports as the model.

The warning reads the field, its list levels and the arms of a union, not other
containers, so `Union[List[Card], str]` and `Tuple[Card, ...]` are not warned
about and still export as `"string"`.
"""

import inspect
import warnings
from typing import Annotated, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel, Field

from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.configuration_helper import (
    ConfigurationHelper,
)
from stickler.structured_object_evaluator.models.json_schema_field_converter import (
    JsonSchemaFieldConverter,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class Plain(BaseModel):
    a: str
    b: str


class Counted(BaseModel):
    n: int


class Card(StructuredModel):
    pan: str = ComparableField(threshold=0.9)


class Bank(StructuredModel):
    iban: str = ComparableField(threshold=0.9)


class Shapes(StructuredModel):
    child: Plain = ComparableField(threshold=0.9)
    maybe_child: Optional[Plain] = ComparableField(threshold=0.9)
    children: List[Plain] = ComparableField(threshold=0.9)
    nested: List[List[float]] = ComparableField(threshold=0.9)
    nested_maps: List[List[Dict[str, str]]] = ComparableField(threshold=0.9)


FIELDS = [name for name in Shapes.model_fields if name != "extra_fields"]

# One value wrong per field, so every field scores between 0 and 1.
GT = {
    "child": {"a": "x", "b": "y"},
    "maybe_child": {"a": "x", "b": "y"},
    "children": [{"a": "x", "b": "y"}],
    "nested": [[1.0, 2.0], [3.0]],
    "nested_maps": [[{"a": "x", "b": "y"}], [{"c": "d"}]],
}
PRED = {
    "child": {"a": "x", "b": "z"},
    "maybe_child": {"a": "x", "b": "z"},
    "children": [{"a": "x", "b": "z"}],
    "nested": [[1.0, 2.0], [4.0]],
    "nested_maps": [[{"a": "x", "b": "y"}], [{"c": "e"}]],
}

REIMPORT = {
    "json_schema": lambda model: StructuredModel.from_json_schema(
        model.to_json_schema()
    ),
    "stickler_config": lambda model: StructuredModel.model_from_json(
        model.to_stickler_config()
    ),
}


@pytest.mark.parametrize("path", REIMPORT)
def test_reimported_model_loads_the_original_data_and_scores_the_same(path):
    Reimported = REIMPORT[path](Shapes)
    original = Shapes(**GT).compare_with(Shapes(**PRED), include_confusion_matrix=True)
    rebuilt = Reimported(**GT).compare_with(
        Reimported(**PRED), include_confusion_matrix=True
    )
    assert rebuilt["field_scores"] == pytest.approx(original["field_scores"])
    assert rebuilt["overall_score"] == pytest.approx(original["overall_score"])
    assert (
        rebuilt["confusion_matrix"]["overall"]
        == original["confusion_matrix"]["overall"]
    )


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("name", FIELDS)
def test_reimported_field_resolves_like_the_original(path, name):
    Reimported = REIMPORT[path](Shapes)
    original = ConfigurationHelper.get_comparison_info(Shapes, name)
    rebuilt = ConfigurationHelper.get_comparison_info(Reimported, name)
    assert type(rebuilt.comparator) is type(original.comparator)
    assert rebuilt.threshold == original.threshold
    assert rebuilt.weight == original.weight
    assert rebuilt.clip_under_threshold is original.clip_under_threshold


@pytest.mark.parametrize(
    "name, expected",
    [
        ("child", {"type": "object"}),
        ("maybe_child", {"type": ["object", "null"]}),
        ("children", {"type": "array", "items": {"type": "object"}}),
        (
            "nested",
            {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        ),
        (
            "nested_maps",
            {"type": "array", "items": {"type": "array", "items": {"type": "object"}}},
        ),
    ],
)
def test_json_schema_writes_the_structure(name, expected):
    prop = Shapes.to_json_schema()["properties"][name]
    assert {key: prop[key] for key in expected} == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("child", "dict"),
        ("children", "List[dict]"),
        # Pins the documented limit: the inner element type is not written.
        ("nested", "List[List]"),
    ],
)
def test_stickler_config_writes_the_type(name, expected):
    assert Shapes.to_stickler_config()["fields"][name]["type"] == expected


def test_value_types_are_not_exported():
    """Pins the documented limit: the re-imported field is a `dict`."""

    class Typed(StructuredModel):
        counted: Counted = ComparableField()

    for Reimported in (REIMPORT[path](Typed) for path in REIMPORT):
        assert Typed(counted={"n": "5"}).counted == Counted(n=5)
        assert Reimported(counted={"n": "5"}).counted == {"n": "5"}


def test_an_annotated_scalar_exports_as_its_own_type():
    class Holder(StructuredModel):
        count: Optional[Annotated[int, Field(description="d")]] = ComparableField()

    assert Holder.to_json_schema()["properties"]["count"]["type"] == ["integer", "null"]
    assert Holder.to_stickler_config()["fields"]["count"]["type"] == "int"


# Pydantic keeps `Annotated` inside `Optional`, so `Annotated[Card, Field(...)] | None`
# wraps the type the exporters dispatch on. Card's threshold is 0.9, so 0.9375 on
# the first pan is a partial match.
ANNOTATED_MODELS = {
    "Optional[Annotated[Card, Field]]": (
        Optional[Annotated[Card, Field(description="d")]],
        {"pan": "pan-000000000001"},
        {"pan": "pan-000000000002"},
    ),
    "List[Annotated[Card]]": (
        List[Annotated[Card, "m"]],
        [{"pan": "pan-000000000001"}, {"pan": "pan-000000000099"}],
        [{"pan": "pan-000000000002"}, {"pan": "pan-000000000099"}],
    ),
    "Optional[Annotated[List[Card], Field]]": (
        Optional[Annotated[List[Card], Field(description="d")]],
        [{"pan": "pan-000000000001"}, {"pan": "pan-000000000099"}],
        [{"pan": "pan-000000000002"}, {"pan": "pan-000000000099"}],
    ),
}


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", ANNOTATED_MODELS)
def test_an_annotated_nested_model_round_trips(path, spelling):
    annotation, gt, pred = ANNOTATED_MODELS[spelling]

    class Holder(StructuredModel):
        held: annotation = ComparableField()

    Reimported = REIMPORT[path](Holder)
    original = Holder(held=gt).compare_with(
        Holder(held=pred), include_confusion_matrix=True
    )
    rebuilt = Reimported(held=gt).compare_with(
        Reimported(held=pred), include_confusion_matrix=True
    )
    assert rebuilt["field_scores"] == pytest.approx(original["field_scores"])
    assert rebuilt["overall_score"] == pytest.approx(original["overall_score"])
    assert (
        rebuilt["confusion_matrix"]["overall"]
        == original["confusion_matrix"]["overall"]
    )


UNION_ANNOTATIONS = {
    "Union[Card, Bank]": Union[Card, Bank],
    "Optional[Union[Card, Bank]]": Optional[Union[Card, Bank]],
    "Card | Bank": Card | Bank,
    "Union[Plain, str]": Union[Plain, str],
    "List[Union[Card, Bank]]": List[Union[Card, Bank]],
    "List[List[Union[Card, Bank]]]": List[List[Union[Card, Bank]]],
    "Optional[Annotated[Union[Card, Bank]]]": Optional[
        Annotated[Union[Card, Bank], "m"]
    ],
    "Union[Annotated[Card], Bank]": Union[Annotated[Card, "m"], Bank],
    "List[Annotated[Union[Card, Bank]]]": List[Annotated[Union[Card, Bank], "m"]],
}


# What 1.0.0 wrote for each spelling, measured on v1.0.0: the JSON Schema
# `type`/`items` and the Stickler config `type`. 1.0.0 unwrapped `Optional` once
# and nothing else, so only the spelling with `Optional` outside `Annotated` is
# nullable, and a list of lists is one array level.
LEGACY_UNION_EXPORTS = {
    "Union[Card, Bank]": ({"type": "string"}, "str"),
    "Optional[Union[Card, Bank]]": ({"type": "string"}, "str"),
    "Card | Bank": ({"type": "string"}, "str"),
    "Union[Plain, str]": ({"type": "string"}, "str"),
    "List[Union[Card, Bank]]": (
        {"type": "array", "items": {"type": "string"}},
        "List[Union]",
    ),
    "List[List[Union[Card, Bank]]]": (
        {"type": "array", "items": {"type": "string"}},
        "List[List]",
    ),
    "Optional[Annotated[Union[Card, Bank]]]": ({"type": ["string", "null"]}, "str"),
    "Union[Annotated[Card], Bank]": ({"type": "string"}, "str"),
    "List[Annotated[Union[Card, Bank]]]": (
        {"type": "array", "items": {"type": "string"}},
        "List[Annotated]",
    ),
}

LEGACY_JSON_EXTENSIONS = {
    "x-aws-stickler-clip-under-threshold": True,
    "x-aws-stickler-comparator": "LevenshteinComparator",
    "x-aws-stickler-comparator-config": {"normalize": True},
    "x-aws-stickler-threshold": 0.9,
    "x-aws-stickler-weight": 1.0,
}

LEGACY_STICKLER_SETTINGS = {
    "clip_under_threshold": True,
    "comparator": "LevenshteinComparator",
    "comparator_config": {"normalize": True},
    "default": None,
    "required": False,
    "threshold": 0.9,
    "weight": 1.0,
}


def _export(model, path):
    """Export, returning the one field's output and the legacy-shape warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if path == "json_schema":
            exported = model.to_json_schema()["properties"]
        else:
            exported = model.to_stickler_config()["fields"]
    legacy = [
        w for w in caught if "This becomes a ValueError in 1.0.2" in str(w.message)
    ]
    return exported, legacy


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", UNION_ANNOTATIONS)
def test_a_union_with_a_model_arm_exports_as_1_0_0_did_and_warns(path, spelling):
    annotation = UNION_ANNOTATIONS[spelling]

    class Holder(StructuredModel):
        either: annotation = ComparableField(threshold=0.9)

    exported, legacy = _export(Holder, path)
    json_type, stickler_type = LEGACY_UNION_EXPORTS[spelling]
    if path == "json_schema":
        assert exported["either"] == {**json_type, **LEGACY_JSON_EXTENSIONS}
    else:
        assert exported["either"] == {"type": stickler_type, **LEGACY_STICKLER_SETTINGS}
    assert len(legacy) == 1
    assert legacy[0].category is UserWarning
    message = str(legacy[0].message)
    assert f"field 'Holder.either' cannot export {annotation!r}" in message
    assert "has no type for a union with a model arm" in message
    assert "Declare the field as a single model type" in message


NESTED_MODEL_LISTS = {
    "List[List[Card]]": List[List[Card]],
    "Optional[List[List[Card]]]": Optional[List[List[Card]]],
    "List[List[Optional[Card]]]": List[List[Optional[Card]]],
    "List[List[List[Card]]]": List[List[List[Card]]],
    "List[List[Annotated[Card]]]": List[List[Annotated[Card, "m"]]],
}


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", NESTED_MODEL_LISTS)
def test_a_structured_model_in_a_nested_list_exports_as_1_0_0_did_and_warns(
    path, spelling
):
    annotation = NESTED_MODEL_LISTS[spelling]

    class Holder(StructuredModel):
        grid: annotation = ComparableField()

    exported, legacy = _export(Holder, path)
    # 1.0.0: an array of strings, `List[List]` in a Stickler config.
    if path == "json_schema":
        assert exported["grid"] == {
            "type": "array",
            "items": {"type": "string"},
            **LEGACY_JSON_EXTENSIONS,
            "x-aws-stickler-threshold": 0.5,
        }
    else:
        assert exported["grid"] == {
            "type": "List[List]",
            **LEGACY_STICKLER_SETTINGS,
            "threshold": 0.5,
        }
    assert len(legacy) == 1
    message = str(legacy[0].message)
    assert "has no type for a StructuredModel inside a nested list" in message
    assert "Wrap the inner list in its own StructuredModel" in message


@pytest.mark.parametrize("path", REIMPORT)
def test_the_legacy_export_warning_points_at_the_callers_line(path):
    class Holder(StructuredModel):
        either: Union[Card, Bank] = ComparableField()
        grid: List[List[Card]] = ComparableField()

    export = (
        Holder.to_json_schema if path == "json_schema" else Holder.to_stickler_config
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        this_line = inspect.currentframe().f_lineno + 1
        export()
    legacy = [w for w in caught if "becomes a ValueError in 1.0.2" in str(w.message)]
    # One per field per call: an export runs once, so it is not memoised.
    assert [w.filename for w in legacy] == [__file__, __file__]
    assert [w.lineno for w in legacy] == [this_line, this_line]
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        export()
    assert len([w for w in again if "1.0.2" in str(w.message)]) == 2


def test_a_union_built_from_a_schema_exports_as_1_0_0_did_and_warns():
    """The schema-only loop from #349 C: import builds the union, export keeps 1.0.0's string."""
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "anyOf": [
                    {
                        "type": "object",
                        "x-aws-stickler-model-name": "Card",
                        "properties": {"pan": {"type": "string"}},
                        "required": ["pan"],
                    },
                    {
                        "type": "object",
                        "x-aws-stickler-model-name": "Bank",
                        "properties": {"iban": {"type": "string"}},
                        "required": ["iban"],
                    },
                ]
            }
        },
        "required": ["payload"],
    }
    Imported = StructuredModel.from_json_schema(schema)
    Imported(payload={"iban": "DE89"})
    exported, legacy = _export(Imported, "json_schema")
    assert exported["payload"] == {
        "type": "string",
        "x-aws-stickler-clip-under-threshold": True,
        "x-aws-stickler-comparator": "ExactComparator",
        "x-aws-stickler-threshold": 1.0,
        "x-aws-stickler-weight": 1.0,
    }
    assert len(legacy) == 1
    exported, legacy = _export(Imported, "stickler_config")
    assert exported["payload"] == {
        "type": "str",
        "comparator": "ExactComparator",
        "threshold": 1.0,
        "weight": 1.0,
        "clip_under_threshold": True,
        "required": True,
    }
    assert len(legacy) == 1


class TestFieldToProperty:
    """The converter itself, which the issue measured directly."""

    field_info = Shapes.model_fields["child"]

    def _convert(self, annotation):
        return JsonSchemaFieldConverter({}).field_to_property(
            annotation, self.field_info
        )

    def test_a_union_of_models_is_written_as_a_string_with_a_warning(self):
        with pytest.warns(UserWarning, match="union with a model arm"):
            prop = self._convert(Union[Card, Bank])
        assert prop["type"] == "string"

    def test_a_structured_model_is_still_refused(self):
        with pytest.raises(ValueError, match="StructuredModel"):
            self._convert(Card)

    def test_a_single_list_of_models_is_not_called_a_nested_list(self):
        with pytest.raises(ValueError, match="as a scalar"):
            self._convert(Optional[Annotated[List[Card], Field(description="d")]])

    def test_a_plain_model_is_written_as_an_object(self):
        assert self._convert(Plain)["type"] == "object"
