"""Plain models, lists of lists and model unions are not exported as scalars (#349).

`to_json_schema()` and `to_stickler_config()` fell back to `"string"` / `"str"`
for any annotation missing from their type tables. A plain `BaseModel`, a list
of them, a list of lists and a union of models all took that fallback, so the
re-imported model could not load the data the original model accepted.

A plain model is scored as one object, as a mapping is, so it now exports as a
free-form object and comes back as `dict`, and a list of lists exports as nested
arrays. The exporters have no type for a union with a model arm or for a
StructuredModel inside a nested list. 1.0.x wrote those as a string, and 1.0.1
and 1.0.2 also warned; from 1.1.0 both exporters raise `ValueError` naming the
field. `Optional` and `Annotated` are stripped first, so a nested model under
`Field(...)` on an optional field exports as the model.

The check reads the field, its list levels and the arms of a union, not other
containers, so `Union[List[Card], str]` and `Tuple[Card, ...]` are not refused
and still export as `"string"`. Those, and the other shapes model-export.md
names as not round-tripping, are pinned here, so a test fails when one starts
working and the page needs updating.
"""

import warnings
from typing import Annotated, Dict, List, Optional, Tuple, Union

import pytest
from pydantic import BaseModel, Field, ValidationError

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
    "Union[Annotated[Card], str]": Union[Annotated[Card, "m"], str],
    "List[Annotated[Union[Card, Bank]]]": List[Annotated[Union[Card, Bank], "m"]],
}


def _export(model, path):
    if path == "json_schema":
        return model.to_json_schema()
    return model.to_stickler_config()


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", UNION_ANNOTATIONS)
def test_a_union_with_a_model_arm_is_refused(path, spelling):
    annotation = UNION_ANNOTATIONS[spelling]

    class Holder(StructuredModel):
        either: annotation = ComparableField(threshold=0.9)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as raised:
            _export(Holder, path)
    message = str(raised.value)
    method = "to_json_schema()" if path == "json_schema" else "to_stickler_config()"
    assert message.startswith(
        f"{method} field 'Holder.either' cannot export {annotation!r}"
    )
    assert "no type for a union with a model arm" in message
    assert "Declare the field (or its list element) as a single model type" in message


NESTED_MODEL_LISTS = {
    "List[List[Card]]": List[List[Card]],
    "Optional[List[List[Card]]]": Optional[List[List[Card]]],
    "List[List[Optional[Card]]]": List[List[Optional[Card]]],
    "List[List[List[Card]]]": List[List[List[Card]]],
    "List[List[Annotated[Card]]]": List[List[Annotated[Card, "m"]]],
}


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", NESTED_MODEL_LISTS)
def test_a_structured_model_in_a_nested_list_is_refused(path, spelling):
    annotation = NESTED_MODEL_LISTS[spelling]

    class Holder(StructuredModel):
        grid: annotation = ComparableField()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as raised:
            _export(Holder, path)
    message = str(raised.value)
    method = "to_json_schema()" if path == "json_schema" else "to_stickler_config()"
    assert message.startswith(
        f"{method} field 'Holder.grid' cannot export {annotation!r}"
    )
    assert "no type for a StructuredModel inside a nested list" in message
    assert (
        "Wrap each inner list in its own StructuredModel, so that no list "
        "directly holds a list of models" in message
    )


NESTED_MODEL_UNIONS = {
    "List[List[Union[Card, Bank]]]": List[List[Union[Card, Bank]]],
    "Optional[List[List[Union[Card, Bank]]]]": Optional[List[List[Union[Card, Bank]]]],
    "list[list[Card | Bank]] | None": list[list[Card | Bank]] | None,
    "List[Annotated[List[Union[Annotated[Card], Bank]]]]": List[
        Annotated[List[Union[Annotated[Card, "m"], Bank]], "m"]
    ],
    "List[List[Union[Annotated[Card], str]]]": List[
        List[Union[Annotated[Card, "m"], str]]
    ],
}


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", NESTED_MODEL_UNIONS)
def test_a_model_union_in_a_nested_list_gets_both_hints(path, spelling):
    """A single StructuredModel there would still be a model in a nested list."""
    annotation = NESTED_MODEL_UNIONS[spelling]

    class Holder(StructuredModel):
        grid: annotation = ComparableField()

    with pytest.raises(ValueError) as raised:
        _export(Holder, path)
    message = str(raised.value)
    assert "no type for a union with a model arm" in message
    assert (
        "Declare the field (or its list element) as a single model type, and if "
        "you choose a StructuredModel, wrap each inner list in its own "
        "StructuredModel" in message
    )


@pytest.mark.parametrize("path", REIMPORT)
def test_a_plain_model_union_in_a_nested_list_gets_only_the_union_hint(path):
    """`List[List[Plain]]` exports, so a single plain model is enough there."""

    class Holder(StructuredModel):
        grid: List[List[Union[Plain, str]]] = ComparableField()

    with pytest.raises(ValueError) as raised:
        _export(Holder, path)
    message = str(raised.value)
    assert "no type for a union with a model arm" in message
    assert "Declare the field (or its list element) as a single model type" in message
    assert "wrap each inner list" not in message


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", ["Union[Card, Bank]", "List[Union[Card, Bank]]"])
def test_a_union_under_one_list_level_gets_only_the_union_hint(path, spelling):
    annotation = UNION_ANNOTATIONS[spelling]

    class Holder(StructuredModel):
        either: annotation = ComparableField()

    with pytest.raises(ValueError) as raised:
        _export(Holder, path)
    assert "wrap each inner list" not in str(raised.value)


class _Row(StructuredModel):
    cards: List[Card] = ComparableField()


class _Plane(StructuredModel):
    rows: List[_Row] = ComparableField()


@pytest.mark.parametrize("path", REIMPORT)
def test_following_the_nested_list_hint_exports_at_any_depth(path):
    """Wrapping one level of `List[List[List[Card]]]` is still refused;
    wrapping each inner list, as the message says, exports and reloads."""

    class OneWrap(StructuredModel):
        grid: List[List[_Row]] = ComparableField()

    with pytest.raises(ValueError, match="inside a nested list"):
        _export(OneWrap, path)

    class Grid(StructuredModel):
        planes: List[_Plane] = ComparableField()

    data = {"planes": [{"rows": [{"cards": [{"pan": "1"}]}]}]}
    rebuilt = REIMPORT[path](Grid)
    assert rebuilt(**data).compare_with(rebuilt(**data))["overall_score"] == 1.0


@pytest.mark.parametrize("path", REIMPORT)
def test_only_the_export_is_refused(path):
    """The model still builds and scores; only writing it out raises."""

    class Holder(StructuredModel):
        either: Union[Card, Bank] = ComparableField()
        grid: List[List[Card]] = ComparableField()

    gt = Holder(either={"pan": "1"}, grid=[[{"pan": "1"}]])
    pred = Holder(either={"pan": "1"}, grid=[[{"pan": "1"}]])
    assert gt.compare_with(pred)["overall_score"] == 1.0
    with pytest.raises(ValueError, match="'Holder.either'"):
        _export(Holder, path)


# Shapes model-export.md names as still not round-tripping. Each test below asserts
# the stated limitation, so it fails (and the page needs updating) once a shape works.
OTHER_CONTAINERS = {
    "Union[List[Card], str]": (Union[List[Card], str], [{"pan": "4111"}]),
    "Union[Dict[str, Card], str]": (
        Union[Dict[str, Card], str],
        {"main": {"pan": "4111"}},
    ),
    "Tuple[Card, ...]": (Tuple[Card, ...], ({"pan": "4111"},)),
}


def _holder(annotation):
    class Holder(StructuredModel):
        other: annotation = ComparableField()

    return Holder


@pytest.mark.parametrize("path", REIMPORT)
@pytest.mark.parametrize("spelling", OTHER_CONTAINERS)
def test_other_containers_are_not_refused(path, spelling):
    """The check reads only the field, its list levels and a union's arms."""
    annotation, data = OTHER_CONTAINERS[spelling]
    Holder = _holder(annotation)
    Holder(other=data)

    exported = _export(Holder, path)
    fields = exported["properties"] if path == "json_schema" else exported["fields"]
    assert fields["other"]["type"] == ("string" if path == "json_schema" else "str")
    with pytest.raises(ValidationError):
        REIMPORT[path](Holder)(other=data)


@pytest.mark.parametrize("path", REIMPORT)
def test_a_nested_list_of_scalar_unions_round_trips_only_through_the_config(path):
    Holder = _holder(List[List[Union[str, int]]])
    data = [["a", 1]]
    Holder(other=data)
    Reimported = REIMPORT[path](Holder)

    if path == "json_schema":
        exported = Holder.to_json_schema()["properties"]["other"]
        assert exported["items"] == {"type": "array", "items": {"type": "string"}}
        with pytest.raises(ValidationError):
            Reimported(other=data)
    else:
        assert Holder.to_stickler_config()["fields"]["other"]["type"] == "List[List]"
        assert Reimported(other=data).other == data


@pytest.mark.parametrize("path", REIMPORT)
def test_a_model_in_a_mapping_in_a_nested_list_loses_its_schema(path):
    Holder = _holder(List[List[Dict[str, Card]]])
    gt = [[{"main": {"pan": "4111111"}}]]
    pred = [[{"main": {"pan": "4111112"}}]]
    Reimported = REIMPORT[path](Holder)

    if path == "json_schema":
        exported = Holder.to_json_schema()["properties"]["other"]
        assert exported["items"] == {"type": "array", "items": {"type": "object"}}
    assert isinstance(Holder(other=gt).other[0][0]["main"], Card)
    assert type(Reimported(other=gt).other[0][0]["main"]) is dict
    original = Holder(other=gt).compare_with(Holder(other=pred))["overall_score"]
    rebuilt = Reimported(other=gt).compare_with(Reimported(other=pred))["overall_score"]
    assert rebuilt != pytest.approx(original)


@pytest.mark.parametrize("path", REIMPORT)
def test_a_union_built_from_a_schema_is_refused_on_export(path):
    """The schema-only loop from #349 C: import builds the union, export refuses it."""
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
    with pytest.raises(ValueError, match="no type for a union with a model arm"):
        _export(Imported, path)


class TestFieldToProperty:
    """The converter itself, which the issue measured directly."""

    field_info = Shapes.model_fields["child"]

    def _convert(self, annotation):
        return JsonSchemaFieldConverter({}).field_to_property(
            annotation, self.field_info
        )

    def _to_config(self, annotation):
        return JsonSchemaFieldConverter({}).field_to_stickler_config(
            annotation, self.field_info
        )

    @pytest.mark.parametrize(
        ("method", "annotation", "what"),
        [
            ("field_to_property", Union[Card, Bank], "a union with a model arm"),
            ("field_to_stickler_config", Union[Card, Bank], "a union with a model arm"),
            (
                "field_to_property",
                List[List[Card]],
                "a StructuredModel inside a nested list",
            ),
            (
                "field_to_stickler_config",
                List[List[Card]],
                "a StructuredModel inside a nested list",
            ),
        ],
    )
    def test_the_converters_refuse_these_shapes_and_name_the_type(
        self, method, annotation, what
    ):
        convert = self._convert if method == "field_to_property" else self._to_config
        with pytest.raises(ValueError) as raised:
            convert(annotation)
        message = str(raised.value)
        assert message.startswith(f"{method}() cannot export {annotation!r}:")
        assert f"no type for {what}" in message
        assert message.endswith("or remove the field from the model you export.")

    @pytest.mark.parametrize(
        "method", ["field_to_property", "field_to_stickler_config"]
    )
    def test_both_converters_name_the_type_as_passed(self, method):
        """field_to_property() unwraps Optional, but names the type it was given."""
        convert = self._convert if method == "field_to_property" else self._to_config
        annotation = Optional[List[List[Card]]]
        with pytest.raises(ValueError) as raised:
            convert(annotation)
        assert str(raised.value).startswith(f"{method}() cannot export {annotation!r}:")

    def test_a_structured_model_is_still_refused(self):
        with pytest.raises(ValueError, match="StructuredModel"):
            self._convert(Card)

    def test_a_single_list_of_models_is_not_called_a_nested_list(self):
        with pytest.raises(ValueError, match="as a scalar"):
            self._convert(Optional[Annotated[List[Card], Field(description="d")]])

    @pytest.mark.parametrize(
        ("method", "written"),
        [("field_to_property", "string"), ("field_to_stickler_config", "str")],
    )
    def test_a_list_of_lists_is_written_as_a_string_by_the_converters(
        self, method, written
    ):
        """Named in model-export.md; to_json_schema() writes the same field correctly."""
        convert = self._convert if method == "field_to_property" else self._to_config
        assert convert(List[List[float]])["type"] == written
        assert Shapes.to_json_schema()["properties"]["nested"]["items"] == {
            "type": "array",
            "items": {"type": "number"},
        }

    def test_a_plain_model_is_written_as_an_object(self):
        assert self._convert(Plain)["type"] == "object"
