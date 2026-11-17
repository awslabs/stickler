"""
Regression test for ComparableField rejecting default_factory.

ComparableField always forwarded its own ``default`` to pydantic's ``Field``,
so a caller-supplied ``default_factory`` collided with it and every collection
field raised ``TypeError: cannot specify both default and default_factory``.

See: https://github.com/awslabs/stickler/issues/306
"""

import inspect
import json
from typing import Annotated, Dict, List, Optional, Set

import pytest
from pydantic import Field

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class FactoryModel(StructuredModel):
    name: str = ComparableField()
    tags: List[str] = ComparableField(default_factory=list)
    meta: Dict[str, str] = ComparableField(default_factory=dict)


def test_list_factory_produces_empty_list():
    """A list factory field defaults to [], not None."""
    assert FactoryModel(name="x").tags == []


def test_dict_factory_produces_empty_dict():
    """A dict factory field defaults to {}, not None."""
    assert FactoryModel(name="x").meta == {}


def test_factory_default_is_not_shared_between_instances():
    """Each instance gets its own mutable default."""
    first = FactoryModel(name="x")
    second = FactoryModel(name="y")
    first.tags.append("only-first")
    assert first.tags == ["only-first"]
    assert second.tags == []


def test_factory_field_still_compares():
    """A factory field takes part in comparison like any other field."""
    result = FactoryModel(name="Alexandra").compare_with(FactoryModel(name="Alexandre"))
    assert 0.0 < result["overall_score"] <= 1.0


def test_supplying_both_default_and_factory_raises():
    """Passing both must still raise, not silently drop one."""
    with pytest.raises(TypeError, match="default_factory"):
        ComparableField(default=5, default_factory=list)


def test_supplying_explicit_none_default_and_factory_raises():
    """An explicit default=None is a caller's choice, so pairing it still raises.

    The omitted-default case is indistinguishable from default=None by value,
    which is why the check looks at whether the caller supplied the key.
    """
    with pytest.raises(TypeError, match="default_factory"):
        ComparableField(default=None, default_factory=list)


def test_an_ellipsis_default_beside_a_factory_is_ignored_as_in_pydantic():
    """Pydantic reads `default=...` as "no default", so it does not collide with
    a factory: the field is optional and takes the factory's value. Mirrored,
    not refused, so a ComparableField behaves as the Field it wraps."""
    assert not Field(default=..., default_factory=list).is_required()

    class EllipsisModel(StructuredModel):
        tags: List[str] = ComparableField(default=..., default_factory=list)

    assert not EllipsisModel.model_fields["tags"].is_required()
    assert EllipsisModel().tags == []


def test_omitted_default_is_still_none():
    """Without a factory, an omitted default keeps its historical None."""

    class PlainModel(StructuredModel):
        code: str = ComparableField()

    assert PlainModel().code is None


def test_explicit_default_still_honoured():
    """A plain default is unaffected by the factory handling."""

    class DefaultModel(StructuredModel):
        code: str = ComparableField(default="unset")
        items: List[str] = ComparableField(default=[])

    model = DefaultModel()
    assert model.code == "unset"
    assert model.items == []


def test_factory_field_is_optional_in_json_schema():
    """A factory is a real default, so the field is not required."""
    assert "tags" not in FactoryModel.model_json_schema().get("required", [])


def test_default_factory_none_stays_optional():
    """default_factory=None is pydantic's own signature default, not a real factory."""

    class NoteModel(StructuredModel):
        note: str = ComparableField(default_factory=None, comparator=ExactComparator())

    field_info = NoteModel.model_fields["note"]
    assert not field_info.is_required()
    assert field_info.default is None
    assert NoteModel().note is None


def test_stickler_config_default_factory_round_trips_through_json():
    """to_stickler_config() must be JSON-serializable and rebuild tags as optional."""
    config = FactoryModel.to_stickler_config()

    serialized = json.dumps(config)

    rebuilt = StructuredModel.model_from_json(json.loads(serialized))
    assert not rebuilt.model_fields["tags"].is_required()
    assert rebuilt(name="a").tags == []


def test_json_schema_round_trip_keeps_factory_default():
    """from_json_schema() must rebuild a factory-backed field with [], not None."""
    schema = FactoryModel.to_json_schema()

    rebuilt = StructuredModel.from_json_schema(schema)

    assert rebuilt(name="a").tags == []


def test_a_set_factory_exports_as_a_list():
    """JSON has no set type, so exporting the set itself is the same bug again.

    A list-annotated field may legally use `default_factory=set`, and a set in
    the exported config makes `json.dumps` raise exactly as PydanticUndefined
    did.
    """

    class SetModel(StructuredModel):
        name: str = ComparableField(comparator=ExactComparator())
        tags: List[str] = ComparableField(
            default_factory=set, comparator=ExactComparator()
        )

    config = SetModel.to_stickler_config()
    assert config["fields"]["tags"]["default"] == []
    json.dumps(config)
    json.dumps(SetModel.to_json_schema())


class Sub(StructuredModel):
    sku: str = ComparableField(comparator=ExactComparator())


class ParentModel(StructuredModel):
    name: str = ComparableField(comparator=ExactComparator())
    subs: List[Sub] = ComparableField(default_factory=list)


def test_list_of_models_factory_round_trips_through_stickler_config():
    """The list-of-models branch must export the factory's [] like the primitive one."""
    config = ParentModel.to_stickler_config()
    assert config["fields"]["subs"]["default"] == []

    rebuilt = StructuredModel.model_from_json(json.loads(json.dumps(config)))
    assert rebuilt(name="a").subs == []


def test_list_of_models_factory_round_trips_through_json_schema():
    """from_json_schema() must rebuild a factory-backed List[Model] with [], not None."""
    schema = ParentModel.to_json_schema()
    assert schema["properties"]["subs"]["default"] == []

    rebuilt = StructuredModel.from_json_schema(schema)
    assert rebuilt(name="a").subs == []


def test_list_of_models_without_factory_exports_no_default():
    """Only the factory case is new; a plain List[Model] field exports as before."""

    class PlainParent(StructuredModel):
        name: str = ComparableField(comparator=ExactComparator())
        subs: List[Sub] = ComparableField()

    assert "default" not in PlainParent.to_stickler_config()["fields"]["subs"]
    assert "default" not in PlainParent.to_json_schema()["properties"]["subs"]


def test_dict_factory_round_trips_through_json_schema():
    """A mapping exports as an object, so its factory's {} survives the trip.

    It used to take the scalar branch, which exported it as "string" with no
    default, so the rebuilt field was None.
    """
    prop = FactoryModel.to_json_schema()["properties"]["meta"]
    assert prop["type"] == "object"
    assert prop["additionalProperties"] == {"type": "string"}
    assert prop["default"] == {}

    rebuilt = StructuredModel.from_json_schema(
        json.loads(json.dumps(FactoryModel.to_json_schema()))
    )
    assert rebuilt(name="a").meta == {}
    assert rebuilt(name="a", meta={"k": "v"}).meta == {"k": "v"}


def test_dict_factory_round_trips_through_stickler_config():
    """`"default": {}` used to land in a field exported as "str", so the
    rebuilt model rejected every mapping except that default."""
    config = FactoryModel.to_stickler_config()
    assert config["fields"]["meta"]["type"] == "Dict[str, str]"
    assert config["fields"]["meta"]["default"] == {}

    rebuilt = StructuredModel.model_from_json(json.loads(json.dumps(config)))
    assert rebuilt(name="a").meta == {}
    assert rebuilt(name="a", meta={"k": "v"}).meta == {"k": "v"}


def test_a_round_tripped_mapping_field_scores_like_the_original():
    """Both exports rebuild a mapping field that scores what the original does."""
    gt = {"name": "a", "meta": {"vendor": "Acme Corporation", "city": "Seattle"}}
    pred = {"name": "a", "meta": {"vendor": "Acme Corp", "city": "Seatle"}}
    original = FactoryModel(**gt).compare_with(FactoryModel(**pred))
    expected = original["field_scores"]["meta"]
    assert 0.0 < expected < 1.0

    for rebuilt in (
        StructuredModel.model_from_json(FactoryModel.to_stickler_config()),
        StructuredModel.from_json_schema(FactoryModel.to_json_schema()),
    ):
        result = rebuilt(**gt).compare_with(rebuilt(**pred))
        assert result["field_scores"]["meta"] == pytest.approx(expected)


def test_a_factory_default_is_not_written_into_a_scalar_export():
    """`Set[str]` exports as "str" / "string", which cannot hold the factory's
    value, so no default is written rather than one the rebuilt field rejects."""

    class TagSet(StructuredModel):
        tags: Set[str] = ComparableField(
            default_factory=set, comparator=ExactComparator()
        )

    assert "default" not in TagSet.to_stickler_config()["fields"]["tags"]
    assert "default" not in TagSet.to_json_schema()["properties"]["tags"]


def test_a_literal_set_default_exports_as_a_list():
    """A literal set broke json.dumps exactly as the set factory did."""

    class SetLiteral(StructuredModel):
        tags: List[str] = ComparableField(
            default={"d", "b", "e", "a", "c"}, comparator=ExactComparator()
        )

    config = SetLiteral.to_stickler_config()
    assert config["fields"]["tags"]["default"] == ["a", "b", "c", "d", "e"]
    json.dumps(config)


def test_an_unknown_factory_is_matched_by_identity():
    """`in` compares with ==, so export consulted the factory object's __eq__.

    The trap is armed only after the class exists: on Python < 3.13, pydantic's
    `inspect.signature(factory)` at class definition compares it with == too.
    """

    class Factory:
        armed = False

        def __call__(self):
            return []

        def __eq__(self, other):
            if self.armed:
                raise AssertionError("export compared the factory with ==")
            return NotImplemented

        __hash__ = object.__hash__

    factory = Factory()

    class Custom(StructuredModel):
        tags: List[str] = ComparableField(
            default_factory=factory, comparator=ExactComparator()
        )

    factory.armed = True
    assert "default" not in Custom.to_json_schema()["properties"]["tags"]
    assert "default" not in Custom.to_stickler_config()["fields"]["tags"]


def test_the_omitted_default_reads_as_unset_in_the_signature():
    """A bare object() sentinel showed as `<object object at 0x...>`."""
    param = inspect.signature(ComparableField).parameters["default"]
    assert str(param) == "default: Any = unset"


def test_a_required_list_of_models_stays_required_through_stickler_config():
    """The list-of-models branch wrote no `required`, which model_from_json()
    reads as False, so a required list rebuilt as optional."""

    class RequiredParent(StructuredModel):
        name: str = ComparableField(comparator=ExactComparator())
        subs: List[Sub] = ComparableField(default=...)

    config = RequiredParent.to_stickler_config()
    assert config["fields"]["subs"]["required"] is True
    assert StructuredModel.model_from_json(config).model_fields["subs"].is_required()
    assert ParentModel.to_stickler_config()["fields"]["subs"]["required"] is False


def _optional_factory_model(annotation, factory):
    return type(
        "OptionalFactory",
        (StructuredModel,),
        {
            "__annotations__": {"name": str, "tags": annotation},
            "name": ComparableField(comparator=ExactComparator()),
            "tags": ComparableField(default_factory=factory),
        },
    )


_UNSET = object()

# Pydantic keeps `Annotated` inside a union, so the first two reach the
# exporters as `Optional[Annotated[List[...], ...]]`, which read as a scalar.
_OPTIONAL_FACTORY_SHAPES = {
    "annotated-str-list": (Optional[Annotated[List[str], "m"]], list, ["a"]),
    "annotated-model-list": (Optional[Annotated[List[Sub], "m"]], list, [{"sku": "a"}]),
    "int-list": (Optional[List[int]], list, [1]),
    "int-mapping": (Optional[Dict[str, int]], dict, {"a": 1}),
}


@pytest.mark.parametrize(
    "annotation, factory, sample",
    list(_OPTIONAL_FACTORY_SHAPES.values()),
    ids=list(_OPTIONAL_FACTORY_SHAPES),
)
def test_an_optional_factory_field_round_trips(annotation, factory, sample):
    """Both rebuilt models accept None and score every pair as the original does.

    JSON Schema carries the factory's empty value. A Stickler config has no
    nullable flag and `model_from_json()` widens to Optional only for a None
    default, so it writes none: written, the field rebuilt non-nullable and
    rejected None. An empty list or mapping scores the same as None.
    """
    original = _optional_factory_model(annotation, factory)
    schema = original.to_json_schema()
    config = original.to_stickler_config()
    assert schema["properties"]["tags"]["default"] == factory()
    assert "default" not in config["fields"]["tags"]

    values = [_UNSET, None, factory(), sample]

    def scores(model):
        def build(value):
            return model(name="n") if value is _UNSET else model(name="n", tags=value)

        return [
            build(gt).compare_with(build(pred))["field_scores"]["tags"]
            for gt in values
            for pred in values
        ]

    expected = scores(original)
    for rebuilt in (
        StructuredModel.from_json_schema(json.loads(json.dumps(schema))),
        StructuredModel.model_from_json(json.loads(json.dumps(config))),
    ):
        assert rebuilt(name="n", tags=None).tags is None
        assert scores(rebuilt) == expected


def _config_round_trip(model):
    return StructuredModel.model_from_json(
        json.loads(json.dumps(model.to_stickler_config()))
    )


def _accepts_none(model, field):
    try:
        model(**{field: None})
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    "field_kwargs",
    [{"default_factory": list}, {"default": []}],
    ids=["factory", "literal"],
)
def test_a_non_optional_list_of_models_stays_non_nullable_over_two_round_trips(
    field_kwargs,
):
    """The list-of-models branch dropped a literal `[]`, so the first rebuild's
    default went missing in the second export, and model_from_json() widened
    the missing default to Optional."""
    original = type(
        "Leaves",
        (StructuredModel,),
        {
            "__annotations__": {"subs": List[Sub]},
            "subs": ComparableField(**field_kwargs),
        },
    )
    once = _config_round_trip(original)
    twice = _config_round_trip(once)
    assert once.to_stickler_config()["fields"]["subs"]["default"] == []
    for model in (original, once, twice):
        assert not _accepts_none(model, "subs")
        assert model().subs == []


@pytest.mark.parametrize("required", [False, True])
def test_a_hand_written_list_of_models_is_nullable_unless_required(required):
    """model_from_json() widens a None default to Optional on this path as on
    the primitive one (#149); a required list stays `List[Model]`."""
    config = {
        "model_name": "Hand",
        "fields": {
            "subs": {
                "type": "list_structured_model",
                "required": required,
                "fields": {"sku": {"type": "str", "comparator": "ExactComparator"}},
            }
        },
    }
    model = StructuredModel.model_from_json(config)
    assert _accepts_none(model, "subs") is not required
