"""
Regression test for ComparableField rejecting default_factory.

ComparableField always forwarded its own ``default`` to pydantic's ``Field``,
so a caller-supplied ``default_factory`` collided with it and every collection
field raised ``TypeError: cannot specify both default and default_factory``.

See: https://github.com/awslabs/stickler/issues/306
"""

import json
from typing import Dict, List

import pytest

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
