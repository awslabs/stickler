"""
Regression test for ComparableField rejecting default_factory.

ComparableField always forwarded its own ``default`` to pydantic's ``Field``,
so a caller-supplied ``default_factory`` collided with it and every collection
field raised ``TypeError: cannot specify both default and default_factory``.

See: https://github.com/awslabs/stickler/issues/306
"""

from typing import Dict, List

import pytest

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
