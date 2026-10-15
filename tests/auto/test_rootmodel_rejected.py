"""A RootModel is refused when the evaluator is built, not when it is used.

``eval_for`` used to accept a ``RootModel`` and return a spec. What happened next
depended on the root type, and one of the four outcomes was silent:

======================================  ==========================================
``RootModel[list[Point]]``              ``AttributeError: 'list' object has no
                                        attribute 'keys'``
``RootModel[int]``                      ``AttributeError: 'int' object has no
                                        attribute 'copy'``
``RootModel[dict[str, int]]``           ``ValidationError``, missing ``root``
``RootModel[Point]``                    **scored 1.0** for two values differing
                                        in every field
======================================  ==========================================

The three errors arrived once per document rather than once at configuration
time, and named neither ``RootModel`` nor the field they could not key. The
fourth reported a perfect match for a wrong prediction.

These tests pin the refusal and its message across all four root shapes, and pin
the two shapes that must keep working, so the check cannot widen into them.

See https://github.com/awslabs/stickler/issues/366
"""

import pytest
from pydantic import BaseModel, RootModel, ValidationError

import stickler
from stickler import ComparableField, eval_for
from stickler.auto.builder import structured_model_for
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class Point(BaseModel):
    x: float
    y: float


class NamedPoints(RootModel[list[Point]]):
    """A named subclass, since that is how a caller usually declares one."""


# Each root shape failed differently before the guard, so each is pinned. The
# values differ in every field, which is what made the `RootModel[Point]` row
# score 1.0.
ROOTS = [
    (RootModel[list[Point]], [Point(x=1.0, y=2.0)], [Point(x=9.0, y=9.0)]),
    (NamedPoints, [Point(x=1.0, y=2.0)], [Point(x=9.0, y=9.0)]),
    (RootModel[int], 1, 99),
    (RootModel[dict[str, int]], {"a": 1}, {"a": 99}),
    (RootModel[Point], Point(x=1.0, y=2.0), Point(x=9.0, y=9.0)),
]
ROOT_IDS = ["list", "named-subclass", "int", "dict", "model"]


@pytest.mark.parametrize("root_cls, gt_value, pred_value", ROOTS, ids=ROOT_IDS)
def test_eval_for_refuses_every_root_shape(root_cls, gt_value, pred_value):
    with pytest.raises(TypeError, match="RootModel") as exc:
        eval_for(root_cls)
    assert root_cls.__name__ in str(exc.value)


@pytest.mark.parametrize("root_cls, gt_value, pred_value", ROOTS, ids=ROOT_IDS)
def test_evaluate_refuses_every_root_shape(root_cls, gt_value, pred_value):
    """The facade refuses too, so neither the old errors nor the false 1.0 remain."""
    with pytest.raises(TypeError, match="RootModel"):
        stickler.evaluate(root_cls(gt_value), root_cls(pred_value))


def test_the_message_says_what_to_do_instead():
    with pytest.raises(TypeError) as exc:
        structured_model_for(RootModel[int])
    message = str(exc.value)
    assert "RootModel" in message
    assert "BaseModel" in message


def test_a_plain_model_still_compiles_and_scores():
    """The guard must not widen: a BaseModel is unaffected."""

    class Pair(BaseModel):
        label: str
        value: float

    result = stickler.evaluate(
        Pair(label="same", value=1.0), Pair(label="same", value=1.0)
    )
    assert result.overall_score == pytest.approx(1.0)


def test_a_structured_model_still_takes_the_configured_path():
    """A StructuredModel never reaches the guard, because eval_for returns early."""

    class Configured(StructuredModel):
        label: str = ComparableField(default="")

    spec = eval_for(Configured)
    assert spec.eval_model is Configured


def test_a_model_holding_a_root_model_field_still_compiles():
    """The guard reads the top-level class only.

    Refusing the outer model would reject one whose other fields are scoreable.
    A nested root value fails on its own terms when it is compared, with a
    pydantic ``ValidationError`` that names the field, which is a different shape
    from the one this guard covers.
    """

    class Holder(BaseModel):
        name: str
        points: RootModel[list[Point]]

    spec = eval_for(Holder)
    assert spec is not None

    instance = Holder(name="a", points=RootModel[list[Point]]([Point(x=1.0, y=2.0)]))
    with pytest.raises(ValidationError) as exc:
        spec.evaluate(instance, instance)
    assert "points" in str(exc.value)
