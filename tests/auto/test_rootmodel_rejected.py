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
time. The two ``AttributeError``s named neither the shape nor a field; the
mapping case named ``root``, the shadow's own field rather than the model field
that held the value. None of them said the shape was unsupported. The fourth
reported a perfect match for a wrong prediction.

The refusal lives in ``_build``, the function that assembles a shadow class for
every model the zero-config builder mirrors, so it covers the class passed in
and every model reached from it: a ``RootModel`` field, and a ``RootModel`` list
element. ``RootModel[Point]`` nested in either position scored 1.0 before that.

These tests pin the refusal and its message across the five root shapes and both
nesting positions, and pin the shapes that must keep working, so the check
cannot widen into them.

See https://github.com/awslabs/stickler/issues/366
"""

from typing import List, Optional

import pytest
from pydantic import BaseModel, RootModel

import stickler
from stickler import ComparableField, EvalSpec, eval_for
from stickler.auto.builder import specs_for, structured_model_for
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


@pytest.mark.parametrize("root_cls", [row[0] for row in ROOTS], ids=ROOT_IDS)
def test_eval_for_refuses_every_root_shape(root_cls):
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


def test_a_root_model_field_is_refused_and_the_message_names_it():
    """A nested RootModel is refused when the evaluator is built.

    ``RootModel[Point]`` here is the silent case: the shadow carried one ``root``
    field that no dumped value ever filled, so both sides read None and two
    points differing in every field scored 1.0. The message names the field,
    since the class alone does not say which of several fields to fix.
    """

    class Holder(BaseModel):
        name: str
        pt: RootModel[Point]

    with pytest.raises(TypeError, match="RootModel") as exc:
        eval_for(Holder)
    assert "Holder.pt" in str(exc.value)


def test_a_root_model_list_element_is_refused_and_marked_as_an_element():
    """A list element reaches the builder through its own branch.

    It scored 1.0 the same way a single nested field did. ``[]`` in the path
    distinguishes the element from the field that holds the list.
    """

    class ListHolder(BaseModel):
        items: List[RootModel[Point]]

    with pytest.raises(TypeError, match="RootModel") as exc:
        eval_for(ListHolder)
    assert "ListHolder.items[]" in str(exc.value)


def test_a_list_rooted_field_is_refused_at_build_time_too():
    """This shape used to fail per document, with a pydantic ``ValidationError``.

    Refusing it when the evaluator is built reports it once, at configuration
    time, instead of once per document in a corpus run.
    """

    class Holder(BaseModel):
        name: str
        points: RootModel[list[Point]]

    with pytest.raises(TypeError, match="RootModel") as exc:
        eval_for(Holder)
    assert "Holder.points" in str(exc.value)


def test_from_pydantic_refuses_a_root_model():
    """``from_pydantic`` delegates to the same builder, so it refuses too."""
    with pytest.raises(TypeError, match="RootModel"):
        StructuredModel.from_pydantic(RootModel[int])


def test_a_structured_model_holding_a_root_model_field_is_untouched():
    """The guard must not widen past the models the builder mirrors.

    An explicitly configured ``StructuredModel`` is returned as configured and
    is never mirrored, so its fields are scored by the comparators the caller
    declared. That boundary is deliberate: re-inspecting an explicit
    configuration would discard the caller's decision.
    """

    class Configured(StructuredModel):
        name: str = ComparableField(default="")
        pt: Optional[RootModel[Point]] = ComparableField(default=None)

    spec = eval_for(Configured)
    assert spec.eval_model is Configured

    result = spec.evaluate(
        Configured(name="a", pt=RootModel[Point](Point(x=1.0, y=2.0))),
        Configured(name="a", pt=RootModel[Point](Point(x=9.0, y=9.0))),
    )
    assert result.field_scores["name"] == pytest.approx(1.0)
    # The boundary is what `pt` does, not what `name` does: the field is still
    # compared by the declared comparator, and two differing roots do not score
    # a perfect match. Asserting only `name` would pass even if `pt` returned
    # the false 1.0 this work removes everywhere else.
    assert result.field_scores["pt"] < 1.0


class _Holder(BaseModel):
    name: str
    pt: RootModel[Point]


class _ListHolder(BaseModel):
    items: List[RootModel[Point]]


@pytest.mark.parametrize(
    "cls, where",
    [(row[0], row[0].__name__) for row in ROOTS]
    + [(_Holder, "_Holder.pt"), (_ListHolder, "_ListHolder.items[]")],
    ids=ROOT_IDS + ["field", "list-element"],
)
def test_specs_for_refuses_a_root_model_and_names_where(cls, where):
    """``specs_for`` reported ``root.*`` paths, such as ``pt.root.x``, that are
    never scored (#423). It now refuses with the builder's RootModel message,
    which names the class or the field that holds the root.
    """
    with pytest.raises(TypeError, match="a RootModel holds one unnamed value") as exc:
        specs_for(cls)
    assert where in str(exc.value)


def test_specs_for_refuses_a_recursive_model_too():
    """The check is the builder's, so it covers more than RootModel: a
    recursive model, which ``eval_for`` refuses, is refused here as well."""

    class Node(BaseModel):
        name: str
        children: List["Node"] = []

    with pytest.raises(TypeError, match="recursive"):
        specs_for(Node)


def test_explain_on_a_hand_written_eval_model_does_not_run_the_builder():
    """An ``EvalSpec`` given its own ``eval_model`` never uses the builder, so
    the builder refusing ``source_cls`` must not break ``explain()``. Here the
    source is recursive, which the builder refuses, and the spec still scores.
    """

    class Node(BaseModel):
        name: str
        children: List["Node"] = []

    class NodeSM(StructuredModel):
        name: str = ComparableField()

    spec = EvalSpec(Node, NodeSM, weight_hints=False, match_threshold=0.7)
    assert spec.evaluate(Node(name="a"), Node(name="a")).overall_score == pytest.approx(1.0)
    assert "name" in spec.explain()


def test_explain_on_a_structured_model_holding_a_root_model_field_still_works():
    """The configured path is never built, so ``specs_for`` does not refuse it."""

    class Configured(StructuredModel):
        name: str = ComparableField(default="")
        pt: Optional[RootModel[Point]] = ComparableField(default=None)

    assert "name" in eval_for(Configured).explain()
