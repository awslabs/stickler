"""A sub-field of a list item is classified as it is outside a list.

For each matched pair, `StructuredListComparator._handle_non_hierarchical_field`
counted a primitive sub-field by calling the field's comparator directly, not
through `_dispatch_field_comparison`. That skipped the dispatcher's guards, so:

- a dict the field's comparator cannot score raised `TypeError` out of
  `compare_with()`, where the same field at top level warns and scores an FD;
- a pair the comparator is not given at top level (with an inferred
  comparator, a tuple, a set, `bytes`, or a `date` against its ISO string; a
  plain pydantic model under `LevenshteinComparator`) counted a TP that the
  item's score did not credit, and that `non_matches` reported as an FD;
- a list in an `Any` sub-field counted one unit instead of one per element;
- a whitespace-only string counted as null.

The list path now goes through the dispatcher. Three rules hold on top of it:
an empty list is null by its value, as `classification-logic.md` says; a pair
is read on its GT item's own fields, so a list of mixed model classes does not
raise; and two models of different classes in a sub-field that is neither
routed as a nested model nor list-annotated, such as `Any` or `Union[A, B]`,
are classified by `compare_with()`, as before. Such a sub-field keeps its
nested `fields` only when every pair has them, so every pair counts under
`aggregate`. Every expected count below is a literal, so a change to the
dispatcher shows up here. See https://github.com/awslabs/stickler/issues/400
"""

import datetime
import warnings
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

from stickler import (
    ComparableField,
    LevenshteinComparator,
    NonMatchType,
    StructuredModel,
)

COUNTS = ("tp", "fa", "fd", "fp", "tn", "fn")


def _expect(**counts):
    """Six counts from the non-zero ones; `fp` is `fa + fd`."""
    full = {k: counts.get(k, 0) for k in COUNTS}
    full["fp"] = full["fa"] + full["fd"]
    return full


class Element(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Any = ComparableField()
    match_threshold = 0.4


class Holder(StructuredModel):
    items: List[Element] = ComparableField()


class ObjectElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: object = ComparableField()
    match_threshold = 0.4


class ObjectHolder(StructuredModel):
    items: List[ObjectElement] = ComparableField()


class MappingElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Dict[str, str] = ComparableField(comparator=LevenshteinComparator())
    match_threshold = 0.4


class MappingHolder(StructuredModel):
    items: List[MappingElement] = ComparableField()


class Point(BaseModel):
    x: int


def _counts(node, key="overall"):
    return {k: node[key][k] for k in COUNTS}


def _in_list(gt, pred, holder=Holder, element=Element, **kwargs):
    return holder(items=[element(key="k", value=gt)]).compare_with(
        holder(items=[element(key="k", value=pred)]),
        include_confusion_matrix=True,
        **kwargs,
    )


def _value_node(result):
    return result["confusion_matrix"]["fields"]["items"]["fields"]["value"]


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        pytest.param({"text": "p"}, {"text": "q"}, _expect(fd=1), id="dict"),
        pytest.param({"text": "p"}, {"text": "p"}, _expect(fd=1), id="identical-dict"),
        pytest.param(("a", "b"), ("a", "c"), _expect(fd=1), id="tuple"),
        pytest.param({"a"}, {"b"}, _expect(fd=1), id="set"),
        pytest.param(b"ab", b"ab", _expect(fd=1), id="bytes"),
        pytest.param(Point(x=1), Point(x=2), _expect(fd=1), id="plain-model"),
        pytest.param(
            datetime.date(2020, 1, 1), "2020-01-01", _expect(fd=1), id="date-vs-str"
        ),
        pytest.param(["a", "b"], ["a", "c"], _expect(tp=1, fd=1), id="list"),
        pytest.param(["a", "b"], ["b", "a"], _expect(tp=2), id="reordered-list"),
        pytest.param(["a", "b"], [], _expect(fn=2), id="list-vs-empty"),
        pytest.param([], ["a", "b"], _expect(fa=2), id="empty-vs-list"),
        # One list is absent, so this is one value against `None`, as at top
        # level. `non_matches` reports one FN per element.
        pytest.param(["a", "b"], None, _expect(fn=1), id="list-vs-none"),
        pytest.param("  ", "x", _expect(fd=1), id="whitespace"),
        pytest.param("  ", None, _expect(fn=1), id="whitespace-vs-none"),
        pytest.param((), None, _expect(fn=1), id="empty-tuple-vs-none"),
        pytest.param(None, (), _expect(fa=1), id="none-vs-empty-tuple"),
        pytest.param("abc", "abd", _expect(tp=1), id="str"),
        pytest.param(None, "x", _expect(fa=1), id="none-vs-str"),
    ],
)
def test_sub_field_counts(gt, pred, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _counts(_value_node(_in_list(gt, pred))) == expected


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        pytest.param(None, [], _expect(tn=1), id="none-vs-empty"),
        pytest.param([], None, _expect(tn=1), id="empty-vs-none"),
        pytest.param([], [], _expect(tn=1), id="both-empty"),
        pytest.param([None], [None], _expect(tn=1), id="only-none-elements"),
        pytest.param("", [], _expect(tn=1), id="empty-str-vs-empty"),
        pytest.param({}, [], _expect(tn=1), id="empty-dict-vs-empty"),
        pytest.param(float("nan"), [], _expect(tn=1), id="nan-vs-empty"),
        pytest.param("x", [], _expect(fn=1), id="str-vs-empty"),
        pytest.param([], "x", _expect(fa=1), id="empty-vs-str"),
        # A whitespace string and an empty tuple are values, not null.
        pytest.param("  ", [], _expect(fn=1), id="whitespace-vs-empty"),
        pytest.param([], (), _expect(fa=1), id="empty-vs-empty-tuple"),
    ],
)
@pytest.mark.parametrize(
    "holder, element",
    [
        pytest.param(Holder, Element, id="any"),
        pytest.param(ObjectHolder, ObjectElement, id="object"),
    ],
)
def test_an_empty_list_in_an_open_sub_field_is_null(
    holder, element, gt, pred, expected
):
    node = _value_node(_in_list(gt, pred, holder=holder, element=element))
    assert _counts(node) == expected


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        pytest.param("  ", "", _expect(fn=1), id="whitespace-vs-empty-str"),
        pytest.param("", "  ", _expect(fa=1), id="empty-str-vs-whitespace"),
        pytest.param((), {}, _expect(fn=1), id="empty-tuple-vs-empty-dict"),
        pytest.param((), float("nan"), _expect(fn=1), id="empty-tuple-vs-nan"),
        pytest.param(b"", "", _expect(fn=1), id="empty-bytes-vs-empty-str"),
        pytest.param(set(), {}, _expect(fn=1), id="empty-set-vs-empty-dict"),
        pytest.param("  ", "  ", _expect(tp=1), id="whitespace-vs-itself"),
        pytest.param((), (), _expect(fd=1), id="empty-tuple-vs-itself"),
    ],
)
@pytest.mark.parametrize(
    "holder, element",
    [
        pytest.param(Holder, Element, id="any"),
        pytest.param(ObjectHolder, ObjectElement, id="object"),
    ],
)
def test_an_empty_value_in_an_open_sub_field_counts_as_at_top_level(
    holder, element, gt, pred, expected
):
    # A whitespace string and an empty tuple, set or bytes are values, while
    # "", {} and NaN are null, in a list element as at top level.
    node = _value_node(_in_list(gt, pred, holder=holder, element=element))
    assert _counts(node) == expected

    alone = element(key="k", value=gt).compare_with(
        element(key="k", value=pred), include_confusion_matrix=True
    )
    assert _counts(alone["confusion_matrix"]["fields"]["value"]) == expected


class OptionalStrElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Optional[str] = ComparableField()
    match_threshold = 0.4


class OptionalStrHolder(StructuredModel):
    items: List[OptionalStrElement] = ComparableField()


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        pytest.param("  ", "", _expect(fn=1), id="whitespace-vs-empty-str"),
        pytest.param("", "  ", _expect(fa=1), id="empty-str-vs-whitespace"),
        pytest.param("  ", "  ", _expect(tp=1), id="whitespace-vs-itself"),
    ],
)
def test_a_whitespace_string_in_a_str_sub_field_is_a_value(gt, pred, expected):
    node = _value_node(
        _in_list(gt, pred, holder=OptionalStrHolder, element=OptionalStrElement)
    )
    assert _counts(node) == expected


class StrOrListElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Optional[Union[str, List[str]]] = ComparableField()
    match_threshold = 0.4


class StrOrListHolder(StructuredModel):
    items: List[StrOrListElement] = ComparableField()


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        pytest.param(None, "", _expect(fa=1), id="none-vs-empty-str"),
        pytest.param("", None, _expect(fn=1), id="empty-str-vs-none"),
        pytest.param("", [], _expect(fn=1), id="empty-str-vs-empty-list"),
        pytest.param("", ["a", "b"], _expect(fa=1), id="empty-str-vs-list"),
        pytest.param(["a", "b"], "", _expect(fn=1), id="list-vs-empty-str"),
        pytest.param(None, [], _expect(tn=1), id="none-vs-empty-list"),
        pytest.param(["a", "b"], ["a", "c"], _expect(tp=1, fd=1), id="lists"),
    ],
)
def test_a_list_annotated_sub_field_counts_as_top_level(gt, pred, expected):
    # The dispatcher takes a list-annotated field's null rules from its
    # annotation, so `""` is a value here, at top level and in a list alike.
    # Only a field that is not list-annotated reads an empty list as null by
    # its value.
    node = _value_node(
        _in_list(gt, pred, holder=StrOrListHolder, element=StrOrListElement)
    )
    assert _counts(node) == expected

    alone = StrOrListElement(key="k", value=gt).compare_with(
        StrOrListElement(key="k", value=pred), include_confusion_matrix=True
    )
    assert _counts(alone["confusion_matrix"]["fields"]["value"]) == expected


class Inner(StructuredModel):
    a: str = ComparableField()
    b: Optional[str] = ComparableField(default=None)


class Other(StructuredModel):
    c: str = ComparableField()


class Search(StructuredModel):
    query: str = ComparableField()


class Browse(StructuredModel):
    url: str = ComparableField()


class ModelElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Any = ComparableField(threshold=0.9)
    match_threshold = 0.4


class ModelHolder(StructuredModel):
    items: List[ModelElement] = ComparableField()


class UnionElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Union[Search, Browse] = ComparableField(threshold=0.9)
    match_threshold = 0.4


class UnionHolder(StructuredModel):
    items: List[UnionElement] = ComparableField()


@pytest.mark.parametrize(
    "holder, element, gt, pred, overall, aggregate",
    [
        pytest.param(
            ModelHolder,
            ModelElement,
            Inner(a="x", b="y"),
            Inner(a="x", b="y"),
            _expect(tp=1),
            _expect(tp=2),
            id="same-model",
        ),
        pytest.param(
            ModelHolder,
            ModelElement,
            Inner(a="x", b="y"),
            Inner(a="x", b="z"),
            _expect(fd=1),
            _expect(tp=1, fd=1),
            id="model-below-threshold",
        ),
        pytest.param(
            UnionHolder,
            UnionElement,
            Search(query="hotels"),
            Search(query="motels"),
            _expect(fd=1),
            _expect(tp=1),
            id="union-below-threshold",
        ),
    ],
)
def test_a_model_in_a_sub_field_keeps_its_leaves(
    holder, element, gt, pred, overall, aggregate
):
    node = _value_node(_in_list(gt, pred, holder=holder, element=element))
    assert _counts(node) == overall
    assert _counts(node, "aggregate") == aggregate

    # The same field at top level, read from its own item.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = element(key="k", value=gt).compare_with(
            element(key="k", value=pred), include_confusion_matrix=True
        )
    top = alone["confusion_matrix"]["fields"]["value"]
    assert _counts(top) == overall
    assert _counts(top, "aggregate") == aggregate


class Pet(StructuredModel):
    name: str = ComparableField()


class Cat(Pet):
    lives: int = ComparableField()


@pytest.mark.parametrize(
    "holder, element, gt, pred, expected",
    [
        pytest.param(
            ModelHolder,
            ModelElement,
            Inner(a="x", b="y"),
            Other(c="z"),
            _expect(fd=1),
            id="other-class",
        ),
        pytest.param(
            ObjectHolder,
            ObjectElement,
            Inner(a="x", b="y"),
            Other(c="z"),
            _expect(fd=1),
            id="object-other-class",
        ),
        # compare_with() reads `b`, which `Other` lacks, as None, so the unset
        # `b` matches: 0.5, a TP at the default threshold of 0.5.
        pytest.param(
            Holder,
            Element,
            Inner(a="x"),
            Other(c="z"),
            _expect(tp=1),
            id="other-class-default-threshold",
        ),
        pytest.param(
            UnionHolder,
            UnionElement,
            Search(query="hotels"),
            Browse(url="x.com"),
            _expect(fd=1),
            id="union-other-class",
        ),
        pytest.param(
            ModelHolder,
            ModelElement,
            Cat(name="tom", lives=9),
            Pet(name="tom"),
            _expect(fd=1),
            id="base-for-subclass",
        ),
        # compare_with() scores the GT's own fields, and a Pet has only `name`.
        # The exact-class rule does not cover StructuredModel yet (#327).
        pytest.param(
            ModelHolder,
            ModelElement,
            Pet(name="tom"),
            Cat(name="tom", lives=9),
            _expect(tp=1),
            id="subclass-for-base",
        ),
    ],
)
def test_a_model_of_another_class_in_a_sub_field_is_scored_by_compare_with(
    holder, element, gt, pred, expected
):
    # compare_with() reads a field the prediction lacks as missing. compare(),
    # which the dispatcher uses, skips it, and would score `Inner(a, b)` vs
    # `Other(c)` 1.0, a TP (#327). No nested fields are kept for such a pair.
    node = _value_node(_in_list(gt, pred, holder=holder, element=element))
    assert _counts(node) == expected
    assert _counts(node, "aggregate") == expected
    assert not node.get("fields")


class OptionalUnionElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Optional[Union[Search, Browse]] = ComparableField(threshold=0.9)
    match_threshold = 0.4


class OptionalUnionHolder(StructuredModel):
    items: List[OptionalUnionElement] = ComparableField()


class ListArmUnionElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Union[Search, Browse, List[str]] = ComparableField(threshold=0.9)
    match_threshold = 0.4


class ListArmUnionHolder(StructuredModel):
    items: List[ListArmUnionElement] = ComparableField()


class ListArmAnyElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Union[Any, List[str]] = ComparableField(threshold=0.9)
    match_threshold = 0.4


class ListArmAnyHolder(StructuredModel):
    items: List[ListArmAnyElement] = ComparableField()


@pytest.mark.parametrize(
    "holder, element, gt, pred, aggregate",
    [
        pytest.param(
            OptionalUnionHolder,
            OptionalUnionElement,
            Search(query="hotels"),
            Browse(url="x.com"),
            _expect(fn=1),
            id="optional-union",
        ),
        pytest.param(
            ListArmUnionHolder,
            ListArmUnionElement,
            Search(query="hotels"),
            Browse(url="x.com"),
            _expect(fn=1),
            id="union-with-list-arm",
        ),
        pytest.param(
            ListArmAnyHolder,
            ListArmAnyElement,
            Inner(a="x", b="y"),
            Other(c="z"),
            _expect(fn=2),
            id="any-with-list-arm",
        ),
    ],
)
def test_another_class_in_a_nested_or_list_sub_field_counts_as_top_level(
    holder, element, gt, pred, aggregate
):
    # A field routed as a nested model, or annotated as a list, sends every
    # pair to the dispatcher, so the pair is a TP and its leaves report the
    # fields the prediction lacks, as at top level (#327).
    node = _value_node(_in_list(gt, pred, holder=holder, element=element))
    assert _counts(node) == _expect(tp=1)
    assert _counts(node, "aggregate") == aggregate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = element(key="k", value=gt).compare_with(
            element(key="k", value=pred), include_confusion_matrix=True
        )
    top = alone["confusion_matrix"]["fields"]["value"]
    assert _counts(top) == _expect(tp=1)
    assert _counts(top, "aggregate") == aggregate


def _pairs_in_list(gts, preds, holder=Holder, element=Element):
    def items(values):
        return [element(key=f"k{i}", value=v) for i, v in enumerate(values)]

    result = holder(items=items(gts)).compare_with(
        holder(items=items(preds)), include_confusion_matrix=True
    )
    return result["confusion_matrix"]


@pytest.mark.parametrize(
    "holder, element, gts, preds, value, root",
    [
        pytest.param(
            Holder,
            Element,
            [Inner(a="x"), Inner(a="y")],
            [Inner(a="x"), None],
            _expect(tp=1, fn=1),
            _expect(tp=3, fn=1),
            id="model-and-none",
        ),
        pytest.param(
            ObjectHolder,
            ObjectElement,
            [Inner(a="x"), Inner(a="y")],
            [Inner(a="x"), None],
            _expect(tp=1, fn=1),
            _expect(tp=3, fn=1),
            id="object-model-and-none",
        ),
        pytest.param(
            Holder,
            Element,
            ["abc", Inner(a="x")],
            ["abc", Inner(a="x")],
            _expect(tp=2),
            _expect(tp=4),
            id="primitive-and-model",
        ),
        pytest.param(
            ModelHolder,
            ModelElement,
            [Inner(a="x", b="y"), Inner(a="p", b="q")],
            [Inner(a="x", b="y"), Other(c="z")],
            _expect(tp=1, fd=1),
            _expect(tp=3, fd=1),
            id="same-and-other-class",
        ),
    ],
)
def test_a_sub_field_of_mixed_shapes_counts_every_pair(
    holder, element, gts, preds, value, root
):
    # `aggregate` sums a node's children when it has any. If one pair kept a
    # model's fields, the other pair's flat counts would drop out of it.
    matrix = _pairs_in_list(gts, preds, holder=holder, element=element)
    node = matrix["fields"]["items"]["fields"]["value"]
    assert _counts(node) == value
    assert _counts(node, "aggregate") == value
    assert _counts(matrix, "aggregate") == root


class Pets(StructuredModel):
    pets: List[Pet] = ComparableField()


class Actions(StructuredModel):
    actions: List[Union[Search, Browse]] = ComparableField()


def _list_fields(result, field):
    fields = result["confusion_matrix"]["fields"][field]["fields"]
    return {name: _counts(node) for name, node in fields.items()}


def test_a_list_of_a_model_and_its_subclass_does_not_raise():
    pets = [Cat(name="tom", lives=9), Pet(name="rex")]

    result = Pets(pets=pets).compare_with(
        Pets(pets=list(pets)), include_confusion_matrix=True
    )

    assert result["overall_score"] == 1.0
    # `lives` is counted for the Cat pair only, which declares it.
    assert _list_fields(result, "pets") == {
        "name": _expect(tp=2),
        "lives": _expect(tp=1),
    }


def test_a_list_of_a_model_union_does_not_raise():
    actions = [Search(query="q"), Browse(url="u")]

    result = Actions(actions=actions).compare_with(
        Actions(actions=list(actions)), include_confusion_matrix=True
    )

    assert result["overall_score"] == 1.0
    assert _list_fields(result, "actions") == {
        "query": _expect(tp=1),
        "url": _expect(tp=1),
    }


def test_a_list_with_the_base_class_first_counts_the_subclass_fields():
    result = Pets(pets=[Pet(name="rex"), Cat(name="tom", lives=9)]).compare_with(
        Pets(pets=[Pet(name="rex"), Pet(name="tom")]), include_confusion_matrix=True
    )

    assert result["overall_score"] == 0.75
    # The Cat pair is read on its own fields, so the missing `lives` counts.
    assert _list_fields(result, "pets") == {
        "name": _expect(tp=2),
        "lives": _expect(fn=1),
    }


def test_a_model_union_pair_of_other_classes_pairs_as_a_tp():
    # The pairing verdict comes from compare(), which skips the fields the
    # prediction lacks, so the pair is a TP while the field scores 0.0 (#327).
    result = Actions(actions=[Search(query="hotels")]).compare_with(
        Actions(actions=[Browse(url="x.com")]), include_confusion_matrix=True
    )

    node = result["confusion_matrix"]["fields"]["actions"]
    assert result["overall_score"] == 0.0
    assert _counts(node) == _expect(tp=1)
    assert _counts(node, "aggregate") == _expect(fn=1)
    assert _list_fields(result, "actions") == {"query": _expect(fn=1)}


class AnyField(StructuredModel):
    value: Any = ComparableField()


class ObjectField(StructuredModel):
    value: object = ComparableField()


class AnyOrListField(StructuredModel):
    value: Union[Any, List[str]] = ComparableField()


@pytest.mark.parametrize("model", [AnyField, ObjectField, AnyOrListField])
@pytest.mark.parametrize(
    "pred, overall",
    [
        pytest.param(["a"], _expect(tp=1), id="str"),
        pytest.param([1, 2], _expect(tp=1, fa=1), id="numbers"),
        pytest.param([{"a": "x"}], _expect(tp=1), id="dict"),
        pytest.param([["a"]], _expect(tp=1), id="list"),
    ],
)
def test_a_list_of_models_in_an_open_field_against_non_models_does_not_raise(
    model, pred, overall
):
    result = model(value=[Inner(a="x")]).compare_with(
        model(value=pred), include_confusion_matrix=True
    )

    node = result["confusion_matrix"]["fields"]["value"]
    assert _counts(node) == overall
    assert _counts(node, "aggregate") == _expect(fn=1, tn=1)


def test_a_list_of_models_in_an_any_sub_field_against_non_models_does_not_raise():
    node = _value_node(_in_list([Inner(a="x")], ["a"]))

    assert _counts(node) == _expect(tp=1)
    assert _counts(node, "aggregate") == _expect(fn=1, tn=1)


class ListAnyField(StructuredModel):
    value: List[Any] = ComparableField()


class ListAnyElement(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: List[Any] = ComparableField()
    match_threshold = 0.4


class ListAnyHolder(StructuredModel):
    items: List[ListAnyElement] = ComparableField()


@pytest.mark.parametrize("at_top_level", [True, False], ids=["top-level", "sub-field"])
@pytest.mark.parametrize(
    "field, holder, element, pred",
    [
        pytest.param(
            ListAnyField, ListAnyHolder, ListAnyElement, ["a"], id="list-of-any-str"
        ),
        pytest.param(AnyField, Holder, Element, [Other(c="z")], id="any-other-class"),
        pytest.param(
            ObjectField,
            ObjectHolder,
            ObjectElement,
            [Other(c="z")],
            id="object-other-class",
        ),
    ],
)
def test_a_list_of_models_against_a_list_lacking_their_fields_does_not_raise(
    field, holder, element, pred, at_top_level
):
    # The prediction lacks `a` and `b`, which read as missing. Pairing uses
    # compare(), which skips them, so the pair is a TP (#327).
    gt = [Inner(a="x")]
    if at_top_level:
        result = field(value=gt).compare_with(
            field(value=pred), include_confusion_matrix=True
        )
        node = result["confusion_matrix"]["fields"]["value"]
    else:
        node = _value_node(_in_list(gt, pred, holder=holder, element=element))

    assert _counts(node) == _expect(tp=1)
    assert _counts(node, "aggregate") == _expect(fn=1, tn=1)


def _at_top_level(gt, pred):
    result = AnyField(value=gt).compare_with(
        AnyField(value=pred), include_confusion_matrix=True
    )
    return result, result["confusion_matrix"]["fields"]["value"]


def _as_sub_field(gt, pred):
    result = _in_list(gt, pred)
    return result, _value_node(result)


WHERE = [
    pytest.param(_at_top_level, id="top-level"),
    pytest.param(_as_sub_field, id="sub-field"),
]


@pytest.mark.parametrize("compare", WHERE)
@pytest.mark.parametrize("value", ["b", 1, {"a": "x"}], ids=["str", "int", "dict"])
def test_a_ground_truth_list_of_a_model_and_a_non_model_still_raises(compare, value):
    # The first GT element is a model, so the list is scored as a list of
    # models, which calls each matched GT item's compare_with(). A value
    # without one raises, so field classification never sees such a pair. This
    # raised before #400 too: only the prediction side's crash is fixed.
    values = [Inner(a="x"), value]
    with pytest.raises(AttributeError, match="compare_with"):
        compare(values, list(values))


@pytest.mark.parametrize("compare", WHERE)
@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        pytest.param(
            ["b", Inner(a="x")],
            ["b", Inner(a="x")],
            _expect(tp=2),
            id="non-model-first",
        ),
        pytest.param(
            [Inner(a="x"), "b"],
            [Inner(a="x")],
            _expect(tp=1, fn=1),
            id="non-model-unmatched",
        ),
    ],
)
def test_a_ground_truth_list_of_a_model_and_a_non_model_runs_otherwise(
    compare, gt, pred, expected
):
    # The raise needs both a model as the first GT element and a matched GT
    # element without compare_with(). Either alone runs.
    _, node = compare(gt, pred)
    assert _counts(node) == expected


class Scored(BaseModel):
    """Not a StructuredModel, but scores itself like one."""

    a: str = "x"

    def compare_with(self, other, **kwargs):
        return {"overall_score": 1.0}


class Duck:
    """A plain class that scores itself like a StructuredModel."""

    def compare_with(self, other, **kwargs):
        return {"overall_score": 1.0}

    def __eq__(self, other):
        return isinstance(other, Duck)


@pytest.mark.parametrize("compare", WHERE)
@pytest.mark.parametrize("element", [Scored, Duck])
def test_a_ground_truth_list_of_a_model_and_a_self_scoring_value_runs(
    compare, element
):
    # The list scores, since every GT item has compare_with(). Field
    # classification then skips the GT item that is not a StructuredModel,
    # which has no `_dispatch_field_comparison()` to call.
    values = [Inner(a="x"), element()]
    result, node = compare(values, list(values))

    assert result["overall_score"] == 1.0
    assert _counts(node) == _expect(tp=2)


class FixedScoreElement(ModelElement):
    """Scores an item without recursing, as a custom `compare_with()` may."""

    def compare_with(self, other, **kwargs):
        return {"overall_score": 1.0}


class FixedScoreHolder(StructuredModel):
    items: List[FixedScoreElement] = ComparableField()


class FixedScoreObjectElement(StructuredModel):
    """`FixedScoreElement` with an `object` sub-field."""

    key: str = ComparableField(threshold=1.0)
    value: object = ComparableField(threshold=0.9)
    match_threshold = 0.4

    def compare_with(self, other, **kwargs):
        return {"overall_score": 1.0}


class FixedScoreObjectHolder(StructuredModel):
    items: List[FixedScoreObjectElement] = ComparableField()


class FixedScoreDefaultElement(Element):
    """`FixedScoreElement` at the default field threshold."""

    def compare_with(self, other, **kwargs):
        return {"overall_score": 1.0}


class FixedScoreDefaultHolder(StructuredModel):
    items: List[FixedScoreDefaultElement] = ComparableField()


def test_a_dict_against_a_model_in_a_sub_field_list_does_not_raise():
    result = _in_list(
        [Inner(a="x")], [{"a": "x"}], holder=ModelHolder, element=ModelElement
    )

    node = _value_node(result)
    # compare() skips a field the prediction lacks, so the pair is a TP, as
    # at top level (#327). The leaves report the miss. The same pair under a
    # fixed-score compare_with() is pinned below.
    assert _counts(node) == _expect(tp=1)
    assert _counts(node, "aggregate") == _expect(fn=1, tn=1)


@pytest.mark.parametrize(
    "holder, element",
    [
        pytest.param(FixedScoreHolder, FixedScoreElement, id="any"),
        pytest.param(FixedScoreObjectHolder, FixedScoreObjectElement, id="object"),
        pytest.param(
            FixedScoreDefaultHolder, FixedScoreDefaultElement, id="default-threshold"
        ),
    ],
)
@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(["a"], id="str"),
        pytest.param([1], id="int"),
        pytest.param([["a"]], id="list"),
        pytest.param([{"a": "x"}], id="dict"),
        pytest.param([Other(c="z")], id="other-class"),
    ],
)
def test_a_fixed_score_element_counts_a_list_of_models_as_without_it(
    holder, element, pred
):
    # The element's compare_with() returns a score without recursing. The
    # pair still counts what it does without the override and at top level:
    # a TP, with the leaves reporting the miss.
    result = _in_list([Inner(a="x")], pred, holder=holder, element=element)

    node = _value_node(result)
    assert result["overall_score"] == 1.0
    assert _counts(node) == _expect(tp=1)
    assert _counts(node, "aggregate") == _expect(fn=1, tn=1)


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        pytest.param(("a", "b"), ("a", "b"), NonMatchType.FALSE_DISCOVERY, id="tuple"),
        pytest.param("  ", "x", NonMatchType.FALSE_DISCOVERY, id="whitespace"),
        pytest.param("  ", None, NonMatchType.FALSE_NEGATIVE, id="whitespace-vs-none"),
    ],
)
def test_sub_field_counts_agree_with_non_matches(gt, pred, expected):
    result = _in_list(gt, pred, document_non_matches=True)

    reported = [
        n["non_match_type"]
        for n in result["non_matches"]
        if n["field_path"] == "items[0].value"
    ]
    assert reported == [expected]
    count = {
        NonMatchType.FALSE_DISCOVERY: "fd",
        NonMatchType.FALSE_NEGATIVE: "fn",
    }[expected]
    assert _counts(_value_node(result))[count] == 1


@pytest.mark.parametrize("include_confusion_matrix", [False, True])
@pytest.mark.parametrize(
    "holder, element",
    [
        pytest.param(Holder, Element, id="any"),
        pytest.param(MappingHolder, MappingElement, id="dict-explicit-levenshtein"),
    ],
)
def test_dict_in_sub_field_warns_instead_of_raising(
    holder, element, include_confusion_matrix
):
    gt = holder(items=[element(key="k", value={"text": "p"})])
    pred = holder(items=[element(key="k", value={"text": "q"})])

    with pytest.warns(UserWarning, match="holds a mapping"):
        result = gt.compare_with(
            pred, include_confusion_matrix=include_confusion_matrix
        )

    # The item scores as it does alone: `key` matches, `value` is an FD.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = gt.items[0].compare_with(pred.items[0])["overall_score"]
    assert alone == 0.5
    assert result["overall_score"] == alone
