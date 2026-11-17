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
    ],
)
def test_an_empty_list_in_an_any_sub_field_is_null(gt, pred, expected):
    assert _counts(_value_node(_in_list(gt, pred))) == expected


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


class FixedScoreElement(ModelElement):
    """Scores an item without recursing, as a custom `compare_with()` may."""

    def compare_with(self, other, **kwargs):
        return {"overall_score": 1.0}


class FixedScoreHolder(StructuredModel):
    items: List[FixedScoreElement] = ComparableField()


@pytest.mark.parametrize(
    "holder, element",
    [
        pytest.param(ModelHolder, ModelElement, id="plain"),
        pytest.param(FixedScoreHolder, FixedScoreElement, id="compare-with-override"),
    ],
)
def test_a_dict_against_a_model_in_a_sub_field_list_does_not_raise(holder, element):
    result = _in_list([Inner(a="x")], [{"a": "x"}], holder=holder, element=element)

    node = _value_node(result)
    # compare() skips a field the prediction lacks, so the pair is a TP, as
    # at top level (#327). The leaves report the miss.
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
