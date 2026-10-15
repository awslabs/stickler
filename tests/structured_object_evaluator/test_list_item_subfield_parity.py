"""A primitive sub-field of a list item is classified as it is outside a list.

For each matched pair, `StructuredListComparator._handle_non_hierarchical_field`
counted a primitive sub-field by calling the field's comparator directly, not
through `_dispatch_field_comparison`. That skipped the dispatcher's guards, so:

- a dict in an `Any` sub-field raised `TypeError` out of `compare_with()`, where
  the same field at top level warns and scores an FD;
- a tuple, a set, or a `date` against its ISO string counted a TP that the
  item's score did not credit;
- a list in an `Any` sub-field counted one unit instead of one per element;
- a whitespace-only string counted as null.

The assertions compare the list path to the top-level path, because parity is
the requirement. See https://github.com/awslabs/stickler/issues/400
"""

import datetime
import warnings
from typing import Any, List

import pytest

from stickler import ComparableField, StructuredModel

COUNTS = ("tp", "fa", "fd", "fp", "tn", "fn")


class Element(StructuredModel):
    key: str = ComparableField(threshold=1.0)
    value: Any = ComparableField()
    match_threshold = 0.4


class Holder(StructuredModel):
    items: List[Element] = ComparableField()


class Flat(StructuredModel):
    value: Any = ComparableField()


def _counts(node):
    return {k: node["overall"][k] for k in COUNTS}


def _top_level(gt, pred):
    result = Flat(value=gt).compare_with(
        Flat(value=pred), include_confusion_matrix=True
    )
    return _counts(result["confusion_matrix"]["fields"]["value"])


def _in_list(gt, pred):
    result = Holder(items=[Element(key="k", value=gt)]).compare_with(
        Holder(items=[Element(key="k", value=pred)]), include_confusion_matrix=True
    )
    items = result["confusion_matrix"]["fields"]["items"]
    return _counts(items["fields"]["value"])


@pytest.mark.parametrize(
    "gt, pred",
    [
        pytest.param({"text": "p"}, {"text": "q"}, id="dict"),
        pytest.param({"text": "p"}, {"text": "p"}, id="identical-dict"),
        pytest.param(("a", "b"), ("a", "c"), id="tuple"),
        pytest.param({"a"}, {"b"}, id="set"),
        pytest.param(datetime.date(2020, 1, 1), "2020-01-01", id="date-vs-str"),
        pytest.param(["a", "b"], ["a", "c"], id="list"),
        pytest.param(["a", "b"], ["b", "a"], id="reordered-list"),
        pytest.param("  ", "x", id="whitespace"),
        pytest.param("abc", "abd", id="str"),
        pytest.param(None, "x", id="none-vs-str"),
    ],
)
def test_sub_field_counts_as_at_top_level(gt, pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _in_list(gt, pred) == _top_level(gt, pred)


@pytest.mark.parametrize("include_confusion_matrix", [False, True])
def test_dict_in_any_sub_field_warns_instead_of_raising(include_confusion_matrix):
    gt = Holder(items=[Element(key="k", value={"text": "p"})])
    pred = Holder(items=[Element(key="k", value={"text": "q"})])

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
