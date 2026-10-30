"""`evaluator_format=True` without `document_non_matches=True` collects no records.

It always wrote `"non_matches": []`, so "not collected" read as "no
non-matches": `print_non_matches` printed a success message over a document
with a wrong field (#413). 1.0.1 keeps the key, so `result["non_matches"]`
works as on 1.0.0, but as an empty `UncollectedNonMatches` placeholder that the
printers and the HTML report recognise, and warns once that 1.0.2 omits it.
The HTML side of this lives in `tests/reporting/html/test_html_non_match_records.py`.
"""

import copy
import json
import pickle
import warnings
from typing import Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import ExactComparator
from stickler.structured_object_evaluator.models.non_match_field import (
    UncollectedNonMatches,
    has_non_match_records,
    is_uncollected_non_matches,
)
from stickler.structured_object_evaluator.utils.pretty_print import (
    print_evaluation_results,
    print_non_matches,
)


class Doc(StructuredModel):
    a: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)
    b: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)


GT = Doc(a="x", b="y")
WRONG = Doc(a="x", b="z")


WARNING_TEXT = (
    "compare_with(evaluator_format=True) without document_non_matches=True "
    "returns 'non_matches' as an empty placeholder: the records were not "
    "collected, so it does not mean every field matched. Pass "
    "document_non_matches=True to collect them. From 1.0.2 the 'non_matches' "
    "key is omitted unless document_non_matches=True."
)


def placeholder_warnings(caught):
    return [
        w
        for w in caught
        if "returns 'non_matches' as an empty placeholder" in str(w.message)
    ]


def compare(*args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = GT.compare_with(*args, **kwargs)
    return result, placeholder_warnings(caught)


def test_unrequested_key_is_the_1_0_0_empty_list_as_a_placeholder():
    """`result["non_matches"]` works as on 1.0.0: it is `[]`, not a KeyError."""
    result, _ = compare(WRONG, evaluator_format=True)

    assert result["non_matches"] == []
    assert type(result["non_matches"]) is UncollectedNonMatches
    assert is_uncollected_non_matches(result["non_matches"])
    assert not has_non_match_records(result)
    # The default dict is unchanged: no key without records.
    assert "non_matches" not in GT.compare_with(WRONG)


def test_placeholder_serialises_as_an_empty_list_and_survives_copies():
    result, _ = compare(WRONG, evaluator_format=True)

    assert json.dumps(result["non_matches"]) == "[]"
    assert json.loads(json.dumps(result, default=str))["non_matches"] == []
    assert type(copy.deepcopy(result)["non_matches"]) is UncollectedNonMatches
    assert type(copy.copy(result["non_matches"])) is UncollectedNonMatches
    assert (
        type(pickle.loads(pickle.dumps(result))["non_matches"]) is UncollectedNonMatches
    )


def test_unrequested_records_warn_once_per_process():
    result, first = compare(WRONG, evaluator_format=True)
    _, second = compare(GT, evaluator_format=True)

    assert [str(w.message) for w in first] == [WARNING_TEXT]
    assert first[0].category is DeprecationWarning
    assert first[0].filename == __file__
    assert second == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"document_non_matches": True},
        {"evaluator_format": True, "document_non_matches": True},
    ],
)
def test_no_warning_when_records_were_requested_or_format_not_used(kwargs):
    _, caught = compare(WRONG, **kwargs)

    assert caught == []


class Item(StructuredModel):
    code: str = ComparableField(comparator=ExactComparator())


class Parent(StructuredModel):
    items: list[Item]


def test_list_items_scored_internally_do_not_warn_for_the_caller():
    """Per-item metrics call `compare_with(evaluator_format=True)` themselves."""
    gt = Parent(items=[Item(code="a"), Item(code="b")])
    pred = Parent(items=[Item(code="a"), Item(code="c")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gt.compare_with(pred, evaluator_format=True, document_non_matches=True)

    assert placeholder_warnings(caught) == []
    assert [nm["field_path"] for nm in result["non_matches"]] == ["items[1]"]
    # Each item carries the placeholder, as 1.0.0 carried `[]`.
    assert [type(i["non_matches"]) for i in result["fields"]["items"]["items"]] == [
        UncollectedNonMatches,
        UncollectedNonMatches,
    ]


def test_records_present_when_requested():
    result = GT.compare_with(WRONG, evaluator_format=True, document_non_matches=True)

    assert [nm["field_path"] for nm in result["non_matches"]] == ["b"]


def test_requested_and_none_found_is_an_empty_list():
    result = GT.compare_with(GT, evaluator_format=True, document_non_matches=True)

    assert result["non_matches"] == []
    assert type(result["non_matches"]) is list
    assert has_non_match_records(result)


@pytest.mark.parametrize("pred", [WRONG, GT], ids=["wrong", "clean"])
def test_printer_says_records_are_missing_not_that_all_matched(capsys, pred):
    result, _ = compare(pred, evaluator_format=True)
    for shown in (result, copy.deepcopy(result)):
        print_non_matches(shown, use_color=False)
        out = capsys.readouterr().out

        assert "No non-matches found" not in out
        assert "No non-match records in these results." in out
        # No EvalResult has an evaluator-format .raw, so that hint does not apply.
        assert "EvalResult" not in out


def test_evaluation_printer_says_records_are_missing(capsys):
    result, _ = compare(GT, evaluator_format=True, include_confusion_matrix=True)
    print_evaluation_results(result, use_color=False)
    out = capsys.readouterr().out

    assert "No non-matches found" not in out
    assert "No non-match records in these results." in out


def test_printers_report_a_collected_empty_list_as_clean(capsys):
    result = GT.compare_with(
        GT,
        evaluator_format=True,
        document_non_matches=True,
        include_confusion_matrix=True,
    )
    print_non_matches(result, use_color=False)
    assert "No non-matches found." in capsys.readouterr().out
    print_evaluation_results(result, use_color=False)
    out = capsys.readouterr().out
    assert "No non-matches found." in out
    assert "No non-match records" not in out


def test_printer_lists_requested_records(capsys):
    print_non_matches(
        GT.compare_with(WRONG, evaluator_format=True, document_non_matches=True),
        use_color=False,
    )

    assert '📍 b: "y" → "z"' in capsys.readouterr().out
