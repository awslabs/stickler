"""`evaluator_format=True` writes `non_matches` only with `document_non_matches=True`.

Through 1.0.x it wrote `"non_matches": []` without records, so "not collected"
read as "no non-matches": `print_non_matches` printed a success message over a
document with a wrong field (#413). It now omits the key, as the default
`compare_with()` dict does, and the printers say the records were not
collected (#608). The HTML side of this lives in
`tests/reporting/html/test_html_non_match_records.py`.
"""

import warnings
from typing import Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import ExactComparator
from stickler.structured_object_evaluator.utils.pretty_print import (
    print_evaluation_results,
    print_non_matches,
)


class Doc(StructuredModel):
    a: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)
    b: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)


GT = Doc(a="x", b="y")
WRONG = Doc(a="x", b="z")


@pytest.mark.parametrize("pred", [WRONG, GT], ids=["wrong", "same"])
def test_unrequested_key_is_absent(pred):
    """Without records there is no key, as in the default dict."""
    result = GT.compare_with(pred, evaluator_format=True)

    assert "non_matches" not in result
    assert set(result) == {"overall", "fields", "confusion_matrix"}
    assert "non_matches" not in GT.compare_with(pred)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"evaluator_format": True},
        {"evaluator_format": True, "document_non_matches": True},
        {"evaluator_format": True, "include_confusion_matrix": True},
    ],
)
def test_evaluator_format_emits_no_non_matches_warning(kwargs):
    """The 1.0.x DeprecationWarning about the `[]` placeholder is gone."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GT.compare_with(WRONG, **kwargs)

    assert [str(w.message) for w in caught if "non_matches" in str(w.message)] == []


class Item(StructuredModel):
    code: str = ComparableField(comparator=ExactComparator())


class Parent(StructuredModel):
    items: list[Item]


def test_list_item_metrics_carry_no_non_matches_key():
    """Per-item metrics call `compare_with(evaluator_format=True)` without records."""
    gt = Parent(items=[Item(code="a"), Item(code="b")])
    pred = Parent(items=[Item(code="a"), Item(code="c")])
    result = gt.compare_with(pred, evaluator_format=True, document_non_matches=True)

    assert [nm["field_path"] for nm in result["non_matches"]] == ["items[1]"]
    items = result["fields"]["items"]["items"]
    assert len(items) == 2
    assert all("non_matches" not in item for item in items)


def test_records_present_when_requested():
    result = GT.compare_with(WRONG, evaluator_format=True, document_non_matches=True)

    assert [nm["field_path"] for nm in result["non_matches"]] == ["b"]


def test_requested_and_none_found_is_an_empty_list():
    result = GT.compare_with(GT, evaluator_format=True, document_non_matches=True)

    assert type(result["non_matches"]) is list
    assert result["non_matches"] == []


def test_printer_says_unrequested_evaluator_records_were_not_collected(capsys):
    """The #298/#411 rule now holds for evaluator-format output too (#413)."""
    result = GT.compare_with(
        WRONG, evaluator_format=True, include_confusion_matrix=True
    )
    for printer in (print_non_matches, print_evaluation_results):
        printer(result, use_color=False)
        out = capsys.readouterr().out

        assert "No non-matches found" not in out
        assert "No non-match records in these results." in out


def test_printer_says_a_default_dict_without_records_was_not_collected(capsys):
    """The #298/#411 rule still holds for the default `compare_with()` dict."""
    print_non_matches(GT.compare_with(WRONG), use_color=False)
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
