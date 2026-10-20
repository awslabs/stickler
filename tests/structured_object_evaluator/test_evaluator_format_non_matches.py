"""`evaluator_format=True` carries `non_matches` only when records were requested.

It always wrote `"non_matches": []`, so "not collected" read as "no
non-matches": `print_non_matches` printed a success message over a document
with a wrong field (#413). The default `compare_with()` dict only has the key
with `document_non_matches=True`; the evaluator format now does the same. The
HTML side of this lives in `tests/reporting/html/test_html_non_match_records.py`.
"""

from typing import Optional

from stickler import ComparableField, StructuredModel
from stickler.comparators import ExactComparator
from stickler.structured_object_evaluator.utils.pretty_print import print_non_matches


class Doc(StructuredModel):
    a: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)
    b: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)


GT = Doc(a="x", b="y")
WRONG = Doc(a="x", b="z")


def test_key_absent_unless_requested():
    result = GT.compare_with(WRONG, evaluator_format=True)

    assert "non_matches" not in result
    assert "non_matches" not in GT.compare_with(WRONG)


def test_records_present_when_requested():
    result = GT.compare_with(WRONG, evaluator_format=True, document_non_matches=True)

    assert [nm["field_path"] for nm in result["non_matches"]] == ["b"]


def test_requested_and_none_found_is_an_empty_list():
    result = GT.compare_with(GT, evaluator_format=True, document_non_matches=True)

    assert result["non_matches"] == []


def test_printer_says_records_are_missing_not_that_all_matched(capsys):
    print_non_matches(GT.compare_with(WRONG, evaluator_format=True), use_color=False)
    out = capsys.readouterr().out

    assert "No non-matches found" not in out
    assert "document_non_matches=True" in out
    # No EvalResult has an evaluator-format .raw, so that hint does not apply.
    assert "EvalResult" not in out


def test_printer_lists_requested_records(capsys):
    print_non_matches(
        GT.compare_with(WRONG, evaluator_format=True, document_non_matches=True),
        use_color=False,
    )

    assert '📍 b: "y" → "z"' in capsys.readouterr().out
