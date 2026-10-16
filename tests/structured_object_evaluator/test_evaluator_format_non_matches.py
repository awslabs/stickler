"""`evaluator_format=True` carries `non_matches` only when records were requested.

It always wrote `"non_matches": []`, so "not collected" read as "no
non-matches": `print_non_matches` printed a success message over a document
with a wrong field (#413). The default `compare_with()` dict only has the key
with `document_non_matches=True`; the evaluator format now does the same.
"""

from typing import Optional

from stickler import ComparableField, StructuredModel
from stickler.comparators import ExactComparator
from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.section_generator import SectionGenerator
from stickler.reporting.html.utils import DataExtractor
from stickler.structured_object_evaluator.utils.pretty_print import print_non_matches
from stickler.utils.process_evaluation import ProcessEvaluation


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


def test_printer_lists_requested_records(capsys):
    print_non_matches(
        GT.compare_with(WRONG, evaluator_format=True, document_non_matches=True),
        use_color=False,
    )

    assert "b" in capsys.readouterr().out


def _html_non_matches(results):
    return SectionGenerator(results, None).generate_non_matches(ReportConfig())


def test_html_report_says_records_were_not_collected():
    results = GT.compare_with(WRONG, evaluator_format=True)

    assert not DataExtractor.has_non_match_records(results)
    html = _html_non_matches(results)
    assert "No non-matches found." not in html
    assert "not collected" in html


def test_html_report_empty_records_still_reads_none_found():
    results = GT.compare_with(GT, evaluator_format=True, document_non_matches=True)

    assert DataExtractor.has_non_match_records(results)
    assert "No non-matches found." in _html_non_matches(results)


def test_html_report_bulk_without_records_says_not_collected():
    no_records = ProcessEvaluation(document_count=1, metrics={}, field_metrics={})
    empty = ProcessEvaluation(
        document_count=1, metrics={}, field_metrics={}, non_matches=[]
    )

    assert "not collected" in _html_non_matches(no_records)
    assert "No non-matches found." in _html_non_matches(empty)
