"""The HTML report must not announce success over a document it has no records for.

Every other test in this directory mocks `DataExtractor.extract_non_matches`, so none of
them can see the defect in #411: the conflation happened at the boundary between a real
`compare_with()` dict and the extractor, and a mocked return value replaces exactly that
boundary. These tests drive the real objects end to end.
"""

from typing import Optional

from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.section_generator import SectionGenerator
from stickler.reporting.html.utils.data_extractors import DataExtractor
from stickler.reporting.html.visualization_engine import VisualizationEngine
from stickler.structured_object_evaluator import StructuredModel
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
    aggregate_from_comparisons,
)
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.utils.pretty_print import print_non_matches


class _Person(StructuredModel):
    name: Optional[str] = ComparableField(default=None)
    city: Optional[str] = ComparableField(default=None)


def _section(results) -> str:
    return SectionGenerator(results, VisualizationEngine()).generate_non_matches(
        ReportConfig()
    )


def _wrong_pair():
    """Ground truth and a prediction whose `name` scores 0.0."""
    return _Person(name="Ada", city="London"), _Person(name="Grace", city="London")


def test_a_default_comparison_dict_is_not_reported_as_clean():
    """The #411 reproduction: no `non_matches` key, one field at 0.0."""
    gt, pred = _wrong_pair()
    results = gt.compare_with(pred, include_confusion_matrix=True)

    assert "non_matches" not in results
    assert results["field_scores"]["name"] == 0.0

    html = _section(results)

    assert "No non-matches found" not in html
    assert "document_non_matches=True" in html


def test_a_comparison_dict_with_records_lists_them():
    """The records path still works when they were requested."""
    gt, pred = _wrong_pair()
    results = gt.compare_with(
        pred, include_confusion_matrix=True, document_non_matches=True
    )

    assert len(results["non_matches"]) == 1

    html = _section(results)

    assert "Found 1 non-matches." in html
    assert "name" in html


def test_a_clean_document_with_records_requested_is_a_success():
    """An empty record list is the one case that may claim success."""
    same = _Person(name="Ada", city="London")
    results = same.compare_with(
        same, include_confusion_matrix=True, document_non_matches=True
    )

    assert results["non_matches"] == []

    assert "<p>No non-matches found.</p>" in _section(results)


def test_a_bulk_result_without_records_is_not_reported_as_clean():
    """`ProcessEvaluation.non_matches` is None when the evaluator skipped records."""
    gt, pred = _wrong_pair()
    evaluator = BulkStructuredModelEvaluator(
        target_schema=_Person, document_non_matches=False
    )
    evaluator.update(gt, pred)
    results = evaluator.compute()

    assert results.non_matches is None
    assert DataExtractor.extract_non_matches(results) is None

    html = _section(results)

    assert "No non-matches found" not in html
    assert "document_non_matches=True" in html


def test_a_bulk_result_with_records_lists_them():
    """The bulk counterpart of the records path."""
    gt, pred = _wrong_pair()
    evaluator = BulkStructuredModelEvaluator(target_schema=_Person)
    evaluator.update(gt, pred)
    results = evaluator.compute()

    assert results.non_matches

    assert "No non-matches found" not in _section(results)
    assert "document_non_matches=True" not in _section(results)


def test_aggregating_results_without_records_is_not_reported_as_clean(capsys):
    """#425: a missing `non_matches` key was read as "no non-matches"."""
    gt, pred = _wrong_pair()
    results = aggregate_from_comparisons(
        [gt.compare_with(pred, include_confusion_matrix=True)]
    )

    assert results.non_matches is None
    assert "No non-matches found" not in _section(results)
    print_non_matches(results)
    assert "No non-match records in these results." in capsys.readouterr().out


def test_one_result_without_records_is_enough_to_withhold_success():
    """A clean result with records says nothing about the one without them."""
    same = _Person(name="Ada", city="London")
    gt, pred = _wrong_pair()
    clean = same.compare_with(
        same, include_confusion_matrix=True, document_non_matches=True
    )
    missing = gt.compare_with(pred, include_confusion_matrix=True)

    assert aggregate_from_comparisons([clean, missing]).non_matches is None
    assert aggregate_from_comparisons([missing, clean]).non_matches is None


def test_collected_records_survive_a_result_without_them():
    gt, pred = _wrong_pair()
    recorded = gt.compare_with(
        pred, include_confusion_matrix=True, document_non_matches=True
    )
    missing = gt.compare_with(pred, include_confusion_matrix=True)

    results = aggregate_from_comparisons([missing, recorded])

    assert [nm["field_path"] for nm in results.non_matches] == ["name"]


def test_aggregating_clean_results_with_records_is_a_success():
    same = _Person(name="Ada", city="London")
    clean = same.compare_with(
        same, include_confusion_matrix=True, document_non_matches=True
    )

    results = aggregate_from_comparisons([clean])

    assert results.non_matches == []
    assert "<p>No non-matches found.</p>" in _section(results)


def test_reset_forgets_a_result_without_records():
    same = _Person(name="Ada", city="London")
    evaluator = BulkStructuredModelEvaluator(target_schema=_Person)
    evaluator.update_from_comparison_result(
        same.compare_with(same, include_confusion_matrix=True)
    )
    assert evaluator.compute().non_matches is None

    evaluator.reset()
    evaluator.update(same, same)

    assert evaluator.compute().non_matches == []


def test_a_bulk_run_where_every_document_raised_is_not_reported_as_clean(capsys):
    """#425: documents that raised have no records because they were not compared."""
    evaluator = BulkStructuredModelEvaluator(target_schema=_Person)
    evaluator.update(None, None, "doc-1")
    evaluator.update(None, None, "doc-2")
    results = evaluator.compute()

    assert len(results.errors) == 2
    assert results.non_matches == []

    html = _section(results)
    assert "No non-matches found" not in html
    assert "2 could not be compared" in html

    print_non_matches(results)
    out = capsys.readouterr().out
    assert "No non-matches found" not in out
    assert "2 could not be compared" in out


class _FailingAccumulator:
    name = "failing"

    def reset(self):
        pass

    def accumulate(self, comparison_result, prediction_raw):
        raise RuntimeError("accumulator failed")

    def compute(self):
        return None


def test_a_failed_accumulator_does_not_withhold_success(capsys):
    """Its document was compared, and the records say it was clean."""
    same = _Person(name="Ada", city="London")
    evaluator = BulkStructuredModelEvaluator(
        target_schema=_Person, accumulators=[_FailingAccumulator()]
    )
    evaluator.update(same, same)
    results = evaluator.compute()

    assert len(results.errors) == 1
    assert "<p>No non-matches found.</p>" in _section(results)
    print_non_matches(results)
    assert "No non-matches found." in capsys.readouterr().out


def test_an_evaluator_format_dict_without_records_reads_as_1_0_0_did():
    """Known limit until 1.1.0: its `[]` cannot say "not collected" (#413).

    1.0.1 keeps 1.0.0's plain `"non_matches": []` so 1.0.0 callers keep
    working, and the report reads it as 1.0.0 did. 1.1.0 omits the key, and
    the report then says the records were not collected.
    """
    gt, pred = _wrong_pair()
    results = gt.compare_with(pred, evaluator_format=True)

    assert type(results["non_matches"]) is list
    assert results["non_matches"] == []
    assert DataExtractor.extract_non_matches(results) == []

    assert "<p>No non-matches found.</p>" in _section(results)


def test_an_evaluator_format_dict_with_empty_records_is_a_success():
    """Requested and none produced: the evaluator-format `[]` still means clean."""
    same = _Person(name="Ada", city="London")
    results = same.compare_with(same, evaluator_format=True, document_non_matches=True)

    assert DataExtractor.extract_non_matches(results) == []

    assert "<p>No non-matches found.</p>" in _section(results)


def _report_page(results, tmp_path) -> str:
    from stickler.reporting.html.html_reporter import EvaluationHTMLReporter

    path = tmp_path / "report.html"
    EvaluationHTMLReporter().generate_report(results, str(path))
    return path.read_text(encoding="utf-8")


def test_the_full_report_says_a_default_dict_without_records_was_not_collected(
    tmp_path,
):
    """The #411 rule still holds for the default `compare_with()` dict."""
    same = _Person(name="Ada", city="London")
    page = _report_page(
        same.compare_with(same, include_confusion_matrix=True), tmp_path
    )

    assert "No non-matches found" not in page
    assert "No non-match records in these results." in page


def test_the_full_report_says_a_collected_empty_list_is_clean(tmp_path):
    same = _Person(name="Ada", city="London")
    results = same.compare_with(same, evaluator_format=True, document_non_matches=True)

    page = _report_page(results, tmp_path)

    assert "<p>No non-matches found.</p>" in page
    assert "No non-match records in these results." not in page
