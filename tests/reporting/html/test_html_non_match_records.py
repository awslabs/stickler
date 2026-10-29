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
)
from stickler.structured_object_evaluator.models.comparable_field import ComparableField


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


def test_an_evaluator_format_dict_without_records_is_not_reported_as_clean():
    """`evaluator_format=True` writes `non_matches` only when requested (#413)."""
    gt, pred = _wrong_pair()
    results = gt.compare_with(pred, evaluator_format=True)

    assert "non_matches" not in results
    assert DataExtractor.extract_non_matches(results) is None

    html = _section(results)

    assert "No non-matches found" not in html
    assert "document_non_matches=True" in html


def test_an_evaluator_format_dict_with_empty_records_is_a_success():
    """Requested and none produced: the evaluator-format `[]` still means clean."""
    same = _Person(name="Ada", city="London")
    results = same.compare_with(same, evaluator_format=True, document_non_matches=True)

    assert DataExtractor.extract_non_matches(results) == []

    assert "<p>No non-matches found.</p>" in _section(results)
