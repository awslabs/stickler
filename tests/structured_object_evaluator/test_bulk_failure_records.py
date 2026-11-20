"""Pin the bulk evaluator's document and accumulator failure output."""

from unittest.mock import Mock

import pytest

from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


@pytest.mark.parametrize("entry_point", ["update", "update_from_comparison_result"])
@pytest.mark.parametrize("elide_errors", [False, True])
def test_verbose_document_failure(entry_point, elide_errors, capsys, monkeypatch):
    evaluator = BulkStructuredModelEvaluator(
        verbose=True, elide_errors=elide_errors, accumulators=[]
    )
    capsys.readouterr()  # Discard initialization messages.
    failure = ValueError("document comparison failed")

    if entry_point == "update":
        gt_model = Mock(spec=StructuredModel)
        gt_model.compare_with.side_effect = failure
        evaluator.update(gt_model, Mock(spec=StructuredModel), doc_id="doc-1")
    else:
        monkeypatch.setattr(
            evaluator, "_accumulate_confusion_matrix", Mock(side_effect=failure)
        )
        evaluator.update_from_comparison_result({"confusion_matrix": {}}, "doc-1")

    captured = capsys.readouterr()
    assert (
        captured.out == "Error processing document doc-1: document comparison failed\n"
    )
    assert captured.err == ""
    assert evaluator._processed_count == 0
    assert evaluator._confusion_matrix["overall"]["fn"] == (0 if elide_errors else 1)
    if elide_errors:
        assert evaluator._errors == []
    else:
        assert evaluator._errors == [
            {
                "doc_id": "doc-1",
                "error": "document comparison failed",
                "error_type": "ValueError",
            }
        ]
        assert list(evaluator._errors[0]) == ["doc_id", "error", "error_type"]


@pytest.mark.parametrize("elide_errors", [False, True])
def test_verbose_accumulator_failure(elide_errors, capsys):
    accumulator = Mock()
    accumulator.name = "broken"
    accumulator.accumulate.side_effect = RuntimeError("accumulator failed")
    evaluator = BulkStructuredModelEvaluator(
        verbose=True, elide_errors=elide_errors, accumulators=[accumulator]
    )
    capsys.readouterr()

    evaluator.update_from_comparison_result(
        {"confusion_matrix": {"overall": {"tp": 1}, "fields": {}}}, "doc-1"
    )

    captured = capsys.readouterr()
    assert captured.out == "Accumulator 'broken' failed on doc-1: accumulator failed\n"
    assert captured.err == ""
    assert evaluator._processed_count == 1
    assert evaluator._accumulator_errors == {"broken": 1}
    assert evaluator._confusion_matrix["overall"]["tp"] == 1
    assert evaluator._confusion_matrix["overall"]["fn"] == 0
    if elide_errors:
        assert evaluator._errors == []
    else:
        assert evaluator._errors == [
            {
                "doc_id": "doc-1",
                "error": "accumulator failed",
                "error_type": "RuntimeError",
                "accumulator": "broken",
            }
        ]
        assert list(evaluator._errors[0]) == [
            "doc_id",
            "error",
            "error_type",
            "accumulator",
        ]
