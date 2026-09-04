"""Regression coverage for pretty-printing the public ``EvalResult`` API."""

import pytest

from stickler.auto import EvalResult
from stickler.structured_object_evaluator.utils.pretty_print import (
    print_confusion_matrix,
    print_evaluation_results,
    print_non_matches,
)


@pytest.fixture
def evaluation_result() -> EvalResult:
    """Return an EvalResult containing one failed field comparison."""
    raw = {
        "overall_score": 0.0,
        "field_scores": {"name": 0.0},
        "all_fields_matched": False,
        "confusion_matrix": {
            "overall": {
                "tp": 0,
                "fp": 1,
                "tn": 0,
                "fn": 1,
                "fd": 0,
                "fa": 0,
                "derived": {
                    "cm_precision": 0.0,
                    "cm_recall": 0.0,
                    "cm_f1": 0.0,
                    "cm_accuracy": 0.0,
                },
            },
            "fields": {},
        },
        "non_matches": [
            {
                "field_path": "name",
                "type": "mismatch",
                "ground_truth": "Ada",
                "prediction": "Grace",
            }
        ],
    }
    return EvalResult(raw, spec=None)


def test_print_confusion_matrix_accepts_eval_result(evaluation_result, capsys):
    """The public confusion-matrix printer unwraps EvalResult.raw."""
    print_confusion_matrix(evaluation_result, use_color=False)

    assert "CONFUSION MATRIX SUMMARY" in capsys.readouterr().out


def test_print_non_matches_accepts_eval_result(evaluation_result, capsys):
    """The public non-match printer unwraps EvalResult.raw."""
    print_non_matches(evaluation_result, use_color=False)

    assert "name" in capsys.readouterr().out


def test_print_evaluation_results_accepts_eval_result(evaluation_result, capsys):
    """The combined printer reports the actual failed comparison."""
    print_evaluation_results(evaluation_result, use_color=False)

    output = capsys.readouterr().out
    assert "CONFUSION MATRIX SUMMARY" in output
    assert "name" in output
    assert "No non-matches found" not in output
