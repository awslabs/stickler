"""Tests for print_confusion_matrix, including empty-fields edge cases (issue #307)."""

import pytest

from stickler import StructuredModel
from stickler.structured_object_evaluator.utils.pretty_print import (
    print_confusion_matrix,
)


class EmptyModel(StructuredModel):
    pass


class PersonModel(StructuredModel):
    name: str
    age: int


def test_print_confusion_matrix_empty_model_does_not_raise(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty StructuredModel produces an empty fields dict and must not raise."""
    cm = EmptyModel().compare_with(EmptyModel(), include_confusion_matrix=True)
    print_confusion_matrix(cm, use_color=False)

    captured = capsys.readouterr()
    assert "CONFUSION MATRIX SUMMARY" in captured.out
    assert "FIELD-LEVEL METRICS" in captured.out


def test_print_confusion_matrix_empty_dict_does_not_raise(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A hand-built results dict with empty fields must not raise."""
    raw_cm = {
        "confusion_matrix": {
            "fields": {},
            "overall": {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "fd": 0,
                "derived": {
                    "cm_precision": 0.0,
                    "cm_recall": 0.0,
                    "cm_f1": 0.0,
                    "cm_accuracy": 0.0,
                },
            },
        }
    }
    print_confusion_matrix(raw_cm, use_color=False)

    captured = capsys.readouterr()
    assert "CONFUSION MATRIX SUMMARY" in captured.out
    assert "FIELD-LEVEL METRICS" in captured.out


def test_print_confusion_matrix_filter_matching_nothing_does_not_raise(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When field_filter matches zero fields, the empty filtered set must not raise."""
    gt = PersonModel(name="Alice", age=30)
    pred = PersonModel(name="Alice", age=30)
    cm = gt.compare_with(pred, include_confusion_matrix=True)

    print_confusion_matrix(cm, field_filter="^nomatch_.*", use_color=False)

    captured = capsys.readouterr()
    assert "FIELD-LEVEL METRICS" in captured.out


def test_print_confusion_matrix_normal_model_renders_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-empty models continue to format field rows as before."""
    gt = PersonModel(name="Alice", age=30)
    pred = PersonModel(name="Alice", age=30)
    cm = gt.compare_with(pred, include_confusion_matrix=True)

    print_confusion_matrix(cm, use_color=False)

    captured = capsys.readouterr()
    assert "FIELD-LEVEL METRICS" in captured.out
    assert "name" in captured.out
    assert "age" in captured.out
