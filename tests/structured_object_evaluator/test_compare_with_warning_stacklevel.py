"""Verify that warnings from compare_with point at the caller's line (#541)."""

import inspect
import warnings
from typing import Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.utils import deprecation


@pytest.fixture(autouse=True)
def _fresh_sentinels(monkeypatch):
    """Clear warn_once memoisation so warnings always trigger."""
    monkeypatch.setattr(deprecation, "_warned", set())


class SimpleDoc(StructuredModel):
    name: Optional[str] = ComparableField()


def test_confidence_metrics_warnings_point_at_caller_line():
    """Both UserWarning and DeprecationWarning should point at compare_with caller."""
    gt = SimpleDoc(name="test")
    pred = SimpleDoc.from_json({"name": {"_value": "test", "_confidence": 0.9}})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1
        gt.compare_with(pred, add_confidence_metrics=True)

    confidence_user_warnings = [
        w for w in caught if "Single-document confidence metrics" in str(w.message)
    ]
    assert len(confidence_user_warnings) == 1
    assert confidence_user_warnings[0].category is UserWarning
    assert confidence_user_warnings[0].filename == __file__
    assert confidence_user_warnings[0].lineno == expected_line

    deprecation_warnings = [
        w
        for w in caught
        if "auroc_confidence_metric' result key is deprecated" in str(w.message)
    ]
    assert len(deprecation_warnings) == 1
    assert deprecation_warnings[0].category is DeprecationWarning
    assert deprecation_warnings[0].filename == __file__
    assert deprecation_warnings[0].lineno == expected_line


def test_bbox_metrics_warning_points_at_caller_line():
    """Single-document mAP UserWarning should point at compare_with caller."""
    gt = SimpleDoc(name="test")
    pred = SimpleDoc(name="test")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1
        gt.compare_with(pred, add_bbox_metrics=True)

    bbox_user_warnings = [
        w for w in caught if "Single-document mAP metrics" in str(w.message)
    ]
    assert len(bbox_user_warnings) == 1
    assert bbox_user_warnings[0].category is UserWarning
    assert bbox_user_warnings[0].filename == __file__
    assert bbox_user_warnings[0].lineno == expected_line


def test_bbox_metrics_evaluator_format_warning_points_at_caller_line():
    """evaluator_format drop warning should point at compare_with caller."""
    gt = SimpleDoc(name="test")
    pred = SimpleDoc(name="test")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1
        gt.compare_with(pred, add_bbox_metrics=True, evaluator_format=True)

    evaluator_warnings = [
        w
        for w in caught
        if "add_bbox_metrics has no effect with evaluator_format=True" in str(w.message)
    ]
    assert len(evaluator_warnings) == 1
    assert evaluator_warnings[0].category is UserWarning
    assert evaluator_warnings[0].filename == __file__
    assert evaluator_warnings[0].lineno == expected_line
