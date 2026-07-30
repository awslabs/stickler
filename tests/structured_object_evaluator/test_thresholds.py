"""Tests for is_above_threshold's floating-point boundary handling.

Pins the epsilon-tolerance invariant (abs(score - threshold) < 1e-10):
a score that is mathematically equal to the threshold but differs by
floating-point rounding error must still count as a match.
"""

from stickler.structured_object_evaluator.models.thresholds import is_above_threshold


def test_score_above_threshold():
    assert is_above_threshold(0.9, 0.7) is True


def test_score_below_threshold():
    assert is_above_threshold(0.5, 0.7) is False


def test_score_exactly_equal_to_threshold():
    assert is_above_threshold(0.7, 0.7) is True


def test_score_within_floating_point_epsilon_of_threshold():
    # 0.1 + 0.2 != 0.3 exactly in binary floating point; this is the classic
    # case the epsilon tolerance exists to handle.
    score = 0.1 + 0.2
    threshold = 0.3
    assert score != threshold  # confirms this is actually testing the edge case
    assert is_above_threshold(score, threshold) is True


def test_score_meaningfully_below_threshold_is_not_rescued_by_epsilon():
    assert is_above_threshold(0.6999, 0.7) is False
