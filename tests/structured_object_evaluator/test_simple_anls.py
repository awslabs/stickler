"""
Tests for the basic ANLS score functionality in structured_object_evaluator.

These tests focus on the core functionality of the anls_score function,
without the more complex structured model tests.
"""

from pytest import approx

from stickler.structured_object_evaluator.utils.anls_score import anls_score


# Basic tests
def test_anls_same_value():
    """Test that identical values get a perfect score."""
    assert anls_score("hello", "hello") == approx(1.0)


def test_anls_different_values():
    """Test that different values get a lower score."""
    assert anls_score("hello", "world") < 1.0


def test_anls_with_return_gt():
    """Test returning the ground truth value."""
    score, gt = anls_score("hello", "hello", return_gt=True)
    assert score == approx(1.0)
    assert gt == "hello"


def test_anls_with_return_key_scores():
    """Test returning key scores."""
    score, gt, key_scores = anls_score(
        "hello", "hello", return_gt=True, return_key_scores=True
    )
    assert score == approx(1.0)
    assert gt == "hello"
    assert isinstance(key_scores, dict)


def test_anls_score_case_insensitive():
    """Test that case differences are handled well."""
    score = anls_score("Hello", "hello")
    assert score > 0.5  # Not perfect but should be high
