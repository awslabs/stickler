"""Tests for the semantic comparator."""

from stickler.comparators.semantic import SemanticComparator


def test_model_id_remains_a_string_when_using_default_embedding():
    comparator = SemanticComparator(model_id="test-model")

    assert comparator.model_id == "test-model"
    assert isinstance(comparator.model_id, str)
