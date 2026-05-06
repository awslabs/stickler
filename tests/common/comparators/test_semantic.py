"""Tests for the semantic comparator."""

import logging

from stickler.comparators.semantic import SemanticComparator


def test_model_id_remains_a_string_when_using_default_embedding():
    comparator = SemanticComparator(model_id="test-model")

    assert comparator.model_id == "test-model"
    assert isinstance(comparator.model_id, str)


def test_compare_logs_embedding_failures_before_fallback(caplog):
    def raise_embedding_failure(_value):
        raise RuntimeError("simulated embedding outage")

    comparator = SemanticComparator(embedding_function=raise_embedding_failure)

    with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
        score = comparator.compare("cat on mat", "feline on rug")

    assert score == 0.0
    assert "Semantic embedding comparison failed" in caplog.text
    assert "simulated embedding outage" in caplog.text


def test_compare_keeps_exact_match_fallback_when_embedding_fails(caplog):
    def raise_embedding_failure(_value):
        raise RuntimeError("simulated embedding outage")

    comparator = SemanticComparator(embedding_function=raise_embedding_failure)

    with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
        score = comparator.compare("same", "same")

    assert score == 1.0
    assert "Semantic embedding comparison failed" in caplog.text
