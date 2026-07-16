"""Tests for SemanticComparator."""

import logging

from stickler.comparators import SemanticComparator


class TestSemanticComparator:
    """Test semantic comparator fallback behavior."""

    def test_model_id_remains_a_string_when_using_default_embedding(self):
        """Default embedding setup stores model_id as a string."""
        comparator = SemanticComparator(model_id="test-model")

        assert comparator.model_id == "test-model"
        assert isinstance(comparator.model_id, str)

    def test_compare_logs_embedding_failures_before_fallback(self, caplog):
        """Embedding failures are logged before returning unequal fallback."""

        def raise_embedding_failure(_value):
            raise RuntimeError("simulated embedding outage")

        comparator = SemanticComparator(
            embedding_function=raise_embedding_failure, model_id="custom-model"
        )

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare("cat on mat", "feline on rug")

        record = caplog.records[0]
        assert score == 0.0
        assert "Semantic embedding comparison failed" in caplog.text
        assert "simulated embedding outage" in caplog.text
        assert record.embedding_function == "raise_embedding_failure"
        assert record.model_id == "custom-model"
        assert record.input_1_length == len("cat on mat")
        assert record.input_2_length == len("feline on rug")
        assert record.exception_type == "RuntimeError"

    def test_compare_keeps_exact_match_fallback_when_embedding_fails(self, caplog):
        """Embedding failures keep exact-match fallback semantics."""

        def raise_embedding_failure(_value):
            raise RuntimeError("simulated embedding outage")

        comparator = SemanticComparator(embedding_function=raise_embedding_failure)

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare("same", "same")

        assert score == 1.0
        assert "Semantic embedding comparison failed" in caplog.text

    def test_compare_logs_non_string_input_lengths_before_fallback(self, caplog):
        """Fallback logging does not raise for primitive non-string inputs."""

        def raise_embedding_failure(_value):
            raise RuntimeError("simulated embedding outage")

        comparator = SemanticComparator(embedding_function=raise_embedding_failure)

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare(123, 456)

        record = caplog.records[0]
        assert score == 0.0
        assert record.input_1_length == len("123")
        assert record.input_2_length == len("456")

    def test_default_bedrock_fallback_log_includes_function_and_model(
        self, caplog, monkeypatch
    ):
        """Default Bedrock partial logs the wrapped function name and model ID."""

        def raise_bedrock_failure(_value, model_id):
            raise RuntimeError(f"simulated {model_id} outage")

        monkeypatch.setattr(
            "stickler.comparators.semantic.generate_bedrock_embedding",
            raise_bedrock_failure,
        )
        comparator = SemanticComparator(model_id="amazon.test-model")

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare("cat", "dog")

        record = caplog.records[0]
        assert score == 0.0
        assert record.embedding_function == "raise_bedrock_failure"
        assert record.model_id == "amazon.test-model"
        assert record.exception_type == "RuntimeError"
