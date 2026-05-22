"""
Tests for LLMComparator.

Note: This test module mocks the strands-agents and botocore dependencies
to allow tests to run without these optional packages installed. The mocking
is done at module level using sys.modules before importing LLMComparator.
This logic is located in the conftest.py file in this directory.
"""

import re
import socket
from unittest.mock import MagicMock, patch

import jinja2
import pytest

from stickler.comparators import BaseComparator, LLMComparator


# Mock AWS exception classes to avoid botocore dependency in tests
class MockClientError(Exception):
    """Mock version of botocore.exceptions.ClientError for testing."""

    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        error_code = error_response.get("Error", {}).get("Code", "Unknown")
        error_message = error_response.get("Error", {}).get("Message", "Unknown error")
        super().__init__(f"An error occurred ({error_code}): {error_message}")


class MockNoCredentialsError(Exception):
    """Mock version of botocore.exceptions.NoCredentialsError for testing."""

    pass


# Use mock exceptions instead of real botocore exceptions
ClientError = MockClientError
NoCredentialsError = MockNoCredentialsError


@patch("stickler.comparators.llm.STRANDS_AVAILABLE", True)
class TestLLMComparator:
    """
    Test cases for the LLMComparator class used for comparing values using LLM models.
    """

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        with patch("stickler.comparators.llm.STRANDS_AVAILABLE", True):
            # Mock the Agent class
            self.agent_patcher = patch("stickler.comparators.llm.Agent")
            self.mock_agent_class = self.agent_patcher.start()
            self.mock_agent = MagicMock()
            # Pinning return_value before construction means self.comparator.agent
            # and self.mock_agent are the same object — reprogramming
            # self.mock_agent.side_effect/return_value reprograms the comparator's
            # agent. Tests that construct a fresh LLMComparator share the same
            # mock instance for the same reason.
            self.mock_agent_class.return_value = self.mock_agent

            # Create comparator instance
            self.comparator = LLMComparator(
                model="us.anthropic.claude-3-haiku-20240307-v1:0"
            )

        yield

        # Cleanup
        self.agent_patcher.stop()

    def _mock_agent_response(self, content_text):
        """Helper to mock Agent response."""
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": content_text}]}
        self.mock_agent.return_value = mock_result

    def test_init(self):
        """Test the initialization of the LLMComparator."""
        comparator = LLMComparator(model="test-model")
        assert comparator.model == "test-model"
        assert comparator.eval_guidelines is None
        assert comparator.threshold == 0.7
        # Pin the system prompt to its default and to the parsing invariant
        # downstream ("true"/"false" detection in compare()).
        assert comparator.system_prompt == comparator._default_system_prompt()
        assert "Only return one word" in comparator.system_prompt
        assert isinstance(comparator.prompt_template, jinja2.Template)

    def test_init_with_model_instance(self):
        """Test initialization with a strands Model object instead of a string model ID."""
        mock_model = MagicMock()
        comparator = LLMComparator(model=mock_model)
        assert comparator.model is mock_model
        assert comparator.agent is self.mock_agent_class.return_value

    def test_client_initialization(self):
        """Test that the Agent is initialized eagerly during __init__."""
        self.mock_agent_class.reset_mock()
        new_comp = LLMComparator(model="eager-init-model")
        self.mock_agent_class.assert_called_once_with(
            model="eager-init-model",
            system_prompt=new_comp.system_prompt,
            callback_handler=None,
        )
        assert new_comp.agent is self.mock_agent_class.return_value

    def test_compare_values_equal(self):
        """Test comparison of values that are considered equal by the LLM."""
        self._mock_agent_response("true")

        result = self.comparator.compare("hello world", "hello world")

        assert result == 1.0
        self.mock_agent.assert_called_once()
        prompt = self.mock_agent.call_args[0][0]
        assert "hello world" in prompt

    def test_compare_values_not_equal(self):
        """Test comparison of values that are not considered equal by the LLM."""
        self._mock_agent_response("false")

        result = self.comparator.compare("apple", "orange")

        assert result == 0.0
        self.mock_agent.assert_called_once()
        prompt = self.mock_agent.call_args[0][0]
        assert "apple" in prompt
        assert "orange" in prompt

    def test_compare_with_special_values(self):
        """Test that HTML-special characters in values are escaped in the prompt."""
        self._mock_agent_response("true")

        result = self.comparator.compare("<script>", "<script>")

        assert result == 1.0
        prompt = self.mock_agent.call_args[0][0]
        # Values are HTML-escaped before insertion into the prompt
        assert "&lt;script&gt;" in prompt
        # Scope the negative to the rendered Value 1 / Value 2 slots so that
        # future template additions (e.g. <example>, <output>, few-shot blocks)
        # don't break this test even though the escape path is still correct.
        value_lines = re.findall(r"Value [12]:.*", prompt)
        assert value_lines, "expected Value 1: / Value 2: lines in rendered prompt"
        for line in value_lines:
            assert "<script>" not in line

    def test_compare_escapes_eval_guidelines(self):
        """Test that eval_guidelines are HTML-escaped and rendered inside the
        <guidelines> block of the prompt."""
        self._mock_agent_response("true")

        guidelines = "<rule> Use strict & exact matching"
        comparator = LLMComparator(model="test-model", eval_guidelines=guidelines)
        result = comparator.compare("value1", "value2")

        assert result == 1.0
        prompt = self.mock_agent.call_args[0][0]
        # The escaped guidelines must land inside the <guidelines>...</guidelines>
        # block, not just somewhere in the prompt.
        assert re.search(
            r"<guidelines>.*?&lt;rule&gt; Use strict &amp; exact matching.*?</guidelines>",
            prompt,
            re.DOTALL,
        )

    def test_inheritance(self):
        """Test that LLMComparator inherits from BaseComparator."""
        assert isinstance(self.comparator, BaseComparator)

    def test_compare_exception_handling(self):
        """Test that NoCredentialsError raised by the agent propagates via the
        dedicated except branch, and that the comparator stays usable after."""
        # Use the module's own NoCredentialsError so we actually hit the
        # `except NoCredentialsError` branch in compare(), not the generic one.
        from stickler.comparators.llm import NoCredentialsError as LLMNoCredentialsError

        self.mock_agent.side_effect = LLMNoCredentialsError()

        with pytest.raises(LLMNoCredentialsError):
            self.comparator.compare("value1", "value2")

        # Comparator remains usable after the exception is cleared. Use a
        # 'true' → 1.0 roundtrip so a regression to always-zero would fail here.
        self.mock_agent.side_effect = None
        self._mock_agent_response("true")
        assert self.comparator.compare("value1", "value2") == 1.0

    def test_ambiguous_response(self):
        """Test that ambiguous responses default to 0.0."""
        ambiguous_responses = [
            "maybe",
            "I don't know",
            "uncertain",
            "both are valid",
            "",
            "neither",
        ]

        for response in ambiguous_responses:
            self._mock_agent_response(response)
            result = self.comparator.compare("value1", "value2")
            assert result == 0.0, f"Failed for response: {response}"

    def test_none_values(self):
        """Test that None values are handled properly."""
        # Both None should return 1.0 without calling agent
        result = self.comparator.compare(None, None)
        assert result == 1.0
        self.mock_agent.assert_not_called()

        # Reset mock for next tests
        self.mock_agent.reset_mock()

        # None vs value should return 0.0 without calling agent
        result = self.comparator.compare(None, "test")
        assert result == 0.0
        self.mock_agent.assert_not_called()

        result = self.comparator.compare("test", None)
        assert result == 0.0
        self.mock_agent.assert_not_called()

    def test_empty_strings(self):
        """Test that empty strings are handled properly."""
        self._mock_agent_response("true")

        result = self.comparator.compare("", "")
        assert result == 1.0

        # Should call the agent for empty strings
        self.mock_agent.assert_called_once()

    def test_numeric_inputs(self):
        """Test that numeric inputs are converted to strings."""
        self._mock_agent_response("true")

        result = self.comparator.compare(123, 123)
        assert result == 1.0

        # Verify the agent was called with a prompt containing string representations
        self.mock_agent.assert_called_once()
        call_args = self.mock_agent.call_args[0][
            0
        ]  # First positional argument (prompt)
        assert "123" in call_args

    def test_binary_compare(self):
        """Test binary_compare returns correct (tp, fp) tuples."""
        # Test true response with default threshold (0.7)
        self._mock_agent_response("true")
        result = self.comparator.binary_compare("test", "test")
        assert result == (1, 0)  # True positive

        # Test false response
        self._mock_agent_response("false")
        result = self.comparator.binary_compare("test", "different")
        assert result == (0, 1)  # False positive

        # Test with different threshold
        high_threshold = LLMComparator(model="test-model", eval_guidelines=None)
        high_threshold.threshold = 0.9
        self._mock_agent_response("true")
        result = high_threshold.binary_compare("value1", "value2")
        assert result == (1, 0)

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        custom_guidelines = "Custom evaluation guidelines"
        comparator = LLMComparator(
            model="custom-model", eval_guidelines=custom_guidelines
        )
        assert comparator.model == "custom-model"
        assert comparator.eval_guidelines == custom_guidelines
        assert comparator.threshold == 0.7  # BaseComparator default

    def test_agent_response_format_error(self):
        """Test handling of unexpected agent response format."""
        # Mock agent response with missing expected structure
        mock_result = MagicMock()
        mock_result.message = {"unexpected_field": "value"}
        self.mock_agent.return_value = mock_result

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_prompt_template_without_guidelines(self):
        """Test that prompt works correctly without eval_guidelines."""
        self._mock_agent_response("false")

        comparator_no_guidelines = LLMComparator(
            model="test-model", eval_guidelines=None
        )

        result = comparator_no_guidelines.compare("value1", "value2")
        assert result == 0.0

        # Check that guidelines section is not included
        call_args = self.mock_agent.call_args[0][0]
        assert "<guidelines>" not in call_args

    def test_get_comparison_details(self):
        """Test get_comparison_details method."""
        self._mock_agent_response("true")

        details = self.comparator.get_comparison_details("value1", "value2")

        assert "prompt" in details
        assert "llm_response" in details
        assert "model_id" in details
        assert "comparison_result" in details

        assert details["llm_response"] == "true"
        assert details["model_id"] == "us.anthropic.claude-3-haiku-20240307-v1:0"
        assert details["comparison_result"] == 1.0

    def test_get_comparison_details_error_handling(self):
        """Test get_comparison_details error handling."""
        self.mock_agent.side_effect = Exception("Agent Error")

        details = self.comparator.get_comparison_details("value1", "value2")

        assert "error" in details
        assert "comparison_result" in details
        assert not details["comparison_result"]

    def test_string_representation(self):
        """Test string representations for serialization."""
        assert str(self.comparator) == "LLMComparator"
        assert "LLMComparator" in repr(self.comparator)
        assert "threshold" in repr(self.comparator)

    # Enhanced Error Handling Tests

    def test_client_error_handling(self):
        """Test handling of AWS ClientError."""
        error_response = {
            "Error": {"Code": "ValidationException", "Message": "Invalid model"}
        }
        client_error = ClientError(error_response, "InvokeModel")
        self.mock_agent.side_effect = client_error

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        self.mock_agent.side_effect = socket.timeout("Connection timed out")

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        self.mock_agent.side_effect = ConnectionError("Connection failed")

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_missing_message(self):
        """Test handling of response missing 'message' key."""
        mock_result = MagicMock()
        mock_result.message = None
        self.mock_agent.return_value = mock_result

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_missing_content(self):
        """Test handling of response missing 'content' key."""
        mock_result = MagicMock()
        mock_result.message = {"no_content": "value"}
        self.mock_agent.return_value = mock_result

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_empty_content_array(self):
        """Test handling of response with empty content array."""
        mock_result = MagicMock()
        mock_result.message = {"content": []}
        self.mock_agent.return_value = mock_result

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_missing_text_key(self):
        """Test handling of response missing 'text' key in content."""
        mock_result = MagicMock()
        mock_result.message = {"content": [{"no_text": "value"}]}
        self.mock_agent.return_value = mock_result

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_get_comparison_details_comprehensive_error_handling(self):
        """Test comprehensive error handling in get_comparison_details."""
        # Test NoCredentialsError
        self.mock_agent.side_effect = NoCredentialsError()
        details = self.comparator.get_comparison_details("value1", "value2")
        assert "error" in details
        assert not details["comparison_result"]

        # Test ClientError
        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        self.mock_agent.side_effect = ClientError(error_response, "InvokeModel")
        details = self.comparator.get_comparison_details("value1", "value2")
        assert "error" in details
        assert not details["comparison_result"]

        # Test generic exception
        self.mock_agent.side_effect = Exception("Generic error")
        details = self.comparator.get_comparison_details("value1", "value2")
        assert "error" in details
        assert not details["comparison_result"]

    def test_model_initialization_error(self):
        """Test error handling during model initialization."""
        with patch("stickler.comparators.llm.Agent") as mock_agent_class:
            mock_agent_class.side_effect = Exception("Model initialization failed")

            with pytest.raises(Exception):
                LLMComparator(model="invalid-model")

    def test_none_model_initialization_error(self):
        """Test error when model is None during initialization."""
        with pytest.raises(ValueError) as context:
            LLMComparator(model=None)

        assert "Model must be provided" in str(context.value)

    def test_rate_limiting_simulation(self):
        """Test handling of rate limiting errors."""
        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate limit exceeded"}
        }
        throttling_error = ClientError(error_response, "InvokeModel")
        self.mock_agent.side_effect = throttling_error

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_service_unavailable_simulation(self):
        """Test handling of service unavailable errors."""
        error_response = {
            "Error": {
                "Code": "ServiceUnavailableException",
                "Message": "Service temporarily unavailable",
            }
        }
        service_error = ClientError(error_response, "InvokeModel")
        self.mock_agent.side_effect = service_error

        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")
