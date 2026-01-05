import json
import socket
import pytest
from unittest.mock import patch, MagicMock

from botocore.exceptions import NoCredentialsError, ClientError
from stickler.comparators import BaseComparator, LLMComparator


class TestLLMComparator(unittest.TestCase):
    """
    Test cases for the LLMComparator class used for comparing values using LLM models.
    """

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the Agent class
        self.agent_patcher = patch("stickler.comparators.llm.Agent")
        self.mock_agent_class = self.agent_patcher.start()
        self.mock_agent = MagicMock()
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
        mock_result.message = {
            'content': [{'text': content_text}]
        }
        self.mock_agent.return_value = mock_result

    @pytest.mark.skip(reason="Not implemented yet")
    def test_init(self):
        """Test the initialization of the LLMComparator."""
        comparator = LLMComparator(model_name="test-model", temperature=0.5)
        assert comparator.model_name == "test-model"
        assert comparator.temperature == 0.5
        assert comparator.client is None

    @pytest.mark.skip(reason="Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_init_with_client(self, mock_bedrock):
        """Test initialization with a client."""
        mock_client = MagicMock()
        comparator = LLMComparator(model_name="test-model", client=mock_client)
        assert comparator.client == mock_client
        mock_bedrock.assert_not_called()

    @pytest.mark.skip(reason="Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_client_initialization(self, mock_bedrock):
        """Test client initialization when no client is provided."""
        mock_client = MagicMock()
        mock_bedrock.return_value = mock_client

        comparator = LLMComparator(model_name="test-model")
        # Access the client property to trigger initialization
        client = comparator.client

        mock_bedrock.assert_called_once()
        assert client == mock_client

    @pytest.mark.skip(reason="Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_compare_values_equal(self, mock_bedrock):
        """Test comparison of values that are considered equal by the LLM."""
        # Setup mock response for equal values
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.read.return_value = json.dumps(
            {
                "completion": "After comparing the values, they are semantically equivalent."
            }
        ).encode()
        mock_client.invoke_model.return_value = mock_response
        mock_bedrock.return_value = mock_client

        comparator = LLMComparator(model_name="test-model")
        result = comparator.compare("value1", "value2")

        # Verify the compare method returns True for equal values
        assert result is True
        mock_client.invoke_model.assert_called_once()

    @pytest.mark.skip(reason="Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_compare_values_not_equal(self, mock_bedrock):
        """Test comparison of values that are not considered equal by the LLM."""
        # Setup mock response for unequal values
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.read.return_value = json.dumps(
            {
                "completion": "After comparing the values, they are not semantically equivalent."
            }
        ).encode()
        mock_client.invoke_model.return_value = mock_response
        mock_bedrock.return_value = mock_client

        comparator = LLMComparator(model_name="test-model")
        result = comparator.compare("value1", "completely different value")

        # Verify the compare method returns False for unequal values
        assert result is False
        mock_client.invoke_model.assert_called_once()

    @pytest.mark.skip(reason="Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_compare_with_special_values(self, mock_bedrock):
        """Test comparison with special values like None and empty strings."""
        mock_client = MagicMock()
        mock_bedrock.return_value = mock_client

        comparator = LLMComparator(model_name="test-model")

        # Compare None with None (should be equal without calling the LLM)
        assert comparator.compare(None, None) is True
        mock_client.invoke_model.assert_not_called()

        # Compare empty string with None (should not be equal without calling the LLM)
        assert comparator.compare("", None) is False
        mock_client.invoke_model.assert_not_called()

        # Compare None with a value (should not be equal without calling the LLM)
        assert comparator.compare(None, "value") is False
        mock_client.invoke_model.assert_not_called()

        # Reset mock for next test
        mock_client.reset_mock()

        # Setup mock response for comparing empty strings
        mock_response = MagicMock()
        mock_response.body.read.return_value = json.dumps(
            {
                "completion": "After comparing the values, they are semantically equivalent."
            }
        ).encode()
        mock_client.invoke_model.return_value = mock_response

        # Compare empty strings (should call LLM)
        assert comparator.compare("", "") is True
        mock_client.invoke_model.assert_called_once()

    @pytest.mark.skip(reason="Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_compare_with_custom_prompt(self, mock_bedrock):
        """Test comparison with a custom prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.read.return_value = json.dumps(
            {
                "completion": "After comparing the values, they are semantically equivalent."
            }
        ).encode()
        mock_client.invoke_model.return_value = mock_response
        mock_bedrock.return_value = mock_client

        custom_prompt = "Custom prompt {value1} vs {value2}"
        comparator = LLMComparator(
            model_name="test-model", prompt_template=custom_prompt
        )

    def test_inheritance(self):
        """Test that LLMComparator inherits from BaseComparator."""
        assert isinstance(self.comparator, BaseComparator)

    @pytest.mark.skip(reason="Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_compare_exception_handling(self, mock_bedrock):
        """Test exception handling during comparison."""
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("API Error")
        mock_bedrock.return_value = mock_client

    def test_no_match(self):
        """Test that non-matching values return 0.0."""
        self._mock_agent_response("false")
        
        result = self.comparator.compare("test", "completely different")
        assert result == 0.0

    def test_case_variations(self):
        """Test different case variations of true/false responses."""
        # Test true variations
        true_cases = ["TRUE", "True", "true", " true ", "  TRUE  "]
        for response in true_cases:
            self._mock_agent_response(response)
            result = self.comparator.compare("value1", "value2")
            assert result == 1.0, f"Failed for response: {response}"

        # Test false variations  
        false_cases = ["FALSE", "False", "false", " false ", "  FALSE  "]
        for response in false_cases:
            self._mock_agent_response(response)
            result = self.comparator.compare("value1", "value2")
            assert result == 0.0, f"Failed for response: {response}"

    def test_ambiguous_response(self):
        """Test that ambiguous responses default to 0.0."""
        ambiguous_responses = [
            "maybe", "I don't know", "uncertain", 
            "both are valid", "", "neither"
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
        call_args = self.mock_agent.call_args[0][0]  # First positional argument (prompt)
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
            model="custom-model",
            eval_guidelines=custom_guidelines
        )
        assert comparator.model == "custom-model"
        assert comparator.eval_guidelines == custom_guidelines
        assert comparator.threshold == 0.7  # BaseComparator default

    def test_default_initialization(self):
        """Test default initialization parameters."""
        comparator = LLMComparator(
            model="us.anthropic.claude-3-haiku-20240307-v1:0"
        )
        assert comparator.model == "us.anthropic.claude-3-haiku-20240307-v1:0"
        assert comparator.eval_guidelines is None
        assert comparator.threshold == 0.7  # BaseComparator default

    def test_agent_exception_handling(self):
        """Test that Agent exceptions are handled gracefully."""
        self.mock_agent.side_effect = Exception("Agent Error")
        
        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_agent_response_format_error(self):
        """Test handling of unexpected agent response format."""
        # Mock agent response with missing expected structure
        mock_result = MagicMock()
        mock_result.message = {"unexpected_field": "value"}
        self.mock_agent.return_value = mock_result
        
        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_agent_initialization(self):
        """Test that Agent is properly initialized."""
        # Verify Agent was called with correct parameters
        self.mock_agent_class.assert_called_once_with(
            model="us.anthropic.claude-3-haiku-20240307-v1:0",
            system_prompt="You are a helpful assistant that compares two values and determines if they are equivalent. Only return one word: 'true' or 'false'.",
            callback_handler=None
        )

    def test_prompt_template_with_guidelines(self):
        """Test that eval_guidelines are included in prompt when provided."""
        self._mock_agent_response("true")
        
        comparator_with_guidelines = LLMComparator(
            model="test-model",
            eval_guidelines="Use strict comparison rules"
        )
        
        result = comparator_with_guidelines.compare("value1", "value2")
        assert result == 1.0
        
        # Check that guidelines were included in the prompt
        call_args = self.mock_agent.call_args[0][0]
        assert "Use strict comparison rules" in call_args
        assert "<guidelines>" in call_args

    def test_prompt_template_without_guidelines(self):
        """Test that prompt works correctly without eval_guidelines."""
        self._mock_agent_response("false")
        
        comparator_no_guidelines = LLMComparator(
            model="test-model",
            eval_guidelines=None
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
        assert details["comparison_result"] == False

    def test_string_representation(self):
        """Test string representations for serialization."""
        assert str(self.comparator) == "LLMComparator"
        assert "LLMComparator" in repr(self.comparator)
        assert "threshold" in repr(self.comparator)

    # Enhanced Error Handling Tests

    def test_no_credentials_error_handling(self):
        """Test handling of AWS NoCredentialsError."""
        self.mock_agent.side_effect = NoCredentialsError()
        
        with pytest.raises(NoCredentialsError):
            self.comparator.compare("value1", "value2")

    def test_client_error_handling(self):
        """Test handling of AWS ClientError."""
        error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Invalid model'}}
        client_error = ClientError(error_response, 'InvokeModel')
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

    def test_error_recovery_after_exception(self):
        """Test that comparator recovers properly after an exception."""
        # First call raises exception
        self.mock_agent.side_effect = Exception("Temporary error")
        
        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")
        
        # Reset mock and verify subsequent calls work
        self.mock_agent.side_effect = None
        self._mock_agent_response("true")
        
        result = self.comparator.compare("value3", "value4")
        assert result == 1.0

    def test_get_comparison_details_comprehensive_error_handling(self):
        """Test comprehensive error handling in get_comparison_details."""
        # Test NoCredentialsError
        self.mock_agent.side_effect = NoCredentialsError()
        details = self.comparator.get_comparison_details("value1", "value2")
        assert "error" in details
        assert details["comparison_result"] == False
        
        # Test ClientError
        error_response = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}}
        self.mock_agent.side_effect = ClientError(error_response, 'InvokeModel')
        details = self.comparator.get_comparison_details("value1", "value2")
        assert "error" in details
        assert details["comparison_result"] == False
        
        # Test generic exception
        self.mock_agent.side_effect = Exception("Generic error")
        details = self.comparator.get_comparison_details("value1", "value2")
        assert "error" in details
        assert details["comparison_result"] == False

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
        error_response = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate limit exceeded'}}
        throttling_error = ClientError(error_response, 'InvokeModel')
        self.mock_agent.side_effect = throttling_error
        
        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")

    def test_service_unavailable_simulation(self):
        """Test handling of service unavailable errors."""
        error_response = {'Error': {'Code': 'ServiceUnavailableException', 'Message': 'Service temporarily unavailable'}}
        service_error = ClientError(error_response, 'InvokeModel')
        self.mock_agent.side_effect = service_error
        
        with pytest.raises(Exception):
            self.comparator.compare("value1", "value2")
