"""Tests for LLMComparator."""

import unittest
from unittest.mock import patch, MagicMock

from botocore.exceptions import NoCredentialsError, ClientError
from stickler.comparators import BaseComparator, LLMComparator


class TestLLMComparator(unittest.TestCase):
    """Test the LLMComparator implementation."""

    def setUp(self):
        """Set up test environment."""
        # Mock the strands Agent instead of boto3
        self.agent_patcher = patch("stickler.comparators.llm.Agent")
        self.mock_agent_class = self.agent_patcher.start()
        
        # Configure mock agent instance
        self.mock_agent = MagicMock()
        self.mock_agent_class.return_value = self.mock_agent
        
        # Create test comparator
        self.comparator = LLMComparator(
            model="us.anthropic.claude-3-haiku-20240307-v1:0",
            eval_guidelines="Test guidelines"
        )

    def tearDown(self):
        """Clean up after tests."""
        self.agent_patcher.stop()

    def _mock_agent_response(self, content_text):
        """Helper to mock Agent response."""
        mock_result = MagicMock()
        mock_result.message = {
            'content': [{'text': content_text}]
        }
        self.mock_agent.return_value = mock_result

    def test_inheritance(self):
        """Test that LLMComparator inherits from BaseComparator."""
        self.assertIsInstance(self.comparator, BaseComparator)

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        self._mock_agent_response("true")
        
        result = self.comparator.compare("test", "test")
        self.assertEqual(result, 1.0)
        
        # Test __call__ interface
        result = self.comparator("test", "test") 
        self.assertEqual(result, 1.0)

    def test_no_match(self):
        """Test that non-matching values return 0.0."""
        self._mock_agent_response("false")
        
        result = self.comparator.compare("test", "completely different")
        self.assertEqual(result, 0.0)

    def test_case_variations(self):
        """Test different case variations of true/false responses."""
        # Test true variations
        true_cases = ["TRUE", "True", "true", " true ", "  TRUE  "]
        for response in true_cases:
            with self.subTest(response=response):
                self._mock_agent_response(response)
                result = self.comparator.compare("value1", "value2")
                self.assertEqual(result, 1.0)

        # Test false variations  
        false_cases = ["FALSE", "False", "false", " false ", "  FALSE  "]
        for response in false_cases:
            with self.subTest(response=response):
                self._mock_agent_response(response)
                result = self.comparator.compare("value1", "value2")
                self.assertEqual(result, 0.0)

    def test_ambiguous_response(self):
        """Test that ambiguous responses default to 0.0."""
        ambiguous_responses = [
            "maybe", "I don't know", "uncertain", 
            "both are valid", "", "neither"
        ]
        
        for response in ambiguous_responses:
            with self.subTest(response=response):
                self._mock_agent_response(response)
                result = self.comparator.compare("value1", "value2")
                self.assertEqual(result, 0.0)

    def test_none_values(self):
        """Test that None values are handled properly."""
        # Both None should return 1.0 without calling agent
        result = self.comparator.compare(None, None)
        self.assertEqual(result, 1.0)
        self.mock_agent.assert_not_called()

        # Reset mock for next tests
        self.mock_agent.reset_mock()

        # None vs value should return 0.0 without calling agent
        result = self.comparator.compare(None, "test")
        self.assertEqual(result, 0.0)
        self.mock_agent.assert_not_called()

        result = self.comparator.compare("test", None)
        self.assertEqual(result, 0.0)
        self.mock_agent.assert_not_called()

    def test_empty_strings(self):
        """Test that empty strings are handled properly."""
        self._mock_agent_response("true")
        
        result = self.comparator.compare("", "")
        self.assertEqual(result, 1.0)
        
        # Should call the agent for empty strings
        self.mock_agent.assert_called_once()

    def test_numeric_inputs(self):
        """Test that numeric inputs are converted to strings."""
        self._mock_agent_response("true")
        
        result = self.comparator.compare(123, 123)
        self.assertEqual(result, 1.0)
        
        # Verify the agent was called with a prompt containing string representations
        self.mock_agent.assert_called_once()
        call_args = self.mock_agent.call_args[0][0]  # First positional argument (prompt)
        self.assertIn("123", call_args)

    def test_binary_compare(self):
        """Test binary_compare returns correct (tp, fp) tuples."""
        # Test true response with default threshold (0.7)
        self._mock_agent_response("true")
        result = self.comparator.binary_compare("test", "test")
        self.assertEqual(result, (1, 0))  # True positive

        # Test false response
        self._mock_agent_response("false")
        result = self.comparator.binary_compare("test", "different")
        self.assertEqual(result, (0, 1))  # False positive

        # Test with different threshold
        high_threshold = LLMComparator(model="test-model", eval_guidelines=None)
        high_threshold.threshold = 0.9
        self._mock_agent_response("true")
        result = high_threshold.binary_compare("value1", "value2")
        self.assertEqual(result, (1, 0))

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        custom_guidelines = "Custom evaluation guidelines"
        comparator = LLMComparator(
            model="custom-model",
            eval_guidelines=custom_guidelines
        )
        self.assertEqual(comparator.model, "custom-model")
        self.assertEqual(comparator.eval_guidelines, custom_guidelines)
        self.assertEqual(comparator.threshold, 0.7)  # BaseComparator default

    def test_default_initialization(self):
        """Test default initialization parameters."""
        comparator = LLMComparator(
            model="us.anthropic.claude-3-haiku-20240307-v1:0"
        )
        self.assertEqual(comparator.model, "us.anthropic.claude-3-haiku-20240307-v1:0")
        self.assertIsNone(comparator.eval_guidelines)
        self.assertEqual(comparator.threshold, 0.7)  # BaseComparator default

    def test_agent_exception_handling(self):
        """Test that Agent exceptions are handled gracefully."""
        self.mock_agent.side_effect = Exception("Agent Error")
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_agent_response_format_error(self):
        """Test handling of unexpected agent response format."""
        # Mock agent response with missing expected structure
        mock_result = MagicMock()
        mock_result.message = {"unexpected_field": "value"}
        self.mock_agent.return_value = mock_result
        
        with self.assertRaises(Exception):
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
        self.assertEqual(result, 1.0)
        
        # Check that guidelines were included in the prompt
        call_args = self.mock_agent.call_args[0][0]
        self.assertIn("Use strict comparison rules", call_args)
        self.assertIn("<guidelines>", call_args)

    def test_prompt_template_without_guidelines(self):
        """Test that prompt works correctly without eval_guidelines."""
        self._mock_agent_response("false")
        
        comparator_no_guidelines = LLMComparator(
            model="test-model",
            eval_guidelines=None
        )
        
        result = comparator_no_guidelines.compare("value1", "value2")
        self.assertEqual(result, 0.0)
        
        # Check that guidelines section is not included
        call_args = self.mock_agent.call_args[0][0]
        self.assertNotIn("<guidelines>", call_args)

    def test_get_comparison_details(self):
        """Test get_comparison_details method."""
        self._mock_agent_response("true")
        
        details = self.comparator.get_comparison_details("value1", "value2")
        
        self.assertIn("prompt", details)
        self.assertIn("llm_response", details)
        self.assertIn("model_id", details)
        self.assertIn("comparison_result", details)
        
        self.assertEqual(details["llm_response"], "true")
        self.assertEqual(details["model_id"], "us.anthropic.claude-3-haiku-20240307-v1:0")
        self.assertEqual(details["comparison_result"], 1.0)

    def test_get_comparison_details_error_handling(self):
        """Test get_comparison_details error handling."""
        self.mock_agent.side_effect = Exception("Agent Error")
        
        details = self.comparator.get_comparison_details("value1", "value2")
        
        self.assertIn("error", details)
        self.assertIn("comparison_result", details)
        self.assertEqual(details["comparison_result"], False)

    def test_string_representation(self):
        """Test string representations for serialization."""
        self.assertEqual(str(self.comparator), "LLMComparator")
        self.assertIn("LLMComparator", repr(self.comparator))
        self.assertIn("threshold", repr(self.comparator))

    # Enhanced Error Handling Tests

    def test_no_credentials_error_handling(self):
        """Test handling of AWS NoCredentialsError."""
        self.mock_agent.side_effect = NoCredentialsError()
        
        with self.assertRaises(NoCredentialsError):
            self.comparator.compare("value1", "value2")

    def test_client_error_handling(self):
        """Test handling of AWS ClientError."""
        error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Invalid model'}}
        client_error = ClientError(error_response, 'InvokeModel')
        self.mock_agent.side_effect = client_error
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        import socket
        self.mock_agent.side_effect = socket.timeout("Connection timed out")
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        import socket
        self.mock_agent.side_effect = ConnectionError("Connection failed")
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_missing_message(self):
        """Test handling of response missing 'message' key."""
        mock_result = MagicMock()
        mock_result.message = None
        self.mock_agent.return_value = mock_result
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_missing_content(self):
        """Test handling of response missing 'content' key."""
        mock_result = MagicMock()
        mock_result.message = {"no_content": "value"}
        self.mock_agent.return_value = mock_result
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_empty_content_array(self):
        """Test handling of response with empty content array."""
        mock_result = MagicMock()
        mock_result.message = {"content": []}
        self.mock_agent.return_value = mock_result
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_malformed_response_missing_text_key(self):
        """Test handling of response missing 'text' key in content."""
        mock_result = MagicMock()
        mock_result.message = {"content": [{"no_text": "value"}]}
        self.mock_agent.return_value = mock_result
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_error_recovery_after_exception(self):
        """Test that comparator recovers properly after an exception."""
        # First call raises exception
        self.mock_agent.side_effect = Exception("Temporary error")
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")
        
        # Reset mock and verify subsequent calls work
        self.mock_agent.side_effect = None
        self._mock_agent_response("true")
        
        result = self.comparator.compare("value3", "value4")
        self.assertEqual(result, 1.0)

    def test_get_comparison_details_comprehensive_error_handling(self):
        """Test comprehensive error handling in get_comparison_details."""
        # Test NoCredentialsError
        self.mock_agent.side_effect = NoCredentialsError()
        details = self.comparator.get_comparison_details("value1", "value2")
        self.assertIn("error", details)
        self.assertEqual(details["comparison_result"], False)
        
        # Test ClientError
        error_response = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}}
        self.mock_agent.side_effect = ClientError(error_response, 'InvokeModel')
        details = self.comparator.get_comparison_details("value1", "value2")
        self.assertIn("error", details)
        self.assertEqual(details["comparison_result"], False)
        
        # Test generic exception
        self.mock_agent.side_effect = Exception("Generic error")
        details = self.comparator.get_comparison_details("value1", "value2")
        self.assertIn("error", details)
        self.assertEqual(details["comparison_result"], False)

    def test_model_initialization_error(self):
        """Test error handling during model initialization."""
        with patch("stickler.comparators.llm.Agent") as mock_agent_class:
            mock_agent_class.side_effect = Exception("Model initialization failed")
            
            with self.assertRaises(Exception):
                LLMComparator(model="invalid-model")

    def test_none_model_initialization_error(self):
        """Test error when model is None during initialization."""
        with self.assertRaises(ValueError) as context:
            LLMComparator(model=None)
        
        self.assertIn("Model must be provided", str(context.exception))

    def test_rate_limiting_simulation(self):
        """Test handling of rate limiting errors."""
        error_response = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate limit exceeded'}}
        throttling_error = ClientError(error_response, 'InvokeModel')
        self.mock_agent.side_effect = throttling_error
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")

    def test_service_unavailable_simulation(self):
        """Test handling of service unavailable errors."""
        error_response = {'Error': {'Code': 'ServiceUnavailableException', 'Message': 'Service temporarily unavailable'}}
        service_error = ClientError(error_response, 'InvokeModel')
        self.mock_agent.side_effect = service_error
        
        with self.assertRaises(Exception):
            self.comparator.compare("value1", "value2")


if __name__ == "__main__":
    unittest.main()
