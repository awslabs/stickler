"""Tests for LLMComparator."""

import json
import unittest
from unittest.mock import patch, MagicMock

from stickler.comparators import BaseComparator, LLMComparator


class TestLLMComparator(unittest.TestCase):
    """Test the LLMComparator implementation."""

    def setUp(self):
        """Set up test environment."""
        self.boto_patcher = patch("boto3.Session")
        self.mock_session = self.boto_patcher.start()
        
        # Configure mock client
        self.mock_client = MagicMock()
        self.mock_session.return_value.client.return_value = self.mock_client
        
        # Create test comparator
        self.comparator = LLMComparator(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name="us-east-1"
        )

    def tearDown(self):
        """Clean up after tests."""
        self.boto_patcher.stop()

    def _mock_bedrock_response(self, content_text):
        """Helper to mock Bedrock API response."""
        mock_response = {
            'body': MagicMock()
        }
        response_body = {
            'content': [{'text': content_text}]
        }
        mock_response['body'].read.return_value = json.dumps(response_body)
        self.mock_client.invoke_model.return_value = mock_response

    def test_inheritance(self):
        """Test that LLMComparator inherits from BaseComparator."""
        self.assertIsInstance(self.comparator, BaseComparator)

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        self._mock_bedrock_response("true")
        
        result = self.comparator.compare("test", "test")
        self.assertEqual(result, 1.0)
        
        # Test __call__ interface
        result = self.comparator("test", "test") 
        self.assertEqual(result, 1.0)

    def test_no_match(self):
        """Test that non-matching values return 0.0."""
        self._mock_bedrock_response("false")
        
        result = self.comparator.compare("test", "completely different")
        self.assertEqual(result, 0.0)

    def test_case_variations(self):
        """Test different case variations of true/false responses."""
        # Test true variations
        true_cases = ["TRUE", "True", "true", " true ", "  TRUE  "]
        for response in true_cases:
            with self.subTest(response=response):
                self._mock_bedrock_response(response)
                result = self.comparator.compare("value1", "value2")
                self.assertEqual(result, 1.0)

        # Test false variations  
        false_cases = ["FALSE", "False", "false", " false ", "  FALSE  "]
        for response in false_cases:
            with self.subTest(response=response):
                self._mock_bedrock_response(response)
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
                self._mock_bedrock_response(response)
                result = self.comparator.compare("value1", "value2")
                self.assertEqual(result, 0.0)

    def test_none_values(self):
        """Test that None values are handled properly."""
        # Both None should return 1.0 without calling API
        result = self.comparator.compare(None, None)
        self.assertEqual(result, 1.0)
        self.mock_client.invoke_model.assert_not_called()

        # Reset mock for next tests
        self.mock_client.reset_mock()

        # None vs value should return 0.0 without calling API
        result = self.comparator.compare(None, "test")
        self.assertEqual(result, 0.0)
        self.mock_client.invoke_model.assert_not_called()

        result = self.comparator.compare("test", None)
        self.assertEqual(result, 0.0)
        self.mock_client.invoke_model.assert_not_called()

    def test_empty_strings(self):
        """Test that empty strings are handled properly."""
        self._mock_bedrock_response("true")
        
        result = self.comparator.compare("", "")
        self.assertEqual(result, 1.0)
        
        # Should call the API for empty strings
        self.mock_client.invoke_model.assert_called_once()

    def test_numeric_inputs(self):
        """Test that numeric inputs are converted to strings."""
        self._mock_bedrock_response("true")
        
        result = self.comparator.compare(123, 123)
        self.assertEqual(result, 1.0)
        
        # Verify the prompt contains string representations
        call_args = self.mock_client.invoke_model.call_args
        body = json.loads(call_args[1]['body'])
        prompt_content = body['messages'][0]['content']
        self.assertIn("123", prompt_content)

    def test_binary_compare(self):
        """Test binary_compare returns correct (tp, fp) tuples."""
        # Test true response with default threshold (0.7)
        self._mock_bedrock_response("true")
        result = self.comparator.binary_compare("test", "test")
        self.assertEqual(result, (1, 0))  # True positive

        # Test false response
        self._mock_bedrock_response("false")
        result = self.comparator.binary_compare("test", "different")
        self.assertEqual(result, (0, 1))  # False positive

        # Test with different threshold
        high_threshold = LLMComparator(threshold=0.9)
        self._mock_bedrock_response("true")
        result = high_threshold.binary_compare("value1", "value2")
        self.assertEqual(result, (1, 0))

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        custom_prompt = "Custom prompt: {value1} vs {value2}"
        comparator = LLMComparator(
            model_id="custom-model",
            region_name="us-west-2", 
            prompt_template=custom_prompt,
            threshold=0.8
        )
        self.assertEqual(comparator.model_id, "custom-model")
        self.assertEqual(comparator.prompt_template, custom_prompt)
        self.assertEqual(comparator.threshold, 0.8)

    def test_default_initialization(self):
        """Test default initialization parameters."""
        comparator = LLMComparator()
        self.assertEqual(comparator.model_id, "anthropic.claude-3-haiku-20240307-v1:0")
        self.assertIn("Compare these two values", comparator.prompt_template)
        self.assertEqual(comparator.threshold, 0.7)  # BaseComparator default


    def test_bedrock_exception_handling(self):
        """Test that Bedrock API exceptions are handled gracefully."""
        self.mock_client.invoke_model.side_effect = Exception("API Error")
        
        result = self.comparator.compare("value1", "value2")
        self.assertEqual(result, 0.0)

    def test_json_parse_error(self):
        """Test that JSON parsing errors are handled."""
        # Mock invalid JSON response
        mock_response = {'body': MagicMock()}
        mock_response['body'].read.return_value = b"invalid json"
        self.mock_client.invoke_model.return_value = mock_response
        
        result = self.comparator.compare("value1", "value2")
        self.assertEqual(result, 0.0)

    def test_missing_content_field(self):
        """Test handling of responses missing expected fields."""
        # Mock response without 'content' field
        mock_response = {'body': MagicMock()}
        response_body = {'some_other_field': 'value'}
        mock_response['body'].read.return_value = json.dumps(response_body)
        self.mock_client.invoke_model.return_value = mock_response
        
        result = self.comparator.compare("value1", "value2")
        self.assertEqual(result, 0.0)

    def test_model_request_format(self):
        """Test that the request to Bedrock is properly formatted."""
        self._mock_bedrock_response("true")
        
        self.comparator.compare("value1", "value2")
        
        # Verify the request format
        call_args = self.mock_client.invoke_model.call_args
        self.assertEqual(call_args[1]['modelId'], "anthropic.claude-3-haiku-20240307-v1:0")
        self.assertEqual(call_args[1]['contentType'], 'application/json')
        
        body = json.loads(call_args[1]['body'])
        self.assertEqual(body['anthropic_version'], 'bedrock-2023-05-31')
        self.assertEqual(body['max_tokens'], 100)
        self.assertIn('messages', body)
        self.assertEqual(len(body['messages']), 1)
        self.assertEqual(body['messages'][0]['role'], 'user')

    def test_string_representation(self):
        """Test string representations for serialization."""
        self.assertEqual(str(self.comparator), "LLMComparator")
        self.assertIn("LLMComparator", repr(self.comparator))
        self.assertIn("threshold", repr(self.comparator))


if __name__ == "__main__":
    unittest.main()
