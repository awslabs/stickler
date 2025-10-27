import json
from unittest.mock import patch, MagicMock
from unittest import skip

from stickler.comparators.llm import LLMComparator


class TestLLMComparator:
    """
    Test cases for the LLMComparator class used for comparing values using LLM models.
    """

    @skip("Not implemented yet")
    def test_init(self):
        """Test the initialization of the LLMComparator."""
        comparator = LLMComparator(model_name="test-model", temperature=0.5)
        assert comparator.model_name == "test-model"
        assert comparator.temperature == 0.5
        assert comparator.client is None

    @skip("Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_init_with_client(self, mock_bedrock):
        """Test initialization with a client."""
        mock_client = MagicMock()
        comparator = LLMComparator(model_name="test-model", client=mock_client)
        assert comparator.client == mock_client
        mock_bedrock.assert_not_called()

    @skip("Not implemented yet")
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

    @skip("Not implemented yet")
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

    @skip("Not implemented yet")
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

    @skip("Not implemented yet")
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

    @skip("Not implemented yet")
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
        result = comparator.compare("value1", "value2")

        # Verify the custom prompt is used
        assert result is True
        mock_client.invoke_model.assert_called_once()

        # Extract the body parameter used in the invoke_model call
        call_args = mock_client.invoke_model.call_args[1]
        body = json.loads(call_args["body"])

        # Verify the prompt in the request body
        assert "Custom prompt value1 vs value2" in body["prompt"]

    @skip("Not implemented yet")
    @patch("stickler.comparators.llm.BedrockRuntime")
    def test_compare_exception_handling(self, mock_bedrock):
        """Test exception handling during comparison."""
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("API Error")
        mock_bedrock.return_value = mock_client

        comparator = LLMComparator(model_name="test-model")

        # The comparison should return False when an exception occurs
        assert comparator.compare("value1", "value2") is False
        mock_client.invoke_model.assert_called_once()
