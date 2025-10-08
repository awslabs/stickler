"""Unit tests for the LLMComparator.

This module tests the functionality of the LLMComparator to ensure it works
correctly and maintains compatibility with existing code.
"""

import unittest
from unittest.mock import patch, MagicMock

from stickler.comparators import BaseComparator
from stickler.comparators.llm import LLMComparator
from unittest import skip


class TestLLMComparator(unittest.TestCase):
    """Test the LLMComparator implementation."""

    def setUp(self):
        """Set up test environment."""
        # Create a mock for ClaudeInvoker to avoid actual API calls
        self.patcher = patch("stickler.comparators.llm.ClaudeInvoker")
        self.mock_claude_invoker = self.patcher.start()

        # Configure the mock to return specific responses
        self.mock_instance = MagicMock()
        self.mock_claude_invoker.return_value = self.mock_instance

        # Create the comparator with a dummy prompt and model
        self.comparator = LLMComparator(
            prompt="Compare {value1} and {value2}", model_id="test-model-id"
        )

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()

    @skip("Not implemented yet")
    def test_inheritance(self):
        """Test that LLMComparator inherits from BaseComparator."""
        self.assertIsInstance(self.comparator, BaseComparator)

    @skip("Not implemented yet")
    def test_true_response(self):
        """Test that 'TRUE' response returns 1.0."""
        self.mock_instance.inference.return_value = "TRUE"

        with patch("stickler.comparators.llm.print") as mock_print:
            result = self.comparator.compare("value1", "value2")

            # Verify the result is 1.0
            self.assertEqual(result, 1.0)

            # Verify warning was printed
            mock_print.assert_called_once()
            self.assertIn("WARNING", mock_print.call_args[0][0])

    @skip("Not implemented yet")
    def test_false_response(self):
        """Test that 'FALSE' response returns 0.0."""
        self.mock_instance.inference.return_value = "FALSE"

        result = self.comparator.compare("value1", "value2")
        self.assertEqual(result, 0.0)

    @skip("Not implemented yet")
    def test_none_values(self):
        """Test that None values are handled properly."""
        result = self.comparator.compare(None, "value2")
        self.assertEqual(result, 0.0)

        result = self.comparator.compare("value1", None)
        self.assertEqual(result, 0.0)

        result = self.comparator.compare(None, None)
        self.assertEqual(result, 0.0)

    @skip("Not implemented yet")
    def test_retry_on_error(self):
        """Test that errors are retried once."""
        # Configure mock to raise exception on first call, then succeed
        self.mock_instance.inference.side_effect = [Exception("API Error"), "TRUE"]

        with patch("stickler.comparators.llm.time.sleep") as mock_sleep:
            with patch("stickler.comparators.llm.print"):
                result = self.comparator.compare("value1", "value2")

                # Verify sleep was called for retry
                mock_sleep.assert_called_once()

                # Verify the result is from the second call
                self.assertEqual(result, 1.0)

    @skip("Not implemented yet")
    def test_callable_interface(self):
        """Test that the comparator is callable via __call__."""
        self.mock_instance.inference.return_value = "FALSE"

        # Test using the __call__ interface
        result = self.comparator("value1", "value2")
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
