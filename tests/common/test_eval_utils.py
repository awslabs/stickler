import pytest

import string

# Import utility functions from the evaluation module
from stickler.utils.text_normalizers import lowercase, strip_punctuation_space


class TestEvaluationUtils:
    """Test cases for utility functions in the common evaluation module."""

    def test_lowercase(self):
        """Test the lowercase function."""
        # Basic functionality tests
        assert lowercase("TEST") == "test"
        assert lowercase("Test String") == "test string"
        assert lowercase(None) is None
        assert lowercase("") == ""

        # Test with mixed case
        assert lowercase("MiXeD cAsE") == "mixed case"

        # Test with numbers and special characters
        assert lowercase("ABC123!@#") == "abc123!@#"

        # Test with non-English characters
        assert lowercase("CAFÉ") == "café"
        assert lowercase("ÑANDÚ") == "ñandú"

        # Test with whitespace
        assert lowercase("  SPACES  ") == "  spaces  "

    def test_strip_punctuation_space(self):
        """Test the strip_punctuation_space function."""
        # Basic functionality tests
        assert strip_punctuation_space("hello!") == "hello"
        assert strip_punctuation_space("test, string.") == "teststring"
        assert strip_punctuation_space(None) is None
        assert strip_punctuation_space("") == ""

        # Test with multiple punctuation marks
        assert strip_punctuation_space("a!b@c#d$e%f^g&h*i(j)k") == "abcdefghijk"

        # Test with spaces and punctuation
        assert strip_punctuation_space("hello, world!") == "helloworld"
        assert (
            strip_punctuation_space("  spaces  and  punctuation!  ")
            == "spacesandpunctuation"
        )

        # Test with all punctuation characters
        all_punct = string.punctuation
        assert strip_punctuation_space(all_punct) == ""

        # Test with numbers
        assert strip_punctuation_space("123,456.78") == "12345678"

        # Test with mixed content
        assert strip_punctuation_space("Phone: (123) 456-7890") == "Phone1234567890"
