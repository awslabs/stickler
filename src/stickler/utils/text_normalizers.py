"""
Common utility functions for key information evaluation.

This module contains utilities shared across both flat and structured evaluation systems.
"""

import re
import string


def lowercase(text):
    """
    Convert text to lowercase.

    Args:
        text: Input text to convert

    Returns:
        Lowercase version of the input text, or None if input is None
    """
    if text is None:
        return None
    if not text:
        return ""
    return str(text).lower()


def strip_punctuation_space(text):
    """
    Remove punctuation and spaces from text.

    Args:
        text: Input text to process

    Returns:
        Text with punctuation and spaces removed
    """
    if text is None:
        return None
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove punctuation and spaces
    translator = str.maketrans("", "", string.punctuation + string.whitespace)
    text = text.translate(translator)

    return text
