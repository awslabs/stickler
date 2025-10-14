"""
Tests for the structured_model_evaluator module.
"""

from unittest.mock import MagicMock, patch

from stickler.structured_object_evaluator.evaluator import (
    get_memory_usage,
)


def test_import_from_new_location():
    """Test that we can import from the new module structure."""
    from stickler.structured_object_evaluator.evaluator import (
        StructuredModelEvaluator,
        get_memory_usage,
    )

    # Simple test to validate imports work
    assert StructuredModelEvaluator is not None
    assert get_memory_usage is not None


def test_backward_compatibility():
    """Test that the backward compatibility imports work."""
    # Import directly from the module
    from stickler.structured_object_evaluator.evaluator import (
        StructuredModelEvaluator as OldEvaluator,
    )

    # Import from the new location
    from stickler.structured_object_evaluator.evaluator import (
        StructuredModelEvaluator as NewEvaluator,
    )

    # They should be the same class
    assert OldEvaluator is NewEvaluator


@patch("stickler.structured_object_evaluator.evaluator.psutil.Process")
def test_get_memory_usage(mock_process):
    """Test the get_memory_usage function."""
    # Setup mock
    mock_process_instance = MagicMock()
    mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB
    mock_process.return_value = mock_process_instance

    # Call the function
    memory = get_memory_usage()

    # Check result
    assert memory == 100.0  # 100 MB
