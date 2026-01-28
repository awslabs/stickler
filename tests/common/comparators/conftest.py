# tests/common/comparators/conftest.py
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module", autouse=True)
def mock_strands_module():
    """Mock strands-agents module for tests that don't have it installed."""
    mock_strands = MagicMock()
    mock_strands_models = MagicMock()

    sys.modules["strands"] = mock_strands
    sys.modules["strands.models"] = mock_strands_models

    yield

    # Cleanup
    del sys.modules["strands"]
    del sys.modules["strands.models"]
