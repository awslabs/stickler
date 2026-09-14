# tests/common/comparators/conftest.py
import sys
from unittest.mock import MagicMock

import pytest

from stickler.utils import deprecation as _deprecation


@pytest.fixture(autouse=True)
def _reset_deprecation_sentinels():
    """Clear ``warn_once`` sentinels so each test sees a fresh warning.

    ``warn_once`` remembers ``(deprecation_id, context)`` per process, so a
    ``pytest.warns`` assertion would silently fail if a peer test had already
    tripped the same sentinel. ``BaseComparator.__init_subclass__`` warns
    through ``warn_once``, and the shadowing tests in
    ``test_none_handling.py`` assert it fires. Mirrors the fixture in
    ``tests/structured_object_evaluator/conftest.py``.
    """
    _deprecation._warned.clear()
    yield
    _deprecation._warned.clear()


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
