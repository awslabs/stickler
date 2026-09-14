"""Shared fixtures for comparator tests."""

import pytest

from stickler.utils import deprecation as _deprecation


@pytest.fixture(autouse=True)
def _reset_deprecation_sentinels():
    """Clear ``warn_once`` sentinels so each test sees fresh warnings.

    The library suppresses repeat warnings by remembering
    ``(deprecation_id, context)`` tuples in a process-scoped set. Tests that
    assert ``with pytest.warns(...):`` would otherwise silently fail when run
    after a peer test that already tripped the same sentinel.
    """
    _deprecation._warned.clear()
    yield
    _deprecation._warned.clear()
