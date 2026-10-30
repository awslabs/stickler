"""`warn_once` records a warning only after it was emitted.

Recorded first, a warning the filters turned into an exception (`-W error`,
pytest's `filterwarnings = error`) raised on the first call only, and every later
call, including one under `pytest.warns`, passed silently.
"""

import warnings

import pytest

from stickler.utils import deprecation


@pytest.fixture(autouse=True)
def _fresh_sentinels(monkeypatch):
    monkeypatch.setattr(deprecation, "_warned", set())


def _warn():
    deprecation.warn_once("test-id", "ctx", "test message", category=UserWarning)


def test_every_call_raises_under_an_error_filter():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            with pytest.raises(UserWarning, match="test message"):
                _warn()
    with pytest.warns(UserWarning, match="test message"):
        _warn()


def test_an_emitted_warning_is_not_repeated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn()
        _warn()
    assert [str(w.message) for w in caught] == ["test message"]
