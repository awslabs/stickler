"""The warning filters documented in confidence-evaluation.md must silence the warnings."""

import warnings

import pytest

from stickler import StructuredModel
from stickler.utils import deprecation


class _Doc(StructuredModel):
    a: str = "x"


@pytest.fixture(autouse=True)
def _fresh_sentinels(monkeypatch):
    monkeypatch.setattr(deprecation, "_warned", set())


def _recorded(*filters):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for kwargs in filters:
            warnings.filterwarnings("ignore", **kwargs)
        _Doc().compare_with(_Doc(), add_confidence_metrics=True)
    return caught


AUROC = {
    "message": "The 'auroc_confidence_metric' result key is deprecated",
    "category": DeprecationWarning,
}
SINGLE_DOC = {
    "message": "Single-document confidence metrics are a quick sanity check",
    "category": UserWarning,
}


def test_without_filters_both_warnings_are_emitted():
    categories = {w.category for w in _recorded()}
    assert {DeprecationWarning, UserWarning} <= categories


def test_documented_filters_silence_both_warnings():
    assert _recorded(AUROC, SINGLE_DOC) == []


def test_auroc_filter_leaves_the_user_warning():
    categories = [w.category for w in _recorded(AUROC)]
    assert DeprecationWarning not in categories
    assert UserWarning in categories
