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


# Keep these filters aligned with docs/docs/Guides/Evaluation/
# confidence-evaluation.md, "Silencing the legacy AUROC deprecation warning".
AUROC = {
    "message": "The 'auroc_confidence_metric' result key is deprecated",
    "category": DeprecationWarning,
}
SINGLE_DOC = {
    "message": "Single-document confidence metrics are a quick sanity check",
    "category": UserWarning,
}


def test_without_filters_both_warnings_are_emitted():
    messages = [str(w.message) for w in _recorded()]
    for warning_filter in (AUROC, SINGLE_DOC):
        assert any(m.startswith(warning_filter["message"]) for m in messages)


def test_documented_filters_silence_both_warnings():
    messages = [str(w.message) for w in _recorded(AUROC, SINGLE_DOC)]
    for warning_filter in (AUROC, SINGLE_DOC):
        assert not any(m.startswith(warning_filter["message"]) for m in messages)


def test_auroc_filter_leaves_the_user_warning():
    messages = [str(w.message) for w in _recorded(AUROC)]
    assert not any(m.startswith(AUROC["message"]) for m in messages)
    assert any(m.startswith(SINGLE_DOC["message"]) for m in messages)
