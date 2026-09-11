"""Regression for #315: text normalization must not silently score mappings."""

from typing import Any, Dict, Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators.normalized import NormalizedComparator
from stickler.utils import deprecation


@pytest.mark.parametrize("method", ["compare", "compare_with"])
@pytest.mark.parametrize("prediction", [{"k": "v", "n": 1}, {"n": 1, "k": "v"}])
def test_normalized_mapping_warns_and_completes(monkeypatch, method, prediction):
    # Each case must observe its own warning, independent of earlier tests.
    monkeypatch.setattr(deprecation, "_warned", set())

    class Model(StructuredModel):
        meta: Optional[Dict[str, Any]] = ComparableField(
            comparator=NormalizedComparator(), default=None
        )

    expected = Model(meta={"k": "v", "n": 1})
    actual = Model(meta=prediction)
    with pytest.warns(UserWarning, match="NormalizedComparator.*ANLSStarComparator"):
        result = getattr(expected, method)(actual)

    score = result["overall_score"] if method == "compare_with" else result
    assert score == 0.0
