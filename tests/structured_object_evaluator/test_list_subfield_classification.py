"""How a sub-field of a matched ``List[StructuredModel]`` pair is classified.

For each matched pair, ``StructuredListComparator._handle_non_hierarchical_field``
asks ``StructuredModel._classify_field_for_confusion_matrix`` for a verdict on
every non-list sub-field and adds up the counts (#395).

- The verdict is the six counts and nothing else. The caller only adds them up.
- A sub-field annotated ``Union[A, B]`` or ``Any`` is not routed as a nested
  model, so model values reach the classifier, which compares them as models.
"""

from typing import Any, List, Union

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators.base import BaseComparator

THRESHOLD = 0.8


class _FixedScore(BaseComparator):
    """Scores equal values 1.0 and every other pair at ``score``."""

    def __init__(self, score: float):
        super().__init__(threshold=THRESHOLD)
        self.score = score

    def _compare(self, str1, str2):
        return 1.0 if str1 == str2 else self.score


def _list_models(score: float):
    """An element model whose ``name`` scores ``score``, and a list of it."""

    class Item(StructuredModel):
        key: str = ComparableField(threshold=1.0)
        name: str = ComparableField(comparator=_FixedScore(score), threshold=THRESHOLD)
        # Low enough that the pair is always matched and its sub-fields
        # classified, whatever ``name`` scores.
        match_threshold = 0.4

    class Holder(StructuredModel):
        items: List[Item] = ComparableField()

    return Item, Holder


class TestVerdictShape:
    def test_a_match_returns_only_the_six_counts(self):
        Item, _ = _list_models(0.5)
        verdict = Item(key="k", name="a")._classify_field_for_confusion_matrix(
            "name", "a"
        )
        assert verdict == {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0}

    def test_a_non_match_returns_only_the_six_counts(self):
        Item, _ = _list_models(0.5)
        verdict = Item(key="k", name="a")._classify_field_for_confusion_matrix(
            "name", "b"
        )
        assert verdict == {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0}


class _Inner(StructuredModel):
    text: str = ComparableField(threshold=0.5)


class _Other(StructuredModel):
    code: str = ComparableField(threshold=0.5)


class TestModelValuedSubField:
    """``Union[A, B]`` and ``Any`` sub-fields holding models are compared as models.

    ``ConfigurationHelper.is_structured_field_type`` recognises neither
    annotation, so the pair takes the primitive route. The field's default
    comparator is ``LevenshteinComparator``, which scores the text of
    ``_Inner(text="p")`` against ``_Inner(text="q")`` at about 0.96. Compared as
    models they score 0.0, which is the right verdict.
    """

    @pytest.mark.parametrize("annotation", [Union[_Inner, _Other], Any])
    @pytest.mark.parametrize(
        "pred_text, expected",
        [("p", {"tp": 1, "fd": 0}), ("q", {"tp": 0, "fd": 1})],
    )
    def test_the_pair_is_scored_as_models(self, annotation, pred_text, expected):
        class Element(StructuredModel):
            key: str = ComparableField(threshold=1.0)
            value: annotation = ComparableField()
            match_threshold = 0.4

        class Holder(StructuredModel):
            items: List[Element] = ComparableField()

        result = Holder(items=[Element(key="k", value=_Inner(text="p"))]).compare_with(
            Holder(items=[Element(key="k", value=_Inner(text=pred_text))]),
            include_confusion_matrix=True,
        )
        counts = result["confusion_matrix"]["fields"]["items"]["fields"]["value"][
            "overall"
        ]
        assert {"tp": counts["tp"], "fd": counts["fd"]} == expected
