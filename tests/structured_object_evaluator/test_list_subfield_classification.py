"""How a model in a sub-field of a matched ``List[StructuredModel]`` pair is classified.

For each matched pair, ``StructuredListComparator._handle_non_hierarchical_field``
classifies every non-list sub-field through the GT item's dispatcher, as at top
level. A sub-field annotated ``Union[A, B]`` or ``Any`` is not routed as a nested
model, so its model values reach the dispatcher, which compares them as models.
"""

from typing import Any, List, Union

import pytest

from stickler import ComparableField, StructuredModel


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
