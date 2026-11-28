"""Tied optimal assignments must not make list scores depend on element order."""

from itertools import permutations
from typing import List

import pytest

from stickler import ComparableField, StructuredModel


class _Model(StructuredModel):
    values: List[str] = ComparableField(threshold=0.7)


@pytest.mark.parametrize("clip", [True, False])
def test_tied_assignments_score_independently_of_element_order(clip):
    class Model(StructuredModel):
        values: List[str] = ComparableField(threshold=0.7, clip_under_threshold=clip)

    gt = ["bcc", "caaa", "cbac"]
    results = []
    for gt_order in permutations(gt):
        for pred_order in permutations(["baaa", "bbaa", "bbac"]):
            result = Model(values=list(gt_order)).compare_with(
                Model(values=list(pred_order)), include_confusion_matrix=True
            )
            overall = result["confusion_matrix"]["fields"]["values"]["overall"]
            results.append((result["overall_score"], overall["tp"], overall["fd"]))

    assert {(tp, fd) for _, tp, fd in results} == {(2, 1)}
    assert [score for score, _, _ in results] == pytest.approx(
        [results[0][0]] * len(results)
    )


def test_tie_break_reports_the_pairs_that_were_counted():
    result = _Model(values=["bcc", "caaa", "cbac"]).compare_with(
        _Model(values=["baaa", "bbac", "bbaa"]), document_non_matches=True
    )

    assert len(result["non_matches"]) == 1
