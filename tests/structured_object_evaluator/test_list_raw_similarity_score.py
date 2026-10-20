"""`raw_similarity_score` on a list node is unclipped (#330, item 2).

Every other node reports its comparator output unclipped in
`raw_similarity_score` and the clipped score in `threshold_applied_score`.
A list node reported the clipped mean in both, so its "raw" score disagreed with
`compare()` for the same field and the unclipped number could not be recovered
from `compare_with()` output. These tests pin the split, and pin that the score
and the counts are what they were.
"""

from typing import List, Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.auto import eval_for
from stickler.comparators import ExactComparator, LevenshteinComparator


class Tags(StructuredModel):
    tags: Optional[List[str]] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, default=None
    )


class TagsKeep(StructuredModel):
    tags: Optional[List[str]] = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.9,
        clip_under_threshold=False,
        default=None,
    )


class Item(StructuredModel):
    match_threshold = 0.9
    a: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)
    b: Optional[str] = ComparableField(comparator=ExactComparator(), default=None)


class Items(StructuredModel):
    items: Optional[List[Item]] = ComparableField(default=None)


def _node(gt, pred, field):
    result = gt.compare_with(pred, include_confusion_matrix=True)
    return result, result["confusion_matrix"]["fields"][field]


def test_clipped_list_reports_unclipped_raw_score():
    """`banana`/`banxxx` scores 0.5 against a 0.9 bar, so it is clipped to 0.0
    in the score. Raw keeps it: (1.0 + 0.5) / 2 = 0.75, as `compare()` says."""
    gt = Tags(tags=["apple", "banana"])
    pred = Tags(tags=["apple", "banxxx"])

    result, node = _node(gt, pred, "tags")

    assert node["raw_similarity_score"] == pytest.approx(0.75)
    assert node["raw_similarity_score"] == pytest.approx(gt.compare(pred))
    # The score is unchanged: clipped mean, (1.0 + 0.0) / 2.
    assert node["similarity_score"] == pytest.approx(0.5)
    assert node["threshold_applied_score"] == pytest.approx(0.5)
    assert result["field_scores"]["tags"] == pytest.approx(0.5)
    assert result["overall_score"] == pytest.approx(0.5)
    assert {k: node["overall"][k] for k in ("tp", "fd", "fa", "fn")} == {
        "tp": 1,
        "fd": 1,
        "fa": 0,
        "fn": 0,
    }


def test_raw_score_is_coverage_scaled():
    """An unmatched element counts in the denominator: 1.5 / max(2, 3)."""
    gt = Tags(tags=["apple", "banana"])
    pred = Tags(tags=["apple", "banxxx", "cherry"])

    _, node = _node(gt, pred, "tags")

    assert node["raw_similarity_score"] == pytest.approx(1.5 / 3)
    assert node["threshold_applied_score"] == pytest.approx(1.0 / 3)


def test_all_elements_below_threshold_raw_is_not_zero():
    """A list whose every pair missed the bar scores 0.0, and raw says by how much."""
    gt = Tags(tags=["banana"])
    pred = Tags(tags=["banxxx"])

    result, node = _node(gt, pred, "tags")

    assert result["overall_score"] == 0.0
    assert node["threshold_applied_score"] == 0.0
    assert node["raw_similarity_score"] == pytest.approx(0.5)


def test_unclipped_list_raw_equals_score():
    """With clipping off nothing is zeroed, so raw and score agree exactly."""
    _, node = _node(
        TagsKeep(tags=["apple", "banana"]), TagsKeep(tags=["apple", "banxxx"]), "tags"
    )

    assert node["raw_similarity_score"] == node["threshold_applied_score"]
    assert node["threshold_applied_score"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "gt,pred,expected",
    [
        ([], [], 1.0),
        (["a"], [], 0.0),
        ([], ["a"], 0.0),
        (["a", "b"], ["a", "b"], 1.0),
    ],
    ids=["both-empty", "pred-empty", "gt-empty", "identical"],
)
def test_raw_equals_score_where_nothing_is_clipped(gt, pred, expected):
    _, node = _node(Tags(tags=gt), Tags(tags=pred), "tags")

    assert node["raw_similarity_score"] == expected
    assert node["threshold_applied_score"] == expected


def test_structured_list_raw_score_matches_compare():
    """A `List[StructuredModel]` pair is scored by the item's own
    `compare_with`; raw is the unclipped mean of those, as `compare()` reports."""
    gt = Items(items=[Item(a="x", b="y"), Item(a="p", b="q")])
    pred = Items(items=[Item(a="x", b="y"), Item(a="p", b="zz")])

    result, node = _node(gt, pred, "items")

    assert node["raw_similarity_score"] == pytest.approx(gt.compare(pred))
    assert node["raw_similarity_score"] == pytest.approx(0.75)
    assert node["threshold_applied_score"] == pytest.approx(0.75)
    assert result["field_scores"]["items"] == pytest.approx(0.75)


def test_explain_list_verdict_does_not_read_raw_as_one_score():
    """A list's raw is a coverage-scaled mean, so the verdict does not compare
    it to the per-element threshold: 'banxxx' scored 0.5, not the 0.25 shown."""
    from pydantic import BaseModel

    import stickler

    class Doc(BaseModel):
        name: str
        tags: List[str]

    explained = stickler.evaluate(
        Doc(name="banana", tags=["banana", "cherry"]),
        Doc(name="banxxx", tags=["banxxx"]),
    ).explain()

    assert explained["tags"]["raw_similarity"] == pytest.approx(0.25)
    assert explained["tags"]["verdict"] == (
        "each paired element < threshold 0.7 -> clipped to 0.0 "
        "(raw 0.25 is their coverage-scaled mean)"
    )
    assert explained["name"]["verdict"] == (
        "raw 0.50 < threshold 0.85 -> clipped to 0.0"
    )


class Code(StructuredModel):
    code: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, default=None
    )


class Codes(StructuredModel):
    items: Optional[List[Code]] = ComparableField(default=None)


def test_structured_list_raw_is_clipped_inside_each_item():
    """Each pair's score is the item's own `overall_score`, which has already
    clipped `code` (0.5 < 0.9), so the list's raw score is 0.0, not 0.5."""
    _, node = _node(
        Codes(items=[Code(code="banana")]), Codes(items=[Code(code="banxxx")]), "items"
    )

    assert node["raw_similarity_score"] == 0.0
    assert node["threshold_applied_score"] == 0.0


class TinyCode(StructuredModel):
    code: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.9,
        clip_under_threshold=False,
        default=None,
    )


class TinyCodes(StructuredModel):
    items: Optional[List[TinyCode]] = ComparableField(default=None)


def test_explain_gives_no_verdict_for_a_list_of_models():
    """The list clips a pair below 0.01, not at the field's threshold, so a
    verdict naming that threshold would state the wrong cause."""
    result = eval_for(TinyCodes).evaluate(
        TinyCodes(items=[TinyCode(code="a")]),
        TinyCodes(items=[TinyCode(code="a" + "b" * 199)]),
    )
    explained = result.explain()

    assert explained["items"]["score"] == 0.0
    assert explained["items"]["raw_similarity"] == pytest.approx(0.005)
    assert "verdict" not in explained["items"]
