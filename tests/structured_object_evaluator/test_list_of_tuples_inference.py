"""A tuple inside a list is scored as one positional value, not by its repr.

``List[Tuple[int, int]]`` with ``ComparableField()`` reached the scalar fallback,
so every element was handed to ``LevenshteinComparator`` over ``str(element)``.
The edit distance between ``"(1, 2)"`` and ``"(2, 1)"`` is small, so a swapped
pair kept most of the credit and the list looked almost right (#659):

    [(1, 2), (3, 4)]  against  [(2, 1), (3, 4)]   ->  0.8333

A tuple is positional. ``(1, 2)`` and ``(2, 1)`` are different values, so the
inferred comparator for a list of tuples is now ``ExactComparator``, which is the
rule #391 proposes for a tuple: one value, compared with equality.

Two boundaries are pinned here so the fix cannot widen:

- An explicit ``comparator=`` is consent and is never replaced, matching the
  object-grade substitution above it.
- A tuple is deliberately NOT routed to ``ANLSStarComparator``. It reads a tuple
  as one of several allowed answers, so it scored the swapped pair ``1.0``.
"""

from typing import Annotated, List, Optional, Tuple

import pytest

from stickler import (
    ANLSStarComparator,
    ComparableField,
    ExactComparator,
    LevenshteinComparator,
    StructuredModel,
)


def _inferred_comparator(annotation) -> str:
    """The comparator inferred for a field, by class name."""
    model = type(
        "Holder",
        (StructuredModel,),
        {"__annotations__": {"f": annotation}, "f": ComparableField()},
    )
    return type(model._get_comparison_info("f").comparator).__name__


def _list_score(gt, pred, threshold: float = 0.5) -> float:
    """Score two ``List[Tuple[int, int]]`` values against each other."""

    class Model(StructuredModel):
        pts: List[Tuple[int, int]] = ComparableField(threshold=threshold)

    return Model(pts=gt).compare_with(Model(pts=pred))["field_scores"]["pts"]


class TestASwappedPairIsNotMostlyRight:
    def test_the_issue_reproduction_scores_half(self):
        """One of two elements is right, so the list is half right."""
        assert _list_score([(1, 2), (3, 4)], [(2, 1), (3, 4)]) == 0.5

    def test_identical_lists_score_one(self):
        assert _list_score([(1, 2), (3, 4)], [(1, 2), (3, 4)]) == 1.0

    def test_every_element_swapped_scores_zero(self):
        assert _list_score([(1, 2), (3, 4)], [(2, 1), (4, 3)]) == 0.0

    def test_order_of_the_list_itself_does_not_matter(self):
        """The list is still matched unordered; only the tuple inside is positional."""
        assert _list_score([(1, 2), (3, 4)], [(3, 4), (1, 2)]) == 1.0

    def test_one_missing_element_scores_half(self):
        assert _list_score([(1, 2), (3, 4)], [(1, 2)]) == 0.5


class TestTheListPathMatchesTheComparator:
    """Parity with ``ExactComparator`` is the requirement, not the literal."""

    def test_the_inferred_comparator_is_exact(self):
        assert _inferred_comparator(List[Tuple[int, int]]) == "ExactComparator"

    @pytest.mark.parametrize(
        "gt_element, pred_element",
        [((1, 2), (1, 2)), ((1, 2), (2, 1)), ((1, 2), (1, 3)), ((1, 2), (2, 2))],
    )
    def test_an_element_scores_as_exact_comparator_scores_it(
        self, gt_element, pred_element
    ):
        """A one-element list scores what ``ExactComparator`` gives the pair."""
        direct = ExactComparator().compare(gt_element, pred_element)
        assert _list_score(
            [gt_element], [pred_element], threshold=0.0
        ) == pytest.approx(direct)

    def test_a_wrong_length_tuple_never_reaches_the_comparator(self):
        """Pydantic rejects it at construction, so it is not a score at all.

        ``ExactComparator`` would give ``0.0`` for ``(1, 2)`` against ``(1, 2, 3)``,
        but a ``Tuple[int, int]`` field cannot hold the second value.
        """
        with pytest.raises(ValueError):
            _list_score([(1, 2)], [(1, 2, 3)])


class TestAnExplicitComparatorIsConsent:
    def test_an_explicit_levenshtein_is_kept(self):
        model = type(
            "Holder",
            (StructuredModel,),
            {
                "__annotations__": {"f": List[Tuple[int, int]]},
                "f": ComparableField(comparator=LevenshteinComparator()),
            },
        )
        assert type(model._get_comparison_info("f").comparator) is LevenshteinComparator

    def test_an_explicit_levenshtein_still_scores_the_old_way(self):
        """Whoever names Levenshtein gets Levenshtein, swapped pairs and all."""

        class Model(StructuredModel):
            pts: List[Tuple[int, int]] = ComparableField(
                comparator=LevenshteinComparator()
            )

        score = Model(pts=[(1, 2), (3, 4)]).compare_with(Model(pts=[(2, 1), (3, 4)]))
        assert score["field_scores"]["pts"] == pytest.approx(0.8333333333333334)

    def test_an_explicit_anls_star_is_kept(self):
        class Model(StructuredModel):
            pts: List[Tuple[int, int]] = ComparableField(
                comparator=ANLSStarComparator()
            )

        info = Model._get_comparison_info("pts")
        assert type(info.comparator) is ANLSStarComparator
        assert info.comparator_explicit is True


class TestTheAnnotationIsRecognisedHoweverItIsSpelled:
    @pytest.mark.parametrize(
        "annotation",
        [
            List[Tuple[int, int]],
            List[tuple[int, int]],
            List[Tuple],
            List[tuple],
            List[Tuple[int, ...]],
            Optional[List[Tuple[int, int]]],
            List[Optional[Tuple[int, int]]],
            List[Annotated[Tuple[int, int], "a point"]],
        ],
    )
    def test_every_spelling_of_a_tuple_list_infers_exact(self, annotation):
        assert _inferred_comparator(annotation) == "ExactComparator"


class TestNothingElseMoved:
    @pytest.mark.parametrize(
        "annotation, expected",
        [
            (List[str], "LevenshteinComparator"),
            (List[int], "LevenshteinComparator"),
            (List[float], "LevenshteinComparator"),
            (List[List[int]], "LevenshteinComparator"),
            (Tuple[int, int], "LevenshteinComparator"),
        ],
    )
    def test_other_shapes_keep_their_comparator(self, annotation, expected):
        """A tuple FIELD is #391's to settle, and a nested list is out of scope."""
        assert _inferred_comparator(annotation) == expected

    def test_a_list_of_mappings_is_still_object_grade(self):
        assert _inferred_comparator(List[dict[str, int]]) == "ANLSStarComparator"
