"""A value scores the same whether or not it sits in a list.

``ComparisonHelper.compare_unordered_lists`` builds its ``HungarianMatcher``
with ``normalize_values`` left at the default ``True``, so ``_prepare_lists``
coerced every item to ``str``, lowercased it and collapsed its whitespace
*before the comparator ever saw it*. The field's declared comparator was
handed values the caller never wrote.

Three shapes were affected. Two of them now score the way the same comparator
already scored the same pair as a scalar; the third deliberately does not:

- ``ExactComparator`` on a case- or whitespace-only difference matched, which
  silently undid #199 for every list-typed field. Now parity.
- ``BBoxIoUComparator`` scored every list of boxes ``0.0``, even against
  identical input, because ``_normalize_bbox`` rejects the stringified
  coordinates. Now real IoU. There is no scalar spelling of a box field to
  compare against -- a box is itself a list -- so those assertions call the
  comparator directly instead.
- ``List[dict]`` under ``LevenshteinComparator`` scored ``1.0`` by comparing
  ``str(dict)``, then raised once #278 let the comparator see the dict. It now
  scores sorted-key JSON, but only after the comparator has refused the raw
  dict. This one is deliberately **not** parity with the scalar spelling, which
  still raises; ``test_dict_items_diverge_from_the_scalar_spelling`` pins that
  gap so it cannot widen unnoticed.

Most assertions here compare the list path to the scalar path rather than to a
literal, because parity is the actual requirement: pinning two independent
constants lets them drift apart while both stay green, which is how this
survived #199 in the first place.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import (
    BBoxIoUComparator,
    ExactComparator,
    LevenshteinComparator,
    NumericComparator,
)
from stickler.comparators.base import BaseComparator
from stickler.comparators.fuzzy import FuzzyComparator


def _scores(comparator, threshold, gt_value, pred_value, item_type=str) -> Tuple[float, float]:
    """Score one pair twice: once as a list field, once as a scalar field.

    Returns ``(list_score, scalar_score)``. The two models are identical apart
    from the list wrapper, and share a comparator instance and threshold, so any
    difference is attributable to the list path alone.
    """

    class ListModel(StructuredModel):
        v: List[item_type] = ComparableField(comparator=comparator, threshold=threshold)

    class ScalarModel(StructuredModel):
        v: item_type = ComparableField(comparator=comparator, threshold=threshold)

    list_result = ListModel(v=[gt_value]).compare_recursive(ListModel(v=[pred_value]))
    scalar_result = ScalarModel(v=gt_value).compare_recursive(ScalarModel(v=pred_value))
    return (
        list_result["fields"]["v"]["similarity_score"],
        scalar_result["fields"]["v"]["similarity_score"],
    )


def _list_counts(comparator, threshold, gt_values, pred_values, item_type=str) -> Dict[str, Any]:
    """Confusion-matrix counts for a list field."""

    class ListModel(StructuredModel):
        v: List[item_type] = ComparableField(comparator=comparator, threshold=threshold)

    result = ListModel(v=gt_values).compare_recursive(ListModel(v=pred_values))
    return result["fields"]["v"]["overall"]


class TestExactComparatorReachesListFields:
    """#199's strictness must apply to a list field, not only a scalar one."""

    def test_case_only_difference_is_not_a_match(self):
        """The exact pair #199 cites, which a List[str] used to match."""
        list_score, scalar_score = _scores(
            ExactComparator(), 1.0, "SHP-2024-001", "shp-2024-001"
        )
        assert list_score == scalar_score
        assert list_score == 0.0

    def test_case_only_difference_is_classified_as_a_non_match(self):
        counts = _list_counts(ExactComparator(), 1.0, ["SHP-2024-001"], ["shp-2024-001"])
        assert counts["tp"] == 0
        assert counts["fd"] == 1

    def test_whitespace_only_difference_is_not_a_match(self):
        list_score, scalar_score = _scores(ExactComparator(), 1.0, "A  B", "A B")
        assert list_score == scalar_score
        assert list_score == 0.0

    def test_an_equal_pair_still_matches(self):
        """The fix must not simply score everything 0.0."""
        list_score, scalar_score = _scores(ExactComparator(), 1.0, "SHP-1", "SHP-1")
        assert list_score == scalar_score
        assert list_score == 1.0

    def test_case_insensitive_is_the_supported_opt_out(self):
        """A caller wanting the old lenient behavior asks for it explicitly.

        This is the same knob a scalar field uses, which is the point: the
        opt-out is a comparator setting, not a property of being a list.
        """
        list_score, scalar_score = _scores(
            ExactComparator(case_sensitive=False), 1.0, "SHP-2024-001", "shp-2024-001"
        )
        assert list_score == scalar_score
        assert list_score == 1.0

    def test_multi_item_lists_are_not_rescued_by_list_membership(self):
        """Guards the general path, not just the 1x1 fast path.

        ``HungarianMatcher.calculate_metrics`` short-circuits a single-item
        list, so a two-item list exercises the matrix path instead.
        """
        counts = _list_counts(
            ExactComparator(), 1.0, ["AB", "CD"], ["cd", "ab"]
        )
        assert counts["tp"] == 0
        assert counts["fd"] == 2


class TestComparatorsThatNormalizeThemselvesAreUnaffected:
    """The no-regression half: self-normalizing comparators keep working."""

    def test_levenshtein_case_only_difference(self):
        list_score, scalar_score = _scores(
            LevenshteinComparator(), 0.7, "Hello World", "hello world"
        )
        assert list_score == scalar_score
        assert list_score == 1.0

    def test_levenshtein_collapses_internal_whitespace(self):
        list_score, scalar_score = _scores(LevenshteinComparator(), 0.7, "A  B", "A B")
        assert list_score == scalar_score
        assert list_score == 1.0

    def test_fuzzy_case_only_difference(self):
        list_score, scalar_score = _scores(
            FuzzyComparator(), 0.7, "Hello World", "hello world"
        )
        assert list_score == scalar_score
        assert list_score == 1.0

    def test_fuzzy_internal_whitespace_is_the_comparators_business(self):
        """No literal: ``FuzzyComparator._normalize`` deliberately does not
        collapse internal whitespace, so the list path must land on whatever
        the scalar path already returns rather than on 1.0.
        """
        list_score, scalar_score = _scores(FuzzyComparator(), 0.7, "A  B", "A B")
        assert list_score == scalar_score
        assert list_score < 1.0


def _bbox_list_score(gt_boxes, pred_boxes, threshold, item_type) -> float:
    """Score a list-of-bboxes field.

    A bounding box *is* a list, so there is no scalar spelling of a bbox field
    to compare against -- ``v: List[int]`` is a list field over coordinates,
    not one box. These cases therefore assert against
    ``BBoxIoUComparator().compare(...)`` called directly on whole boxes.
    """

    class ListModel(StructuredModel):
        v: List[item_type] = ComparableField(
            comparator=BBoxIoUComparator(), threshold=threshold
        )

    result = ListModel(v=gt_boxes).compare_recursive(ListModel(v=pred_boxes))
    return result["fields"]["v"]["similarity_score"]


class TestItemTypesReachTheComparatorIntact:
    """Items must arrive as the caller supplied them, not as ``str(item)``."""

    def test_identical_flat_bboxes_match(self):
        """Every List[bbox] field scored 0.0 before this fix, including here.

        The stringified ``"[0, 0, 10, 10]"`` is not a list, so
        ``_normalize_bbox`` returned None and IoU fell to 0.0 -- so a
        list-of-bboxes field could not score a match against identical input.
        """
        box = [0, 0, 10, 10]
        assert _bbox_list_score([box], [box], 0.5, List[int]) == 1.0
        assert BBoxIoUComparator().compare(box, box) == 1.0

    def test_identical_two_point_bboxes_match(self):
        box = [[0, 0], [10, 10]]
        assert _bbox_list_score([box], [box], 0.5, List[List[int]]) == 1.0
        assert BBoxIoUComparator().compare(box, box) == 1.0

    def test_partially_overlapping_bboxes_score_real_iou(self):
        gt, pred = [0, 0, 10, 10], [5, 5, 15, 15]
        direct = BBoxIoUComparator().compare(gt, pred)
        assert 0.0 < direct < 1.0
        assert _bbox_list_score([gt], [pred], 0.01, List[int]) == direct

    def test_numeric_items_supplied_as_numbers(self):
        list_score, scalar_score = _scores(NumericComparator(), 1.0, 42, 42, item_type=int)
        assert list_score == scalar_score
        assert list_score == 1.0

    def test_numeric_float_matches_across_spellings(self):
        list_score, scalar_score = _scores(
            NumericComparator(), 1.0, 1.50, 1.5, item_type=float
        )
        assert list_score == scalar_score
        assert list_score == 1.0

    def test_dict_items_are_canonicalized_not_stringified(self):
        """A dict item is compared as sorted-key JSON, so key order stops mattering.

        ``LevenshteinComparator`` raises for a dict, and the comparators that
        accept one only do so by way of ``str(dict)``, which preserves insertion
        order -- so key order was significant and the comparison meaningless.
        0.6.0 scored two dicts with identical content ``0.5556`` for exactly that
        reason.

        Once #278 stopped ``_prepare_lists`` stringifying items, the comparator's
        guard fired and the ``TypeError`` escaped to the caller, so a field that
        scored in 0.6.0 crashed instead. ``_score`` now retries a refused pair
        against ``_comparable_form``, which renders a dict as sorted-key JSON.
        That removes both the raise and the key-order sensitivity.

        The retry is a fallback, never a pre-filter: the comparator is offered
        the raw dict first, so one that understands mappings keeps receiving one
        (``test_a_dict_aware_comparator_still_receives_the_dict``). Anything but
        a dict re-raises rather than being canonicalized -- a bounding box *is* a
        list and ``_normalize_bbox`` needs the raw value, which is what
        ``TestItemTypesReachTheComparatorIntact`` covers above.

        This also aligns the explicit path with the zero-config path, which
        already canonicalizes a dict field to sorted-key JSON
        (``src/stickler/auto/README.md``, inference precedence table).

        Whether a dict deserves per-key comparison rather than JSON-string
        similarity at all is #277; refusing the annotation outright on the
        explicit path is #276. Neither is settled here.
        """

        class ListModel(StructuredModel):
            v: List[Dict[str, Any]] = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7
            )

        def score(gt, pred):
            return ListModel(v=[gt]).compare_with(ListModel(v=[pred]))["field_scores"]["v"]

        # Identical content scores 1.0 regardless of key order. On 0.6.0 the
        # reordered pair scored 0.5556.
        assert score({"a": 1, "b": 2}, {"a": 1, "b": 2}) == 1.0
        assert score({"a": 1, "b": 2}, {"b": 2, "a": 1}) == 1.0

        # Different content still scores below a perfect match.
        assert score({"a": 1, "b": 2}, {"a": 9, "b": 8}) < 1.0

        # And it does not raise, which is the regression this pins (#278 exposed
        # the comparator's guard on the list path for the first time).
        assert isinstance(score({"a": 1}, {"a": 1}), float)

    def test_a_dict_aware_comparator_still_receives_the_dict(self):
        """Canonicalization is a fallback, so a mapping-aware comparator is intact.

        Rendering every dict to JSON *before* the comparator call would fix the
        crash by making a dict unreachable: a comparator that scores mappings
        per-key -- the shape #277 is about -- would be handed a ``str`` it never
        asked for and would have no way to opt out. Offering the raw item first
        keeps that door open, and is why ``_score`` retries rather than
        pre-filters.
        """

        class KeyOverlap(BaseComparator):
            """Jaccard over keys; refuses anything that is not a mapping."""

            def _compare(self, value1, value2):
                if not isinstance(value1, dict) or not isinstance(value2, dict):
                    raise TypeError(f"expected dicts, got {type(value1).__name__}")
                keys1, keys2 = set(value1), set(value2)
                union = keys1 | keys2
                return len(keys1 & keys2) / len(union) if union else 1.0

        class ListModel(StructuredModel):
            v: List[Dict[str, Any]] = ComparableField(
                comparator=KeyOverlap(), threshold=0.5
            )

        def score(gt, pred):
            return ListModel(v=[gt]).compare_with(ListModel(v=[pred]))["field_scores"]["v"]

        # Same keys, different values: a key-wise comparator says 1.0 where JSON
        # string similarity would not.
        assert score({"a": 1, "b": 2}, {"a": 9, "b": 8}) == 1.0
        # Half the keys shared is below the field's threshold, so it scores 0.0
        # rather than being rescued by canonicalization.
        assert score({"a": 1, "b": 2}, {"a": 1, "c": 3}) == 0.0

    def test_dict_items_diverge_from_the_scalar_spelling(self):
        """The list path scores a dict pair the scalar path still refuses.

        ``_score`` lives in ``HungarianMatcher``, so only the list path has the
        fallback: a scalar ``Dict`` field hits the comparator's guard with
        nothing to retry against. That is the mirror image of the asymmetry #278
        removed, and it is pinned here rather than left to be rediscovered --
        closing it means deciding whether a scalar dict field should be
        canonicalized too, which is #276 and #277 territory.
        """

        class ScalarModel(StructuredModel):
            v: Dict[str, Any] = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7
            )

        class ListModel(StructuredModel):
            v: List[Dict[str, Any]] = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.7
            )

        gt, pred = {"a": 1, "b": 2}, {"b": 2, "a": 1}

        assert ListModel(v=[gt]).compare(ListModel(v=[pred])) == 1.0
        with pytest.raises(TypeError):
            ScalarModel(v=gt).compare(ScalarModel(v=pred))

        # compare_with degrades to 0.0 instead of raising, which is its own
        # per-field policy -- the divergence in the score survives either way.
        assert ListModel(v=[gt]).compare_with(ListModel(v=[pred]))["field_scores"]["v"] == 1.0
        assert ScalarModel(v=gt).compare_with(ScalarModel(v=pred))["field_scores"]["v"] == 0.0

    def test_a_comparator_error_is_not_swallowed(self):
        """Canonicalization must not become a catch-all for comparator failures.

        The rejected alternative was catching ``TypeError`` around the comparator
        call. That fixes the dict case and also silently converts any genuine
        comparator bug into a ``0.0`` score, which for a metrics library is worse
        than crashing: a wrong number is harder to notice than an exception. It
        would also have created a fresh list-versus-scalar divergence, since the
        scalar path does not catch anything.
        """

        class Broken(BaseComparator):
            def _compare(self, value1, value2):
                return len(value1) / len(value2)  # TypeError on ints

        class ListModel(StructuredModel):
            v: List[int] = ComparableField(comparator=Broken(), threshold=0.7)

        with pytest.raises(TypeError):
            ListModel(v=[1]).compare_with(ListModel(v=[1]))


class TestMissingValuePolicyOnTheListPath:
    """``None`` items follow the shared policy, not a placeholder substitution."""

    def test_both_items_missing_match(self):
        list_score, _ = _scores(
            ExactComparator(), 1.0, None, None, item_type=Optional[str]
        )
        assert list_score == 1.0

    def test_one_item_missing_does_not_match(self):
        list_score, _ = _scores(
            ExactComparator(), 1.0, None, "x", item_type=Optional[str]
        )
        assert list_score == 0.0

    def test_missing_against_empty_string_follows_the_comparator(self):
        """The one case where list and scalar deliberately differ.

        ``NullHelper.is_effectively_null_for_primitives`` treats ``""`` as
        equivalent to ``None`` for a *scalar* primitive field, so the scalar
        path scores this pair 1.0 by design. The list path has no such rule:
        items go to the comparator, and ``BaseComparator``'s ``None`` policy
        makes missing-against-present a non-match.

        Before this change the list path reached 1.0 too, but only by accident
        -- ``_normalize_value`` mapped ``None`` to ``""`` and the two then
        compared equal as strings. Removing the normalization removes that
        coincidence, which surfaces a real asymmetry: field-level null
        equivalence is applied on the scalar path and not the list path. Pinned
        here so it is a recorded decision rather than a silent difference.
        """
        list_score, scalar_score = _scores(
            ExactComparator(), 1.0, None, "", item_type=Optional[str]
        )
        assert list_score == ExactComparator().compare(None, "") == 0.0
        assert scalar_score == 1.0


class TestNestedModelListsKeepTheirOwnPath:
    """Model-element lists are scored by recursive comparison, as before."""

    class _Inner(StructuredModel):
        city: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

    def test_identical_nested_models_match(self):
        class Doc(StructuredModel):
            items: List["TestNestedModelListsKeepTheirOwnPath._Inner"] = ComparableField()

        inner = TestNestedModelListsKeepTheirOwnPath._Inner
        result = Doc(items=[inner(city="Seattle")]).compare_recursive(
            Doc(items=[inner(city="Seattle")])
        )
        assert result["fields"]["items"]["similarity_score"] == 1.0

    def test_differing_nested_models_do_not_match(self):
        class Doc(StructuredModel):
            items: List["TestNestedModelListsKeepTheirOwnPath._Inner"] = ComparableField()

        inner = TestNestedModelListsKeepTheirOwnPath._Inner
        result = Doc(items=[inner(city="Seattle")]).compare_recursive(
            Doc(items=[inner(city="Portland")])
        )
        assert result["fields"]["items"]["similarity_score"] == 0.0
