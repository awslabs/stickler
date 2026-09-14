"""The single-item shortcut must classify exactly like the general path (#224).

``HungarianMatcher.calculate_metrics`` has a fast path for ``len == 1`` on both
sides that bypasses the assignment algorithm. Nothing pinned that it agreed
with the general multi-item path, and it did not: a zero-similarity 1-vs-1 pair
was dropped (yielding FN + FA instead of FD), and the shortcut gated on
``score > 0`` instead of ``score >= match_threshold``, so any non-zero
similarity counted as a true positive no matter how far below threshold.

The documented contract (``docs/docs/Advanced/hungarian-matching.md``) is:

- a pair the algorithm assigns is matched; ``match_threshold`` splits matched
  pairs into TP (``>=``) and FD (``<``), it does not un-match them
- unmatched GT items are FN, unmatched Pred items are FA
- equal-length lists pair every element, so only TP and FD are possible

These tests pin that contract against list length, which is the property that
would have caught #224.
"""

import pytest

from stickler.algorithms import HungarianMatcher
from stickler.comparators.levenshtein import LevenshteinComparator

# Values with zero pairwise Levenshtein similarity (no shared characters).
DISSIMILAR_GT = ["AAA", "BBB", "CCC"]
DISSIMILAR_PRED = ["XXX", "YYY", "ZZZ"]


def _matcher(threshold: float) -> HungarianMatcher:
    return HungarianMatcher(
        comparator=LevenshteinComparator(), match_threshold=threshold
    )


class TestZeroSimilarityIsAMatchedPair:
    """A zero-similarity assignment is still an assignment."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_zero_similarity_pairs_are_kept_at_every_length(self, n):
        """The pair stays in ``matched_pairs`` regardless of list length.

        Dropping it is what made a 1-item list report FN + FA where a 2-item
        list reported FD for the identical situation.
        """
        result = _matcher(0.5).calculate_metrics(
            DISSIMILAR_GT[:n], DISSIMILAR_PRED[:n]
        )

        assert len(result["matched_pairs"]) == n
        assert all(score == 0.0 for _, _, score in result["matched_pairs"])

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_zero_similarity_counts_are_length_independent(self, n):
        """n unrelated pairs produce the same per-pair counts at any n.

        What #224 is about is the *length independence*, not the values: a
        1-item list must not classify differently from a 2-item one.

        The values themselves changed in #231, which separated the ``fn``
        semantics from the threshold. Every item here is paired, so nothing
        is left without a partner and ``fn`` is zero at every ``n``. The n
        pairs all score below the threshold, so they are false discoveries.
        """
        result = _matcher(0.5).calculate_metrics(
            DISSIMILAR_GT[:n], DISSIMILAR_PRED[:n]
        )

        # Below threshold, so no true positives. The n pairs are all fd, and
        # the fp rollup still counts them.
        assert result["tp"] == 0
        assert result["fp"] == n
        assert result["fd"] == n
        assert result["fn"] == 0
        assert result["fa"] == 0

    def test_single_item_agrees_with_multi_item(self):
        """The regression in one assertion: 1-vs-1 must scale to 2-vs-2."""
        one = _matcher(0.5).calculate_metrics(DISSIMILAR_GT[:1], DISSIMILAR_PRED[:1])
        two = _matcher(0.5).calculate_metrics(DISSIMILAR_GT[:2], DISSIMILAR_PRED[:2])

        assert len(one["matched_pairs"]) == 1
        assert len(two["matched_pairs"]) == 2
        assert (one["tp"], one["fp"], one["fd"], one["fn"]) == (0, 1, 1, 0)
        assert (two["tp"], two["fp"], two["fd"], two["fn"]) == (0, 2, 2, 0)


class TestFastPathHonorsMatchThreshold:
    """The shortcut must gate on ``match_threshold``, not on ``score > 0``."""

    def test_below_threshold_single_pair_is_not_a_true_positive(self):
        """``score > 0`` gating made any non-zero similarity a TP."""
        # "abc" vs "abd" is ~0.667 similar: above 0.5, below 0.9.
        result = _matcher(0.9).calculate_metrics(["abc"], ["abd"])

        assert result["matched_pairs"][0][2] == pytest.approx(0.667, abs=1e-3)
        assert result["tp"] == 0, "a below-threshold pair is not a true positive"

    def test_above_threshold_single_pair_is_a_true_positive(self):
        result = _matcher(0.5).calculate_metrics(["abc"], ["abd"])

        assert result["tp"] == 1

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.5, 0.66, 0.7, 0.9, 1.0])
    def test_single_and_multi_item_agree_across_thresholds(self, threshold):
        """Same pair, same verdict, whether or not a second pair is present.

        The second pair is identical on both sides, so it is always a TP and
        contributes a known constant; the pair under test is what varies.
        """
        single = _matcher(threshold).calculate_metrics(["abc"], ["abd"])
        multi = _matcher(threshold).calculate_metrics(["abc", "QQQ"], ["abd", "QQQ"])

        # Subtract the guaranteed-TP second pair to isolate the shared pair.
        assert single["tp"] == multi["tp"] - 1

    def test_exactly_at_threshold_is_a_true_positive(self):
        """The boundary is inclusive (``>=``), matching the general path."""
        identical = _matcher(1.0).calculate_metrics(["abc"], ["abc"])

        assert identical["matched_pairs"][0][2] == 1.0
        assert identical["tp"] == 1

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_threshold_zero_makes_every_pair_a_true_positive(self, n):
        """``match_threshold=0.0`` is a capture-all sentinel, at any length.

        Because the gate is ``>=``, a zero-similarity pair satisfies a zero
        threshold, so ``tp`` counts *pairs* rather than true positives. This
        pins that the shortcut and the general path agree on the boundary --
        the point of #224 -- and documents the trap.

        ``ComparisonHelper.compare_unordered_lists`` builds a matcher this way
        deliberately to capture all pairs for scoring, and reads only
        ``matched_pairs``, reclassifying against its own
        ``classification_threshold``. Nothing should read ``tp`` from a
        matcher constructed with ``0.0``.
        """
        result = _matcher(0.0).calculate_metrics(DISSIMILAR_GT[:n], DISSIMILAR_PRED[:n])

        assert all(score == 0.0 for _, _, score in result["matched_pairs"])
        assert result["tp"] == n, "every pair clears a zero threshold"
        assert result["fp"] == 0
        assert result["fn"] == 0


class TestUnmatchedItemsAreFnAndFa:
    """Extras on the GT side are FN; extras on the Pred side are FA."""

    def test_extra_ground_truth_items_are_false_negatives(self):
        result = _matcher(0.5).calculate_metrics(["abc", "QQQ"], ["abc"])

        assert result["tp"] == 1
        assert result["fn"] == 1, "the unpaired GT item is a false negative"
        assert result["fp"] == 0
        assert result["fd"] == 0, "the one pair clears the threshold"

    def test_extra_prediction_items_are_false_alarms(self):
        result = _matcher(0.5).calculate_metrics(["abc"], ["abc", "QQQ"])

        assert result["tp"] == 1
        assert result["fp"] == 1, "the unpaired prediction item is a false alarm"
        assert result["fn"] == 0
        assert result["fa"] == 1, "fa names the unpaired prediction on its own"
        assert result["fd"] == 0

    def test_equal_length_lists_produce_no_unmatched_items(self):
        """Documented invariant: equal length means only TP and FD are possible.

        Every pair here is below threshold, so all of them are false
        discoveries. The two unmatched counts are zero because there is no item
        without a partner, which is what the name of this test claims. Before
        #231 that could not be asserted, since ``fn`` counted the low score
        pairs as well.
        """
        result = _matcher(0.9).calculate_metrics(
            ["abc", "def"], ["abd", "deg"]
        )

        assert len(result["matched_pairs"]) == 2, "every element is paired"
        assert result["tp"] == 0
        assert result["fd"] == 2
        assert result["fn"] == 0, "no ground truth item is without a partner"
        assert result["fa"] == 0, "no prediction is without a partner"
