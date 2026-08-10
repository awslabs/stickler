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
        """n unrelated pairs produce the same per-pair counts at any n."""
        result = _matcher(0.5).calculate_metrics(
            DISSIMILAR_GT[:n], DISSIMILAR_PRED[:n]
        )

        # Below threshold, so no true positives; fp/fn reflect the n pairs.
        assert result["tp"] == 0
        assert result["fp"] == n
        assert result["fn"] == n

    def test_single_item_agrees_with_multi_item(self):
        """The regression in one assertion: 1-vs-1 must scale to 2-vs-2."""
        one = _matcher(0.5).calculate_metrics(DISSIMILAR_GT[:1], DISSIMILAR_PRED[:1])
        two = _matcher(0.5).calculate_metrics(DISSIMILAR_GT[:2], DISSIMILAR_PRED[:2])

        assert len(one["matched_pairs"]) == 1
        assert len(two["matched_pairs"]) == 2
        assert (one["tp"], one["fp"], one["fn"]) == (0, 1, 1)
        assert (two["tp"], two["fp"], two["fn"]) == (0, 2, 2)


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

    @pytest.mark.parametrize("threshold", [0.1, 0.5, 0.66, 0.7, 0.9, 1.0])
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


class TestUnmatchedItemsAreFnAndFa:
    """Extras on the GT side are FN; extras on the Pred side are FA."""

    def test_extra_ground_truth_items_are_false_negatives(self):
        result = _matcher(0.5).calculate_metrics(["abc", "QQQ"], ["abc"])

        assert result["tp"] == 1
        assert result["fn"] == 1, "the unpaired GT item is a false negative"
        assert result["fp"] == 0

    def test_extra_prediction_items_are_false_alarms(self):
        result = _matcher(0.5).calculate_metrics(["abc"], ["abc", "QQQ"])

        assert result["tp"] == 1
        assert result["fp"] == 1, "the unpaired prediction item is a false alarm"
        assert result["fn"] == 0

    def test_equal_length_lists_produce_no_unmatched_items(self):
        """Documented invariant: equal length means only TP and FD are possible.

        With every pair below threshold, fp/fn come from below-threshold pairs
        rather than from unmatched items -- there are none to be had.
        """
        result = _matcher(0.9).calculate_metrics(
            ["abc", "def"], ["abd", "deg"]
        )

        assert len(result["matched_pairs"]) == 2, "every element is paired"
        assert result["tp"] == 0
