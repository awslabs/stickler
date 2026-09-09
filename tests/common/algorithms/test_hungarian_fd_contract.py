"""The contract for the counts that ``calculate_metrics`` returns. See #231.

``calculate_metrics`` used to derive ``fn`` and ``fp`` as ``len(list) minus
tp``. So a pair that the assignment did produce, and that the method returns
inside ``matched_pairs``, was also counted as a false negative when its score
sat below ``match_threshold``. The same pair was reported as matched and as
missing at the same time.

The rule these tests hold the method to is the one
``docs/docs/Advanced/hungarian-matching.md`` publishes. The assignment decides
what is paired. The threshold then splits the paired items into TP and FD. It
never puts a pair back into ``fn`` or ``fa``. Only an item with no partner at
all becomes FN on the ground truth side or FA on the prediction side.

This file is the executable form of that rule.

The five categories, with ``m`` and ``n`` as the list lengths after the matcher
prepares its inputs and ``k`` as the number of assigned pairs:

TP
    a pair with a score at or above ``match_threshold``.
FD
    a pair with a score below ``match_threshold``. This is the key #231 adds.
FN
    a ground truth item with no partner, so ``m minus k``.
FA
    a prediction item with no partner, so ``n minus k``.
FP
    the rollup ``fd plus fa``, which is the value the method already returns.

A caller that read ``fn`` before this change now reads a value lower by ``fd``.
The old value is recoverable as ``fn plus fd``, and one test below pins that
identity so the migration note stays true.
"""

import pytest

from stickler.algorithms import HungarianMatcher
from stickler.comparators.levenshtein import LevenshteinComparator

#: Every key the method must return, and nothing else. Exact equality is the
#: point. It pins that ``fd`` and ``fa`` arrive and that no other key does.
CONTRACT_KEYS = {
    "matched_pairs",
    "tp",
    "fa",
    "fd",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
}

#: Letters that build values with zero pairwise Levenshtein similarity. Two
#: values share no character, so no pair can score above zero.
GT_LETTERS = "ABCDEF"
PRED_LETTERS = "UVWXYZ"


def _matcher(threshold: float) -> HungarianMatcher:
    return HungarianMatcher(
        comparator=LevenshteinComparator(), match_threshold=threshold
    )


def _unrelated(letters: str, count: int) -> list[str]:
    """Build ``count`` values that cannot match anything on the other side.

    The length is asserted rather than assumed. Slicing a fixture that is too
    short truncates in silence, and a shape that quietly shrinks would still
    pass while testing something weaker than its name claims.
    """
    values = [letter * 3 for letter in letters[:count]]
    assert len(values) == count, "the fixture cannot build that many values"
    return values


def _gt(count: int) -> list[str]:
    return _unrelated(GT_LETTERS, count)


def _pred(count: int) -> list[str]:
    return _unrelated(PRED_LETTERS, count)


#: Shapes built from plain string lists, so ``m`` and ``n`` are the input
#: lengths with no preparation step to reason about. Every branch of the method
#: is covered. The last two shapes are the ones where an unpaired count is
#: larger than one, which is where a wrong ``fn`` formula shows up.
PLAIN_SHAPES = [
    pytest.param([], [], id="both_empty"),
    pytest.param([], _pred(2), id="ground_truth_empty"),
    pytest.param(_gt(2), [], id="prediction_empty"),
    pytest.param(_gt(1), _pred(1), id="one_vs_one"),
    pytest.param(_gt(2), _pred(2), id="two_vs_two"),
    pytest.param(_gt(3), _pred(3), id="three_vs_three"),
    pytest.param(_gt(1), _pred(4), id="one_vs_four"),
    pytest.param(_gt(4), _pred(1), id="four_vs_one"),
]


class TestTheFullKeySetIsAlwaysPresent:
    """Every input shape returns the same nine keys.

    A caller must not have to know which branch its input took in order to know
    which keys it can read.
    """

    @pytest.mark.parametrize("ground_truth, prediction", PLAIN_SHAPES)
    def test_every_input_shape_carries_the_full_key_set(self, ground_truth, prediction):
        result = _matcher(0.5).calculate_metrics(ground_truth, prediction)

        assert set(result) == CONTRACT_KEYS

    def test_a_scalar_input_carries_the_full_key_set(self):
        """A bare value stands for a list of one, and reports like one."""
        result = _matcher(0.5).calculate_metrics("apple", "apple")

        assert set(result) == CONTRACT_KEYS


class TestAPairedItemIsNeverAlsoMissing:
    """The defect in #231, stated as behaviour rather than as arithmetic."""

    def test_a_pair_below_the_threshold_is_fd_and_not_fn(self):
        """Two lists of equal length pair every item, so nothing is missing.

        Every pair scores zero here, which is below the threshold, so both
        pairs are false discoveries. The old formula reported ``fn`` as two,
        which said both ground truth items had no partner while the method was
        returning both of them inside ``matched_pairs``.
        """
        result = _matcher(0.5).calculate_metrics(_gt(2), _pred(2))

        assert len(result["matched_pairs"]) == 2
        assert result["tp"] == 0
        assert result["fd"] == 2
        assert result["fn"] == 0, "a paired ground truth item is not missing"
        assert result["fa"] == 0, "a paired prediction item is not an extra"
        assert result["fp"] == 2, "the fp rollup is unchanged"

    def test_one_pair_below_the_threshold_is_fd_and_not_fn(self):
        """The same rule on the fast path for one item on each side."""
        result = _matcher(0.5).calculate_metrics(_gt(1), _pred(1))

        assert len(result["matched_pairs"]) == 1
        assert result["tp"] == 0
        assert result["fd"] == 1
        assert result["fn"] == 0
        assert result["fa"] == 0
        assert result["fp"] == 1

    def test_an_unpaired_ground_truth_item_is_fn(self):
        """Only an item with no partner reaches ``fn``.

        The shared value pairs and clears the threshold. The extra ground truth
        item has nothing to pair with, so it is the one false negative.
        """
        result = _matcher(0.5).calculate_metrics(["apple", "QQQ"], ["apple"])

        assert result["tp"] == 1
        assert result["fd"] == 0
        assert result["fn"] == 1
        assert result["fa"] == 0

    def test_an_unpaired_prediction_item_is_fa(self):
        """The mirror case. An extra prediction is a false alarm."""
        result = _matcher(0.5).calculate_metrics(["apple"], ["apple", "QQQ"])

        assert result["tp"] == 1
        assert result["fd"] == 0
        assert result["fn"] == 0
        assert result["fa"] == 1

    def test_fd_and_fn_can_both_be_present(self):
        """A low score pair and a missing item are different things.

        Three ground truth values against two predictions. One prediction
        matches and clears the threshold. The other pairs with a value it
        shares nothing with, so that pair is a false discovery. The third
        ground truth value has no partner, so it is a false negative. Both of
        them used to land in ``fn``, which made the two cases impossible to
        tell apart.
        """
        result = _matcher(0.5).calculate_metrics(
            ["apple", "AAA", "BBB"], ["apple", "ZZZ"]
        )

        assert len(result["matched_pairs"]) == 2
        assert result["tp"] == 1
        assert result["fd"] == 1, "the low score pair"
        assert result["fn"] == 1, "the item with no partner"
        assert result["fa"] == 0
        assert result["fp"] == 1


class TestTheCountsPartitionTheInputs:
    """Five invariants that hold for every shape, and that name the contract.

    These are the cheapest description of the whole rule. If they hold, no item
    is counted twice and none is dropped.
    """

    @pytest.mark.parametrize("ground_truth, prediction", PLAIN_SHAPES)
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_counts_partition_every_shape(self, ground_truth, prediction, threshold):
        m, n = len(ground_truth), len(prediction)
        result = _matcher(threshold).calculate_metrics(ground_truth, prediction)
        k = len(result["matched_pairs"])

        assert k == min(m, n), "the assignment pairs as many items as it can"
        assert result["tp"] + result["fd"] == k, "every pair is a TP or an FD"
        assert result["fp"] == result["fd"] + result["fa"], "fp is the rollup"
        assert result["fn"] == m - k, "fn counts ground truth items with no partner"
        assert result["fa"] == n - k, "fa counts predictions with no partner"
        assert result["tp"] + result["fd"] + result["fn"] + result["fa"] == max(m, n), (
            "the four counts cover every item exactly once"
        )
        assert result["fn"] * result["fa"] == 0, "only one side can have extras"


class TestTheOldFnValueIsRecoverable:
    """The migration note for callers, pinned as a test.

    The old ``fn`` mixed missing items and low score pairs. The two are now
    separate keys, and the old number is ``fn plus fd``. That must hold for
    every shape, so the note in the changelog does not go stale.
    """

    @pytest.mark.parametrize("ground_truth, prediction", PLAIN_SHAPES)
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_old_fn_equals_new_fn_plus_fd(self, ground_truth, prediction, threshold):
        m = len(ground_truth)
        result = _matcher(threshold).calculate_metrics(ground_truth, prediction)

        # ``m minus tp`` is the formula the method used before the fix.
        old_fn = m - result["tp"]

        assert old_fn == result["fn"] + result["fd"]


class TestTheRatesFollowTheCountsThatAreReturned:
    """A caller can recompute the rates from the counts in the same dict.

    Nothing else in the suite checks this. The method computes the rates from
    the list lengths rather than from the counts, which is correct arithmetic
    but a different route, so the two can drift apart unnoticed.
    """

    @pytest.mark.parametrize("ground_truth, prediction", PLAIN_SHAPES)
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_rates_agree_with_the_published_formulas(
        self, ground_truth, prediction, threshold
    ):
        result = _matcher(threshold).calculate_metrics(ground_truth, prediction)
        tp, fp, fn, fd = (result[key] for key in ("tp", "fp", "fn", "fd"))

        # Both denominators can be zero, and the method already picks a limit
        # for each. Precision is one only when there is nothing on either side.
        # An empty prediction against a ground truth that holds items scores
        # zero, so saying nothing never reads as perfect. Recall is one
        # whenever there was nothing to find.
        nothing_at_all = not ground_truth and not prediction
        expected_precision = tp / (tp + fp) if tp + fp else float(nothing_at_all)
        expected_recall = tp / (tp + fn + fd) if tp + fn + fd else 1.0
        rate_sum = expected_precision + expected_recall
        expected_f1 = (
            2 * expected_precision * expected_recall / rate_sum if rate_sum else 0.0
        )

        assert result["precision"] == pytest.approx(expected_precision)
        assert result["recall"] == pytest.approx(expected_recall)
        assert result["f1"] == pytest.approx(expected_f1)


class TestTheFastPathAgreesWithTheGeneralPath:
    """One item on each side must classify like the same pair among many.

    ``calculate_metrics`` skips the assignment algorithm when both sides hold
    one item. That shortcut already drifted away from the general path once, in
    #224. Adding two keys is a second chance to drift, so the agreement is
    pinned on all of them.
    """

    @pytest.mark.parametrize(
        "threshold", [0.0, 0.1, 0.5, 0.66, 0.6666666666666667, 0.7, 0.9, 1.0]
    )
    def test_the_two_paths_report_the_same_counts(self, threshold):
        """The extra pair is identical on both sides, so it is always a TP.

        Subtracting that known pair leaves the shared pair on its own, and the
        two paths must say the same thing about it.
        """
        one = _matcher(threshold).calculate_metrics(["abc"], ["abd"])
        many = _matcher(threshold).calculate_metrics(["abc", "QQQ"], ["abd", "QQQ"])

        assert one["tp"] == many["tp"] - 1
        assert one["fd"] == many["fd"]
        assert one["fn"] == many["fn"] == 0
        assert one["fa"] == many["fa"] == 0


class TestTheThresholdBoundaryIsInclusive:
    """A score equal to the threshold is a TP. Just below it is an FD."""

    def test_a_score_equal_to_the_threshold_is_a_true_positive(self):
        """``abc`` against ``abd`` scores two thirds, exactly the threshold."""
        result = _matcher(0.6666666666666667).calculate_metrics(["abc"], ["abd"])

        assert result["matched_pairs"][0][2] == pytest.approx(2 / 3)
        assert result["tp"] == 1
        assert result["fd"] == 0

    def test_a_score_just_below_the_threshold_is_a_false_discovery(self):
        result = _matcher(0.6666666666666667 + 1e-9).calculate_metrics(["abc"], ["abd"])

        assert result["tp"] == 0
        assert result["fd"] == 1
        assert result["fn"] == 0


class TestTheValuesThatMustNotChange:
    """A guard on the rest of the contract. These passed before the fix too.

    #231 changes the meaning of one key. Every other value in the returned dict
    stays what it was, and these assertions are what make that claim checkable
    rather than asserted in prose. Read a failure here as a change in scope,
    not as a test to update.
    """

    @pytest.mark.parametrize("ground_truth, prediction", PLAIN_SHAPES)
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_fp_stays_the_number_of_predictions_that_are_not_true_positives(
        self, ground_truth, prediction, threshold
    ):
        n = len(prediction)
        result = _matcher(threshold).calculate_metrics(ground_truth, prediction)

        assert result["fp"] == n - result["tp"]

    @pytest.mark.parametrize("ground_truth, prediction", PLAIN_SHAPES)
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_matched_pairs_holds_one_entry_per_assigned_pair(
        self, ground_truth, prediction, threshold
    ):
        m, n = len(ground_truth), len(prediction)
        result = _matcher(threshold).calculate_metrics(ground_truth, prediction)

        assert len(result["matched_pairs"]) == min(m, n)

    def test_two_empty_lists_agree_on_everything(self):
        """Nothing to find and nothing found, so both rates are one."""
        result = _matcher(0.5).calculate_metrics([], [])

        assert result["matched_pairs"] == []
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_an_empty_ground_truth_makes_every_prediction_wrong(self):
        result = _matcher(0.5).calculate_metrics([], _pred(2))

        assert result["tp"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 1.0, "there was nothing to miss"
        assert result["f1"] == 0.0

    def test_an_empty_prediction_misses_every_ground_truth_item(self):
        result = _matcher(0.5).calculate_metrics(_gt(2), [])

        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 2
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_a_perfect_match_is_all_true_positives(self):
        result = _matcher(0.5).calculate_metrics(
            ["apple", "banana"], ["banana", "apple"]
        )

        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
