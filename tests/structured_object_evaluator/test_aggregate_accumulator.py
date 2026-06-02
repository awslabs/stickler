"""Unit tests for AggregateConfusionMatrixAccumulator.

Tests the accumulator in isolation, without BulkStructuredModelEvaluator,
by feeding hand-crafted comparison_result dicts and asserting on the
returned compute() output / state shape.
"""

import json

import pytest

from stickler.structured_object_evaluator.models.aggregate.accumulator import (
    AggregateConfusionMatrixAccumulator,
)
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)


# ── Helpers ──


def _counts(tp=0, fp=0, fn=0, fa=0, fd=0, tn=0):
    return {"tp": tp, "fp": fp, "fn": fn, "fa": fa, "fd": fd, "tn": tn}


def _make_result(overall_counts, fields=None):
    """Build a comparison_result dict shaped like ConfusionMatrixBuilder output."""
    cm = {"aggregate": dict(overall_counts)}
    if fields is not None:
        cm["fields"] = fields
    return {
        "overall_score": 1.0,
        "confusion_matrix": cm,
    }


# ── 1. ABC conformance ──


class TestConformsToABC:
    def test_instantiates_without_args_and_has_name(self):
        acc = AggregateConfusionMatrixAccumulator()
        assert isinstance(acc, PostComparisonAccumulator)
        assert acc.name == "aggregate_metrics"


# ── 2. reset clears state ──


class TestResetClearsState:
    def test_reset_after_accumulate_returns_none(self):
        acc = AggregateConfusionMatrixAccumulator()
        acc.accumulate(_make_result(_counts(tp=3, fp=1)), None)
        assert acc.compute() is not None  # sanity
        acc.reset()
        assert acc.compute() is None


# ── 3. empty compute returns None ──


class TestEmptyComputeReturnsNone:
    def test_compute_with_no_documents(self):
        acc = AggregateConfusionMatrixAccumulator()
        assert acc.compute() is None


# ── 4. single doc overall ──


class TestSingleDocumentOverall:
    def test_single_doc_counts_match_input(self):
        acc = AggregateConfusionMatrixAccumulator()
        counts = _counts(tp=5, fp=2, fn=1, fa=0, fd=1, tn=4)
        acc.accumulate(_make_result(counts), None)

        result = acc.compute()
        assert result is not None
        overall = result["overall"]
        for key, expected in counts.items():
            assert overall[key] == expected
        # Derived metrics must be present too.
        assert "derived" in overall


# ── 5. multi-doc overall sums ──


class TestMultiDocumentOverallSums:
    def test_three_docs_summed_elementwise(self):
        acc = AggregateConfusionMatrixAccumulator()
        a = _counts(tp=1, fp=2, fn=3, fa=0, fd=0, tn=4)
        b = _counts(tp=5, fp=0, fn=1, fa=2, fd=1, tn=0)
        c = _counts(tp=0, fp=3, fn=0, fa=1, fd=2, tn=7)

        for d in (a, b, c):
            acc.accumulate(_make_result(d), None)

        overall = acc.compute()["overall"]
        for key in ("tp", "fp", "fn", "fa", "fd", "tn"):
            assert overall[key] == a[key] + b[key] + c[key]


# ── 6. nested per-field paths flatten correctly ──


class TestPerFieldPathsFlattenCorrectly:
    def test_two_level_dotted_path(self):
        acc = AggregateConfusionMatrixAccumulator()
        fields = {
            "parent": {
                "aggregate": _counts(tp=1),
                "fields": {
                    "child": {
                        "aggregate": _counts(tp=2, fp=1),
                    },
                },
            },
        }
        acc.accumulate(_make_result(_counts(tp=3, fp=1), fields=fields), None)

        result = acc.compute()
        paths = result["fields"]
        assert "parent" in paths
        assert "parent.child" in paths
        assert paths["parent.child"]["tp"] == 2
        assert paths["parent.child"]["fp"] == 1
        assert paths["parent"]["tp"] == 1


# ── 7. three-level nesting ──


class TestPerFieldPathNestedTwoLevels:
    def test_three_level_dotted_path(self):
        acc = AggregateConfusionMatrixAccumulator()
        fields = {
            "a": {
                "aggregate": _counts(tp=1),
                "fields": {
                    "b": {
                        "aggregate": _counts(tp=2),
                        "fields": {
                            "c": {
                                "aggregate": _counts(tp=4, fp=2),
                            },
                        },
                    },
                },
            },
        }
        acc.accumulate(_make_result(_counts(tp=7, fp=2), fields=fields), None)

        paths = acc.compute()["fields"]
        assert "a" in paths
        assert "a.b" in paths
        assert "a.b.c" in paths
        assert paths["a.b.c"]["tp"] == 4
        assert paths["a.b.c"]["fp"] == 2


# ── 8. derived metrics computed at every level ──


class TestDerivedMetricsRecomputedAtEachLevel:
    def test_derived_keys_present_overall_and_per_field(self):
        acc = AggregateConfusionMatrixAccumulator()
        fields = {
            "name": {"aggregate": _counts(tp=2, fp=1)},
            "price": {"aggregate": _counts(tp=1, fn=1)},
        }
        acc.accumulate(_make_result(_counts(tp=3, fp=1, fn=1), fields=fields), None)

        result = acc.compute()
        expected_keys = {"cm_precision", "cm_recall", "cm_f1", "cm_accuracy"}

        # overall derived
        overall_derived = result["overall"]["derived"]
        assert expected_keys.issubset(set(overall_derived.keys()))

        # per-field derived
        for path in ("name", "price"):
            assert "derived" in result["fields"][path]
            assert expected_keys.issubset(set(result["fields"][path]["derived"].keys()))


# ── 9. handles missing confusion matrix ──


class TestHandlesMissingConfusionMatrix:
    def test_accumulate_without_confusion_matrix_does_not_crash_or_contribute(self):
        acc = AggregateConfusionMatrixAccumulator()
        # No "confusion_matrix" key at all.
        acc.accumulate({"overall_score": 0.5}, None)
        assert acc.compute() is None

    def test_confusion_matrix_none_does_not_crash(self):
        acc = AggregateConfusionMatrixAccumulator()
        acc.accumulate({"overall_score": 0.5, "confusion_matrix": None}, None)
        assert acc.compute() is None


# ── 10. handles missing aggregate key ──


class TestHandlesMissingAggregateKey:
    def test_legacy_confusion_matrix_without_aggregate(self):
        acc = AggregateConfusionMatrixAccumulator()
        # confusion_matrix exists but has no "aggregate" key and no "fields".
        acc.accumulate(
            {"overall_score": 1.0, "confusion_matrix": {"some_other_key": 1}},
            None,
        )
        # No data was accumulated, so compute() should return None.
        assert acc.compute() is None

    def test_confusion_matrix_with_only_fields_no_top_aggregate(self):
        acc = AggregateConfusionMatrixAccumulator()
        # Has fields but no top-level aggregate; fields data should still be picked up.
        acc.accumulate(
            {
                "overall_score": 1.0,
                "confusion_matrix": {
                    "fields": {"x": {"aggregate": _counts(tp=2)}},
                },
            },
            None,
        )
        result = acc.compute()
        assert result is not None
        # Top-level aggregate is all zeros (nothing added there).
        assert result["overall"]["tp"] == 0
        # But the per-field path must have been recorded.
        assert result["fields"]["x"]["tp"] == 2


# ── 11. state round-trip ──


class TestStateRoundTrip:
    def test_get_state_jsonable_then_load_state(self):
        acc1 = AggregateConfusionMatrixAccumulator()
        acc1.accumulate(
            _make_result(
                _counts(tp=3, fp=1, fn=2, fa=1, fd=0, tn=5),
                fields={
                    "a": {"aggregate": _counts(tp=1)},
                    "b": {
                        "aggregate": _counts(tp=2, fp=1),
                        "fields": {"c": {"aggregate": _counts(tp=1)}},
                    },
                },
            ),
            None,
        )
        acc1.accumulate(_make_result(_counts(tp=2, fn=1)), None)

        before = acc1.compute()
        # Round-trip through JSON to prove the state is serializable.
        state_json = json.dumps(acc1.get_state())
        state = json.loads(state_json)

        acc2 = AggregateConfusionMatrixAccumulator()
        acc2.load_state(state)
        after = acc2.compute()

        assert after == before


# ── 12. merge_state is additive ──


class TestMergeStateIsAdditive:
    def test_two_accumulators_merge_sum_counts(self):
        acc1 = AggregateConfusionMatrixAccumulator()
        acc2 = AggregateConfusionMatrixAccumulator()

        acc1.accumulate(
            _make_result(
                _counts(tp=2, fp=1),
                fields={"shared": {"aggregate": _counts(tp=2)}},
            ),
            None,
        )
        acc2.accumulate(
            _make_result(
                _counts(tp=3, fp=2, fn=1),
                fields={
                    "shared": {"aggregate": _counts(tp=1, fp=1)},
                    "only_in_two": {"aggregate": _counts(tp=4)},
                },
            ),
            None,
        )

        acc1.merge_state(acc2.get_state())
        result = acc1.compute()

        # Overall: elementwise sum.
        assert result["overall"]["tp"] == 2 + 3
        assert result["overall"]["fp"] == 1 + 2
        assert result["overall"]["fn"] == 0 + 1

        # Shared field: counts add.
        assert result["fields"]["shared"]["tp"] == 2 + 1
        assert result["fields"]["shared"]["fp"] == 0 + 1

        # Field that only exists in the peer's state shows up after merge.
        assert result["fields"]["only_in_two"]["tp"] == 4


class TestRecallWithFdParameter:
    """``recall_with_fd`` ctor flag controls derived recall/F1 formulas."""

    @staticmethod
    def _accumulate_one(acc):
        # Counts chosen so the formula difference is observable:
        #   tp=2, fn=1, fd=2  -> recall_with_fd=False  ->  2/(2+1)   = 0.667
        #                       recall_with_fd=True   ->  2/(2+1+2) = 0.4
        acc.accumulate(_make_result(_counts(tp=2, fn=1, fd=2)), None)

    def test_default_uses_textbook_recall(self):
        acc = AggregateConfusionMatrixAccumulator()
        self._accumulate_one(acc)
        result = acc.compute()
        derived = result["overall"]["derived"]
        # Textbook recall: TP / (TP + FN) = 2 / 3
        assert derived["cm_recall"] == pytest.approx(2 / 3)

    def test_opt_in_uses_include_fd_recall(self):
        acc = AggregateConfusionMatrixAccumulator(recall_with_fd=True)
        self._accumulate_one(acc)
        result = acc.compute()
        derived = result["overall"]["derived"]
        # Include-FD recall: TP / (TP + FN + FD) = 2 / 5
        assert derived["cm_recall"] == pytest.approx(2 / 5)

    def test_recall_with_fd_propagates_to_per_field_derived(self):
        acc = AggregateConfusionMatrixAccumulator(recall_with_fd=True)
        acc.accumulate(
            _make_result(
                _counts(tp=2, fn=1, fd=2),
                fields={"x": {"aggregate": _counts(tp=2, fn=1, fd=2)}},
            ),
            None,
        )
        result = acc.compute()
        # Same formula must apply at every aggregate level, per docs.
        assert result["fields"]["x"]["derived"]["cm_recall"] == pytest.approx(2 / 5)

    def test_recall_with_fd_does_not_change_raw_counts(self):
        # The flag only affects derived metrics. Raw counts must be identical
        # between the two flavors so a downstream consumer can recompute.
        acc_a = AggregateConfusionMatrixAccumulator(recall_with_fd=False)
        acc_b = AggregateConfusionMatrixAccumulator(recall_with_fd=True)
        for acc in (acc_a, acc_b):
            self._accumulate_one(acc)
        cm_keys = ("tp", "fp", "fn", "fa", "fd", "tn")
        assert {k: acc_a.compute()["overall"][k] for k in cm_keys} == {
            k: acc_b.compute()["overall"][k] for k in cm_keys
        }
