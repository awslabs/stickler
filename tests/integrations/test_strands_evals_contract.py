"""The stickler surface strands-agents-evals 1.6.0 reads, pinned without strands.

`StructuredOutputSimilarity` and `StructuredOutputReport`
(strands_evals/evaluators/deterministic/structured_output.py in the 1.6.0 wheel)
score with stickler. Their suite pins a released stickler, so only a test here
fails on the commit that breaks them. These tests need no strands install and
run in every CI job.

The evaluator reads `EvalResult.raw` with `.get()` defaults, so a renamed key
degrades its verdict to "always pass" instead of raising. That is why these
tests assert that keys EXIST, rather than that a lookup succeeds.

Line numbers below refer to structured_output.py in strands-agents-evals 1.6.0.
"""

import inspect
import json
import math
import warnings
from typing import Any, Dict, List

import pytest

from tests.integrations.strands_corpus import (
    CORPUS,
    MODELS,
    STRANDS_DEFAULT_THRESHOLD,
    Contact,
    Invoice,
    Lead,
    Order,
    stored_comparison,
    strands_test_pass,
)

COUNT_KEYS = ("tp", "fa", "fd", "fp", "tn", "fn")
DERIVED_KEYS = ("cm_precision", "cm_recall", "cm_f1", "cm_accuracy")
EXPLAIN_KEYS = {
    "comparator",
    "threshold",
    "weight",
    "clip_under_threshold",
    "source",
    "why",
}


def _spec(model, threshold=STRANDS_DEFAULT_THRESHOLD, weight_hints=False):
    from stickler import eval_for

    # The exact call at line 344.
    return eval_for(model, match_threshold=threshold, weight_hints=weight_hints)


def _evaluate(case, threshold=STRANDS_DEFAULT_THRESHOLD):
    return _spec(case.model, threshold).evaluate(case.expected, case.actual)


def _case(case_id):
    return next(case for case in CORPUS if case.id == case_id)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_unit_float(value: Any) -> bool:
    return (
        isinstance(value, float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _aggregate(stored: List[Dict[str, Any]]):
    """aggregate_from_comparisons as line 209 calls it, with warnings as errors."""
    from stickler import aggregate_from_comparisons

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return aggregate_from_comparisons(stored)


CASE_IDS = [case.id for case in CORPUS]


# --- import paths -----------------------------------------------------------


class TestImportPaths:
    def test_top_level_names_the_evaluator_imports(self):
        # Line 47, imported lazily on first use.
        from stickler import aggregate_from_comparisons, eval_for

        assert callable(eval_for)
        assert callable(aggregate_from_comparisons)

    def test_process_evaluation_type_path(self):
        # Line 25, the return annotation of StructuredOutputReport.metrics.
        from stickler.utils.process_evaluation import ProcessEvaluation

        assert inspect.isclass(ProcessEvaluation)


# --- eval_for and EvalSpec ----------------------------------------------------


class TestEvalForSignature:
    def test_accepts_the_kwargs_the_evaluator_passes(self):
        from stickler import eval_for

        params = inspect.signature(eval_for).parameters
        for name in ("match_threshold", "weight_hints"):
            assert name in params
            assert params[name].kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        inspect.signature(eval_for).bind(
            Invoice, match_threshold=0.7, weight_hints=False
        )

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    @pytest.mark.parametrize("weight_hints", [False, True])
    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 1.0])
    def test_builds_a_spec_for_every_corpus_model(self, model, weight_hints, threshold):
        spec = _spec(model, threshold, weight_hints)
        # Line 423 calls evaluate(expected, actual) positionally; line 459 explain().
        inspect.signature(spec.evaluate).bind(object(), object())
        inspect.signature(spec.explain).bind()

    def test_weight_hints_true_still_scores_identical_as_one(self):
        case = _case("invoice/identical")
        result = _spec(Invoice, weight_hints=True).evaluate(case.expected, case.actual)
        assert result.overall_score == pytest.approx(1.0)


class TestExplain:
    """Line 459 returns spec.explain() to users verbatim."""

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_entry_shape(self, model):
        explained = _spec(model).explain()
        assert isinstance(explained, dict) and explained
        for path, entry in explained.items():
            assert isinstance(path, str)
            assert EXPLAIN_KEYS <= set(entry), path
            assert isinstance(entry["comparator"], str) and entry["comparator"]
            assert isinstance(entry["source"], str) and entry["source"]
            assert isinstance(entry["threshold"], float)
            assert isinstance(entry["weight"], float)
            assert isinstance(entry["clip_under_threshold"], bool)
            assert isinstance(entry["why"], list)
            assert all(isinstance(step, str) for step in entry["why"])

    def test_keyed_by_dotted_path(self):
        assert set(_spec(Invoice).explain()) == {
            "invoice_number",
            "issue_date",
            "total",
            "currency",
            "vendor",
            "vendor.street",
            "vendor.city",
            "vendor.postal_code",
            "ship_to",
            "ship_to.street",
            "ship_to.city",
            "ship_to.postal_code",
            "tags",
            "line_items",
            "line_items.sku",
            "line_items.description",
            "line_items.quantity",
            "line_items.unit_price",
            "credits",
            "credits.sku",
            "credits.description",
            "credits.quantity",
            "credits.unit_price",
        }

    def test_list_inside_list_paths(self):
        assert set(_spec(Order).explain()) == {
            "order_id",
            "placed_on",
            "shipments",
            "shipments.tracking_id",
            "shipments.carrier",
            "shipments.items",
            "shipments.items.sku",
            "shipments.items.description",
            "shipments.items.quantity",
            "shipments.items.unit_price",
        }

    def test_aliased_field_is_keyed_by_field_name(self):
        explained = _spec(Contact).explain()
        assert "email" in explained
        assert "emailAddress" not in explained


# --- EvalResult ---------------------------------------------------------------


class TestEvalResultShape:
    """Every attribute the evaluator reads (lines 140-147, 424-449)."""

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_scalar_attributes(self, case):
        result = _evaluate(case)
        assert _is_unit_float(result.overall_score)
        assert isinstance(result.matched, bool)
        # Never None: line 147 would read None as a failed gate.
        for name in ("precision", "recall", "f1"):
            assert _is_unit_float(getattr(result, name)), name

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_field_scores_keyed_by_top_level_field(self, case):
        result = _evaluate(case)
        # Line 438 builds the reason from these; line 442 stores dict(...).
        assert isinstance(result.field_scores, dict)
        assert set(result.field_scores) == set(case.model.model_fields)
        assert all(_is_unit_float(v) for v in result.field_scores.values())

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_raw_overall_counts_exist(self, case):
        raw = _evaluate(case).raw
        assert isinstance(raw, dict)
        assert "confusion_matrix" in raw
        assert "overall" in raw["confusion_matrix"]
        overall = raw["confusion_matrix"]["overall"]
        # The guard reads tp, fn, fd and fa (lines 144-146). Assert presence,
        # since a missing key reads as 0 there and the gate passes.
        for key in COUNT_KEYS:
            assert key in overall, key
            assert _is_int(overall[key]) and overall[key] >= 0, key
        assert overall["fp"] == overall["fa"] + overall["fd"]

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_attributes_agree_with_raw(self, case):
        result = _evaluate(case)
        overall = result.raw["confusion_matrix"]["overall"]
        derived = overall["derived"]
        # The report's aggregate reads raw; per_case reads the attributes.
        assert result.precision == derived["cm_precision"]
        assert result.recall == derived["cm_recall"]
        assert result.f1 == derived["cm_f1"]
        assert result.overall_score == result.raw["overall_score"]
        assert result.field_scores == result.raw["field_scores"]

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_recall_excludes_wrong_values(self, case):
        # The 1.6.0 docstring (lines 136-138) relies on this: a wrong value is
        # fd, not fn, so recall measures only what was left empty.
        result = _evaluate(case)
        overall = result.raw["confusion_matrix"]["overall"]
        denominator = overall["tp"] + overall["fn"]
        expected = overall["tp"] / denominator if denominator else 0.0
        assert result.recall == pytest.approx(expected)

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 1.0])
    def test_matched_is_score_against_threshold(self, case, threshold):
        result = _evaluate(case, threshold)
        assert result.matched == (result.overall_score >= threshold)

    def test_recall_is_zero_not_none_with_nothing_to_find(self):
        case = _case("lead/nothing_to_find_empty")
        result = _evaluate(case)
        overall = result.raw["confusion_matrix"]["overall"]
        assert overall["tp"] + overall["fn"] == 0
        assert result.recall == 0.0
        assert result.recall is not None

    def test_findable_is_positive_when_ground_truth_has_values(self):
        overall = _evaluate(_case("lead/sparse_empty_prediction")).raw[
            "confusion_matrix"
        ]["overall"]
        assert overall["tp"] + overall["fn"] + overall["fd"] > 0


class TestStoredComparison:
    """The per-case `metadata["comparison"]` (line 448) and its JSON form."""

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_is_plain_json(self, case):
        stored = stored_comparison(_evaluate(case))
        for key in ("overall_score", "field_scores", "confusion_matrix"):
            assert key in stored, key
        # model_dump_json and from_file carry it as JSON, so it must survive
        # the standard library's encoder with no fallback.
        assert json.loads(json.dumps(stored)) == stored


# --- the nothing-to-find guard ---------------------------------------------------

#: Expected 1.6.0 verdicts that exercise each branch of `_passed`.
GUARD_VERDICTS = {
    # Identical documents pass, whatever their shape.
    "contact/identical": True,
    "invoice/identical": True,
    "order/identical": True,
    "lead/full_identical": True,
    "lead/sparse_identical": True,
    # Something to find, nothing found: fails, though the sparse one is matched.
    "lead/sparse_empty_prediction": False,
    "lead/full_empty_prediction": False,
    # Recall below the threshold fails even with a matched score.
    "lead/sparse_half_found": False,
    # Score below the threshold fails before recall is read.
    "contact/all_wrong": False,
    "invoice/all_wrong": False,
    "invoice/lists_emptied": False,
    # Nothing to find: passes on score unless the prediction invented values.
    "lead/nothing_to_find_empty": True,
    "lead/nothing_to_find_invented": False,
}


class TestGuard:
    @pytest.mark.parametrize("case_id,expected", sorted(GUARD_VERDICTS.items()))
    def test_verdict(self, case_id, expected):
        assert strands_test_pass(_evaluate(_case(case_id)), 0.7) is expected

    def test_sparse_empty_prediction_is_matched_but_fails(self):
        # The case the guard exists for: blank-on-both fields lift the score
        # over the bar, but nothing populated was found.
        result = _evaluate(_case("lead/sparse_empty_prediction"))
        assert result.matched is True
        assert result.recall == 0.0
        assert strands_test_pass(result, 0.7) is False

    def test_nothing_to_find_passes_on_score(self):
        result = _evaluate(_case("lead/nothing_to_find_empty"))
        overall = result.raw["confusion_matrix"]["overall"]
        assert overall["tp"] + overall["fn"] + overall["fd"] == 0
        assert overall["fa"] == 0
        assert result.overall_score == 1.0
        assert strands_test_pass(result, 0.7) is True

    def test_nothing_to_find_with_invention_fails(self):
        result = _evaluate(_case("lead/nothing_to_find_invented"))
        overall = result.raw["confusion_matrix"]["overall"]
        assert overall["tp"] + overall["fn"] + overall["fd"] == 0
        assert overall["fa"] > 0
        assert strands_test_pass(result, 0.7) is False

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_expected_against_itself_passes(self, case):
        result = _spec(case.model).evaluate(case.expected, case.expected)
        assert strands_test_pass(result, 0.7) is True

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_verdict_never_passes_unmatched(self, case):
        result = _evaluate(case)
        if not result.matched:
            assert strands_test_pass(result, 0.7) is False

    def test_a_renamed_key_would_silently_pass(self):
        # Why the shape tests above assert presence: with the subtree gone,
        # every `.get()` falls back to 0 and the empty prediction passes.
        result = _evaluate(_case("lead/sparse_empty_prediction"))
        result.raw = {k: v for k, v in result.raw.items() if k != "confusion_matrix"}
        assert strands_test_pass(result, 0.7) is True


# --- aggregate_from_comparisons ---------------------------------------------------


def _by_model():
    grouped: Dict[type, List[Any]] = {}
    for case in CORPUS:
        grouped.setdefault(case.model, []).append(_evaluate(case))
    return grouped


class TestAggregate:
    """StructuredOutputReport.metrics (line 209), one model class at a time."""

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_returns_process_evaluation_without_warnings(self, model):
        from stickler.utils.process_evaluation import ProcessEvaluation

        results = _by_model()[model]
        evaluation = _aggregate([stored_comparison(r) for r in results])
        assert type(evaluation) is ProcessEvaluation
        assert evaluation.document_count == len(results)

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_field_metrics_keyed_by_explain_paths(self, model):
        results = _by_model()[model]
        evaluation = _aggregate([stored_comparison(r) for r in results])
        assert set(evaluation.field_metrics) == set(_spec(model).explain())
        for path, entry in evaluation.field_metrics.items():
            for key in COUNT_KEYS:
                assert _is_int(entry[key]), (path, key)
            for key in DERIVED_KEYS:
                assert _is_unit_float(entry[key]), (path, key)

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_overall_metrics_sum_the_documents(self, model):
        results = _by_model()[model]
        evaluation = _aggregate([stored_comparison(r) for r in results])
        for key in COUNT_KEYS:
            assert evaluation.metrics[key] == sum(
                r.raw["confusion_matrix"]["overall"][key] for r in results
            ), key
        for key in DERIVED_KEYS:
            assert _is_unit_float(evaluation.metrics[key]), key

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_dropping_prediction_raw_changes_nothing(self, model):
        # Line 447: "does not affect field_metrics".
        from stickler import aggregate_from_comparisons

        results = _by_model()[model]
        stripped = _aggregate([stored_comparison(r) for r in results])
        with warnings.catch_warnings():
            # The unstripped raw warns (ConfidenceAccumulator sees prediction_raw
            # without field_comparisons). Stripping it, as 1.6.0 does, is silent;
            # see test_returns_process_evaluation_without_warnings.
            warnings.simplefilter("ignore")
            full = aggregate_from_comparisons([r.raw for r in results])
        assert stripped.field_metrics == full.field_metrics
        assert stripped.metrics == full.metrics

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_json_reloaded_comparisons_aggregate_the_same(self, model):
        # StructuredOutputReport.from_file(...).metrics() aggregates the JSON form.
        stored = [stored_comparison(r) for r in _by_model()[model]]
        in_memory = _aggregate(stored)
        reloaded = _aggregate(json.loads(json.dumps(stored)))
        assert reloaded.field_metrics == in_memory.field_metrics
        assert reloaded.metrics == in_memory.metrics
        assert reloaded.document_count == in_memory.document_count

    def test_nested_path_counts_only_matched_parents(self):
        # The metrics() docstring (lines 177-179) states this as behaviour.
        result = _evaluate(_case("invoice/line_item_replaced"))
        metrics = _aggregate([stored_comparison(result)]).field_metrics
        assert metrics["line_items"]["tp"] == 2
        assert metrics["line_items"]["fd"] == 1
        for leaf in ("sku", "description", "quantity", "unit_price"):
            entry = metrics[f"line_items.{leaf}"]
            assert sum(entry[k] for k in ("tp", "fa", "fd", "fn")) == 2, leaf

    def test_aliased_field_is_keyed_by_field_name(self):
        result = _evaluate(_case("contact/one_wrong_str"))
        metrics = _aggregate([stored_comparison(result)]).field_metrics
        assert metrics["email"]["fd"] == 1
        assert "emailAddress" not in metrics

    def test_sparse_documents_count_true_negatives(self):
        results = [_evaluate(_case("lead/nothing_to_find_empty"))]
        metrics = _aggregate([stored_comparison(r) for r in results]).metrics
        assert metrics["tn"] == len(Lead.model_fields)
        assert metrics["tp"] == metrics["fn"] == 0
