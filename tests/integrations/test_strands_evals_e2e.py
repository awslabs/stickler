"""End to end against the real strands-agents-evals evaluator, offline.

Drives `StructuredOutputSimilarity` through a strands_evals `Experiment` with
`report_cls=StructuredOutputReport`, using a plain task function that returns
pydantic instances: no agent, no model, no network. Skipped unless strands_evals
is installed; CI installs `strands-agents-evals==1.6.0` without its `stickler`
extra, so the stickler under test is this checkout.
"""

import importlib
import json

import pytest

strands_evals = pytest.importorskip("strands_evals")
_evaluators = importlib.import_module("strands_evals.evaluators")
if not hasattr(_evaluators, "StructuredOutputSimilarity") or not hasattr(
    _evaluators, "StructuredOutputReport"
):
    pytest.skip(
        "this strands_evals has no StructuredOutputSimilarity (needs >=1.6.0)",
        allow_module_level=True,
    )

from strands_evals import Case, Experiment  # noqa: E402
from strands_evals.types.evaluation import EvaluationData  # noqa: E402

from stickler import aggregate_from_comparisons, eval_for  # noqa: E402
from stickler.utils.process_evaluation import ProcessEvaluation  # noqa: E402
from tests.integrations.strands_corpus import (  # noqa: E402
    CORPUS,
    MODELS,
    STRANDS_DEFAULT_THRESHOLD,
    Contact,
    Invoice,
    stored_comparison,
    strands_test_pass,
)
from tests.integrations.test_strands_evals_snapshot import SNAPSHOT_PATH  # noqa: E402

StructuredOutputReport = _evaluators.StructuredOutputReport
StructuredOutputSimilarity = _evaluators.StructuredOutputSimilarity

MARKER = "StructuredOutputSimilarity"
PER_CASE_KEYS = {
    "case",
    "evaluator",
    "model",
    "overall_score",
    "test_pass",
    "matched",
    "precision",
    "recall",
    "f1",
    "field_scores",
}
CASE_IDS = [case.id for case in CORPUS]


@pytest.fixture(scope="module", autouse=True)
def _offline():
    # conftest.py checks that the stickler under test is this checkout.
    with pytest.MonkeyPatch.context() as mp:
        # The only network path in Experiment: CloudWatch logging when set.
        mp.delenv("AGENT_OBSERVABILITY_ENABLED", raising=False)
        yield


def _run(model, cases, **kwargs):
    actuals = {case.id: case.actual for case in cases}
    experiment = Experiment(
        cases=[
            Case(name=case.id, input=case.id, expected_output=case.expected)
            for case in cases
        ],
        evaluators=[StructuredOutputSimilarity(model, **kwargs)],
        report_cls=StructuredOutputReport,
    )
    return experiment.run_evaluations(lambda case: actuals[case.name])


@pytest.fixture(scope="module")
def reports(_offline):
    return {
        model: _run(model, [case for case in CORPUS if case.model is model])
        for model in MODELS
    }


def _row(reports, case):
    report = reports[case.model]
    index = [c["name"] for c in report.cases].index(case.id)
    (row,) = report.detailed_results[index]
    return row


def _direct(case, threshold=STRANDS_DEFAULT_THRESHOLD, weight_hints=False):
    spec = eval_for(case.model, match_threshold=threshold, weight_hints=weight_hints)
    return spec.evaluate(case.expected, case.actual)


def _case(case_id):
    return next(case for case in CORPUS if case.id == case_id)


# --- every case -----------------------------------------------------------------


class TestEveryCase:
    def test_report_type(self, reports):
        for report in reports.values():
            assert type(report) is StructuredOutputReport

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_row_was_scored_not_isolated(self, reports, case):
        # Experiment turns an exception into a score-0 row with no metadata,
        # so a stickler crash would otherwise read as a failing case.
        row = _row(reports, case)
        assert row.metadata is not None
        assert row.metadata.get("evaluated_by") == MARKER
        assert "error" not in row.metadata
        assert row.label == case.model.__name__

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_row_agrees_with_stickler_directly(self, reports, case):
        row = _row(reports, case)
        result = _direct(case)
        assert row.score == result.overall_score
        assert row.metadata["field_scores"] == result.field_scores
        assert row.metadata["matched"] == result.matched
        for name in ("precision", "recall", "f1"):
            assert row.metadata[name] == getattr(result, name), name
        assert row.metadata["comparison"] == stored_comparison(result)

    @pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
    def test_verdict_matches_the_reimplemented_guard(self, reports, case):
        # Proves strands_corpus.strands_test_pass is a faithful copy of 1.6.0.
        assert _row(reports, case).test_pass is strands_test_pass(
            _direct(case), STRANDS_DEFAULT_THRESHOLD
        )

    def test_verdicts_match_the_snapshot(self, reports):
        snapshot = json.loads(SNAPSHOT_PATH.read_text())["cases"]
        for case in CORPUS:
            if case.id in snapshot:
                assert (
                    _row(reports, case).test_pass is snapshot[case.id]["test_pass"]
                ), case.id


# --- behaviour --------------------------------------------------------------------


class TestBehaviour:
    @pytest.mark.parametrize(
        "case_id",
        ["contact/identical", "invoice/identical", "lead/full_identical"],
    )
    def test_identical_scores_one_and_passes(self, reports, case_id):
        row = _row(reports, _case(case_id))
        assert row.score == 1.0
        assert row.test_pass is True
        assert row.reason == "all fields matched"

    def test_wrong_field_lowers_score_and_is_named(self, reports):
        row = _row(reports, _case("contact/one_wrong_str"))
        assert row.score < 1.0
        assert "email=0.00" in row.reason

    def test_wrong_nested_list_is_named(self, reports):
        row = _row(reports, _case("invoice/line_item_missing"))
        assert row.score < 1.0
        assert "line_items=" in row.reason

    def test_empty_prediction_fails_via_the_guard(self, reports):
        row = _row(reports, _case("lead/sparse_empty_prediction"))
        assert row.score >= STRANDS_DEFAULT_THRESHOLD
        assert row.metadata["matched"] is True
        assert row.metadata["recall"] == 0.0
        assert row.test_pass is False

    def test_nothing_to_find_passes(self, reports):
        row = _row(reports, _case("lead/nothing_to_find_empty"))
        assert row.score == 1.0
        assert row.test_pass is True

    def test_nothing_to_find_with_invention_fails(self, reports):
        assert _row(reports, _case("lead/nothing_to_find_invented")).test_pass is False

    @pytest.mark.parametrize("threshold", [0.5, 0.9])
    def test_threshold_reaches_stickler(self, threshold):
        cases = [case for case in CORPUS if case.model is Invoice]
        report = _run(Invoice, cases, match_threshold=threshold)
        for case, rows in zip(cases, report.detailed_results):
            result = _direct(case, threshold)
            assert rows[0].score == result.overall_score, case.id
            assert rows[0].test_pass is strands_test_pass(result, threshold), case.id

    def test_weight_hints_reach_stickler(self):
        case = _case("invoice/wrong_total")
        report = _run(Invoice, [case], weight_hints=True)
        result = _direct(case, weight_hints=True)
        assert report.detailed_results[0][0].score == result.overall_score

    @pytest.mark.parametrize("form", ["dict", "json"])
    def test_dict_and_json_outputs_score_as_the_instance(self, form):
        case = _case("invoice/line_item_replaced")
        dumped = case.actual.model_dump(mode="json")
        actual = dumped if form == "dict" else json.dumps(dumped)
        evaluator = StructuredOutputSimilarity(Invoice)
        (row,) = evaluator.evaluate(
            EvaluationData(
                input=case.id, expected_output=case.expected, actual_output=actual
            )
        )
        assert row.score == _direct(case).overall_score


# --- report rollups ---------------------------------------------------------------


class TestReport:
    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_per_case(self, reports, model):
        cases = [case for case in CORPUS if case.model is model]
        table = reports[model].per_case()
        assert [entry["case"] for entry in table] == [case.id for case in cases]
        for case, entry in zip(cases, table):
            assert set(entry) == PER_CASE_KEYS
            result = _direct(case)
            assert entry["evaluator"] == MARKER
            assert entry["model"] == model.__name__
            assert entry["overall_score"] == result.overall_score
            assert entry["matched"] == result.matched
            assert entry["precision"] == result.precision
            assert entry["recall"] == result.recall
            assert entry["f1"] == result.f1
            assert entry["field_scores"] == result.field_scores
            assert entry["test_pass"] is strands_test_pass(
                result, STRANDS_DEFAULT_THRESHOLD
            )

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_metrics_agree_with_stickler_directly(self, reports, model):
        cases = [case for case in CORPUS if case.model is model]
        evaluation = reports[model].metrics()
        assert type(evaluation) is ProcessEvaluation
        direct = aggregate_from_comparisons(
            [stored_comparison(_direct(case)) for case in cases]
        )
        assert evaluation.document_count == len(cases)
        assert evaluation.field_metrics == direct.field_metrics
        assert evaluation.metrics == direct.metrics
        assert set(evaluation.field_metrics) == set(eval_for(model).explain())

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"match_threshold": 0.9}, {"weight_hints": True}],
        ids=["default", "threshold", "weight_hints"],
    )
    def test_explain_is_stickler_explain(self, model, kwargs):
        evaluator = StructuredOutputSimilarity(model, **kwargs)
        expected = eval_for(
            model,
            match_threshold=kwargs.get("match_threshold", STRANDS_DEFAULT_THRESHOLD),
            weight_hints=kwargs.get("weight_hints", False),
        ).explain()
        assert evaluator.explain() == expected

    @pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
    def test_json_round_trip_preserves_rollups(self, reports, model):
        report = reports[model]
        reloaded = StructuredOutputReport.model_validate_json(report.model_dump_json())
        before, after = report.metrics(), reloaded.metrics()
        assert after.field_metrics == before.field_metrics
        assert after.metrics == before.metrics
        assert after.document_count == before.document_count
        assert reloaded.per_case() == report.per_case()

    @pytest.mark.xfail(
        reason="strands-agents-evals 1.6.0: EvaluationReport.to_file json.dumps the "
        "case data, which fails on datetime.date values (an evals-side issue)",
        raises=TypeError,
        strict=False,
    )
    def test_to_file_round_trip(self, reports, tmp_path):
        report = reports[Contact]
        report.to_file(str(tmp_path / "report.json"))
        reloaded = StructuredOutputReport.from_file(str(tmp_path / "report.json"))
        assert reloaded.metrics().field_metrics == report.metrics().field_metrics
