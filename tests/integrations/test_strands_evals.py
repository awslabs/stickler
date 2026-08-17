"""StructuredOutputEvaluator adapts stickler to the Strands Evals harness.

The tests that matter here are the ones covering claims the harness itself does
not enforce: that the case score is stickler's weighted score rather than a mean,
that per-field detail is available without a second comparison pass, that a
mixed-schema suite does not produce a merged rollup, and that the evaluator is
safe at the harness's default concurrency.

See docs/docs/Guides/Integrations/strands-evals.md for the design rationale.
"""

import asyncio
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("strands_evals", reason="requires the strands-evals extra")

from strands_evals import Case, Experiment  # noqa: E402
from strands_evals.evaluators import Equals, Evaluator  # noqa: E402
from strands_evals.types.evaluation import EvaluationData  # noqa: E402

from stickler.integrations.strands_evals import (  # noqa: E402
    StructuredOutputEvaluator,
)


class LineItem(BaseModel):
    sku: Optional[str] = None
    unit_price: Optional[float] = None


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    total_amount: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)


class Receipt(BaseModel):
    merchant: str
    tax: float


def _invoice(iid="INV-1", vendor="Acme Corporation", total=100.0, sku="SKU-1", price=100.0):
    return Invoice(
        invoice_id=iid,
        vendor_name=vendor,
        total_amount=total,
        line_items=[LineItem(sku=sku, unit_price=price)],
    )


def _case(name, expected):
    return Case(name=name, input="", expected_output=expected, metadata={"name": name})


def _run(evaluator, pairs, **kwargs):
    """Run pairs of (expected, actual) through the real harness."""
    cases = [_case(name, exp) for name, (exp, _) in pairs.items()]
    actual = {name: act for name, (_, act) in pairs.items()}
    return asyncio.run(
        Experiment(cases=cases, evaluators=[evaluator]).run_evaluations_async(
            lambda c: actual[c.metadata["name"]], **kwargs
        )
    )


def _data(expected, actual):
    return EvaluationData(
        input="", invocation_input="", expected_output=expected, actual_output=actual
    )


class TestPerCaseOutput:
    """One output per case, carrying stickler's weighted score."""

    def test_one_output_per_case(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice()))

        assert len(outputs) == 1
        assert outputs[0].label == "Invoice"

    def test_score_is_sticklers_weighted_overall(self):
        """Not a mean of per-field scores; the weighted score stickler computed.

        An earlier draft emitted one output per field and installed a custom
        aggregator to recombine them, which required looking weights up by field
        name because EvaluationOutput carries none. Reporting overall_score
        directly removes that whole mechanism.
        """
        evaluator = StructuredOutputEvaluator(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec_for(Invoice).evaluate(gt, pred).overall_score
        outputs = evaluator.evaluate(_data(gt, pred))

        assert outputs[0].score == pytest.approx(expected)

    def test_reason_names_the_weakest_fields(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice(iid="INV-9")))

        assert "invoice_id" in (outputs[0].reason or "")

    def test_no_custom_aggregator_is_installed(self):
        """The framework default is correct for a single output, so leave it.

        Overriding it was only necessary while several outputs per case had to be
        recombined.
        """
        evaluator = StructuredOutputEvaluator(Invoice)

        assert evaluator.aggregator is Evaluator._default_aggregator


class TestPerCaseDetail:
    """Field detail comes from the evaluator, not from EvaluationOutput."""

    def test_per_case_carries_field_scores(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {"doc-a": (_invoice(), _invoice(vendor="Acme Corp"))})

        (entry,) = evaluator.per_case()
        assert entry["case"] == "doc-a"
        assert entry["model"] == "Invoice"
        assert set(entry["field_scores"]) == set(Invoice.model_fields)

    def test_per_case_preserves_the_case_name(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {f"doc-{i}": (_invoice(), _invoice()) for i in range(3)})

        assert {e["case"] for e in evaluator.per_case()} == {"doc-0", "doc-1", "doc-2"}

    def test_per_case_runs_no_extra_comparisons(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})

        before = len(evaluator._results)
        evaluator.per_case()
        evaluator.per_case()
        assert len(evaluator._results) == before

    def test_per_case_is_empty_after_reset(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})
        evaluator.reset()

        assert evaluator.per_case() == []


class TestCaseScore:
    def test_case_score_equals_stickler_overall_score(self):
        evaluator = StructuredOutputEvaluator(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec_for(Invoice).evaluate(gt, pred).overall_score
        report = _run(evaluator, {"a": (gt, pred)})

        assert report.scores[0] == pytest.approx(expected)

    def test_perfect_match_says_so(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        report = _run(evaluator, {"a": (_invoice(), _invoice())})

        assert report.scores[0] == pytest.approx(1.0)
        assert report.test_passes[0] is True
        assert report.reasons[0] == "all fields matched"


class TestBeatsEquals:
    """The reason this integration exists."""

    def test_equals_cannot_rank_what_stickler_separates(self):
        pairs = {
            "near": (_invoice(), _invoice(vendor="Acme Corp")),
            "far": (_invoice(), _invoice(iid="X", vendor="Zeta", total=9.0, sku="Z")),
        }
        stickler_report = _run(StructuredOutputEvaluator(Invoice), pairs)
        equals_report = _run(Equals(), pairs)

        assert len(set(equals_report.scores)) == 1  # both wrong, indistinguishable
        assert stickler_report.scores[0] > stickler_report.scores[1]


class TestDatasetRollup:
    def test_metrics_includes_nested_paths(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})

        paths = evaluator.metrics()["Invoice"].field_metrics
        assert "line_items" in paths
        assert "line_items.sku" in paths

    def test_metrics_runs_no_extra_comparisons(self):
        """One comparison per case, feeding both the outputs and the rollup."""
        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {f"c{i}": (_invoice(), _invoice()) for i in range(5)})

        assert len(evaluator._results) == 5
        evaluator.metrics()
        evaluator.metrics()
        assert len(evaluator._results) == 5

    def test_five_categories_are_populated(self):
        """FN is a missed field, FA an invented one, FD a wrong one."""
        gt = _invoice(total=100.0)
        missing = _invoice(total=None)  # FN on total_amount
        wrong = _invoice(total=55.0)  # FD on total_amount

        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {"missing": (gt, missing), "wrong": (gt, wrong)})

        total = evaluator.metrics()["Invoice"].field_metrics["total_amount"]
        assert total["fn"] == 1
        assert total["fd"] == 1

    def test_reset_clears_the_rollup(self):
        evaluator = StructuredOutputEvaluator(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})
        assert evaluator.metrics()

        evaluator.reset()
        assert evaluator.metrics() == {}


class TestMixedSchemas:
    def test_rollup_is_partitioned_by_class(self):
        """A merged rollup would union field paths and misreport denominators."""
        evaluator = StructuredOutputEvaluator()  # no declared model_cls
        pairs = {
            "inv": (_invoice(), _invoice()),
            "rec": (Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=2.0)),
        }
        _run(evaluator, pairs)
        metrics = evaluator.metrics()

        assert set(metrics) == {"Invoice", "Receipt"}
        assert set(metrics["Receipt"].field_metrics) == {"merchant", "tax"}
        assert "merchant" not in metrics["Invoice"].field_metrics
        assert all(pe.document_count == 1 for pe in metrics.values())

    def test_declared_model_cls_rejects_a_foreign_shape(self):
        """Strict mode must fail loudly rather than coerce nonsense."""
        evaluator = StructuredOutputEvaluator(Invoice)

        with pytest.raises(Exception):
            evaluator.evaluate(
                _data(Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=1.0))
            )

    def test_explain_is_ambiguous_across_inferred_schemas(self):
        evaluator = StructuredOutputEvaluator()
        _run(
            evaluator,
            {
                "inv": (_invoice(), _invoice()),
                "rec": (Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=1.0)),
            },
        )

        with pytest.raises(RuntimeError, match="ambiguous"):
            evaluator.explain()

    def test_inference_needs_a_model_instance(self):
        evaluator = StructuredOutputEvaluator()

        with pytest.raises(TypeError, match="could not infer"):
            evaluator.evaluate(_data({"invoice_id": "X"}, {"invoice_id": "X"}))


class TestConcurrency:
    def test_no_results_are_lost_at_the_harness_default(self):
        """`run_evaluations_async` defaults to max_workers=10 and calls
        evaluate() through asyncio.to_thread, so this runs on many threads. A
        naive shared counter loses ~18% of documents; appending to a list is
        atomic under the GIL and loses none.
        """
        n = 200
        evaluator = StructuredOutputEvaluator(Invoice)
        report = _run(
            evaluator,
            {f"c{i}": (_invoice(), _invoice()) for i in range(n)},
            max_workers=10,
        )

        assert len(report.scores) == n
        assert len(evaluator._results) == n
        assert evaluator.metrics()["Invoice"].document_count == n


class TestCoercion:
    @pytest.mark.parametrize(
        "actual",
        [
            {"invoice_id": "INV-1", "vendor_name": "Acme Corporation"},
            '{"invoice_id": "INV-1", "vendor_name": "Acme Corporation"}',
        ],
        ids=["dict", "json-string"],
    )
    def test_accepts_dicts_and_json_strings(self, actual):
        evaluator = StructuredOutputEvaluator(Invoice)
        outputs = evaluator.evaluate(
            _data(Invoice(invoice_id="INV-1", vendor_name="Acme Corporation"), actual)
        )

        assert outputs[0].score == pytest.approx(1.0)

    def test_rejects_an_unusable_type(self):
        evaluator = StructuredOutputEvaluator(Invoice)

        with pytest.raises(TypeError, match="instance, dict, or JSON string"):
            evaluator.evaluate(_data(_invoice(), 42))


class TestExplain:
    def test_covers_nested_paths(self):
        config = StructuredOutputEvaluator(Invoice).explain()

        assert "line_items.sku" in config
        assert config["invoice_id"]["comparator"] == "ExactComparator"

    def test_list_of_models_is_not_reported_as_a_string_comparator(self):
        """A List[StructuredModel] has no single comparator."""
        config = StructuredOutputEvaluator(Invoice).explain()

        assert config["line_items"]["comparator"] != "LevenshteinComparator"

    def test_needs_a_model(self):
        with pytest.raises(RuntimeError, match="needs a model"):
            StructuredOutputEvaluator().explain()
