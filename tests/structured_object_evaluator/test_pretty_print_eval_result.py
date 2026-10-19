"""Regression coverage for pretty-printing the public ``EvalResult`` API.

The printers accepted the comparison `dict` and nothing else, so passing the
`EvalResult` that `stickler.evaluate()` returns -- the object the public API hands
back -- printed "No non-matches found - all fields matched successfully!" over a
document with a wrong field. Unwrapping `.raw` is half the fix; the other half is
that `raw` never carried a `non_matches` key, because the facade did not ask for
one.

The fixture uses a real `stickler.evaluate()` result rather than a hand-built dict.
A synthetic `raw` containing `non_matches` hid the second half entirely: the
unwrapping looked sufficient because the key the fixture supplied is one the real
API did not produce.
"""

from typing import Optional

import pytest
from pydantic import BaseModel

import stickler
from stickler.auto import EvalResult
from stickler.structured_object_evaluator import StructuredModel
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.utils.pretty_print import (
    print_confusion_matrix,
    print_evaluation_results,
    print_non_matches,
)


class _Person(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None


@pytest.fixture
def evaluation_result() -> EvalResult:
    """A real `EvalResult` with one field right and one wrong."""
    return stickler.evaluate(
        _Person(name="Ada", city="London"),
        _Person(name="Grace", city="London"),
    )


def test_the_fixture_is_the_public_api(evaluation_result):
    """Guards the thing that hid the bug: this must not become a hand-built dict.

    A synthetic `raw` can supply `non_matches` itself, at which point the tests
    below pass without the facade change and prove nothing about the real API.
    """
    assert isinstance(evaluation_result, EvalResult)
    assert evaluation_result.raw["field_scores"]["name"] == 0.0
    # Computed on demand rather than carried in `raw`, so that `evaluate()` does
    # not pay for records only a report reader wants.
    assert "non_matches" not in evaluation_result.raw
    assert evaluation_result.non_matches


def test_print_confusion_matrix_accepts_eval_result(evaluation_result, capsys):
    """The public confusion-matrix printer unwraps EvalResult.raw."""
    print_confusion_matrix(evaluation_result, use_color=False)

    assert "CONFUSION MATRIX SUMMARY" in capsys.readouterr().out


def test_print_non_matches_accepts_eval_result(evaluation_result, capsys):
    """The public non-match printer unwraps EvalResult.raw."""
    print_non_matches(evaluation_result, use_color=False)

    assert "name" in capsys.readouterr().out


def test_print_evaluation_results_reports_the_failure(evaluation_result, capsys):
    """The headline symptom: a wrong field must not be reported as all-matched."""
    print_evaluation_results(evaluation_result, use_color=False)

    output = capsys.readouterr().out
    assert "CONFUSION MATRIX SUMMARY" in output
    assert "name" in output
    assert "No non-matches found" not in output


def test_the_matching_field_is_not_reported_as_a_non_match(evaluation_result, capsys):
    """`city` matched, so it must not appear in the non-match analysis."""
    print_non_matches(evaluation_result, use_color=False)

    output = capsys.readouterr().out
    assert "Ada" in output and "Grace" in output
    assert "London" not in output


def test_an_all_matching_result_still_says_so(capsys):
    """The success message must survive: it was right whenever nothing failed."""
    identical = _Person(name="Ada", city="London")
    print_evaluation_results(stickler.evaluate(identical, identical), use_color=False)

    assert "No non-matches found" in capsys.readouterr().out


def test_an_unrecognized_object_prints_no_false_success(capsys):
    """Silence is the point: it used to claim success for input it never read."""
    print_evaluation_results(object(), use_color=False)

    assert "No non-matches found" not in capsys.readouterr().out


def test_a_raw_dict_that_already_carries_non_matches_is_used_as_is():
    """Back-compat: an `EvalResult` built straight from a dict still works.

    The original fixture for this PR did exactly that, and user code may too. Such
    a result has no pair to recompare, so the carried key is the only source.
    """
    raw = {
        "overall_score": 0.0,
        "field_scores": {"name": 0.0},
        "non_matches": [{"field_path": "name", "type": "mismatch"}],
    }
    spec = stickler.eval_for(_Person)
    assert EvalResult(raw, spec).non_matches == raw["non_matches"]


def test_a_raw_dict_with_no_pair_and_no_key_is_empty_not_an_error():
    """It must not try to recompare a pair it was never given."""
    spec = stickler.eval_for(_Person)
    assert EvalResult({"overall_score": 1.0}, spec).non_matches == []


def test_the_lazy_computation_happens_once(evaluation_result, monkeypatch):
    """Printing a report twice must not recompare twice."""
    calls = []
    original = type(evaluation_result._ground_truth).compare_with

    def counting(self, *args, **kwargs):
        if kwargs.get("document_non_matches"):
            calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(evaluation_result._ground_truth), "compare_with", counting)
    assert evaluation_result.non_matches
    assert evaluation_result.non_matches
    assert len(calls) == 1


# A `compare_with()` dict carries no `non_matches` key unless it was asked for
# one, so `ev.raw`, or any default comparison dict, with a wrong field still
# printed "all fields matched successfully!" after the fix above. The absence of
# records is not evidence that nothing failed (#298).

_NO_RECORDS = "document_non_matches=True"


def _compare(gt: _Person, pred: _Person, **kwargs) -> dict:
    model = StructuredModel.from_pydantic(_Person)
    return model.from_json(gt.model_dump()).compare_with(
        model.from_json(pred.model_dump()), include_confusion_matrix=True, **kwargs
    )


def test_a_raw_dict_without_records_is_not_reported_as_a_success(
    evaluation_result, capsys
):
    """`ev.raw` has a wrong field and no records: say so, not that all matched."""
    print_non_matches(evaluation_result.raw, use_color=False)

    output = capsys.readouterr().out
    assert "No non-matches found" not in output
    assert _NO_RECORDS in output
    # `evaluate()` takes no `document_non_matches`, so the actionable advice for
    # `.raw` is to pass the `EvalResult`, which computes its records on demand.
    assert "EvalResult" in output


def test_print_evaluation_results_on_a_raw_dict_says_records_are_missing(capsys):
    """The full report over a default `compare_with()` dict, the same case."""
    raw = _compare(_Person(name="Ada", city="London"), _Person(name="Grace"))
    assert "non_matches" not in raw

    print_evaluation_results(raw, use_color=False)

    output = capsys.readouterr().out
    assert "CONFUSION MATRIX SUMMARY" in output
    assert "No non-matches found" not in output
    assert _NO_RECORDS in output


def test_an_empty_record_list_is_a_success_without_the_removed_wording(capsys):
    """Records were requested and there are none: success, minus #287's concept."""
    same = _Person(name="Ada", city="London")
    raw = _compare(same, same, document_non_matches=True)
    assert raw["non_matches"] == []

    print_evaluation_results(raw, use_color=False)

    output = capsys.readouterr().out
    assert "No non-matches found" in output
    assert "all fields matched" not in output
    assert _NO_RECORDS not in output


def test_a_bulk_result_without_records_says_so(capsys):
    """`ProcessEvaluation.non_matches` is None when the bulk evaluator skipped them."""
    model = StructuredModel.from_pydantic(_Person)
    evaluator = BulkStructuredModelEvaluator(
        target_schema=model, document_non_matches=False
    )
    evaluator.update(
        model.from_json({"name": "Ada", "city": "London"}),
        model.from_json({"name": "Grace", "city": "London"}),
    )
    result = evaluator.compute()
    assert result.non_matches is None

    print_non_matches(result, use_color=False)

    output = capsys.readouterr().out
    assert "No non-matches found" not in output
    assert _NO_RECORDS in output


def test_a_bulk_result_with_records_and_no_failures_is_a_success(capsys):
    """The bulk counterpart of the empty-list case: a real list, empty, is clean."""
    model = StructuredModel.from_pydantic(_Person)
    evaluator = BulkStructuredModelEvaluator(target_schema=model)
    same = model.from_json({"name": "Ada", "city": "London"})
    evaluator.update(same, same)
    result = evaluator.compute()
    assert result.non_matches == []

    print_non_matches(result, use_color=False)

    output = capsys.readouterr().out
    assert "No non-matches found" in output
    assert _NO_RECORDS not in output


@pytest.mark.parametrize(
    "not_a_result",
    [object(), None, [{"field_path": "name"}]],
    ids=["object", "None", "list_of_records"],
)
def test_print_non_matches_rejects_input_that_is_not_a_result(not_a_result, capsys):
    """Unreadable input is an error, not a success and not a hint to collect records.

    A list of records is the likely mistake here, and telling its owner to pass
    `document_non_matches=True` would send them after something they already did.
    """
    print_non_matches(not_a_result, use_color=False)

    output = capsys.readouterr().out
    assert "Error" in output
    assert "No non-matches found" not in output
    assert _NO_RECORDS not in output
