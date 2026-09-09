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
    assert EvalResult(raw, spec=None).non_matches == raw["non_matches"]


def test_a_raw_dict_with_no_pair_and_no_key_is_empty_not_an_error():
    """It must not try to recompare a pair it was never given."""
    assert EvalResult({"overall_score": 1.0}, spec=None).non_matches == []


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
