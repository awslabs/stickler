"""The report's embedded data is strict JSON even when a value is NaN or infinite.

``json.dumps`` writes non-finite floats as bare ``NaN``/``Infinity`` tokens.
JavaScript evaluates them, so the page loaded, but anything that extracts the
payload and parses it as JSON failed. A blank numeric cell from pandas is NaN,
so this is the ordinary case. They are now written as ``null``.

Python's ``json.loads`` accepts ``NaN`` by default, so these tests parse with
``parse_constant`` raising, which is what a strict parser does.

See https://github.com/awslabs/stickler/issues/450
"""

import json
import math
from typing import Optional

import numpy as np
import pytest

from stickler import ComparableField, StructuredModel
from stickler.reporting.html.html_reporter import (
    EvaluationHTMLReporter,
    _json_for_script,
)
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

NAN, INF = float("nan"), float("inf")


def _reject(constant):
    raise ValueError(f"non-JSON constant {constant}")


def strict_loads(text):
    return json.loads(text, parse_constant=_reject)


class Invoice(StructuredModel):
    vendor: str = ComparableField()
    total: Optional[float] = ComparableField()


def _embedded_payloads(tmp_path, pairs):
    jsonl = tmp_path / "results.jsonl"
    jsonl.touch()
    evaluator = BulkStructuredModelEvaluator(
        target_schema=Invoice, individual_results_jsonl=str(jsonl)
    )
    for i, (gt, pred) in enumerate(pairs):
        evaluator.update(gt, pred, doc_id=f"doc{i}")
    out = tmp_path / "report.html"
    result = EvaluationHTMLReporter().generate_report(
        evaluator.compute(),
        str(out),
        model_schema=Invoice,
        individual_results_jsonl_path=str(jsonl),
    )
    assert result.success, result.errors
    page = out.read_text()
    start = page.index("initializeDocumentData(", page.index("// Initialize document data"))
    args = page[start + len("initializeDocumentData(") :]
    # Parse each argument strictly from where it starts, the way a consumer
    # extracting the payload would.
    decoder = json.JSONDecoder(parse_constant=_reject)
    docs, end = decoder.raw_decode(args)
    assert args[end : end + 2] == ", "
    thresholds, end = decoder.raw_decode(args, end + 2)
    assert args[end : end + 2] == ");"
    return docs, thresholds


def test_a_report_with_non_finite_values_embeds_strict_json(tmp_path):
    docs, thresholds = _embedded_payloads(
        tmp_path,
        [
            (Invoice(vendor="Acme", total=1.0), Invoice(vendor="Acme", total=NAN)),
            (Invoice(vendor="Acme", total=INF), Invoice(vendor="Bolt", total=-INF)),
        ],
    )

    # Parsing strictly above is the main assertion; the NaN cell itself is null.
    assert "total" in thresholds
    doc0 = next(doc for doc in docs if doc["doc_id"] == "doc0")
    (total,) = [
        nm for nm in doc0["comparison_result"]["non_matches"] if nm["field_path"] == "total"
    ]
    assert total["ground_truth_value"] == 1.0
    assert total["prediction_value"] is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (NAN, None),
        (-INF, None),
        ({"a": [1.5, NAN, {"b": (INF, -INF, 2)}]}, {"a": [1.5, None, {"b": [None, None, 2]}]}),
        ([np.float64("nan"), np.float64(0.25)], [None, 0.25]),
        ({NAN: 1, "k": INF}, {"NaN": 1, "k": None}),
    ],
)
def test_non_finite_numbers_become_null_at_any_depth(value, expected):
    assert strict_loads(_json_for_script(value)) == expected


def test_escaping_still_holds_alongside_a_non_finite_value():
    out = _json_for_script({"v": "</script><script>alert(1)</script>", "n": NAN})

    assert strict_loads(out) == {"v": "</script><script>alert(1)</script>", "n": None}
    for ch in "<>&\u2028\u2029":
        assert ch not in out


@pytest.mark.parametrize(
    "value",
    [
        {"total": 1e308, "tiny": 5e-324, "neg": -0.0, "int": 10**20, "flag": True},
        [0.1, 1 / 3, -2.5e-10, math.pi, None, "nan", "Infinity"],
        {"nested": ({"x": 1.0},), "s": "a & b </script>"},
    ],
)
def test_finite_data_is_written_exactly_as_before(value):
    reference = json.dumps(value)
    for ch, escape in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        reference = reference.replace(ch, escape)

    assert _json_for_script(value) == reference


@pytest.mark.parametrize("member", [1.0, NAN], ids=["finite", "nan"])
def test_a_circular_structure_still_raises_value_error(member):
    """As on dev: only json's non-finite error takes the null fallback, and the
    sanitising pass reports a loop the way ``json.dumps`` does."""
    loop = {"n": member}
    loop["self"] = loop

    with pytest.raises(ValueError, match="Circular reference"):
        _json_for_script(loop)
