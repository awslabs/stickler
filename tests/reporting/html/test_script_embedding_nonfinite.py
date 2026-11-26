"""The report's embedded data is strict JSON even when a value is NaN or infinite.

``json.dumps`` writes non-finite floats as bare ``NaN``/``Infinity`` tokens.
JavaScript evaluates them, so the page loaded, but anything that extracts the
payload and parses it as JSON failed. A blank numeric cell from pandas is NaN,
so this is the ordinary case. The JSONL writer now writes ``null`` (tested in
``tests/structured_object_evaluator/test_jsonl_nonfinite.py``), the reporter
reads any leftover ``NaN`` token as ``None``, and ``_json_for_script`` raises
rather than embed one.

Python's ``json.loads`` accepts ``NaN`` by default, so these tests parse with
``parse_constant`` raising, which is what a strict parser does.

See https://github.com/awslabs/stickler/issues/450
"""

import json
import sys
from typing import Optional

import pytest

from stickler import ComparableField, StructuredModel
from stickler.reporting.html.html_reporter import (
    EvaluationHTMLReporter,
    _json_for_script,
)
from stickler.reporting.html.utils.data_extractors import DataExtractor
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

_JSON_DEEP_NESTING = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="json's C encoder and decoder hit the recursion limit on 5000-deep input before 3.12",
)

NAN, INF = float("nan"), float("inf")


def _reject(constant):
    raise ValueError(f"non-JSON constant {constant}")


def strict_loads(text):
    return json.loads(text, parse_constant=_reject)


class Invoice(StructuredModel):
    vendor: str = ComparableField()
    total: Optional[float] = ComparableField()


def _report_payloads(out, results, model_schema, jsonl):
    result = EvaluationHTMLReporter().generate_report(
        results,
        str(out),
        model_schema=model_schema,
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
    jsonl = tmp_path / "results.jsonl"
    evaluator = BulkStructuredModelEvaluator(
        target_schema=Invoice, individual_results_jsonl=str(jsonl)
    )
    pairs = [
        (Invoice(vendor="Acme", total=1.0), Invoice(vendor="Acme", total=NAN)),
        (Invoice(vendor="Acme", total=INF), Invoice(vendor="Bolt", total=-INF)),
    ]
    for i, (gt, pred) in enumerate(pairs):
        evaluator.update(gt, pred, doc_id=f"doc{i}")
    docs, thresholds = _report_payloads(
        tmp_path / "report.html", evaluator.compute(), Invoice, jsonl
    )

    # Parsing strictly above is the main assertion; the NaN cell itself is null.
    assert "total" in thresholds
    doc0 = next(doc for doc in docs if doc["doc_id"] == "doc0")
    (total,) = [
        nm for nm in doc0["comparison_result"]["non_matches"] if nm["field_path"] == "total"
    ]
    assert total["ground_truth_value"] == 1.0
    assert total["prediction_value"] is None


def test_a_jsonl_file_with_bare_non_finite_tokens_reads_them_as_none(tmp_path):
    """Files from older runs or other tools can still hold ``NaN``/``Infinity``."""
    jsonl = tmp_path / "old.jsonl"
    jsonl.write_text(
        '{"doc_id": "d", "v": [NaN, {"w": Infinity}], "x": -Infinity, "s": "NaN"}\n'
    )

    (doc,) = EvaluationHTMLReporter()._load_individual_results(str(jsonl))

    assert doc == {"doc_id": "d", "v": [None, {"w": None}], "x": None, "s": "NaN"}
    assert strict_loads(_json_for_script([doc])) == [doc]


def test_overflowing_float_literals_render_as_null(tmp_path):
    """``1e999`` is valid JSON but parses to inf, which ``parse_constant`` never sees."""
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text(
        '{"doc_id": "d", "comparison_result": '
        '{"a": [1.5, {"b": [1e999, {"c": -1e999}]}], "x": 2.5}}\n'
    )
    docs, _ = _report_payloads(
        tmp_path / "report.html", {"overall": {}, "fields": {}}, Invoice, jsonl
    )

    assert docs == [
        {
            "doc_id": "d",
            "comparison_result": {"a": [1.5, {"b": [None, {"c": None}]}], "x": 2.5},
        }
    ]


@_JSON_DEEP_NESTING
def test_deep_nesting_with_one_nan_reads_and_embeds(tmp_path):
    """Read and embed are both C json, so neither is bound by the recursion limit."""
    depth = 5000
    jsonl = tmp_path / "deep.jsonl"
    jsonl.write_text('{"doc_id": "d", "v": ' + "[" * depth + "NaN" + "]" * depth + "}\n")

    (doc,) = EvaluationHTMLReporter()._load_individual_results(str(jsonl))
    node = strict_loads(_json_for_script(doc))["v"]
    for _ in range(depth - 1):
        (node,) = node
    assert node == [None]


@pytest.mark.parametrize("value", [NAN, [1, {"a": INF}], {"k": -INF}])
def test_a_non_finite_value_reaching_the_embed_step_raises(value):
    """Sanitising happens where data enters the report; anything that slips
    through fails loudly instead of embedding a bare token."""
    with pytest.raises(ValueError, match="Out of range float values"):
        _json_for_script(value)


def test_escaping_still_holds_alongside_a_value_read_as_none(tmp_path):
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text('{"v": "</script><script>alert(1)</script>", "n": NaN}\n')
    (doc,) = EvaluationHTMLReporter()._load_individual_results(str(jsonl))

    out = _json_for_script(doc)

    assert strict_loads(out) == {"v": "</script><script>alert(1)</script>", "n": None}
    for ch in "<>&\u2028\u2029":
        assert ch not in out


def test_a_non_finite_threshold_is_dropped_rather_than_failing_the_report(tmp_path, monkeypatch):
    """A NaN threshold that reaches the report is treated as unset.

    ``ComparableField(threshold=nan)`` raises at class definition, but a field
    the class check cannot read yet (a string ``Annotated`` annotation naming a
    type defined further down) still loads one. The thresholds are patched in
    here so the test does not depend on that gap staying open.
    """

    class Odd(StructuredModel):
        vendor: str = ComparableField(threshold=0.9)
        total: Optional[float] = ComparableField(threshold=0.5)

    monkeypatch.setattr(
        DataExtractor,
        "extract_all_field_thresholds",
        staticmethod(lambda schema: {"vendor": NAN, "total": 0.5}),
    )
    jsonl = tmp_path / "results.jsonl"
    evaluator = BulkStructuredModelEvaluator(target_schema=Odd, individual_results_jsonl=str(jsonl))
    evaluator.update(Odd(vendor="Acme", total=1.0), Odd(vendor="Acme", total=1.0), doc_id="d")

    _, thresholds = _report_payloads(tmp_path / "report.html", evaluator.compute(), Odd, jsonl)

    assert thresholds == {"total": 0.5}
