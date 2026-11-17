"""The report's document filter must be able to find the document it names.

``main.js`` builds the document dropdown from the embedded document data and
sets ``option.value = doc.doc_id``, which the DOM stores as a string. The change
handler passes that string straight back, and the lookup is strictly equal
(``documentData.find(d => d.doc_id === docId)``), so an id embedded as a JSON
number never matches. Nothing raises, and the ``if (doc)`` guard around the
re-render has no ``else``, so the page keeps whatever was last rendered -- the
aggregate numbers on a first selection, another document's numbers after that.

``evaluate_dataframe`` reads ``doc_id`` straight off a DataFrame column, so an
integer id is the ordinary case for data read from CSV, parquet or a database.

Six of these tests fail without the conversion: they assert the invariant the
JavaScript depends on, that the embedded ``doc_id`` is a JSON string. The rest
guard against a *mis*-fix -- an unconditional ``str()`` would convert cases the
conversion leaves alone, and reaching into a record that is not an object would
abort the load.

These tests do not drive a browser, so they do not prove the dropdown repaints.
"""

import json

import pandas as pd

from stickler import ComparableField, StructuredModel
from stickler.reporting.html.html_reporter import EvaluationHTMLReporter
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)


class Invoice(StructuredModel):
    vendor: str = ComparableField()


def _embedded_document_data(page):
    """Return the first argument ``main.js`` is initialized with.

    Anchored on the comment the generator emits immediately above the call, so
    this finds the invocation rather than the declaration inside ``main.js``.
    """
    marker = "initializeDocumentData("
    start = page.index(marker, page.index("// Initialize document data")) + len(marker)
    value, _ = json.JSONDecoder().raw_decode(page[start:].lstrip())
    return value


def _load(tmp_path, *records):
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
    return EvaluationHTMLReporter()._load_individual_results(str(jsonl))


def test_a_zero_doc_id_becomes_a_string(tmp_path):
    """``0`` is the first id of any 0-based column, and it is falsy.

    Guards a mis-fix that tests the id for truthiness instead of ``is not None``:
    that leaves ``0`` a JSON number, so the first document cannot be selected.
    """
    assert _load(tmp_path, {"doc_id": 0})[0]["doc_id"] == "0"


def test_a_float_doc_id_becomes_a_string(tmp_path):
    """An id column with one missing value reads back as float."""
    assert _load(tmp_path, {"doc_id": 1001.0})[0]["doc_id"] == "1001.0"


def test_a_string_doc_id_is_unchanged(tmp_path):
    """Mis-fix guard, not a fix validation: this passes without the change."""
    assert _load(tmp_path, {"doc_id": "doc1"})[0]["doc_id"] == "doc1"


def test_a_none_doc_id_is_left_alone(tmp_path):
    """Pins a case the conversion leaves alone, so changing it is a decision.

    A ``None`` id stays ``None``. Converting it to ``"None"`` would make it a
    selectable string id; that is a separate behaviour change, and this test
    makes it an explicit one.

    Note this is *not* the ordinary missing-value case -- see
    ``test_a_nan_doc_id_becomes_the_string_nan``.
    """
    assert _load(tmp_path, {"doc_id": None})[0]["doc_id"] is None


def test_a_nan_doc_id_becomes_the_string_nan(tmp_path):
    """The real missing-value case, and it is not ``None``.

    A float id column with one missing value reads back as ``NaN``, which
    ``json.dumps`` writes as a bare ``NaN`` token. The conversion embeds it as the
    string ``"nan"``; pinned so that choice stays visible.
    """
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text('{"doc_id": NaN}\n')
    loaded = EvaluationHTMLReporter()._load_individual_results(str(jsonl))
    assert loaded[0]["doc_id"] == "nan"


def test_a_record_without_a_doc_id_gains_no_key(tmp_path):
    """Mis-fix guard: an unconditional ``str()`` would add ``"None"`` here."""
    assert "doc_id" not in _load(tmp_path, {"overall_score": 1.0})[0]


def test_ids_that_differ_only_by_type_map_to_the_same_string(tmp_path):
    """``str`` is not injective, and the filter takes the first match.

    ``1001`` and ``"1001"`` were distinct ids and are now both ``"1001"``;
    ``True`` and ``"True"`` likewise, because ``bool`` is an ``int``. Pinned as
    known behaviour of the conversion, so a change to it is visible.
    """
    loaded = _load(
        tmp_path,
        {"doc_id": 1001},
        {"doc_id": "1001"},
        {"doc_id": True},
        {"doc_id": "True"},
    )
    ids = [record["doc_id"] for record in loaded]
    assert ids == ["1001", "1001", "True", "True"]
    assert len(set(ids)) == 2


def test_a_line_that_is_not_an_object_does_not_drop_later_records(tmp_path):
    """Regression guard for the conversion itself.

    ``json.loads`` can return a bare scalar or a list. Reaching into one with
    ``.get`` raises, and because the ``except`` wraps the whole file read rather
    than one line, the loop aborts and every later record is silently discarded
    from the loader's result. The conversion skips any
    record that is not an object, which is what the loader did before it
    existed.
    """
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text('{"doc_id": 1}\n123\n{"doc_id": 3}\n["x"]\n{"doc_id": 4}\n')
    loaded = EvaluationHTMLReporter()._load_individual_results(str(jsonl))
    assert len(loaded) == 5
    assert [r["doc_id"] for r in loaded if isinstance(r, dict)] == ["1", "3", "4"]


def test_every_embedded_doc_id_is_a_string(tmp_path):
    """End to end, by the route a DataFrame actually takes.

    One id is above 2**53, which is why embedding it as a JSON *string* matters:
    as a JSON number, JavaScript parses ``9007199254740993`` back as ``...92``
    and the dropdown would offer a value no record carries.

    This test does **not** observe that loss, and must not be credited with
    doing so. ``_embedded_document_data`` decodes the page with Python's own
    arbitrary-precision decoder, so the digit survives here whatever a browser
    would do; the test reddens because the id is embedded unquoted, not because
    it sees ``...92``. What it pins is the invariant that prevents the loss --
    the embedded id is a JSON string -- across the whole ``generate_report``
    path rather than at the loader alone. Proving the browser half needs a
    browser.
    """
    jsonl = tmp_path / "results.jsonl"
    df = pd.DataFrame(
        [
            {
                "doc_id": doc_id,
                "expected": json.dumps({"vendor": "Acme"}),
                "predicted": json.dumps({"vendor": predicted}),
            }
            for doc_id, predicted in ((1001, "Globex"), (9007199254740993, "Acme"))
        ]
    )
    results = BulkStructuredModelEvaluator(
        target_schema=Invoice, individual_results_jsonl=str(jsonl)
    ).evaluate_dataframe(df)

    out = tmp_path / "report.html"
    report = EvaluationHTMLReporter().generate_report(
        results,
        str(out),
        model_schema=Invoice,
        individual_results_jsonl_path=str(jsonl),
    )
    assert report.success, report.errors

    embedded = _embedded_document_data(out.read_text())
    assert embedded, "no document data reached the page"
    assert [doc["doc_id"] for doc in embedded] == ["1001", "9007199254740993"]
