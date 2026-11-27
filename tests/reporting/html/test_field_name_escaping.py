"""A field name with HTML metacharacters reaches the report as text, not markup.

Field names come from a Stickler config or a JSON Schema, so they are
arbitrary strings. These tests drive the public API from each of those two
sources to a generated page and assert on the cells of the tables that print
the name: the field performance table and the non-matches table.

The assertions read table cells instead of searching the whole page. The
non-matches table escapes the same string as ``field_path`` through its own
import, so "the escaped form is present somewhere" stays true when the
performance table's escape is removed. Only the cell text separates the two.

See https://github.com/awslabs/stickler/issues/451
"""

from html.parser import HTMLParser

import pytest

from stickler import StructuredModel
from stickler.reporting.html.html_reporter import EvaluationHTMLReporter
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.model_factory import ModelFactory

NAME = "a<b>c'\"&x"
ESCAPED = "a&lt;b&gt;c&#x27;&quot;&amp;x"


def _from_config():
    return ModelFactory.create_model_from_json(
        {
            "fields": {
                "vendor": {"type": "str", "comparator": "auto"},
                NAME: {"type": "str", "comparator": "auto"},
            }
        }
    )


def _from_json_schema():
    return StructuredModel.from_json_schema(
        {
            "type": "object",
            "properties": {"vendor": {"type": "string"}, NAME: {"type": "string"}},
        }
    )


class _TableCells(HTMLParser):
    """Collect the text of each ``<td>`` per table id, as a browser reads it."""

    def __init__(self):
        super().__init__()
        self._table_ids = []
        self._cell = None
        self.cells = {}

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_ids.append(dict(attrs).get("id"))
        elif tag == "td" and self._table_ids:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "table" and self._table_ids:
            self._table_ids.pop()
        elif tag == "td" and self._cell is not None:
            self.cells.setdefault(self._table_ids[-1], []).append("".join(self._cell))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _table_html(page, table_id):
    start = page.index(f'id="{table_id}"')
    return page[start : page.index("</table>", start)]


@pytest.fixture(params=[_from_config, _from_json_schema], ids=["config", "json_schema"])
def page(request, tmp_path):
    model = request.param()
    assert NAME in model.model_fields

    jsonl = tmp_path / "results.jsonl"
    jsonl.touch()
    evaluator = BulkStructuredModelEvaluator(
        target_schema=model, individual_results_jsonl=str(jsonl)
    )
    evaluator.update(
        model(**{"vendor": "Acme", NAME: "x"}),
        model(**{"vendor": "Acme", NAME: "y"}),
        doc_id="doc1",
    )
    out = tmp_path / "report.html"
    result = EvaluationHTMLReporter().generate_report(
        evaluator.compute(),
        str(out),
        model_schema=model,
        individual_results_jsonl_path=str(jsonl),
    )
    assert result.success
    return out.read_text()


@pytest.mark.parametrize("table_id", ["performance-table", "non-matches-table"])
def test_the_name_is_escaped_in_the_table_markup(page, table_id):
    table = _table_html(page, table_id)

    assert ESCAPED in table
    assert NAME not in table


@pytest.mark.parametrize("table_id", ["performance-table", "non-matches-table"])
def test_the_name_is_one_text_cell_not_markup(page, table_id):
    parser = _TableCells()
    parser.feed(page)

    assert NAME in parser.cells[table_id]
