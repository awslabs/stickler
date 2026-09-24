"""The numbers the docs publish are the numbers the code produces.

Both surfaces here were wrong for as long as they existed and nothing read them:
the README's schema example annotated a 0.562 as ``~0.85``, and its Quick
Reference published ``clip-under-threshold`` as ``false``. See #352.
"""

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

from stickler import StructuredModel

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text()
EXTENSION_REFERENCE = (
    REPO_ROOT / "docs" / "docs" / "Guides" / "Evaluation" / "README.md"
).read_text()


def _section_block(text: str, heading: str, lang: str) -> str:
    match = re.search(rf"{re.escape(heading)}\n.*?```{lang}\n(.*?)```", text, re.S)
    assert match, f"no ```{lang} block under {heading!r}"
    return match.group(1)


def _run_readme_schema_example():
    """Run "Using Your Schema" against the schema in the block above it.

    Returns (schema, [(label, printed value, annotated value)]).
    """
    schema = json.loads(
        _section_block(README, "### Complete Real-World Example", "json")
    )
    code = _section_block(README, "### Using Your Schema", "python")
    load = "with open('invoice_schema.json') as f:\n    schema = json.load(f)"
    assert load in code, "the example no longer loads its schema this way"
    code = code.replace(load, "schema = SCHEMA")

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {"SCHEMA": schema})
    prints = [line for line in code.splitlines() if line.startswith("print(")]
    outputs = out.getvalue().splitlines()
    assert len(prints) == len(outputs)

    rows = []
    for source, output in zip(prints, outputs):
        claim = re.search(r"\)\s*#\s*~?([0-9.]+)", source)
        if claim:
            label, value = output.rsplit(":", 1)
            rows.append((label, float(value), float(claim.group(1))))
    return schema, rows


def test_readme_schema_example_prints_its_annotations():
    _, rows = _run_readme_schema_example()
    assert rows, "no annotated print lines found"
    for label, measured, claimed in rows:
        assert measured == pytest.approx(claimed, abs=0.005), label


def test_readme_schema_example_does_not_annotate_a_miss_as_a_pass():
    """An annotation must not put a below-threshold score above its threshold."""
    schema, rows = _run_readme_schema_example()
    thresholds = {
        name: prop["x-aws-stickler-threshold"]
        for name, prop in schema["properties"].items()
        if "x-aws-stickler-threshold" in prop
    }
    field_for_label = {"Invoice ID": "invoice_id", "Customer": "customer_name"}
    for label, measured, claimed in rows:
        field = field_for_label.get(label)
        if field is None:
            continue
        threshold = thresholds[field]
        assert (measured >= threshold) == (claimed >= threshold), label


def _table_after(text: str, header: str) -> dict:
    """First column -> remaining cells, for the Markdown table under ``header``."""
    lines = text[text.index(header) :].splitlines()[2:]
    rows = {}
    for line in lines:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


UNDECLARED_PROPERTIES = {
    "scalar": {"type": "string"},
    "nested": {"type": "object", "properties": {"b": {"type": "string"}}},
    "mapping": {"type": "object"},
    "scalars": {"type": "array", "items": {"type": "string"}},
    "models": {
        "type": "array",
        "items": {"type": "object", "properties": {"a": {"type": "string"}}},
    },
    "dated": {"type": "string", "format": "date"},
    "enumed": {"type": "string", "enum": ["A", "B"]},
    "constant": {"type": "string", "const": "A"},
}
UNDECLARED = StructuredModel.from_json_schema(
    {"type": "object", "properties": UNDECLARED_PROPERTIES}
)


def test_extension_reference_defaults_match_an_undeclared_schema():
    table = _table_after(EXTENSION_REFERENCE, "| Extension | Type | Default |")
    default = {name.strip("`"): cells[1] for name, cells in table.items()}

    infos = [UNDECLARED._get_comparison_info(f) for f in UNDECLARED_PROPERTIES]
    assert default["x-aws-stickler-clip-under-threshold"] == "`true`"
    assert all(info.clip_under_threshold is True for info in infos)
    assert default["x-aws-stickler-weight"] == "1.0"
    assert all(info.weight == 1.0 for info in infos)
    assert default["x-aws-stickler-model-name"] == '`"DynamicModel"`'
    assert UNDECLARED.__name__ == "DynamicModel"
    assert default["x-aws-stickler-match-threshold"] == "0.7"
    assert UNDECLARED.match_threshold == 0.7


@pytest.mark.parametrize(
    "position, fields",
    [
        ("scalar (`string`, `number`, ...)", ["scalar"]),
        ("object with `properties` (a nested model)", ["nested"]),
        ('free-form `{"type": "object"}` (a `Dict`)', ["mapping"]),
        ("array of scalars", ["scalars"]),
        ("array of models", ["models"]),
        ("string with `format`, `enum` or `const`", ["dated", "enumed", "constant"]),
    ],
)
def test_extension_reference_threshold_by_position(position, fields):
    table = _table_after(EXTENSION_REFERENCE, "| Position | Default threshold |")
    published = float(re.match(r"`([0-9.]+)`", table[position][0]).group(1))
    for field in fields:
        assert UNDECLARED._get_comparison_info(field).threshold == published, field
