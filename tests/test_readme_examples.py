"""The README's annotated example output and the Extension Reference defaults
are the numbers the code produces.

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
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
EXTENSION_REFERENCE = (
    REPO_ROOT / "docs" / "docs" / "Guides" / "Evaluation" / "README.md"
).read_text(encoding="utf-8")


def _section_block(text: str, heading: str, lang: str) -> str:
    match = re.search(rf"{re.escape(heading)}\n.*?```{lang}\n(.*?)```", text, re.S)
    assert match, f"no ```{lang} block under {heading!r}"
    return match.group(1)


QUICK_START = "## Get Started in 30 Seconds"
SCHEMA_EXAMPLE = "### Using Your Schema"
README_EXAMPLES = [QUICK_START, SCHEMA_EXAMPLE]


def _run_readme_example(heading: str):
    """Run the ```python block under ``heading``.

    Returns (namespace, [(label, field, printed value, annotated value)]) for
    every print line annotated ``# <number>``. ``field`` is the
    ``field_scores[...]`` key the line prints, or None for the overall score.
    """
    code = _section_block(README, heading, "python")
    namespace = {}
    if heading == SCHEMA_EXAMPLE:
        load = "with open('invoice_schema.json') as f:\n    schema = json.load(f)"
        assert load in code, "the example no longer loads its schema this way"
        code = code.replace(load, "schema = SCHEMA")
        namespace["SCHEMA"] = json.loads(
            _section_block(README, "### Complete Real-World Example", "json")
        )

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, namespace)
    prints = [line for line in code.splitlines() if line.startswith("print(")]
    outputs = out.getvalue().splitlines()
    assert len(prints) == len(outputs)

    rows = []
    for source, output in zip(prints, outputs):
        claim = re.search(r"\)\s*#\s*~?([0-9.]+)", source)
        if claim:
            field = re.search(r"\['field_scores'\]\['(\w+)'\]", source)
            label, value = output.rsplit(":", 1)
            rows.append(
                (
                    label,
                    field and field.group(1),
                    float(value),
                    float(claim.group(1)),
                )
            )
    return namespace, rows


@pytest.mark.parametrize("heading", README_EXAMPLES)
def test_readme_example_prints_its_annotations(heading):
    _, rows = _run_readme_example(heading)
    assert rows, "no annotated print lines found"
    for label, _, measured, claimed in rows:
        assert measured == pytest.approx(claimed, abs=0.005), label


@pytest.mark.parametrize("heading", README_EXAMPLES)
def test_readme_example_does_not_annotate_a_miss_as_a_pass(heading):
    """An annotation within 0.005 of its score must still sit on the same side
    of the threshold the model applies: the field's, or ``match_threshold``."""
    namespace, rows = _run_readme_example(heading)
    model = namespace["Invoice"]
    for label, field, measured, claimed in rows:
        threshold = (
            model.match_threshold
            if field is None
            else model._get_comparison_info(field).threshold
        )
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
    "timestamped": {"type": "string", "format": "date-time"},
    "linked": {"type": "string", "format": "uri"},
    "identified": {"type": "string", "format": "uuid"},
    "timed": {"type": "string", "format": "time"},
    "enumed": {"type": "string", "enum": ["A", "B"]},
    "constant": {"type": "string", "const": "A"},
    # Declares a comparator but no threshold: the default still follows the format.
    "dated_levenshtein": {
        "type": "string",
        "format": "date",
        "x-aws-stickler-comparator": "LevenshteinComparator",
    },
    # A format that selects no comparator stays a plain scalar.
    "emailed": {"type": "string", "format": "email"},
    # `enum` and `const` default to 1.0 whatever the type.
    "int_enum": {"type": "integer", "enum": [1, 2]},
    "num_const": {"type": "number", "const": 3},
    "typeless_enum": {"enum": ["a", "b"]},
    # The same keywords on array items.
    "dated_items": {"type": "array", "items": {"type": "string", "format": "date"}},
    "enumed_items": {"type": "array", "items": {"type": "integer", "enum": [1, 2]}},
    "emailed_items": {"type": "array", "items": {"type": "string", "format": "email"}},
    "mappings": {"type": "array", "items": {"type": "object"}},
}
FORMAT_ENUM_CONST = (
    "`enum` or `const` on any type, or a string `format` of "
    "`date`/`date-time`/`uri`/`uuid`/`time`, on the property or its array `items`"
)
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
        ("plain scalar (`string`, `number`, ...)", ["scalar", "emailed"]),
        ("object with `properties` (a nested model)", ["nested"]),
        (
            'free-form `{"type": "object"}` (a `Dict`), or an array of them',
            ["mapping", "mappings"],
        ),
        ("array of plain scalars", ["scalars", "emailed_items"]),
        ("array of models", ["models"]),
        (
            FORMAT_ENUM_CONST,
            [
                "dated",
                "timestamped",
                "linked",
                "identified",
                "timed",
                "enumed",
                "constant",
                "dated_levenshtein",
                "int_enum",
                "num_const",
                "typeless_enum",
                "dated_items",
                "enumed_items",
            ],
        ),
    ],
)
def test_extension_reference_threshold_by_position(position, fields):
    table = _table_after(EXTENSION_REFERENCE, "| Position | Default threshold |")
    published = float(re.match(r"`([0-9.]+)`", table[position][0]).group(1))
    for field in fields:
        assert UNDECLARED._get_comparison_info(field).threshold == published, field
