"""The README's annotated example output and the Extension Reference defaults
are the numbers the code produces.

Both surfaces here were wrong for as long as they existed and nothing read them:
the README's schema example annotated a 0.562 as ``~0.85``, and its Quick
Reference published ``clip-under-threshold`` as ``false``. See #352.
"""

import ast
import contextlib
import io
import json
import re
import runpy
from pathlib import Path
from typing import get_args, get_origin

import pytest

from stickler import StructuredModel

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def readme():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def extension_reference():
    return (
        REPO_ROOT / "docs" / "docs" / "Guides" / "Evaluation" / "README.md"
    ).read_text(encoding="utf-8")


def _section_block(text: str, heading: str, lang: str) -> str:
    match = re.search(rf"{re.escape(heading)}\n.*?```{lang}\n(.*?)```", text, re.S)
    assert match, f"no ```{lang} block under {heading!r}"
    return match.group(1)


QUICK_START = "## Get Started in 30 Seconds"
SCHEMA_EXAMPLE = "### Using Your Schema"
README_EXAMPLES = [QUICK_START, SCHEMA_EXAMPLE]


def _annotated_rows(code, output):
    """Match the examples' annotated numeric f-string prints by their labels.

    This supports static labels and one numeric f-string value, not arbitrary
    Python output. Numeric annotations may use signs and scientific notation.
    """
    sources = {}
    lines = code.splitlines()
    for node in ast.walk(ast.parse(code)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            continue
        claim = re.search(r"\)\s*#\s*~?(\S+)", lines[node.end_lineno - 1])
        if not claim or not re.match(r"[+\-.\d]", claim.group(1)):
            continue
        try:
            claimed = float(claim.group(1))
        except ValueError:
            pytest.fail(f"README numeric annotation is invalid: {claim.group(1)!r}")
        message = f"README annotated print format changed at line {node.lineno}"
        assert len(node.args) == 1 and not node.keywords, message
        arg = node.args[0]
        assert isinstance(arg, ast.JoinedStr) and len(arg.values) == 2, message
        prefix, value = arg.values
        assert isinstance(prefix, ast.Constant) and isinstance(prefix.value, str), (
            message
        )
        assert isinstance(value, ast.FormattedValue), message
        label = prefix.value.strip()
        assert label.endswith(":") and "\n" not in label, message
        assert label not in sources, f"README has duplicate annotated label {label!r}"
        field = re.search(
            r"\[['\"]field_scores['\"]\]\[['\"]([^'\"]+)['\"]\]",
            ast.unparse(value.value),
        )
        sources[label] = (field.group(1) if field else None, claimed)

    rows = []
    seen = set()
    for line in output.splitlines():
        for label, (field, claim) in sources.items():
            if line.startswith(label):
                number = line[len(label) :].strip()
                try:
                    measured = float(number)
                except ValueError:
                    pytest.fail(
                        f"README output for {label!r} is not a number: {number!r}"
                    )
                rows.append((label.removesuffix(":"), field, measured, claim))
                seen.add(label)
                break
    assert seen == sources.keys(), (
        f"README output missing labels: {sources.keys() - seen}"
    )
    return rows


def _run_readme_example(readme, heading):
    """Run a README block and return its namespace and labeled numeric claims."""
    code = _section_block(readme, heading, "python")
    namespace = {}
    if heading == SCHEMA_EXAMPLE:
        load = "with open('invoice_schema.json') as f:\n    schema = json.load(f)"
        assert load in code, "the example no longer loads its schema this way"
        code = code.replace(load, "schema = SCHEMA")
        namespace["SCHEMA"] = json.loads(
            _section_block(readme, "### Complete Real-World Example", "json")
        )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, namespace)
    return namespace, _annotated_rows(code, out.getvalue())


def _matching_threshold(model, field):
    if field is None:
        return model.match_threshold
    annotation, _ = model._unwrap_optional(model.model_fields[field].annotation)
    if get_origin(annotation) is list:
        item, _ = model._unwrap_optional(get_args(annotation)[0])
        if isinstance(item, type) and issubclass(item, StructuredModel):
            return item.match_threshold
    return model._get_comparison_info(field).threshold


@pytest.mark.parametrize("heading", README_EXAMPLES)
def test_readme_example_prints_its_annotations(readme, heading):
    _, rows = _run_readme_example(readme, heading)
    assert rows, "no annotated print lines found"
    for label, _, measured, claimed in rows:
        assert measured == pytest.approx(claimed, abs=0.005), label


@pytest.mark.parametrize("heading", README_EXAMPLES)
def test_readme_example_does_not_annotate_a_miss_as_a_pass(readme, heading):
    """An annotation within 0.005 of its score must still sit on the same side
    of the threshold the model applies: the field's, or ``match_threshold``."""
    namespace, rows = _run_readme_example(readme, heading)
    model = namespace["Invoice"]
    for label, field, measured, claimed in rows:
        threshold = _matching_threshold(model, field)
        assert (measured >= threshold) == (claimed >= threshold), label


def _table_after(text: str, header: str) -> dict:
    """First column -> remaining cells, for the Markdown table under ``header``."""
    assert header in text, f"Docs table changed: missing header {header!r}"
    lines = text[text.index(header) :].splitlines()[2:]
    rows = {}
    for line in lines:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert cells[0] not in rows, f"Docs table changed: duplicate row {cells[0]!r}"
        rows[cells[0]] = cells[1:]
    assert rows, f"Docs table changed: no rows under {header!r}"
    return rows


def _row(table, position):
    assert position in table, f"Docs table changed: missing Position {position!r}"
    return table[position]


def _assert_rows_covered(table, cases):
    covered = {position for position, _ in cases}
    assert set(table) == covered, (
        f"Docs table changed: untested rows {set(table) - covered}; "
        f"missing rows {covered - set(table)}"
    )


UNDECLARED_PROPERTIES = {
    "scalar": {"type": "string"},
    "number": {"type": "number"},
    "integer": {"type": "integer"},
    "boolean": {"type": "boolean"},
    "number_items": {"type": "array", "items": {"type": "number"}},
    "integer_items": {"type": "array", "items": {"type": "integer"}},
    "boolean_items": {"type": "array", "items": {"type": "boolean"}},
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


@pytest.fixture(scope="module")
def undeclared():
    return StructuredModel.from_json_schema(
        {"type": "object", "properties": UNDECLARED_PROPERTIES}
    )


def test_extension_reference_defaults_match_an_undeclared_schema(
    extension_reference, undeclared
):
    table = _table_after(extension_reference, "| Extension | Type | Default |")
    default = {name.strip("`"): cells[1] for name, cells in table.items()}

    infos = [undeclared._get_comparison_info(f) for f in UNDECLARED_PROPERTIES]
    assert default["x-aws-stickler-clip-under-threshold"] == "`true`"
    assert all(info.clip_under_threshold is True for info in infos)
    assert default["x-aws-stickler-weight"] == "1.0"
    assert all(info.weight == 1.0 for info in infos)
    assert default["x-aws-stickler-model-name"] == '`"DynamicModel"`'
    assert undeclared.__name__ == "DynamicModel"
    assert default["x-aws-stickler-match-threshold"] == "0.7"
    assert undeclared.match_threshold == 0.7


POSITION_CASES = [
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
]


@pytest.mark.parametrize("position, fields", POSITION_CASES)
def test_extension_reference_threshold_by_position(
    extension_reference, undeclared, position, fields
):
    table = _table_after(extension_reference, "| Position | Default threshold |")
    published = float(re.match(r"`([0-9.]+)`", _row(table, position)[0]).group(1))
    for field in fields:
        assert undeclared._get_comparison_info(field).threshold == published, field


def test_extension_reference_covers_every_position(extension_reference):
    table = _table_after(extension_reference, "| Position | Default threshold |")
    _assert_rows_covered(table, POSITION_CASES)


def _readme_defaults(readme):
    return {
        key: " ".join(value.split())
        for key, value in re.findall(
            r"^#### `(x-aws-stickler-[^`]+)`\n(?:(?!^#### ).)*?"
            r"^\*\*Default:\*\* (.*?)(?=\n\n)",
            readme,
            re.M | re.S,
        )
    }


def test_readme_extension_defaults_match_schema(readme, undeclared):
    defaults = _readme_defaults(readme)
    assert set(defaults) == {
        "x-aws-stickler-comparator",
        "x-aws-stickler-threshold",
        "x-aws-stickler-weight",
        "x-aws-stickler-clip-under-threshold",
        "x-aws-stickler-model-name",
        "x-aws-stickler-match-threshold",
    }, "README Default lines changed; update their test coverage"
    infos = [undeclared._get_comparison_info(f) for f in UNDECLARED_PROPERTIES]
    assert {info.weight for info in infos} == {
        float(defaults["x-aws-stickler-weight"].strip("`"))
    }
    clip = json.loads(defaults["x-aws-stickler-clip-under-threshold"].strip("`"))
    assert isinstance(clip, bool), "README clip Default must be a boolean"
    assert all(info.clip_under_threshold is clip for info in infos)
    assert undeclared.__name__ == json.loads(
        defaults["x-aws-stickler-model-name"].strip("`")
    )
    assert undeclared.match_threshold == float(
        defaults["x-aws-stickler-match-threshold"].strip("`")
    )
    assert defaults["x-aws-stickler-comparator"].startswith(
        "Type-dependent (see table below)"
    )
    threshold = defaults["x-aws-stickler-threshold"]
    patterns = [
        (
            r"`([0-9.]+)` for a plain scalar, an array of plain scalars, and an array of models",
            ["scalar", "number", "integer", "boolean", "scalars", "models"],
        ),
        (r"`([0-9.]+)` for an object with `properties`", ["nested"]),
        (
            r"`([0-9.]+)` for a free-form\s+`\{\"type\": \"object\"\}`\s+\(or an array of them\)",
            ["mapping", "mappings"],
        ),
        # The enum/const/format clause shares that number. Requiring it to
        # follow the number with no new value in between means a README edit
        # that gives the clause its own threshold has to update this pattern.
        (
            r"`([0-9.]+)` for a free-form\s+`\{\"type\": \"object\"\}`\s+"
            r"\(or an array of them\)\s+and for any property or array\s+`items`\s+"
            r"with\s+`enum`,\s+`const`,\s+or a string\s+`format`",
            [
                "dated",
                "timestamped",
                "linked",
                "identified",
                "timed",
                "enumed",
                "constant",
                "dated_items",
                "enumed_items",
            ],
        ),
    ]
    for pattern, fields in patterns:
        match = re.search(pattern, threshold)
        assert match, f"README threshold Default prose changed: {pattern}"
        assert {undeclared._get_comparison_info(f).threshold for f in fields} == {
            float(match.group(1))
        }


COMPARATOR_CASES = [
    ('`"string"`', ["scalar"]),
    ('`"number"`', ["number"]),
    ('`"integer"`', ["integer"]),
    ('`"boolean"`', ["boolean"]),
    ('`"format": "date"` or `"date-time"`', ["dated", "timestamped"]),
    (
        '`"enum"`, `"const"`, `"format": "uri"`/`"uuid"`/`"time"`',
        [
            "enumed",
            "constant",
            "int_enum",
            "num_const",
            "typeless_enum",
            "linked",
            "identified",
            "timed",
        ],
    ),
    (
        '`"array"` (primitives)',
        [
            "scalars",
            "number_items",
            "integer_items",
            "boolean_items",
            "dated_items",
            "enumed_items",
        ],
    ),
    ('`"array"` of objects with `properties`', ["models"]),
    ('`"object"` with `properties`', ["nested"]),
    (
        '`"object"` without `properties`, or an `"array"` of them',
        ["mapping", "mappings"],
    ),
]
COMPARATOR_HEADER = "| JSON Schema | Default Comparator | Default Threshold |"


def test_readme_comparator_table_covers_every_row(readme):
    _assert_rows_covered(_table_after(readme, COMPARATOR_HEADER), COMPARATOR_CASES)


@pytest.mark.parametrize("position, fields", COMPARATOR_CASES)
def test_readme_comparator_table_matches_schema(readme, undeclared, position, fields):
    comparator, threshold, _ = _row(_table_after(readme, COMPARATOR_HEADER), position)
    for field in fields:
        info = undeclared._get_comparison_info(field)
        if field == "models":
            assert comparator == "Hungarian matching"
            match = re.fullmatch(
                r"`([0-9.]+)`, pairing elements at `([0-9.]+)`", threshold
            )
            assert match, "README model-list threshold cell changed"
            assert info.threshold == float(match.group(1))
            assert _matching_threshold(undeclared, field) == float(match.group(2))
        elif field == "nested":
            assert comparator == "Recursive comparison"
            annotation, _ = undeclared._unwrap_optional(
                undeclared.model_fields[field].annotation
            )
            assert issubclass(annotation, StructuredModel)
            assert info.threshold == float(threshold.strip("`"))
        elif position == '`"array"` (primitives)':
            assert comparator == threshold == "Based on item type"
            element_schema = UNDECLARED_PROPERTIES[field]["items"]
            element = StructuredModel.from_json_schema(
                {"type": "object", "properties": {"value": element_schema}}
            )
            element_info = element._get_comparison_info("value")
            assert type(info.comparator) is type(element_info.comparator)
            assert info.threshold == element_info.threshold
        else:
            assert type(info.comparator).__name__ == comparator.strip("`")
            assert info.threshold == float(threshold.strip("`"))


@pytest.mark.parametrize(
    "code, output",
    [
        ('if True:\n    print(f"Score: {1.0}")  # 1.0', "Score: 1.0\n"),
        (
            'for _ in range(2):\n    print(f"Score: {1.0}")  # 1.0',
            "Score: 1.0\nScore: 1.0\n",
        ),
        ('print(\n    f"Score: {1.0}"\n)  # 1.0', "Score: 1.0\n"),
        (
            'print(f"Score: {1.0}")  # 1.0',
            "unannotated\nmultiline output\nScore: 1.0\n",
        ),
    ],
)
def test_annotated_prints_support_indentation_loops_and_multiline(code, output):
    rows = _annotated_rows(code, output)
    assert rows and all(row == ("Score", None, 1.0, 1.0) for row in rows)


def test_annotated_outputs_are_matched_by_label_not_order():
    code = 'print(f"A: {0.1}")  # 0.1\nprint(f"B: {0.9}")  # 0.9'
    assert _annotated_rows(code, "B: 0.9\nA: 0.1\n") == [
        ("B", None, 0.9, 0.9),
        ("A", None, 0.1, 0.1),
    ]


@pytest.mark.parametrize("output", ["Other: 1.0\n", "Score: not a number\n"])
def test_changed_annotated_output_fails_clearly(output):
    with pytest.raises((AssertionError, pytest.fail.Exception), match="README"):
        _annotated_rows('print(f"Score: {1.0}")  # 1.0', output)


def test_missing_position_fails_clearly():
    with pytest.raises(AssertionError, match="Docs table changed: missing Position"):
        _row({"renamed": ["0.5"]}, "old position")


def test_new_table_row_requires_a_case():
    with pytest.raises(AssertionError, match="Docs table changed: untested rows"):
        _assert_rows_covered({"existing": [], "new": []}, [("existing", [])])


def test_model_list_matching_threshold_is_the_element_threshold():
    from stickler import ComparableField

    # Element and parent thresholds differ, and neither is the 0.7 default,
    # so returning the parent's match_threshold cannot pass.
    class Item(StructuredModel):
        value: str = ComparableField()
        match_threshold = 0.8

    class Document(StructuredModel):
        items: list[Item] | None = ComparableField()
        match_threshold = 0.6

    assert _matching_threshold(Document, "items") == 0.8
    assert (0.798 >= _matching_threshold(Document, "items")) != (
        0.802 >= _matching_threshold(Document, "items")
    )


def test_module_import_defers_document_reads_and_schema_building(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("fixture dependency unavailable")

    monkeypatch.setattr(Path, "read_text", unavailable)
    monkeypatch.setattr(StructuredModel, "from_json_schema", unavailable)
    namespace = runpy.run_path(__file__)
    for name in ("readme", "extension_reference", "undeclared"):
        with pytest.raises(RuntimeError, match="fixture dependency unavailable"):
            namespace[name].__wrapped__()


@pytest.mark.parametrize("annotation, claimed", [("-1.000", -1.0), ("1.000e-1", 0.1)])
def test_numeric_annotations_are_not_dropped_or_truncated(annotation, claimed):
    code = f'print(f"Score: {{1.0}}")  # {annotation}'
    assert _annotated_rows(code, "Score: 1.0\n") == [("Score", None, 1.0, claimed)]


@pytest.mark.parametrize(
    "measured, claimed", [(0.798, 0.802), (0.802, 0.798), (0.8, 0.799)]
)
def test_example_verdict_check_uses_element_threshold(monkeypatch, measured, claimed):
    from stickler import ComparableField

    class Item(StructuredModel):
        value: str = ComparableField()
        match_threshold = 0.8

    class Document(StructuredModel):
        items: list[Item] = ComparableField()
        match_threshold = 0.6

    assert Document._get_comparison_info("items").threshold == 0.5
    monkeypatch.setitem(
        globals(),
        "_run_readme_example",
        lambda *_: ({"Invoice": Document}, [("Items", "items", measured, claimed)]),
    )
    with pytest.raises(AssertionError, match="Items"):
        test_readme_example_does_not_annotate_a_miss_as_a_pass("", "unused")
