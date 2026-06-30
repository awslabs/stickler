"""Unit tests for ``.github/scripts/examples_summary.py``.

The script parses pytest+nbmake JUnit XML output and renders a Markdown
summary for the ``Examples`` GitHub Actions workflow. These tests pin two
real-world JUnit shapes captured from this repository's own CI artifacts so
the script's two headline features (slashed paths and notebook cell
numbers) keep working as pytest/nbmake evolve.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / ".github" / "scripts" / "examples_summary.py"


@pytest.fixture(scope="module")
def examples_summary():
    """Load ``examples_summary.py`` as a module (it lives outside ``src/``)."""
    spec = importlib.util.spec_from_file_location(
        "examples_summary",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["examples_summary"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


class TestRender:
    """``render`` formatting and ordering."""

    def test_header_and_table_skeleton(self, examples_summary):
        out = examples_summary.render([("PASS", "examples/scripts/quick_start.py")])
        assert out.startswith(
            "### Example execution summary\n\n| Status | Path |\n|---|---|\n"
        )

    def test_fail_skip_pass_ordering(self, examples_summary):
        rows = [
            ("PASS", "examples/notebooks/Map_Reduce_Evaluation.ipynb"),
            ("SKIP", "examples/scripts/llm_comparator_demo.py (credentialed)"),
            ("FAIL", "examples/notebooks/Quick_start.ipynb (cell 4)"),
        ]
        body = examples_summary.render(rows).splitlines()
        # Header (4 lines) then rows in order: FAIL, SKIP, PASS.
        assert body[4].startswith("| ❌ FAIL |")
        assert body[5].startswith("| ⏭ SKIP |")
        assert body[6].startswith("| ✅ PASS |")

    def test_sorts_alphabetically_within_status(self, examples_summary):
        rows = [
            ("PASS", "examples/b.py"),
            ("PASS", "examples/a.py"),
        ]
        out = examples_summary.render(rows).splitlines()
        assert "examples/a.py" in out[4]
        assert "examples/b.py" in out[5]

    def test_pipe_in_cell_value_is_escaped(self, examples_summary):
        out = examples_summary.render([("FAIL", "weird|path.py")])
        assert "weird\\|path.py" in out


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMain:
    """Drive ``main`` against real-shape JUnit XML fixtures."""

    def test_missing_argument_emits_fallback(
        self, examples_summary, capsys: pytest.CaptureFixture[str]
    ):
        exit_code = examples_summary.main(["examples_summary.py"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "No results to display" in captured

    def test_missing_file_emits_fallback(
        self,
        examples_summary,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        exit_code = examples_summary.main(
            ["examples_summary.py", str(tmp_path / "does_not_exist.xml")]
        )
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "No results to display" in captured

    def test_malformed_xml_emits_fallback(
        self,
        examples_summary,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        broken = tmp_path / "broken.xml"
        broken.write_text("<not-closed>")
        exit_code = examples_summary.main(["examples_summary.py", str(broken)])
        assert exit_code == 0
        assert "No results to display" in capsys.readouterr().out

    def test_real_shape_emits_slashed_paths_and_cell_number(
        self,
        examples_summary,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        """Verify the two headline features against pytest+nbmake JUnit shape.

        Critically, pytest's JUnit writer does NOT set a ``file`` attribute
        on ``<testcase>``; it puts the importable module path in
        ``classname`` (dotted, no extension) and the basename in ``name``.
        nbmake's failure tracebacks identify cells as ``Cell In[N], line M``,
        not the older ``Cell N`` form. The summary script must translate
        both correctly.
        """
        # Hand-crafted to mirror the exact shape this repo's CI produces.
        # ``failure`` text mimics nbmake's ANSI-stripped traceback so the
        # regex has to deal with separators between ``Cell`` and ``In[N]``.
        # ``#x1B[...]`` strings model the entity-rendered ANSI escapes.
        fixture = tmp_path / "junit.xml"
        fixture.write_text(
            dedent(
                """\
                <?xml version="1.0" encoding="utf-8"?>
                <testsuites name="pytest tests">
                  <testsuite name="pytest" errors="0" failures="1" skipped="2" tests="4">
                    <testcase classname="examples.notebooks.Quick_start.ipynb"
                              name="Quick_start.ipynb" time="6.807" />
                    <testcase classname="examples.notebooks.Bulk_Confidence_AUROC.ipynb"
                              name="Bulk_Confidence_AUROC.ipynb" time="5.0">
                      <failure message="ValueError: bad">---
                      #x1B[36mCell#x1B[39m#x1B[36m #x1B[39m#x1B[32mIn[6]#x1B[39m#x1B[32m, line 16#x1B[39m
                      ValueError: bad</failure>
                    </testcase>
                    <testcase classname="examples.scripts.bert_comparator_demo"
                              name="bert_comparator_demo.py" time="0.001">
                      <skipped type="pytest.skip"
                               message="long-running example: scheduled runs only">
                        /tmp/examples/scripts/bert_comparator_demo.py:1: long-running example: scheduled runs only
                      </skipped>
                    </testcase>
                    <testcase classname="examples.scripts.llm_comparator_demo"
                              name="llm_comparator_demo.py" time="0.000">
                      <skipped type="pytest.skip"
                               message="credentialed example: requires AWS credentials">
                        /tmp/examples/scripts/llm_comparator_demo.py:1: credentialed example: requires AWS credentials
                      </skipped>
                    </testcase>
                  </testsuite>
                </testsuites>
                """
            )
        )

        exit_code = examples_summary.main(["examples_summary.py", str(fixture)])
        assert exit_code == 0
        output = capsys.readouterr().out

        # Slashed paths, not the dotted classnames pytest emits.
        assert "examples/notebooks/Quick_start.ipynb" in output
        assert "examples/notebooks/Bulk_Confidence_AUROC.ipynb" in output
        assert "examples/scripts/bert_comparator_demo.py" in output
        assert "examples/scripts/llm_comparator_demo.py" in output
        assert "examples.scripts.bert_comparator_demo" not in output

        # Failure row includes the nbmake cell index.
        fail_rows = [
            line for line in output.splitlines() if line.startswith("| ❌ FAIL |")
        ]
        assert len(fail_rows) == 1
        assert "(cell 6)" in fail_rows[0]
        assert "Bulk_Confidence_AUROC.ipynb" in fail_rows[0]

        # Skip reasons are surfaced.
        assert "credentialed example: requires AWS credentials" in output
        assert "long-running example: scheduled runs only" in output


# ---------------------------------------------------------------------------
# _classname_to_path() (internal helper - explicit pinning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "classname,name,expected",
    [
        # Scripts: pytest strips ``.py`` from classname.
        (
            "examples.scripts.bert_comparator_demo",
            "bert_comparator_demo.py",
            "examples/scripts/bert_comparator_demo.py",
        ),
        # Notebooks: pytest keeps ``.ipynb`` in classname (it's not a valid
        # Python module suffix).
        (
            "examples.notebooks.Quick_start.ipynb",
            "Quick_start.ipynb",
            "examples/notebooks/Quick_start.ipynb",
        ),
        # Defensive: empty classname falls back to name.
        ("", "anything.py", "anything.py"),
        # Defensive: nothing at all.
        ("", "", "<unknown>"),
    ],
)
def test_classname_to_path(examples_summary, classname, name, expected):
    assert examples_summary._classname_to_path(classname, name) == expected


# ---------------------------------------------------------------------------
# _CELL_RE (internal helper - explicit pinning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "haystack,expected_group",
    [
        # nbmake 8+ Jupyter traceback shape (post-IPython 8).
        ("Cell In[6], line 16", "6"),
        # ANSI-wrapped form with XML-escaped ESC sequences.
        ("#x1B[36mCell#x1B[39m #x1B[32mIn[2]#x1B[39m, line 1", "2"),
        # ANSI-wrapped form with raw ESC bytes (live console output).
        ("\x1b[36mCell\x1b[39m \x1b[32mIn[3]\x1b[39m, line 9", "3"),
        # Legacy/plain form.
        ("Cell 4: NameError: name 'x' is not defined", "4"),
    ],
)
def test_cell_regex_matches_real_traceback_shapes(
    examples_summary, haystack, expected_group
):
    cleaned = examples_summary._strip_ansi(haystack)
    match = examples_summary._CELL_RE.search(cleaned)
    assert match is not None, f"regex did not match {haystack!r} (cleaned={cleaned!r})"
    # The regex has two alternatives; the matching group holds the index.
    captured = match.group(1) or match.group(2)
    assert captured == expected_group
