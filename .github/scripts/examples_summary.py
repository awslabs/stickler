#!/usr/bin/env python3
"""Render a JUnit XML report as a Markdown summary for GitHub Actions.

Reads a JUnit XML file (as produced by ``pytest --junitxml=...``) and emits a
Markdown summary suitable for ``$GITHUB_STEP_SUMMARY``. Used by the
``Examples`` workflow's ``Write job summary`` step.

The output has the form::

    ### Example execution summary

    | Status | Path |
    |---|---|
    | ❌ FAIL | examples/notebooks/Quick_start.ipynb (cell 4) |
    | ⏭ SKIP | examples/scripts/llm_comparator_demo.py (credentialed example: requires AWS credentials) |
    | ✅ PASS | examples/notebooks/Map_Reduce_Evaluation.ipynb |

Rows are grouped failures → skips → passes; each group is sorted by path so
the summary is stable across runs.

If the JUnit file is missing or unparsable, a fallback message is emitted and
the script exits 0 so that the summary step never masks the underlying
pytest failure visible in the workflow logs.

XML is parsed with :mod:`defusedxml` to harden against entity-expansion and
external-entity attacks; this also keeps the AWS Security Helper (ASH) lint
clean. See https://github.com/tiran/defusedxml.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree.ElementTree import Element

# defusedxml only patches the parser; the element types are stdlib.
from defusedxml.ElementTree import ParseError, parse

HEADER = "### Example execution summary"
FALLBACK = (
    f"{HEADER}\n\n_No results to display: pytest did not produce a JUnit report._"
)

# nbmake's Jupyter-style tracebacks identify the failing cell as
# ``Cell In[N], line M`` (post-IPython 8). Older notebooks may render the
# index plainly as ``Cell N``. Match either form, case-insensitively, and
# capture the cell index. Failure text is pre-cleaned by
# :func:`_strip_ansi` so this pattern doesn't have to tolerate ANSI escape
# sequences inline.
_CELL_RE = re.compile(
    r"Cell\s+(?:In\s*\[\s*(\d+)\s*\]|(\d+))",
    re.IGNORECASE,
)

# Match real ANSI escape sequences (raw ESC byte) and the XML-rendered
# form pytest's JUnit writer produces (``#x1B[...m``). nbmake colors its
# tracebacks; stripping these makes the cell-index regex tractable.
_ANSI_RE = re.compile(
    r"(?:\x1b|#x1B)\[[0-9;]*[a-zA-Z]",
    re.IGNORECASE,
)


def _strip_ansi(text: str) -> str:
    """Return ``text`` with ANSI color escapes (real or XML-rendered) removed."""
    return _ANSI_RE.sub("", text)


# Sort key: failures first, then skips, then passes.
_STATUS_ORDER = {"FAIL": 0, "SKIP": 1, "PASS": 2}


def _classname_to_path(classname: str, name: str) -> str:
    """Convert pytest's dotted ``classname`` plus ``name`` to a slashed path.

    Pytest's JUnit writer puts the importable module path in ``classname``
    and the test name in ``name``. For example-style tests there is no
    ``file`` attribute, so we have to reconstruct the slashed path.

    Two real-world shapes we must handle:

    * Scripts collected by our custom collector — ``classname`` is dotted
      with the ``.py`` extension stripped (e.g.
      ``"examples.scripts.bert_comparator_demo"``) and ``name`` carries
      the basename with the extension (``"bert_comparator_demo.py"``).
    * Notebooks collected by ``nbmake`` — ``classname`` keeps the
      ``.ipynb`` extension because it is not a valid Python module suffix
      (e.g. ``"examples.notebooks.Quick_start.ipynb"``).

    The implementation translates every dot in ``classname`` to a slash,
    then re-attaches the file extension from ``name`` when ``name`` looks
    like a basename that ``classname`` lost.
    """
    if not classname:
        return name or "<unknown>"

    # ``classname`` already carries a recognized file extension — preserve
    # the final dot-before-extension and slash the rest.
    for ext in (".ipynb", ".py"):
        if classname.endswith(ext):
            stem = classname[: -len(ext)].replace(".", "/")
            return f"{stem}{ext}"

    parts = classname.split(".")
    if name and "." in name:
        # ``name`` is a basename (e.g. ``"bert_comparator_demo.py"``).
        # Drop the trailing dotted segment from ``classname`` if it is the
        # basename's stem; otherwise treat ``classname`` as a directory
        # path and append ``name``.
        name_stem = name.rsplit(".", 1)[0]
        if parts and parts[-1] == name_stem:
            parts = parts[:-1]
        return "/".join(parts + [name]) if parts else name

    # Plain dotted module path with no recognizable extension hint.
    path = "/".join(parts)
    if name and not path.endswith(name):
        path = f"{path}/{name}"
    return path or name or "<unknown>"


def _testcase_path(testcase: Element) -> str:
    """Return the repo-relative POSIX path for a ``<testcase>`` element."""
    file_attr = testcase.get("file")
    if file_attr:
        return file_attr.replace("\\", "/")
    return _classname_to_path(
        testcase.get("classname", ""),
        testcase.get("name", ""),
    )


def _failure_cell_suffix(testcase: Element) -> str:
    """Return ``" (cell N)"`` for nbmake notebook failures, else ``""``."""
    for child in testcase:
        tag = child.tag.rsplit("}", 1)[-1]  # strip XML namespace if present
        if tag not in {"failure", "error"}:
            continue
        haystack = _strip_ansi(
            " ".join(part for part in (child.get("message"), child.text) if part)
        )
        match = _CELL_RE.search(haystack)
        if match:
            return f" (cell {match.group(1) or match.group(2)})"
    return ""


def _skip_reason(testcase: Element) -> str:
    """Return the reason text for a ``<skipped>`` child, or an empty string."""
    for child in testcase:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "skipped":
            # Pytest stores the human-readable reason in ``message``; the
            # element text repeats the path/line prefix.
            return (child.get("message") or child.text or "").strip()
    return ""


def _classify(testcase: Element) -> tuple[str, str]:
    """Return ``(status, path)`` for a ``<testcase>`` element."""
    path = _testcase_path(testcase)
    has_failure = False
    has_skipped = False
    for child in testcase:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in {"failure", "error"}:
            has_failure = True
        elif tag == "skipped":
            has_skipped = True

    if has_failure:
        return "FAIL", f"{path}{_failure_cell_suffix(testcase)}"
    if has_skipped:
        reason = _skip_reason(testcase)
        if reason:
            return "SKIP", f"{path} ({reason})"
        return "SKIP", path
    return "PASS", path


_STATUS_LABEL = {
    "FAIL": "❌ FAIL",
    "SKIP": "⏭ SKIP",
    "PASS": "✅ PASS",
}


def _escape_cell(value: str) -> str:
    """Escape ``|`` and newlines so a value renders safely inside a table cell."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render(rows: list[tuple[str, str]]) -> str:
    """Render ``(status, path)`` rows as a Markdown table with a header."""
    lines = [HEADER, "", "| Status | Path |", "|---|---|"]
    sorted_rows = sorted(rows, key=lambda r: (_STATUS_ORDER[r[0]], r[1]))
    for status, path in sorted_rows:
        lines.append(f"| {_STATUS_LABEL[status]} | {_escape_cell(path)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(FALLBACK)
        return 0

    junit_path = Path(argv[1])
    if not junit_path.is_file():
        print(FALLBACK)
        return 0

    try:
        tree = parse(junit_path)
    except ParseError:
        print(FALLBACK)
        return 0

    root = tree.getroot()
    # Accept either a single <testsuite> root or a <testsuites> wrapper.
    rows: list[tuple[str, str]] = [_classify(tc) for tc in root.iter("testcase")]

    if not rows:
        print(FALLBACK)
        return 0

    sys.stdout.write(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
