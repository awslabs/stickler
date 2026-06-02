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
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HEADER = "### Example execution summary"
FALLBACK = (
    f"{HEADER}\n\n_No results to display: pytest did not produce a JUnit report._"
)

# nbmake formats failing notebook cells as e.g. "Cell 4: NameError: ...".
# Match case-insensitively and capture the cell index.
_CELL_RE = re.compile(r"Cell\s+(\d+)", re.IGNORECASE)

# Sort key: failures first, then skips, then passes.
_STATUS_ORDER = {"FAIL": 0, "SKIP": 1, "PASS": 2}


def _testcase_path(testcase: ET.Element) -> str:
    """Return the repo-relative path for a ``<testcase>`` element.

    Pytest's JUnit writer puts the file path in the ``classname`` attribute
    (dotted) and the test name in ``name``. For example-style tests the
    ``file`` attribute is also present and is the most reliable source.
    """
    file_attr = testcase.get("file")
    if file_attr:
        return file_attr
    classname = testcase.get("classname", "")
    name = testcase.get("name", "")
    # Fall back to a best-effort reconstruction.
    if classname and name:
        return f"{classname}::{name}"
    return name or classname or "<unknown>"


def _failure_cell_suffix(testcase: ET.Element) -> str:
    """Return a ``" (cell N)"`` suffix for nbmake notebook failures, else ``""``."""
    for child in testcase:
        tag = child.tag.rsplit("}", 1)[-1]  # strip XML namespace if present
        if tag not in {"failure", "error"}:
            continue
        # Prefer the message attribute; fall back to the element text.
        haystack = child.get("message") or child.text or ""
        match = _CELL_RE.search(haystack)
        if match:
            return f" (cell {match.group(1)})"
    return ""


def _skip_reason(testcase: ET.Element) -> str:
    """Return the reason text for a ``<skipped>`` child, or an empty string."""
    for child in testcase:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "skipped":
            return (child.get("message") or child.text or "").strip()
    return ""


def _classify(testcase: ET.Element) -> tuple[str, str]:
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


def render(rows: list[tuple[str, str]]) -> str:
    """Render ``(status, path)`` rows as a Markdown table with a header."""
    lines = [HEADER, "", "| Status | Path |", "|---|---|"]
    sorted_rows = sorted(rows, key=lambda r: (_STATUS_ORDER[r[0]], r[1]))
    for status, path in sorted_rows:
        lines.append(f"| {_STATUS_LABEL[status]} | {path} |")
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
        tree = ET.parse(junit_path)
    except ET.ParseError:
        print(FALLBACK)
        return 0

    root = tree.getroot()
    # Accept either a single <testsuite> root or a <testsuites> wrapper.
    rows: list[tuple[str, str]] = []
    for testcase in root.iter("testcase"):
        rows.append(_classify(testcase))

    if not rows:
        print(FALLBACK)
        return 0

    sys.stdout.write(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
