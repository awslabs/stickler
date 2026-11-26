"""The object detection guide's examples print the output the guide shows.

Each ```python block runs in order in one namespace, and its output must equal
the ```text block that follows it.
"""

import contextlib
import io
import re
from pathlib import Path

GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "docs"
    / "Guides"
    / "Use-Cases"
    / "object-detection.md"
).read_text(encoding="utf-8")


def test_guide_examples_print_the_shown_output():
    pairs = re.findall(r"```python\n(.*?)```\s*```text\n(.*?)```", GUIDE, re.S)
    assert len(pairs) == len(re.findall(r"```python\n", GUIDE)) == 7
    namespace = {}
    for code, shown in pairs:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            # Runs this repo's own guide code blocks; no external input.
            exec(code, namespace)  # nosec B102
        assert out.getvalue() == shown
