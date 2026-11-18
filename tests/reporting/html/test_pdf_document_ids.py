"""Exercise the PDF viewer with literal document IDs in Node's VM.

The DOM adapter models ID lookup, not rendering or the browser's CSS parser.
PDF.js is stubbed so loading, page completion, navigation and failure display
can run offline. The browser counterpart is still needed for visual rendering.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "src/stickler/reporting/html/interactive/main.js"
HARNESS = Path(__file__).with_name("pdf_document_ids.cjs")


@pytest.mark.skipif(NODE is None, reason="PDF viewer behavior tests require Node.js")
@pytest.mark.parametrize(
    "doc_id",
    ["invoice_1", "invoice_1.pdf", "inv 001", "a:b", "a[b]", 'a"b', "a#b", "請求書"],
)
@pytest.mark.parametrize("fails_to_load", [False, True])
def test_pdf_viewer_uses_literal_document_ids(doc_id, fails_to_load):
    result = subprocess.run(
        [NODE, str(HARNESS), str(SCRIPT), doc_id, str(fails_to_load)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
