"""The report's PDF viewer must not run a crafted PDF's script.

The report loads pdf.js 3.11.174 from a CDN, which is affected by CVE-2024-4367: with
``isEvalSupported`` at its default of ``true``, a crafted font in an attached PDF runs script in the
page's origin. ``loadPDF`` passes ``isEvalSupported: false`` so that route is closed without
changing the rendered output.

``test_the_pdf_viewer_disables_eval`` in ``test_html_reporter.py`` asserts the option is present.
The tests here add that it is on the page's only loader call, when the viewer is reachable at
all, and which pdf.js the page loads.

**What these tests prove and what they do not.** None of them proves the payload fails to
execute -- that needs a browser, and there is none here and none in CI. That measurement is the reviewer's, in headless Chrome, on PR #452: with only this
change the same PDF still rendered and the payload did not run. Do not read a green here as the
browser-level result.

Upgrading pdf.js to 4.2.67 or later, vendored or pinned with an ``integrity=`` hash, is the complete
fix and is not part of this change. These tests must keep
passing until that lands, and the version assertion below is what tells you it has not.

Note that ``test_the_pinned_pdfjs_version_has_not_moved`` asserts ``integrity=`` is *absent*, so
the upgrade work must invert that assertion rather than treat its failure as a regression.
"""

import json
import re

from stickler.reporting.html.html_reporter import EvaluationHTMLReporter
from stickler.reporting.html.report_config import ReportConfig

RESULTS = {
    "fields": {"vendor": {"cm_precision": 0.5, "cm_recall": 0.5, "cm_f1": 0.5, "tp": 1, "fd": 0, "fa": 0, "fn": 0}},
    "overall_score": 0.5,
}


def _pdf_report(tmp_path, with_jsonl):
    """Generate a PDF-gallery report, optionally on the path that embeds the viewer."""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 not a real pdf")
    kwargs = {}
    if with_jsonl:
        jsonl = tmp_path / "results.jsonl"
        jsonl.write_text(json.dumps({"doc_id": "d1", "field_results": []}) + "\n")
        kwargs["individual_results_jsonl_path"] = str(jsonl)

    out = tmp_path / "report.html"
    report = EvaluationHTMLReporter().generate_report(
        RESULTS,
        str(out),
        config=ReportConfig(document_file_type="pdf"),
        document_files={"d1": str(pdf)},
        **kwargs,
    )
    assert report.success, report.errors
    return out.read_text()


def test_the_page_has_one_pdf_loader_call_and_it_disables_eval(tmp_path):
    """Every ``pdfjsLib.getDocument`` call in the page must carry the option.

    Counted, not only searched for: a check for the safe form alone still passes
    when a second loader is added without it. The count matches a
    ``pdfjsLib.getDocument(...)`` call with an argument, not any ``getDocument(``,
    so a prose mention such as ``pdfjsLib.getDocument()`` in a comment, or a pdf.js
    vendored into the page later, does not trip it.
    """
    page = _pdf_report(tmp_path, with_jsonl=True)
    calls = re.findall(r"pdfjsLib\s*\.\s*getDocument\s*\(\s*[^\s)]", page)
    assert len(calls) == 1, (
        f"expected one pdfjsLib.getDocument call in the page, found {len(calls)}; "
        "every call must pass isEvalSupported: false (CVE-2024-4367)"
    )
    assert "pdfjsLib.getDocument({ url, isEvalSupported: false })" in page


def test_the_viewer_is_not_embedded_without_individual_results(tmp_path):
    """Reachability, pinned because it narrows who was ever exposed.

    ``main.js`` is only embedded when individual results are loaded, so without
    ``individual_results_jsonl_path`` the gallery emits its canvas markup but no
    code ever calls ``getDocument``. That means not passing the JSONL path closed
    the PDF route too, not only the ``<script>`` blob route.
    """
    page = _pdf_report(tmp_path, with_jsonl=False)
    assert "pdf-canvas-" in page, "the gallery markup should still render"
    assert "getDocument" not in page


def test_the_pinned_pdfjs_version_has_not_moved(tmp_path):
    """Fails when pdf.js is upgraded, so the note above gets revisited.

    The pdf.js ``<script>`` tag loads 3.11.174 from a CDN and carries no
    ``integrity`` hash. Both facts are asserted on that tag alone, so a change to
    either is a decision rather than a drift.
    """
    page = _pdf_report(tmp_path, with_jsonl=False)
    tag = re.search(r"<script[^>]*pdf\.js/[^>]*>", page)
    assert tag, "the pdf.js <script> tag is missing from the page"
    assert "pdf.js/3.11.174/pdf.min.js" in tag.group(0), (
        "pdf.js moved off 3.11.174. If this is the upgrade, revisit the CVE-2024-4367 note "
        "above, update the version here, and invert the integrity assertion below rather "
        "than deleting it"
    )
    assert "integrity=" not in tag.group(0), (
        "the pdf.js tag gained an integrity hash -- that is the upgrade: invert this "
        "assertion, do not delete it"
    )
