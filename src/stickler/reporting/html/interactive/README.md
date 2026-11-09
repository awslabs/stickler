# Interactive HTML reports

`main.js` is embedded in reports with individual results. It filters document
metrics and controls the image/PDF gallery. PDF.js supplies loading and rendering.

Document IDs are data, not CSS selectors. Resolve PDF canvases, loading/error
messages and page controls with `document.getElementById`; keep the original IDs
so documents containing spaces or punctuation remain distinct.

The offline PDF behavior tests live in `tests/reporting/html/test_pdf_document_ids.py`.
They require Node.js and stub the DOM and PDF.js, so they do not prove browser
rendering or real PDF decoding.
