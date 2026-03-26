"""PDF page rendering with lazy loading and navigation.

Uses ``pdf2image`` (poppler-based) to convert individual PDF pages to PIL
Images on demand.  Only the currently displayed page is rendered, so
documents with 100+ pages load without blocking the interface.

The viewer is designed to sit in the left column of a side-by-side layout
with the annotation panel on the right.  Page number is persisted in
``st.session_state`` so it survives Streamlit reruns.

Requires the ``poppler-utils`` system package.  If poppler is missing the
module surfaces a clear error message instead of crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — pdf2image depends on poppler being installed.  We defer the
# import so the rest of the annotator package can load even when poppler is
# absent (useful for tests that don't exercise the viewer).
# ---------------------------------------------------------------------------

_PDF2IMAGE_AVAILABLE: bool | None = None


def _check_pdf2image() -> bool:
    """Return True if pdf2image is importable, cache the result."""
    global _PDF2IMAGE_AVAILABLE  # noqa: PLW0603
    if _PDF2IMAGE_AVAILABLE is None:
        try:
            import pdf2image  # noqa: F401

            _PDF2IMAGE_AVAILABLE = True
        except ImportError:
            _PDF2IMAGE_AVAILABLE = False
    return _PDF2IMAGE_AVAILABLE


class PDFViewer:
    """Renders a single PDF with lazy per-page loading and prev/next navigation.

    Parameters
    ----------
    pdf_path:
        Filesystem path to the PDF file.
    pages_per_batch:
        Reserved for future batch-loading support.  In v1 we always render
        one page at a time.
    """

    def __init__(self, pdf_path: Path, pages_per_batch: int = 5) -> None:
        self.pdf_path = Path(pdf_path)
        self.pages_per_batch = pages_per_batch
        self._total_pages: int | None = None

    # ------------------------------------------------------------------
    # Page count
    # ------------------------------------------------------------------

    @property
    def total_pages(self) -> int:
        """Return the total number of pages, fetched lazily once."""
        if self._total_pages is None:
            self._total_pages = self._get_page_count()
        return self._total_pages

    def _get_page_count(self) -> int:
        """Use ``pdfinfo_from_path`` to get page count without rendering."""
        if not _check_pdf2image():
            logger.warning("pdf2image is not installed — cannot determine page count")
            return 0

        from pdf2image import pdfinfo_from_path

        try:
            info = pdfinfo_from_path(str(self.pdf_path))
            return int(info.get("Pages", 0))
        except Exception:
            logger.exception("Failed to read PDF info for %s", self.pdf_path)
            return 0

    # ------------------------------------------------------------------
    # Single-page rendering
    # ------------------------------------------------------------------

    def render_page(self, page_num: int) -> Image.Image:
        """Convert a single 1-indexed page to a PIL Image.

        Parameters
        ----------
        page_num:
            1-indexed page number.

        Returns
        -------
        PIL.Image.Image
            The rendered page as an RGB image.

        Raises
        ------
        RuntimeError
            If ``pdf2image`` / poppler is not available.
        ValueError
            If *page_num* is out of range.
        """
        if not _check_pdf2image():
            raise RuntimeError(
                "pdf2image is not installed.  Install it with "
                "'pip install pdf2image' and ensure poppler-utils is "
                "available on your system."
            )

        if page_num < 1:
            raise ValueError(f"page_num must be >= 1, got {page_num}")

        from pdf2image import convert_from_path

        images = convert_from_path(
            str(self.pdf_path),
            first_page=page_num,
            last_page=page_num,
        )

        if not images:
            raise ValueError(
                f"No image returned for page {page_num} of {self.pdf_path}"
            )

        return images[0]

    # ------------------------------------------------------------------
    # Streamlit UI
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Render the PDF viewer with page navigation in Streamlit."""
        import streamlit as st

        if not _check_pdf2image():
            st.error(
                "**pdf2image is not installed.**  "
                "Install it with `pip install pdf2image` and ensure "
                "`poppler-utils` is available on your system."
            )
            return

        total = self.total_pages
        if total == 0:
            st.warning("Could not determine page count for this PDF.")
            return

        # --- session-state key scoped to this PDF path ---
        state_key = f"pdf_page_{self.pdf_path}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 1

        current_page: int = st.session_state[state_key]

        # --- navigation controls ---
        col_prev, col_info, col_next, col_rotate = st.columns([1, 2, 1, 0.5])

        with col_prev:
            if st.button("⬅ Prev Page", disabled=(current_page <= 1), key=f"prev_{self.pdf_path}"):
                st.session_state[state_key] = max(1, current_page - 1)
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align:center'>Page {current_page} / {total}</div>",
                unsafe_allow_html=True,
            )

        with col_next:
            if st.button("Next Page ➡", disabled=(current_page >= total), key=f"next_{self.pdf_path}"):
                st.session_state[state_key] = min(total, current_page + 1)
                st.rerun()

        with col_rotate:
            rot_key = f"pdf_rotation_{self.pdf_path}"
            if rot_key not in st.session_state:
                st.session_state[rot_key] = 0
            if st.button("↻", key=f"rotate_{self.pdf_path}", help="Rotate 90° clockwise"):
                st.session_state[rot_key] = (st.session_state[rot_key] + 90) % 360
                st.rerun()

        # --- page image ---
        try:
            page_image = self.render_page(current_page)
            rotation = st.session_state.get(f"pdf_rotation_{self.pdf_path}", 0)
            if rotation:
                page_image = page_image.rotate(-rotation, expand=True)
            st.image(page_image, use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to render page {current_page}: {exc}")
