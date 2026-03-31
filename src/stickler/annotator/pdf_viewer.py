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
        self.pdf_path = Path(pdf_path).resolve()
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
    # Bounding box overlay
    # ------------------------------------------------------------------

    # Color palette for field boxes (cycling through for distinct fields)
    _BOX_COLORS = [
        "#e63946",
        "#457b9d",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#264653",
        "#a8dadc",
        "#6a4c93",
        "#1982c4",
        "#8ac926",
    ]

    def _draw_field_boxes(
        self, image: Image.Image, page_num: int, field_locations: dict
    ) -> Image.Image:
        """Draw labeled bounding boxes on the page image.

        Args:
            image: The rendered page as a PIL Image.
            page_num: 1-indexed current page number.
            field_locations: {field_name: FieldLocation} dict.

        Returns:
            A copy of the image with boxes drawn.
        """
        from PIL import ImageDraw, ImageFont

        # Filter for fields on this page
        page_fields = {
            name: loc for name, loc in field_locations.items() if loc.page == page_num
        }
        if not page_fields:
            return image

        # Work on a copy with alpha channel for semi-transparent fills
        img = image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size

        font_size = max(11, h // 100)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", size=font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        used_label_rects: list[tuple[int, int, int, int]] = []

        for idx, (field_name, loc) in enumerate(page_fields.items()):
            color_hex = self._BOX_COLORS[idx % len(self._BOX_COLORS)]
            # Parse hex to RGB tuple
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            color_rgb = (r, g, b)
            color_fill = (r, g, b, 18)  # very light fill so text remains readable

            # Convert 0-1000 scaled coords to pixel coords
            # bbox format: [x1, y1, x2, y2]
            x1 = int(loc.bbox[0] / 1000 * w)
            y1 = int(loc.bbox[1] / 1000 * h)
            x2 = int(loc.bbox[2] / 1000 * w)
            y2 = int(loc.bbox[3] / 1000 * h)

            # Draw semi-transparent fill + solid border
            draw.rectangle(
                [x1, y1, x2, y2], fill=color_fill, outline=color_rgb, width=2
            )

            # Label with pill background — find a non-overlapping position
            label = field_name
            bbox_text = font.getbbox(label)
            tw = bbox_text[2] - bbox_text[0] + 8
            th = bbox_text[3] - bbox_text[1] + 4

            # Try positions: above box, below box, inside top-left
            candidates = [
                (x1, y1 - th - 2),  # above
                (x1, y2 + 2),  # below
                (x1 + 2, y1 + 2),  # inside top-left
            ]

            lx, ly = candidates[0]  # default
            for cx, cy in candidates:
                # Clamp to image bounds
                cx = max(0, min(cx, w - tw))
                cy = max(0, min(cy, h - th))
                label_rect = (cx, cy, cx + tw, cy + th)
                # Check overlap with existing labels
                overlaps = any(
                    not (
                        label_rect[2] < er[0]
                        or label_rect[0] > er[2]
                        or label_rect[3] < er[1]
                        or label_rect[1] > er[3]
                    )
                    for er in used_label_rects
                )
                if not overlaps:
                    lx, ly = cx, cy
                    break
            else:
                lx = max(0, min(lx, w - tw))
                ly = max(0, min(ly, h - th))

            label_rect = (lx, ly, lx + tw, ly + th)
            used_label_rects.append(label_rect)

            # Draw pill background
            draw.rounded_rectangle(label_rect, radius=3, fill=(r, g, b, 200))
            draw.text((lx + 4, ly + 1), label, fill=(255, 255, 255, 255), font=font)

        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    # ------------------------------------------------------------------
    # Streamlit UI
    # ------------------------------------------------------------------

    def render(self, field_locations: dict | None = None) -> None:
        """Render the PDF viewer with page navigation in Streamlit.

        Args:
            field_locations: Optional dict of {field_name: FieldLocation} to
                draw bounding box overlays on the rendered page.
        """
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
            if st.button(
                "⬅ Prev Page", disabled=(current_page <= 1), key=f"prev_{self.pdf_path}"
            ):
                st.session_state[state_key] = max(1, current_page - 1)
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align:center'>Page {current_page} / {total}</div>",
                unsafe_allow_html=True,
            )

        with col_next:
            if st.button(
                "Next Page ➡",
                disabled=(current_page >= total),
                key=f"next_{self.pdf_path}",
            ):
                st.session_state[state_key] = min(total, current_page + 1)
                st.rerun()

        with col_rotate:
            rot_key = f"pdf_rotation_{self.pdf_path}"
            if rot_key not in st.session_state:
                st.session_state[rot_key] = 0
            if st.button(
                "↻", key=f"rotate_{self.pdf_path}", help="Rotate 90° clockwise"
            ):
                st.session_state[rot_key] = (st.session_state[rot_key] + 90) % 360
                st.rerun()

        # --- page image ---
        try:
            page_image = self.render_page(current_page)
            rotation = st.session_state.get(f"pdf_rotation_{self.pdf_path}", 0)
            if rotation:
                page_image = page_image.rotate(-rotation, expand=True)

            # Draw bounding box overlays for fields on this page
            if field_locations:
                page_image = self._draw_field_boxes(
                    page_image, current_page, field_locations
                )

            st.image(page_image, use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to render page {current_page}: {exc}")
