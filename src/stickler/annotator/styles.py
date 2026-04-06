"""Inject CSS and JS from static files into the Streamlit app.

Keeps app.py clean — call ``inject_styles()`` once at the top of ``_app()``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_STATIC_DIR = Path(__file__).parent / "static"


def inject_styles() -> None:
    """Load style.css and init.js from the static/ directory."""
    css = (_STATIC_DIR / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    js = (_STATIC_DIR / "init.js").read_text(encoding="utf-8")
    components.html(f"<script>{js}</script>", height=0)
