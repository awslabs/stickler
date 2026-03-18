"""Main Streamlit entry point for the KIE Annotation Tool.

Orchestrates all modules into a single-page application:

1. **Sidebar** — configuration via :func:`render_config_sidebar`.
2. **Schema Builder** — shown when the "Schema Builder" source is selected.
3. **Document queue** — PDF discovery with status indicators.
4. **Side-by-side layout** — PDF viewer (left) + annotation panel (right).

Launch via the ``stickler-annotate`` console script or::

    streamlit run src/stickler/annotator/app.py
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from stickler.annotator.annotation_panel import AnnotationPanel
from stickler.annotator.config import (
    SCHEMA_SOURCE_BUILDER,
    ConfigResult,
    render_config_sidebar,
)
from stickler.annotator.dataset import DatasetManager
from stickler.annotator.models import AnnotationMode, AnnotationState, DocumentStatus
from stickler.annotator.pdf_viewer import PDFViewer
from stickler.annotator.schema_builder import SchemaBuilder
from stickler.annotator.schema_loader import SchemaLoader
from stickler.annotator.serializer import AnnotationSerializer

logger = logging.getLogger(__name__)

# Status emoji mapping
_STATUS_ICONS: dict[DocumentStatus, str] = {
    DocumentStatus.NOT_STARTED: "🔴",
    DocumentStatus.IN_PROGRESS: "🟡",
    DocumentStatus.COMPLETE: "🟢",
}


def _schema_hash(schema: dict) -> str:
    """Compute a stable MD5 hash of a JSON Schema dict."""
    return hashlib.md5(
        json.dumps(schema, sort_keys=True).encode()
    ).hexdigest()


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _load_or_create_state(
    pdf_path: Path, schema: dict
) -> AnnotationState:
    """Load existing annotations for *pdf_path*, or create a fresh state."""
    existing = AnnotationSerializer.load(pdf_path)
    if existing is not None:
        return existing
    now = _now_iso()
    return AnnotationState(
        schema_hash=_schema_hash(schema),
        fields={},
        created_at=now,
        updated_at=now,
    )


def _handle_schema_builder(config: ConfigResult | None) -> ConfigResult | None:
    """Show the Schema Builder UI and update config when finalized.

    Returns an updated ConfigResult if the user finalizes a schema,
    otherwise returns the original config (which may be None).
    """
    builder = SchemaBuilder()
    schema = builder.render()

    if schema is not None:
        # Convert builder output to a model class
        try:
            validated_schema, model_class = SchemaLoader.from_builder_schema(schema)
        except ValueError as exc:
            st.error(f"Schema validation failed: {exc}")
            return config

        # Store in session state so config sidebar can pick it up
        st.session_state["config_schema"] = validated_schema
        st.session_state["config_model_class"] = model_class
        st.session_state["config_validated"] = True
        st.success("Schema finalized! Click **Apply Configuration** in the sidebar.")

        if config is not None:
            return ConfigResult(
                dataset_dir=config.dataset_dir,
                schema_source=config.schema_source,
                mode=config.mode,
                schema=validated_schema,
                model_class=model_class,
            )

    return config


def _render_document_queue(
    manager: DatasetManager, schema_fields: list[str]
) -> Path | None:
    """Discover PDFs, show the document queue with status indicators.

    Returns the selected PDF path, or None if no selection.
    """
    try:
        documents = manager.discover()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return None

    # Derive status for each document
    for doc in documents:
        doc.status = manager.get_status(doc.path, schema_fields)

    if not documents:
        st.info("No PDF documents found in the dataset directory.")
        return None

    # Build display labels with status icons
    labels = [
        f"{_STATUS_ICONS.get(doc.status, '⚪')} {doc.path.name}"
        for doc in documents
    ]

    selected_idx = st.selectbox(
        "Document queue",
        range(len(documents)),
        format_func=lambda i: labels[i],
        key="doc_queue_select",
    )

    if selected_idx is not None:
        return documents[selected_idx].path
    return None


def _app() -> None:
    """The Streamlit application — called when Streamlit runs this file."""
    st.set_page_config(page_title="KIE Annotation Tool", layout="wide")
    st.title("KIE Annotation Tool")

    # 1. Configuration sidebar
    config = render_config_sidebar()

    # 2. Schema Builder — only shown when builder is selected AND schema
    #    is not yet finalized.  Once config is validated the builder
    #    disappears and the annotation workflow takes over.
    schema_source = st.session_state.get("config_schema_source", "")
    schema_finalized = st.session_state.get("config_validated", False)

    if schema_source == SCHEMA_SOURCE_BUILDER and not schema_finalized:
        config = _handle_schema_builder(config)

    if config is None:
        st.info("Configure the tool in the sidebar to get started.")
        return

    # 2b. Allow re-opening the schema builder if needed
    if schema_source == SCHEMA_SOURCE_BUILDER and schema_finalized:
        if st.button("✏️ Edit Schema", key="reopen_schema_builder"):
            st.session_state["config_validated"] = False
            st.rerun()

    # 3. Dataset discovery and document queue
    try:
        manager = DatasetManager(config.dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    schema_fields = list(config.schema.get("properties", {}).keys())
    selected_pdf = _render_document_queue(manager, schema_fields)

    if selected_pdf is None:
        return

    # 4. Side-by-side layout: PDF viewer (left) + annotation panel (right)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        viewer = PDFViewer(selected_pdf)
        viewer.render()

    with right_col:
        state = _load_or_create_state(selected_pdf, config.schema)
        panel = AnnotationPanel(
            config.schema, config.mode, state, selected_pdf
        )
        panel.render()


def main() -> None:
    """Console script entry point — launches Streamlit."""
    import sys

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.headless",
        "true",
    ]
    from streamlit.web.cli import main as st_main

    st_main()


# When Streamlit runs this file, execute the app
_app()
