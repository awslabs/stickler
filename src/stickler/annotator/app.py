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
    _get_stored_config,
    apply_config_from_query_params,
    render_config_dialog,
    render_config_sidebar,
)
from stickler.annotator.dataset import DatasetManager
from stickler.annotator.models import AnnotationMode, AnnotationState, DocumentStatus
from stickler.annotator.pdf_viewer import PDFViewer
from stickler.annotator.schema_builder import SchemaBuilder
from stickler.annotator.schema_loader import SchemaLoader
from stickler.annotator.serializer import AnnotationManifest, AnnotationSerializer, AnnotationSession

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


def _get_or_create_session(config: ConfigResult) -> AnnotationSession:
    """Get the active session from session state, or create/resume one.

    Priority:
    1. ``?session=<guid>`` query param → resume that specific session
    2. ``session_id`` already in st.session_state → reuse current session
    3. Otherwise → create a new session and store its GUID
    """
    import os
    manifest = AnnotationManifest(config.dataset_dir)
    schema_hash = _schema_hash(config.schema)

    # Check for session GUID in query params (deep link resume)
    params = st.query_params
    qp_session = params.get("session", "").strip()
    if qp_session and not st.session_state.get("_session_id"):
        existing = manifest.get_session(qp_session)
        if existing is not None:
            st.session_state["_session_id"] = qp_session
            return existing

    # Reuse session already active in this browser session
    active_id = st.session_state.get("_session_id")
    if active_id:
        existing = manifest.get_session(active_id)
        if existing is not None:
            return existing

    # Create a new session
    annotator = st.session_state.get("config_annotator") or os.getlogin()
    doc_count = len(list(config.dataset_dir.rglob("*.pdf")))
    session = manifest.create_session(
        schema=config.schema,
        schema_hash=schema_hash,
        annotator=annotator,
        doc_count=doc_count,
    )
    st.session_state["_session_id"] = session.session_id
    return session


def _load_or_create_state(
    pdf_path: Path, schema: dict, session: AnnotationSession | None = None
) -> AnnotationState:
    """Load existing annotations for *pdf_path*, or create a fresh state."""
    existing = AnnotationSerializer.load(pdf_path, session=session)
    if existing is not None:
        return existing
    now = _now_iso()
    return AnnotationState(
        schema_hash=_schema_hash(schema),
        fields={},
        created_at=now,
        updated_at=now,
    )


def _resume_or_fresh_state(
    pdf_path: Path, schema: dict, session: AnnotationSession | None = None
) -> AnnotationState | None:
    """Show Resume / Start Fresh prompt when an annotation already exists.

    The loaded/created state is cached in session_state so it survives
    Streamlit reruns triggered by field saves — keeping the progress counter
    and status dots accurate without re-reading disk on every interaction.
    """
    choice_key = f"resume_choice_{pdf_path}"
    state_key = f"annotation_state_{pdf_path}"
    choice = st.session_state.get(choice_key)

    # Return cached in-memory state if available (survives reruns from saves)
    cached = st.session_state.get(state_key)
    if cached is not None and choice is not None:
        return cached

    if not AnnotationSerializer.exists(pdf_path, session=session):
        state = _load_or_create_state(pdf_path, schema, session=session)
        st.session_state[state_key] = state
        return state

    if choice == "resume":
        state = AnnotationSerializer.load(pdf_path, session=session)
        state = state if state is not None else _load_or_create_state(pdf_path, schema, session=session)
        st.session_state[state_key] = state
        return state

    if choice == "fresh":
        now = _now_iso()
        state = AnnotationState(
            schema_hash=_schema_hash(schema),
            fields={},
            created_at=now,
            updated_at=now,
        )
        st.session_state[state_key] = state
        return state

    # No choice yet — show the prompt
    existing = AnnotationSerializer.load(pdf_path, session=session)
    updated = existing.updated_at[:19].replace("T", " ") if existing else ""
    st.info(f"An existing annotation was found for **{pdf_path.name}** (last saved: {updated} UTC).")
    col_resume, col_fresh = st.columns(2)
    with col_resume:
        if st.button("▶ Resume", key=f"btn_resume_{pdf_path}", use_container_width=True):
            st.session_state.pop(state_key, None)  # clear cache so fresh load happens
            st.session_state[choice_key] = "resume"
            st.rerun()
    with col_fresh:
        if st.button("🗑 Start Fresh", key=f"btn_fresh_{pdf_path}", use_container_width=True):
            st.session_state.pop(state_key, None)  # clear cache for blank state
            st.session_state[choice_key] = "fresh"
            st.rerun()
    return None


def _handle_schema_builder(config: ConfigResult | None) -> ConfigResult | None:
    """Show the Schema Builder UI and update config when finalized.

    Once the user clicks Finalize Schema, the builder is replaced with a
    success prompt so the page is clean before Apply Configuration is clicked.

    Returns an updated ConfigResult if the user finalizes a schema,
    otherwise returns the original config (which may be None).
    """
    # If schema was just finalized this cycle, show only the success prompt
    if st.session_state.get("schema_just_finalized"):
        st.success("Schema finalized! Click **Apply Configuration** in the sidebar to start annotating.")
        return config

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
        st.session_state["schema_just_finalized"] = True
        st.rerun()

    return config


def _render_document_queue(
    manager: DatasetManager, schema_fields: list[str], session=None
) -> Path | None:
    """Discover PDFs, show the document queue with prev/next buttons and a dropdown."""
    try:
        documents = manager.discover()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return None

    for doc in documents:
        doc.status = manager.get_status(doc.path, schema_fields, session=session)

    if not documents:
        st.info("No PDF documents found in the dataset directory.")
        return None

    labels = [
        f"{_STATUS_ICONS.get(doc.status, '⚪')} {doc.path.name}"
        for doc in documents
    ]

    n = len(documents)

    # Use a separate nav key so prev/next can set it before the selectbox renders
    nav_key = "_doc_nav_idx"
    current = st.session_state.get(nav_key, 0)
    current = max(0, min(current, n - 1))

    col_prev, col_select, col_next = st.columns([1, 8, 1])

    with col_prev:
        st.markdown("<div style='padding-top:4px'></div>", unsafe_allow_html=True)
        if st.button("◀", key="doc_prev", disabled=(current == 0), help="Previous document", use_container_width=True):
            st.session_state[nav_key] = current - 1
            st.rerun()

    with col_next:
        st.markdown("<div style='padding-top:4px'></div>", unsafe_allow_html=True)
        if st.button("▶", key="doc_next", disabled=(current == n - 1), help="Next document", use_container_width=True):
            st.session_state[nav_key] = current + 1
            st.rerun()

    with col_select:
        selected_idx = st.selectbox(
            "Document",
            range(n),
            index=current,
            format_func=lambda i: labels[i],
            key="doc_queue_select",
            label_visibility="collapsed",
        )
        # Sync nav key when user picks from dropdown directly
        if selected_idx != current:
            st.session_state[nav_key] = selected_idx

    if selected_idx is not None:
        return documents[selected_idx].path
    return None


def _app() -> None:
    """The Streamlit application — called when Streamlit runs this file."""
    st.set_page_config(page_title="KIE Annotation Tool", layout="wide")

    # Load .env credentials if present (for local dev / LLM backend)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Auto-apply config from URL query params (one-click start)
    apply_config_from_query_params()

    config = _get_stored_config()
    is_configured = config is not None

    # --- Compact top bar: title + config summary + gear ---
    if is_configured:
        dataset_name = Path(config.dataset_dir).name
        schema_path_stored = st.session_state.get("config_schema_path", "")
        schema_name = schema_path_stored.split("/")[-1] if schema_path_stored else (
            config.schema.get("title", "schema")
        )
        mode_label = config.mode.value.replace("_", " ").title()

        # Build deep link — session GUID added after session is created below
        session_id = st.session_state.get("_session_id", "")
        import urllib.parse
        if session_id:
            deep_link_params = urllib.parse.urlencode({
                "dataset": str(config.dataset_dir),
                "session": session_id,
            })
        else:
            deep_link_params = urllib.parse.urlencode({
                "dataset": str(config.dataset_dir),
                "schema": schema_path_stored,
                "mode": config.mode.value,
            })
        deep_link = f"http://localhost:8501/?{deep_link_params}"

        _, link_col, gear_col = st.columns([8, 3, 1])
        with link_col:
            st.markdown(
                f"<div style='padding-top:6px;text-align:right'>"
                f"<span style='font-size:11px;color:#aaa'>🔗 </span>"
                f"<code style='font-size:11px;color:#888;background:none'>{deep_link}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with gear_col:
            if st.button("⚙️", key="open_config", help="Configure dataset, schema, and mode"):
                render_config_dialog()

        st.markdown(
            f"<div style='padding:4px 0 8px 0'>"
            f"<span style='font-size:20px;font-weight:700'>KIE Annotation Tool</span>"
            f"&nbsp;&nbsp;"
            f"<span style='font-size:12px;color:#888'>📁 {dataset_name} &nbsp;·&nbsp; 📋 {schema_name} &nbsp;·&nbsp; ⚡ {mode_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        _, gear_col = st.columns([20, 1])
        with gear_col:
            if st.button("⚙️", key="open_config", help="Configure dataset, schema, and mode"):
                render_config_dialog()
        st.markdown(
            "<div style='padding:4px 0 8px 0'>"
            "<span style='font-size:20px;font-weight:700'>KIE Annotation Tool</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:0 0 8px 0'>", unsafe_allow_html=True)

    # Not yet configured — centered prompt
    if not is_configured:
        st.markdown("---")
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown(
                "<div style='text-align:center;padding:40px 0'>"
                "<div style='font-size:48px'>⚙️</div>"
                "<h3>Get started</h3>"
                "<p style='color:#888'>Click the gear icon above to configure your dataset directory, schema, and annotation mode.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        return

    # 2. Schema Builder — only shown when builder is selected AND schema not yet finalized
    schema_source = st.session_state.get("config_schema_source", "")
    schema_finalized = st.session_state.get("config_validated", False)

    if schema_source == SCHEMA_SOURCE_BUILDER and not schema_finalized:
        config = _handle_schema_builder(config)

    if schema_finalized:
        st.session_state.pop("schema_just_finalized", None)

    if config is None:
        return

    # Edit Schema button (builder mode only)
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

    # Get or create the annotation session (manifest-based)
    session = _get_or_create_session(config)

    # Update deep link to include session GUID
    import urllib.parse
    deep_link_params = urllib.parse.urlencode({
        "dataset": str(config.dataset_dir),
        "session": session.session_id,
    })
    deep_link = f"http://localhost:8502/?{deep_link_params}"

    # 4. Side-by-side layout: PDF viewer (left) + annotation panel (right)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        selected_pdf = _render_document_queue(manager, schema_fields, session=session)
        if selected_pdf is None:
            return

        last_pdf_key = "last_selected_pdf"
        if st.session_state.get(last_pdf_key) != str(selected_pdf):
            # Clear cached annotation state for the previous document
            old_pdf = st.session_state.get(last_pdf_key)
            if old_pdf:
                st.session_state.pop(f"annotation_state_{old_pdf}", None)
            st.session_state[last_pdf_key] = str(selected_pdf)

        viewer = PDFViewer(selected_pdf)
        viewer.render()

    with right_col:
        if selected_pdf is None:
            return
        state = _resume_or_fresh_state(selected_pdf, config.schema, session=session)
        if state is None:
            return

        # Build prefill function — available in LLM Inference and Zero Start modes
        prefill_fn = None
        if config.mode in (AnnotationMode.LLM_INFERENCE, AnnotationMode.ZERO_START):
            try:
                from stickler.annotator.llm_backend import BedrockLLMBackend
                _backend_key = "_llm_backend"
                if _backend_key not in st.session_state:
                    st.session_state[_backend_key] = BedrockLLMBackend()
                backend = st.session_state[_backend_key]
                prefill_fn = backend.prefill
            except Exception as exc:
                logger.debug("LLM backend unavailable: %s", exc)

        panel = AnnotationPanel(config.schema, config.mode, state, selected_pdf, session=session, prefill_fn=prefill_fn)
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
