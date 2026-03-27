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
)
from stickler.annotator.dataset import DatasetManager
from stickler.annotator.models import AnnotationMode, AnnotationState, DocumentStatus
from stickler.annotator.pdf_viewer import PDFViewer
from stickler.annotator.schema_builder import SchemaBuilder
from stickler.annotator.schema_loader import SchemaLoader
from stickler.annotator.serializer import (
    AnnotationManifest,
    AnnotationSerializer,
    AnnotationSession,
)

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
        json.dumps(schema, sort_keys=True).encode(), usedforsecurity=False
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

    # If we're inside an active session, auto-resume — no prompt needed
    if session is not None and st.session_state.get("_session_id"):
        state = AnnotationSerializer.load(pdf_path, session=session)
        state = state if state is not None else _load_or_create_state(pdf_path, schema, session=session)
        st.session_state[state_key] = state
        st.session_state[choice_key] = "resume"
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


@st.dialog("Select Document", width="large")
def _show_document_picker(documents, labels, status_list):
    """Dialog with search, sort, and status filter for document selection."""
    col_search, col_sort, col_filter = st.columns([3, 2, 2])
    with col_search:
        search = st.text_input("Search", placeholder="Type to filter…", key="_doc_search", label_visibility="collapsed")
    with col_sort:
        sort_by = st.selectbox("Sort", ["Name", "Status"], key="_doc_sort", label_visibility="collapsed")
    with col_filter:
        status_filter = st.selectbox("Filter", ["All", "Not Started", "In Progress", "Complete"], key="_doc_filter", label_visibility="collapsed")

    # Build filtered + sorted list
    status_map = {"Not Started": DocumentStatus.NOT_STARTED, "In Progress": DocumentStatus.IN_PROGRESS, "Complete": DocumentStatus.COMPLETE}
    filtered = []
    for i, doc in enumerate(documents):
        name = doc.path.name
        if search and search.lower() not in name.lower():
            continue
        if status_filter != "All" and doc.status != status_map.get(status_filter):
            continue
        filtered.append((i, doc, labels[i]))

    if sort_by == "Status":
        order = {DocumentStatus.NOT_STARTED: 0, DocumentStatus.IN_PROGRESS: 1, DocumentStatus.COMPLETE: 2}
        filtered.sort(key=lambda x: (order.get(x[1].status, 9), x[1].path.name))
    else:
        filtered.sort(key=lambda x: x[1].path.name)

    st.markdown(f"<div style='color:#888;font-size:11px;padding:2px 0'>{len(filtered)} of {len(documents)} documents</div>", unsafe_allow_html=True)

    for idx, doc, label in filtered:
        col_name, col_btn = st.columns([5, 1])
        with col_name:
            st.markdown(f"<div style='padding:4px 0;font-size:13px'>{label}</div>", unsafe_allow_html=True)
        with col_btn:
            if st.button("Select", key=f"_pick_doc_{idx}", use_container_width=True):
                st.session_state["_doc_nav_idx"] = idx
                st.rerun()


def _render_document_queue(
    manager: DatasetManager, schema_fields: list[str], session=None
) -> Path | None:
    """Discover PDFs, show prev/next + current doc label + picker button."""
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
    status_list = [doc.status for doc in documents]

    n = len(documents)
    nav_key = "_doc_nav_idx"

    if nav_key not in st.session_state:
        # Check for doc query param to restore position on reload
        doc_param = st.query_params.get("doc", "").strip()
        initial_idx = 0
        if doc_param:
            for i, doc in enumerate(documents):
                if doc.path.name == doc_param:
                    initial_idx = i
                    break
        st.session_state[nav_key] = initial_idx
    current = max(0, min(st.session_state[nav_key], n - 1))

    # Keep doc query param in sync so reload preserves position
    st.query_params["doc"] = documents[current].path.name

    col_prev, col_next, col_label, col_pick = st.columns([1.2, 1.2, 4, 1.2])

    with col_prev:
        if st.button("◀ Prev Doc", key="doc_prev", disabled=(current == 0), help="Previous document", use_container_width=True):
            st.session_state[nav_key] = current - 1
            st.rerun()

    with col_next:
        if st.button("Next Doc ▶", key="doc_next", disabled=(current == n - 1), help="Next document", use_container_width=True):
            st.session_state[nav_key] = current + 1
            st.rerun()

    with col_label:
        st.markdown(
            f"<div style='padding:8px 0;font-size:13px;text-align:center'>"
            f"{labels[current]}"
            f"<span style='color:#aaa;font-size:11px'> &nbsp;({current + 1}/{n})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_pick:
        if st.button("📋 Select", key="doc_picker_btn", help="Browse all documents", use_container_width=True):
            _show_document_picker(documents, labels, status_list)

    return documents[current].path


def _render_landing_page() -> bool:
    """Show the landing page: dataset directory input + session discovery.

    If the user picks a directory with existing sessions, auto-configures
    and returns True. If the user starts a new annotation, opens the config
    dialog and returns True after apply. Returns False if still waiting for
    user input.
    """
    from stickler.annotator.config import (
        SCHEMA_SOURCE_JSON,
        _KEY_DATASET_DIR,
        _KEY_MODE,
        _KEY_SCHEMA,
        _KEY_SCHEMA_PATH,
        _KEY_SCHEMA_SOURCE,
        _KEY_MODEL_CLASS,
        _KEY_VALIDATED,
    )

    st.markdown(
        "<div style='text-align:center;padding:20px 0 10px 0'>"
        "<span style='font-size:28px;font-weight:700'>KIE Annotation Tool</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        dataset_dir = st.text_input(
            "Dataset directory",
            value=st.session_state.get(_KEY_DATASET_DIR, ""),
            help="Path to a directory containing PDF files (and optionally .annotations/)",
            placeholder="./files",
            key="landing_dataset_dir",
        )

        if not dataset_dir.strip():
            st.markdown(
                "<div style='text-align:center;padding:30px 0;color:#888'>"
                "Enter a dataset directory to get started."
                "</div>",
                unsafe_allow_html=True,
            )
            return False

        dataset_path = Path(dataset_dir.strip())
        if not dataset_path.exists() or not dataset_path.is_dir():
            st.error(f"Directory not found: {dataset_path}")
            return False

        # Check for existing sessions
        manifest = AnnotationManifest(dataset_path)
        sessions = manifest.list_sessions()

        if sessions:
            st.markdown("##### Existing Sessions")
            # Count actual PDFs for accurate totals
            actual_pdf_count = len([p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"])
            for sess in sessions:
                sid = sess["session_id"]
                sid_short = sid[:8]
                annotator = sess.get("annotator", "unknown")
                updated = sess.get("updated_at", "")[:19].replace("T", " ")
                doc_count = actual_pdf_count

                # Count completed/in-progress from actual annotation files
                import json as _json
                _sess_dir = dataset_path / ".annotations" / sid
                _completed = 0
                _annotated = 0
                if _sess_dir.exists():
                    for _f in _sess_dir.glob("*.json"):
                        try:
                            _data = _json.loads(_f.read_text())
                            _metadata = _data.get("metadata", {})
                            _fields_meta = _metadata.get("fields", {})
                            _data_fields = _data.get("data", {})
                            _num_annotated = len(_fields_meta)
                            _num_total = len(_data_fields)
                            _annotated += 1
                            if _num_total > 0 and _num_annotated >= _num_total:
                                _completed += 1
                        except Exception:
                            logger.debug("Skipping unreadable annotation file: %s", _f)
                pct = int(100 * _completed / doc_count) if doc_count else 0

                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"<div style='padding:8px 0'>"
                        f"<code style='font-size:10px;color:#999;background:#f1f5f9;padding:1px 4px;border-radius:3px'>{sid_short}</code>"
                        f"&nbsp;&nbsp;"
                        f"<span style='font-size:13px'>👤 {annotator}</span>"
                        f"&nbsp;&nbsp;"
                        f"<span style='color:#888;font-size:12px'>"
                        f"📄 {_completed}/{doc_count} docs ({pct}%)"
                        f"&nbsp;·&nbsp;{updated} UTC"
                        f"</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("▶ Resume", key=f"resume_session_{sid}", use_container_width=True):
                        session_obj = manifest.get_session(sid)
                        if session_obj and session_obj.schema:
                            try:
                                from stickler.annotator.schema_loader import SchemaLoader
                                _, model_class = SchemaLoader.from_builder_schema(session_obj.schema)
                                st.session_state[_KEY_DATASET_DIR] = dataset_dir.strip()
                                st.session_state[_KEY_SCHEMA_SOURCE] = SCHEMA_SOURCE_JSON
                                st.session_state[_KEY_SCHEMA_PATH] = ""
                                st.session_state[_KEY_MODE] = AnnotationMode.ZERO_START
                                st.session_state[_KEY_SCHEMA] = session_obj.schema
                                st.session_state[_KEY_MODEL_CLASS] = model_class
                                st.session_state[_KEY_VALIDATED] = True
                                st.session_state["_session_id"] = sid
                                st.session_state["_query_params_applied"] = True
                                # Set URL query params so refresh preserves the session
                                st.query_params["dataset"] = dataset_dir.strip()
                                st.query_params["session"] = sid
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not load session schema: {exc}")
                        else:
                            st.error("Session has no embedded schema.")

            st.markdown("---")

        # New annotation section
        st.markdown("##### Start New Annotation")
        with st.expander("Configure schema and mode", expanded=not bool(sessions)):
            from stickler.annotator.config import _render_config_widgets
            st.session_state[_KEY_DATASET_DIR] = dataset_dir.strip()
            _render_config_widgets()

    return False


def _render_dataset_progress(dataset_dir: Path, session_id: str) -> None:
    """Render a compact document-set progress bar in the header."""
    if not session_id:
        return

    # Count actual PDFs in the dataset directory (source of truth)
    total = len([p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"])
    if total == 0:
        return

    # Count completed and in-progress from session annotation files
    session_dir = dataset_dir / ".annotations" / session_id
    completed = 0
    in_progress = 0
    if session_dir.exists():
        import json as _json
        for f in session_dir.glob("*.json"):
            try:
                data = _json.loads(f.read_text())
                metadata = data.get("metadata", {})
                fields_meta = metadata.get("fields", {})
                data_fields = data.get("data", {})
                # A field is annotated if it has provenance metadata
                num_annotated = len(fields_meta)
                num_total = len(data_fields)
                if num_annotated == 0:
                    pass  # not started
                elif num_total > 0 and num_annotated >= num_total:
                    completed += 1
                else:
                    in_progress += 1
            except Exception:
                logger.debug("Skipping unreadable annotation file: %s", f)

    not_started = total - completed - in_progress
    pct = completed / total

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;padding:2px 0 4px 0'>"
        f"<div style='flex:1;background:#e5e7eb;border-radius:4px;height:6px;overflow:hidden'>"
        f"<div style='width:{pct*100:.1f}%;background:#22c55e;height:100%'></div>"
        f"</div>"
        f"<span style='font-size:10px;color:#888;white-space:nowrap'>"
        f"<span class='kie-tooltip' data-tip='Completed'>&#9989;</span> {completed}"
        f" &nbsp;<span class='kie-tooltip' data-tip='In progress'>&#128992;</span> {in_progress}"
        f" &nbsp;<span class='kie-tooltip' data-tip='Not started'>&#9898;</span> {not_started}"
        f" &nbsp;/ {total} docs"
        f"</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _app() -> None:
    """The Streamlit application — called when Streamlit runs this file."""
    st.set_page_config(page_title="KIE Annotation Tool", layout="wide")

    # Inject CSS + JS from static files
    from stickler.annotator.styles import inject_styles
    inject_styles()

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

        import streamlit.components.v1 as _comp
        col_title, col_link, col_gear = st.columns([4, 7, 0.5])
        with col_title:
            abs_dataset = str(config.dataset_dir.resolve())
            abs_annotations = str((config.dataset_dir / ".annotations").resolve())
            session_id_short = session_id[:8] + "…" if session_id else "—"
            tooltip = (
                f"Dataset: {abs_dataset}&#10;"
                f"Annotations: {abs_annotations}&#10;"
                f"Session: {session_id_short}"
            )
            st.markdown(
                f"<div style='padding:2px 0;white-space:nowrap'>"
                f"<span style='font-size:14px;font-weight:700'>KIE Annotation Tool</span>"
                f"&nbsp;<span style='font-size:10px;color:#888'>"
                f"<span class='kie-tooltip' data-tip='{abs_dataset}&#10;Annotations: {abs_annotations}&#10;Session: {session_id_short}'>&#128193; {dataset_name}</span>"
                f" &middot; &#128203; {schema_name} &middot; &#9889; {mode_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_link:
            _comp.html(
                f"""<div style="display:flex;align-items:center;justify-content:flex-end;gap:4px;padding-top:2px">
                <code style="font-size:9px;color:#aaa;background:none;white-space:nowrap">{deep_link}</code>
                <button onclick="navigator.clipboard.writeText('{deep_link}');this.textContent='✓';setTimeout(()=>this.textContent='Copy',600)"
                    style="font-size:9px;padding:1px 6px;border:1px solid #ddd;border-radius:3px;background:#fafafa;cursor:pointer;color:#666">Copy</button>
                </div>""",
                height=24,
            )
        with col_gear:
            if st.button("⚙️", key="open_config", help="Configure dataset, schema, and mode"):
                render_config_dialog()

        # Document set progress bar
        _render_dataset_progress(config.dataset_dir, session_id)
    else:
        _, gear_col = st.columns([20, 1])
        with gear_col:
            if st.button("⚙️", key="open_config", help="Configure dataset, schema, and mode"):
                render_config_dialog()

    st.markdown("<hr style='margin:0 0 2px 0'>", unsafe_allow_html=True)

    # Not yet configured — show landing page with session discovery
    if not is_configured:
        _render_landing_page()
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

    # 4. Side-by-side layout: PDF viewer (left, wider) + annotation panel (right, narrower)
    # Pre-load annotation state to make field locations available for the viewer
    left_col, right_col = st.columns([3, 2])

    with left_col:
        selected_pdf = _render_document_queue(manager, schema_fields, session=session)
        if selected_pdf is None:
            return

        last_pdf_key = "last_selected_pdf"
        if st.session_state.get(last_pdf_key) != str(selected_pdf):
            old_pdf = st.session_state.get(last_pdf_key)
            if old_pdf:
                st.session_state.pop(f"annotation_state_{old_pdf}", None)
                st.session_state.pop(f"pdf_page_{old_pdf}", None)
                st.session_state.pop(f"_field_locations_{old_pdf}", None)
            st.session_state[last_pdf_key] = str(selected_pdf)
            st.session_state[f"pdf_page_{selected_pdf}"] = 1

        # Pre-load annotation state so field locations are available for the viewer
        state = _resume_or_fresh_state(selected_pdf, config.schema, session=session)

        # Populate field locations from loaded state for bbox overlay
        _loc_key = f"_field_locations_{selected_pdf}"
        if state is not None:
            locs = {
                name: fa.location
                for name, fa in state.fields.items()
                if fa.location is not None
            }
            if locs:
                st.session_state[_loc_key] = locs

        viewer = PDFViewer(selected_pdf)
        field_locations = st.session_state.get(_loc_key, {})
        viewer.render(field_locations=field_locations if field_locations else None)

    with right_col:
        if selected_pdf is None:
            return
        if state is None:
            return

        # Build prefill function — available in LLM Inference and Zero Start modes
        prefill_fn = None
        if config.mode in (AnnotationMode.LLM_INFERENCE, AnnotationMode.ZERO_START):
            try:
                from stickler.annotator.llm_backend import (
                    AVAILABLE_MODELS,
                    DEFAULT_MODEL_LABEL,
                    BedrockLLMBackend,
                )

                # Initialise model selection in session state
                if "_llm_model_label" not in st.session_state:
                    st.session_state["_llm_model_label"] = DEFAULT_MODEL_LABEL

                selected_label = st.session_state["_llm_model_label"]
                selected_model_id = AVAILABLE_MODELS.get(selected_label, AVAILABLE_MODELS[DEFAULT_MODEL_LABEL])

                # (Re)create backend when model changes
                _backend_key = "_llm_backend"
                _model_key = "_llm_backend_model_id"
                if (
                    _backend_key not in st.session_state
                    or st.session_state.get(_model_key) != selected_model_id
                ):
                    st.session_state[_backend_key] = BedrockLLMBackend(model_id=selected_model_id)
                    st.session_state[_model_key] = selected_model_id

                backend = st.session_state[_backend_key]
                prefill_fn = backend.prefill
            except Exception as exc:
                logger.debug("LLM backend unavailable: %s", exc)

        # Build localize function
        localize_fn = None
        if config.mode in (AnnotationMode.LLM_INFERENCE, AnnotationMode.ZERO_START):
            try:
                from stickler.annotator.llm_backend import (
                    LOCALIZATION_MODELS,
                    DEFAULT_LOCALIZATION_MODEL_LABEL,
                )

                if "_loc_model_label" not in st.session_state:
                    st.session_state["_loc_model_label"] = DEFAULT_LOCALIZATION_MODEL_LABEL

                loc_label = st.session_state["_loc_model_label"]
                loc_model_id = LOCALIZATION_MODELS.get(loc_label, LOCALIZATION_MODELS[DEFAULT_LOCALIZATION_MODEL_LABEL])

                # Reuse the extraction backend instance but pass localization model_id
                if "_llm_backend" in st.session_state:
                    _be = st.session_state["_llm_backend"]
                    localize_fn = lambda pdf, fv, _be=_be, _mid=loc_model_id: _be.localize(pdf, fv, model_id=_mid)
            except Exception as exc:
                logger.debug("Localization backend unavailable: %s", exc)

        panel = AnnotationPanel(config.schema, config.mode, state, selected_pdf, session=session, prefill_fn=prefill_fn, localize_fn=localize_fn)
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
