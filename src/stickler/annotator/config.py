"""Session configuration sidebar for the KIE Annotation Tool.

Renders a Streamlit sidebar where users configure:

- **Dataset directory path** — validated to exist on disk.
- **Schema source** — one of: JSON Schema file, Pydantic import path,
  or the in-app Schema Builder.

The tool always operates in Zero Start mode (manual annotation with
optional Auto-annotate and Locate buttons).

All configuration is stored in ``st.session_state`` and validated on
apply. Invalid paths or schemas surface via ``st.error()`` with
descriptive messages.

Usage::

    from stickler.annotator.config import render_config_sidebar

    config = render_config_sidebar()
    if config is not None:
        # config is a validated ConfigResult
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import streamlit as st

from .models import AnnotationMode
from .schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

# Session state keys
_KEY_DATASET_DIR = "config_dataset_dir"
_KEY_SCHEMA_SOURCE = "config_schema_source"
_KEY_SCHEMA_PATH = "config_schema_path"
_KEY_PYDANTIC_IMPORT = "config_pydantic_import"
_KEY_MODE = "config_mode"
_KEY_SCHEMA = "config_schema"
_KEY_MODEL_CLASS = "config_model_class"
_KEY_VALIDATED = "config_validated"

# Schema source options
SCHEMA_SOURCE_JSON = "JSON Schema file"
SCHEMA_SOURCE_PYDANTIC = "Pydantic import path"
SCHEMA_SOURCE_BUILDER = "Schema Builder"

SCHEMA_SOURCES = (SCHEMA_SOURCE_JSON, SCHEMA_SOURCE_PYDANTIC, SCHEMA_SOURCE_BUILDER)




@dataclass
class ConfigResult:
    """Validated configuration returned by :func:`render_config_sidebar`.

    Attributes:
        dataset_dir: Validated path to the dataset directory.
        schema_source: Which schema source was selected (display label).
        mode: The selected annotation mode.
        schema: Raw JSON Schema dict.
        model_class: ``StructuredModel`` subclass for the loaded schema.
    """

    dataset_dir: Path
    schema_source: str
    mode: AnnotationMode
    schema: dict[str, Any]
    model_class: Type


def _validate_dataset_dir(dataset_dir: str) -> Path | None:
    """Validate the dataset directory path.

    Returns the resolved ``Path`` on success, or ``None`` after displaying
    an ``st.error()`` on failure.
    """
    if not dataset_dir.strip():
        st.error("Dataset directory path cannot be empty.")
        return None

    path = Path(dataset_dir.strip())
    if not path.exists():
        st.error(f"Dataset directory does not exist: {path}")
        return None
    if not path.is_dir():
        st.error(f"Path is not a directory: {path}")
        return None
    return path


def _validate_schema(
    source: str,
    schema_path: str,
    pydantic_import: str,
) -> tuple[dict, Type] | None:
    """Validate and load the schema based on the selected source.

    Returns ``(raw_schema, model_class)`` on success, or ``None`` after
    displaying an ``st.error()`` on failure.  For the Schema Builder
    source, returns ``None`` (the builder handles its own validation
    in the main panel).
    """
    if source == SCHEMA_SOURCE_JSON:
        path_str = schema_path.strip()
        if not path_str:
            st.error("JSON Schema file path cannot be empty.")
            return None
        schema_file = Path(path_str)
        if not schema_file.exists():
            st.error(f"Schema file not found: {schema_file}")
            return None
        if not schema_file.is_file():
            st.error(f"Schema path is not a file: {schema_file}")
            return None
        try:
            schema, model_class = SchemaLoader.from_json_schema_file(schema_file)
            return schema, model_class
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Invalid schema file: {exc}")
            return None

    if source == SCHEMA_SOURCE_PYDANTIC:
        import_path = pydantic_import.strip()
        if not import_path:
            st.error("Pydantic import path cannot be empty.")
            return None
        try:
            schema, model_class = SchemaLoader.from_pydantic_import(import_path)
            return schema, model_class
        except (ValueError, ImportError, TypeError) as exc:
            st.error(f"Invalid Pydantic import: {exc}")
            return None

    # Schema Builder — no validation here; handled in the main panel
    return None


def render_config_sidebar() -> ConfigResult | None:
    """Render the configuration sidebar and return validated config.

    Draws Streamlit sidebar widgets for dataset directory, schema source,
    and operating mode. When the user clicks *Apply Configuration*,
    validates all inputs and stores the result in session state.

    Returns:
        A :class:`ConfigResult` if configuration is valid and applied,
        or ``None`` if not yet configured or validation failed.
    """
    with st.sidebar:
        st.header("Configuration")
        _render_config_widgets()

    # Return previously validated config if available
    return _get_stored_config()


def _render_config_widgets() -> None:
    """Render the configuration form widgets (shared by sidebar and dialog)."""
    # --- Dataset directory ---
    dataset_dir = st.text_input(
        "Dataset directory",
        value=st.session_state.get(_KEY_DATASET_DIR, ""),
        help="Path to a directory containing PDF files to annotate.",
        key="__config_dataset_dir_input",
    )

    # --- Schema source ---
    current_source = st.session_state.get(_KEY_SCHEMA_SOURCE, SCHEMA_SOURCE_JSON)
    source_index = SCHEMA_SOURCES.index(current_source) if current_source in SCHEMA_SOURCES else 0

    schema_source = st.radio(
        "Schema source",
        SCHEMA_SOURCES,
        index=source_index,
        help="How to provide the annotation schema.",
        key="__config_schema_source_input",
    )

    # Conditional inputs based on schema source
    schema_path = ""
    pydantic_import = ""

    if schema_source == SCHEMA_SOURCE_JSON:
        schema_path = st.text_input(
            "Schema file path",
            value=st.session_state.get(_KEY_SCHEMA_PATH, ""),
            help="Path to a JSON Schema file (.json).",
            key="__config_schema_path_input",
        )
    elif schema_source == SCHEMA_SOURCE_PYDANTIC:
        pydantic_import = st.text_input(
            "Pydantic import path",
            value=st.session_state.get(_KEY_PYDANTIC_IMPORT, ""),
            help="Dotted import path, e.g. mypackage.models.InvoiceModel",
            key="__config_pydantic_import_input",
        )
    else:
        # Inline schema builder
        from .schema_builder import SchemaBuilder
        from .schema_loader import SchemaLoader

        builder = SchemaBuilder()
        built_schema = builder.render()
        if built_schema is not None:
            try:
                validated_schema, model_class = SchemaLoader.from_builder_schema(built_schema)
                st.session_state[_KEY_SCHEMA] = validated_schema
                st.session_state[_KEY_MODEL_CLASS] = model_class
                st.success("✓ Schema finalized. Click **Apply Configuration** to start.")
            except ValueError as exc:
                st.error(f"Schema validation failed: {exc}")
        elif st.session_state.get(_KEY_SCHEMA) is not None:
            st.success("✓ Schema ready. Click **Apply Configuration** to start.")

    # --- Apply button ---
    st.markdown("---")
    apply_clicked = st.button("Apply Configuration", key="__config_apply", type="primary", use_container_width=True)

    if apply_clicked:
        validated_dir = _validate_dataset_dir(dataset_dir)
        if validated_dir is None:
            st.session_state[_KEY_VALIDATED] = False
            return

        if schema_source != SCHEMA_SOURCE_BUILDER:
            result = _validate_schema(schema_source, schema_path, pydantic_import)
            if result is None:
                st.session_state[_KEY_VALIDATED] = False
                return
            schema, model_class = result
        else:
            schema = st.session_state.get(_KEY_SCHEMA)
            model_class = st.session_state.get(_KEY_MODEL_CLASS)
            if schema is None:
                st.warning("Build and finalize a schema below first.")
                return

        st.session_state[_KEY_DATASET_DIR] = dataset_dir.strip()
        st.session_state[_KEY_SCHEMA_SOURCE] = schema_source
        st.session_state[_KEY_SCHEMA_PATH] = schema_path.strip()
        st.session_state[_KEY_PYDANTIC_IMPORT] = pydantic_import.strip()
        st.session_state[_KEY_MODE] = AnnotationMode.ZERO_START
        st.session_state[_KEY_SCHEMA] = schema
        st.session_state[_KEY_MODEL_CLASS] = model_class
        st.session_state[_KEY_VALIDATED] = True
        st.success("✓ Configuration applied.")
        st.rerun()


def _get_stored_config() -> ConfigResult | None:
    """Return the previously validated config from session state, or None."""
    if st.session_state.get(_KEY_VALIDATED):
        stored_dir = st.session_state.get(_KEY_DATASET_DIR, "")
        stored_schema = st.session_state.get(_KEY_SCHEMA)
        stored_model = st.session_state.get(_KEY_MODEL_CLASS)
        stored_mode = st.session_state.get(_KEY_MODE, AnnotationMode.ZERO_START)
        stored_source = st.session_state.get(_KEY_SCHEMA_SOURCE, SCHEMA_SOURCE_JSON)

        if stored_dir and stored_schema is not None and stored_model is not None:
            return ConfigResult(
                dataset_dir=Path(stored_dir),
                schema_source=stored_source,
                mode=stored_mode,
                schema=stored_schema,
                model_class=stored_model,
            )
    return None


@st.dialog("⚙️ Configuration", width="large")
def render_config_dialog() -> None:
    """Render configuration inside a modal dialog."""
    # Show current session info if active
    session_id = st.session_state.get("_session_id", "")
    schema = st.session_state.get(_KEY_SCHEMA)
    if session_id:
        schema_title = schema.get("title", "—") if isinstance(schema, dict) else "—"
        st.markdown(
            f"<div style='padding:4px 8px;background:#f0f9ff;border-radius:4px;margin-bottom:8px;font-size:12px;color:#555'>"
            f"Active session: <code>{session_id[:8]}…</code>"
            f" &nbsp;·&nbsp; Schema: {schema_title}"
            f"</div>",
            unsafe_allow_html=True,
        )
    _render_config_widgets()


def apply_config_from_query_params() -> bool:
    """Auto-apply configuration from URL query parameters.

    Two supported URL patterns:

    1. New session (schema file required):
       ``?dataset=./files&schema=./files/schema.json``

    2. Resume existing session (schema loaded from manifest):
       ``?dataset=./files&session=<guid>``

    Returns True if config was successfully applied, False otherwise.
    """
    if st.session_state.get("_query_params_applied"):
        return False

    params = st.query_params
    dataset = params.get("dataset", "").strip()
    if not dataset:
        return False

    from .schema_loader import SchemaLoader
    from .serializer import AnnotationManifest

    dataset_path = Path(dataset)
    if not dataset_path.exists() or not dataset_path.is_dir():
        logger.warning("Query param dataset dir not found: %s", dataset)
        return False

    session_id = params.get("session", "").strip()
    schema_path = params.get("schema", "").strip()

    # Pattern 2: resume by session GUID — load schema from manifest
    if session_id:
        manifest = AnnotationManifest(dataset_path)
        session = manifest.get_session(session_id)
        if session is None:
            logger.warning("Session %s not found in manifest", session_id)
            return False
        schema = session.schema
        if schema is None:
            logger.warning("Session %s has no embedded schema", session_id)
            return False
        try:
            _, model_class = SchemaLoader.from_builder_schema(schema)
        except (ValueError, Exception) as exc:
            logger.warning("Could not load schema from session manifest: %s", exc)
            return False

        st.session_state[_KEY_DATASET_DIR] = dataset
        st.session_state[_KEY_SCHEMA_SOURCE] = SCHEMA_SOURCE_JSON
        st.session_state[_KEY_SCHEMA_PATH] = ""  # schema came from manifest, not a file
        st.session_state[_KEY_MODE] = AnnotationMode.ZERO_START
        st.session_state[_KEY_SCHEMA] = schema
        st.session_state[_KEY_MODEL_CLASS] = model_class
        st.session_state[_KEY_VALIDATED] = True
        st.session_state["_query_params_applied"] = True
        st.session_state["_session_id"] = session_id
        logger.info("Resumed session %s from query params", session_id)
        return True

    # Pattern 1: new session with schema file
    if not schema_path:
        return False

    schema_file = Path(schema_path)
    if not schema_file.exists() or not schema_file.is_file():
        logger.warning("Query param schema file not found: %s", schema_path)
        return False

    try:
        schema, model_class = SchemaLoader.from_json_schema_file(schema_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Query param schema invalid: %s", exc)
        return False

    st.session_state[_KEY_DATASET_DIR] = dataset
    st.session_state[_KEY_SCHEMA_SOURCE] = SCHEMA_SOURCE_JSON
    st.session_state[_KEY_SCHEMA_PATH] = schema_path
    st.session_state[_KEY_MODE] = AnnotationMode.ZERO_START
    st.session_state[_KEY_SCHEMA] = schema
    st.session_state[_KEY_MODEL_CLASS] = model_class
    st.session_state[_KEY_VALIDATED] = True
    st.session_state["_query_params_applied"] = True
    logger.info("Config auto-applied from query params: dataset=%s schema=%s", dataset, schema_path)
    return True
