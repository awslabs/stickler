"""Annotation panel — field entry and review UI.

Renders schema fields in the right-side panel using the Zero Start workflow:

- All fields shown with empty inputs. User types values or marks None.
  Provenance is always ``source="human"`` for manual entry.
- Optional Auto-annotate button pre-fills fields via LLM.
- Optional Locate button finds field values in the PDF.

Auto-saves on every field change via :class:`AnnotationSerializer`.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from .models import (
    AnnotationState,
    FieldAnnotation,
    FieldProvenance,
)
from .serializer import AnnotationSerializer


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _field_is_annotated(field: FieldAnnotation) -> bool:
    return field.is_none or (
        field.value is not None and field.value != "" and field.value != []
    )


class AnnotationPanel:
    """Schema-driven annotation panel with mode-specific rendering."""

    def __init__(
        self,
        schema: dict,
        annotation_state: AnnotationState,
        pdf_path: Path,
        session=None,
        prefill_fn=None,
        localize_fn=None,
    ) -> None:
        self.schema = schema
        self.state = annotation_state
        self.pdf_path = pdf_path
        self.session = session  # AnnotationSession | None
        self.prefill_fn = prefill_fn  # callable(pdf_path, schema) -> dict | None
        self.localize_fn = (
            localize_fn  # callable(pdf_path, field_values) -> dict | None
        )
        self.fields = list(schema.get("properties", {}).keys())
        self._field_schemas = schema.get("properties", {})
        self._doc_key = hashlib.md5(
            str(pdf_path).encode(), usedforsecurity=False
        ).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        self.state.updated_at = _now_iso()
        AnnotationSerializer.save(self.state, self.pdf_path, session=self.session)
        st.toast("✓ Saved", icon="💾")

    def _update_field(
        self,
        field_name: str,
        value,
        is_none: bool,
        source: str = "human",
        checked: bool = False,
    ) -> None:
        self.state.fields[field_name] = FieldAnnotation(
            value=value,
            is_none=is_none,
            provenance=FieldProvenance(source=source, checked=checked),
        )
        self._auto_save()

    def _annotated_count(self) -> tuple[int, int]:
        """Return (annotated, total) field counts."""
        total = len(self.fields)
        annotated = sum(
            1
            for f in self.fields
            if f in self.state.fields and _field_is_annotated(self.state.fields[f])
        )
        return annotated, total

    def _render_progress(self) -> None:
        """Render a progress bar + fraction text."""
        annotated, total = self._annotated_count()
        pct = annotated / total if total else 0
        st.progress(pct, text=f"**{annotated} / {total}** fields annotated")

    def _render_model_selector(self) -> None:
        """Render a compact model picker popover for extraction and localization."""
        try:
            from .llm_backend import (
                AVAILABLE_MODELS,
                DEFAULT_LOCALIZATION_MODEL_LABEL,
                DEFAULT_MODEL_LABEL,
                LOCALIZATION_MODELS,
            )
        except ImportError:
            return

        current_ext = st.session_state.get("_llm_model_label", DEFAULT_MODEL_LABEL)
        current_loc = st.session_state.get(
            "_loc_model_label", DEFAULT_LOCALIZATION_MODEL_LABEL
        )
        ext_labels = list(AVAILABLE_MODELS.keys())
        loc_labels = list(LOCALIZATION_MODELS.keys())

        st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
        with st.popover("⚙", help="Model settings"):
            st.markdown("**Extraction**")
            ext_choice = st.radio(
                "Extraction model",
                ext_labels,
                index=ext_labels.index(current_ext) if current_ext in ext_labels else 0,
                key="_model_selector_radio",
                label_visibility="collapsed",
            )
            if ext_choice != current_ext:
                st.session_state["_llm_model_label"] = ext_choice
                st.rerun()

            st.markdown("**Localization**")
            loc_choice = st.radio(
                "Localization model",
                loc_labels,
                index=loc_labels.index(current_loc) if current_loc in loc_labels else 0,
                key="_loc_model_selector_radio",
                label_visibility="collapsed",
            )
            if loc_choice != current_loc:
                st.session_state["_loc_model_label"] = loc_choice
                st.rerun()

    @staticmethod
    def _show_llm_error(exc: Exception) -> None:
        """Display a user-friendly error with remediation steps."""
        msg = str(exc)
        if "UnrecognizedClientException" in msg or "security token" in msg.lower():
            st.error(
                "**AWS credentials expired or missing.**  \n"
                "Set `AWS_PROFILE=<your-profile>` in your `.env` file and restart the app.  \n"
                "Or export `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`."
            )
        elif "AccessDeniedException" in msg:
            st.error(
                "**Access denied.** Your AWS role may not have Bedrock permissions.  \n"
                "Ensure your credentials include `bedrock:InvokeModel` access."
            )
        else:
            st.error(f"LLM pre-fill failed: {exc}")

    # ------------------------------------------------------------------
    # Scalar field rendering
    # ------------------------------------------------------------------

    def _render_scalar_field(
        self, field_name: str, label: str, source: str = "human"
    ) -> None:
        """Render a scalar field with inline N/A toggle."""
        field_schema = self._field_schemas.get(field_name, {})
        description = field_schema.get("description", "")

        existing = self.state.fields.get(field_name)
        current_is_none = existing.is_none if existing else False
        current_value = ""
        if existing and not existing.is_none and existing.value is not None:
            current_value = str(existing.value)

        col_input, col_none = st.columns([4, 0.7])

        with col_none:
            st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
            is_none = st.checkbox(
                "N/A",
                value=current_is_none,
                key=f"zero_none_{self._doc_key}_{field_name}",
                help="Mark this field as not applicable / not present in the document",
            )

        with col_input:
            value = st.text_input(
                label,
                value="" if is_none else current_value,
                disabled=is_none,
                key=f"zero_val_{self._doc_key}_{field_name}",
                help=description or None,
                placeholder="—" if is_none else "",
            )

        new_value = None if is_none else (value if value != "" else None)
        needs_update = False
        if existing is None:
            if is_none or new_value is not None:
                needs_update = True
        else:
            if existing.is_none != is_none or existing.value != new_value:
                needs_update = True
        if needs_update:
            self._update_field(field_name, new_value, is_none, source=source)

    # ------------------------------------------------------------------
    # Array field rendering — table-style for object arrays
    # ------------------------------------------------------------------

    def _render_array_field(self, field_name: str, field_schema: dict) -> None:
        """Render an array field.

        Object arrays render as a mini-table (columns per sub-field).
        Primitive arrays render as a vertical list of inputs.
        """
        items_schema = field_schema.get("items", {})
        items_type = items_schema.get("type", "string")
        items_props = items_schema.get("properties", {})
        description = field_schema.get("description", "")

        existing = self.state.fields.get(field_name)
        current_items: list = []
        if existing and not existing.is_none and isinstance(existing.value, list):
            current_items = list(existing.value)

        items_key = f"array_items_{self._doc_key}_{field_name}"
        if items_key not in st.session_state:
            st.session_state[items_key] = current_items if current_items else []

        items: list = st.session_state[items_key]
        n = len(items)

        # Header row
        help_text = description or None
        st.markdown(
            f"**{field_name}**  "
            f"<span style='color:#888;font-size:13px'>array &middot; {n} item{'s' if n != 1 else ''}</span>",
            unsafe_allow_html=True,
            help=help_text,
        )

        if items_type == "object" and items_props:
            self._render_object_array(field_name, items, items_props, items_key)
        else:
            self._render_primitive_array(field_name, items, items_key)

    def _render_object_array(
        self, field_name: str, items: list, items_props: dict, items_key: str
    ) -> None:
        """Render array of objects as a table with one row per item."""
        sub_fields = list(items_props.keys())

        if items:
            # Table header
            header_cols = st.columns([*[3] * len(sub_fields), 1])
            for i, sf in enumerate(sub_fields):
                with header_cols[i]:
                    st.markdown(
                        f"<small style='color:#888'>{sf}</small>",
                        unsafe_allow_html=True,
                    )
            with header_cols[-1]:
                st.markdown(
                    "<small style='color:#888'>remove</small>", unsafe_allow_html=True
                )

            # Table rows
            indices_to_remove = []
            changed = False
            for idx, item in enumerate(items):
                row_cols = st.columns([*[3] * len(sub_fields), 1])
                updated_item = (
                    dict(item)
                    if isinstance(item, dict)
                    else {k: None for k in sub_fields}
                )
                for i, sf in enumerate(sub_fields):
                    with row_cols[i]:
                        sub_key = f"arr_{self._doc_key}_{field_name}_{idx}_{sf}"
                        current_sub = updated_item.get(sf, "") or ""
                        new_sub = st.text_input(
                            sf,
                            value=str(current_sub),
                            key=sub_key,
                            label_visibility="collapsed",
                        )
                        if new_sub != str(current_sub):
                            updated_item[sf] = new_sub if new_sub != "" else None
                            changed = True
                        elif sf not in updated_item:
                            updated_item[sf] = current_sub
                with row_cols[-1]:
                    st.markdown(
                        "<div style='padding-top:4px'></div>", unsafe_allow_html=True
                    )
                    if st.button(
                        "✕",
                        key=f"arr_rm_{self._doc_key}_{field_name}_{idx}",
                        help="Remove this item",
                    ):
                        indices_to_remove.append(idx)
                items[idx] = updated_item

            if indices_to_remove:
                for i in sorted(indices_to_remove, reverse=True):
                    items.pop(i)
                st.session_state[items_key] = items
                self._update_field(field_name, items or None, not items, source="human")
                st.rerun()

            if changed:
                st.session_state[items_key] = items
                self._update_field(field_name, items, False, source="human")

        # Add row button
        if st.button(
            "＋ Add row", key=f"arr_add_{self._doc_key}_{field_name}", type="secondary"
        ):
            items.append({k: None for k in sub_fields})
            st.session_state[items_key] = items
            self._update_field(field_name, items, False, source="human")
            st.rerun()

    def _render_primitive_array(
        self, field_name: str, items: list, items_key: str
    ) -> None:
        """Render array of primitives as a vertical list."""
        indices_to_remove = []
        for idx, item in enumerate(items):
            col_val, col_rm = st.columns([5, 1])
            with col_val:
                new_val = st.text_input(
                    f"Item {idx + 1}",
                    value=str(item) if item is not None else "",
                    key=f"arr_{self._doc_key}_{field_name}_{idx}",
                    label_visibility="collapsed",
                )
                if new_val != str(item):
                    items[idx] = new_val
            with col_rm:
                if st.button("✕", key=f"arr_rm_{self._doc_key}_{field_name}_{idx}"):
                    indices_to_remove.append(idx)

        if indices_to_remove:
            for i in sorted(indices_to_remove, reverse=True):
                items.pop(i)
            st.session_state[items_key] = items
            self._update_field(field_name, items or None, not items, source="human")
            st.rerun()

        if st.button(
            "＋ Add item", key=f"arr_add_{self._doc_key}_{field_name}", type="secondary"
        ):
            items.append("")
            st.session_state[items_key] = items
            self._update_field(field_name, items, False, source="human")
            st.rerun()

    def _render_field(self, field_name: str, label: str, source: str = "human") -> None:
        """Dispatch to scalar or array renderer based on schema type."""
        field_schema = self._field_schemas.get(field_name, {})
        if field_schema.get("type") == "array":
            self._render_array_field(field_name, field_schema)
        else:
            self._render_scalar_field(field_name, label, source=source)

    # ------------------------------------------------------------------
    # Zero Start
    # ------------------------------------------------------------------

    def render_zero_start(self) -> None:
        """Render all schema fields for manual annotation."""
        # Header with optional auto-annotate button + model selector
        col_title, col_btn, col_loc, col_cfg = st.columns([2.5, 1.3, 1, 0.4])
        with col_title:
            st.subheader("Annotation")
        with col_btn:
            if self.prefill_fn is not None:
                st.markdown(
                    "<div style='padding-top:8px'></div>", unsafe_allow_html=True
                )
                if st.button(
                    "🤖 Auto-annotate",
                    key="zero_prefill_btn",
                    width="stretch",
                    help="Pre-fill all fields using the selected Bedrock model. "
                    "Sends PDF pages as images to the LLM and extracts field values.",
                ):
                    with st.spinner(f"Extracting fields from {self.pdf_path.name}…"):
                        try:
                            predictions = self.prefill_fn(self.pdf_path, self.schema)
                        except Exception as exc:
                            self._show_llm_error(exc)
                            predictions = None
                    if predictions:
                        for field_name in self.fields:
                            raw_val = predictions.get(field_name)
                            is_none = raw_val is None
                            field_schema = self._field_schemas.get(field_name, {})
                            if field_schema.get("type") == "array":
                                value = raw_val if isinstance(raw_val, list) else None
                                is_none = value is None
                                st.session_state[
                                    f"array_items_{self._doc_key}_{field_name}"
                                ] = value or []
                            else:
                                value = (
                                    None
                                    if is_none
                                    else str(raw_val)
                                    if raw_val is not None
                                    else None
                                )
                                # Sync widget keys so text inputs reflect new values after rerun
                                st.session_state[
                                    f"zero_val_{self._doc_key}_{field_name}"
                                ] = (
                                    ""
                                    if is_none
                                    else (str(value) if value is not None else "")
                                )
                                st.session_state[
                                    f"zero_none_{self._doc_key}_{field_name}"
                                ] = is_none
                            self.state.fields[field_name] = FieldAnnotation(
                                value=value,
                                is_none=is_none,
                                provenance=FieldProvenance(source="llm", checked=False),
                            )
                        self._auto_save()
                        st.rerun()
        with col_loc:
            annotated_count, _ = self._annotated_count()
            if self.localize_fn is not None and annotated_count > 0:
                has_locations = any(
                    f.location is not None for f in self.state.fields.values()
                )
                loc_label = "🔄 Re-locate" if has_locations else "📍 Locate"
                st.markdown(
                    "<div style='padding-top:8px'></div>", unsafe_allow_html=True
                )
                if st.button(
                    loc_label,
                    key="localize_btn",
                    width="stretch",
                    help="Find where each field value appears in the PDF and draw bounding boxes.",
                ):
                    field_values = {
                        name: fa.value
                        for name, fa in self.state.fields.items()
                        if fa.value is not None and not fa.is_none
                    }
                    with st.spinner(f"Localizing {len(field_values)} fields…"):
                        try:
                            locations = self.localize_fn(self.pdf_path, field_values)
                        except Exception as exc:
                            self._show_llm_error(exc)
                            locations = None
                    if locations:
                        from .models import FieldLocation

                        for field_name, loc_data in locations.items():
                            if field_name in self.state.fields:
                                self.state.fields[field_name].location = FieldLocation(
                                    page=loc_data["page"],
                                    bbox=loc_data["bbox"],
                                )
                        self._auto_save()
                        loc_key = f"_field_locations_{self.pdf_path}"
                        st.session_state[loc_key] = {
                            name: fa.location
                            for name, fa in self.state.fields.items()
                            if fa.location is not None
                        }
                        st.rerun()
        with col_cfg:
            if self.prefill_fn is not None:
                self._render_model_selector()

        # Reserve a slot for the progress bar — filled after fields render
        progress_slot = st.empty()
        completion_slot = st.empty()

        # Separate scalar fields from array fields for cleaner layout
        scalar_fields = [
            f
            for f in self.fields
            if self._field_schemas.get(f, {}).get("type") != "array"
        ]
        array_fields = [
            f
            for f in self.fields
            if self._field_schemas.get(f, {}).get("type") == "array"
        ]

        # Scalar fields
        if scalar_fields:
            for field_name in scalar_fields:
                self._render_field(field_name, label=field_name)

        # Array fields in their own section
        if array_fields:
            st.markdown("---")
            for field_name in array_fields:
                self._render_field(field_name, label=field_name)

        # Now render progress with up-to-date counts (after fields have been processed)
        annotated, total = self._annotated_count()
        pct = annotated / total if total else 0
        progress_slot.progress(pct, text=f"**{annotated} / {total}** fields annotated")
        if annotated == total and total > 0:
            completion_slot.success(
                "🎉 All fields annotated! Move to the next document."
            )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Render the annotation panel (always Zero Start workflow)."""
        self.render_zero_start()
