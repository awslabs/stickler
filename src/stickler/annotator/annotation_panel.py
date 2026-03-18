"""Annotation panel — mode-specific field entry and review UI.

Renders schema fields in the right-side panel with mode-specific workflows:

- **Zero Start**: All fields shown with empty inputs. User types values or
  marks None. Provenance is always ``source="human"``.
- **LLM Inference**: Pre-fill button, batch accept/reject, individual edit.
  LLM values visually distinguished with 🤖 prefix.
- **HITL**: Fields presented one at a time for accept/reject/edit review.

Auto-saves on every field change via :class:`AnnotationSerializer`.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from .models import (
    AnnotationMode,
    AnnotationState,
    FieldAnnotation,
    FieldProvenance,
)
from .serializer import AnnotationSerializer


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _field_is_annotated(field: FieldAnnotation) -> bool:
    return field.is_none or (field.value is not None and field.value != "" and field.value != [])


class AnnotationPanel:
    """Schema-driven annotation panel with mode-specific rendering."""

    def __init__(
        self,
        schema: dict,
        mode: AnnotationMode,
        annotation_state: AnnotationState,
        pdf_path: Path,
        session=None,
        prefill_fn=None,
    ) -> None:
        self.schema = schema
        self.mode = mode
        self.state = annotation_state
        self.pdf_path = pdf_path
        self.session = session  # AnnotationSession | None
        self.prefill_fn = prefill_fn  # callable(pdf_path, schema) -> dict | None
        self.fields = list(schema.get("properties", {}).keys())
        self._field_schemas = schema.get("properties", {})
        self._doc_key = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]

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

    # ------------------------------------------------------------------
    # Scalar field rendering
    # ------------------------------------------------------------------

    def _render_scalar_field(self, field_name: str, label: str, source: str = "human") -> None:
        """Render a scalar field with inline None toggle and status dot."""
        field_schema = self._field_schemas.get(field_name, {})
        description = field_schema.get("description", "")

        existing = self.state.fields.get(field_name)
        current_is_none = existing.is_none if existing else False
        current_value = ""
        if existing and not existing.is_none and existing.value is not None:
            current_value = str(existing.value)

        # Status indicator
        is_done = existing is not None and _field_is_annotated(existing)
        status = "✅" if is_done else "⬜"

        col_status, col_input, col_none = st.columns([0.08, 3, 0.7])

        with col_status:
            st.markdown(f"<div style='padding-top:32px;font-size:18px'>{status}</div>", unsafe_allow_html=True)

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
        is_done = n > 0

        # Header row
        col_status, col_header = st.columns([0.08, 4])
        with col_status:
            st.markdown(
                f"<div style='padding-top:8px;font-size:18px'>{'✅' if is_done else '⬜'}</div>",
                unsafe_allow_html=True,
            )
        with col_header:
            help_text = description or None
            st.markdown(
                f"**{field_name}**  "
                f"<span style='color:#888;font-size:13px'>array · {n} item{'s' if n != 1 else ''}</span>",
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
                    st.markdown(f"<small style='color:#888'>{sf}</small>", unsafe_allow_html=True)
            with header_cols[-1]:
                st.markdown("<small style='color:#888'>remove</small>", unsafe_allow_html=True)

            # Table rows
            indices_to_remove = []
            changed = False
            for idx, item in enumerate(items):
                row_cols = st.columns([*[3] * len(sub_fields), 1])
                updated_item = dict(item) if isinstance(item, dict) else {k: None for k in sub_fields}
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
                    st.markdown("<div style='padding-top:4px'></div>", unsafe_allow_html=True)
                    if st.button("✕", key=f"arr_rm_{self._doc_key}_{field_name}_{idx}", help="Remove this item"):
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
        if st.button(f"＋ Add row", key=f"arr_add_{self._doc_key}_{field_name}", type="secondary"):
            items.append({k: None for k in sub_fields})
            st.session_state[items_key] = items
            self._update_field(field_name, items, False, source="human")
            st.rerun()

    def _render_primitive_array(self, field_name: str, items: list, items_key: str) -> None:
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

        if st.button("＋ Add item", key=f"arr_add_{self._doc_key}_{field_name}", type="secondary"):
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
        annotated, total = self._annotated_count()

        # Header with progress + optional auto-annotate button
        col_title, col_btn, col_hint = st.columns([3, 1.5, 1])
        with col_title:
            st.subheader("Manual Annotation")
        with col_btn:
            if self.prefill_fn is not None:
                st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
                if st.button("🤖 Auto-annotate", key="zero_prefill_btn", use_container_width=True, help="Pre-fill all fields using LLM"):
                    with st.spinner("Extracting fields with LLM…"):
                        try:
                            predictions = self.prefill_fn(self.pdf_path, self.schema)
                        except Exception as exc:
                            st.error(f"LLM pre-fill failed: {exc}")
                            predictions = None
                    if predictions:
                        for field_name in self.fields:
                            raw_val = predictions.get(field_name)
                            is_none = raw_val is None
                            field_schema = self._field_schemas.get(field_name, {})
                            if field_schema.get("type") == "array":
                                value = raw_val if isinstance(raw_val, list) else None
                                is_none = value is None
                                st.session_state[f"array_items_{self._doc_key}_{field_name}"] = value or []
                            else:
                                value = None if is_none else str(raw_val) if raw_val is not None else None
                            self.state.fields[field_name] = FieldAnnotation(
                                value=value,
                                is_none=is_none,
                                provenance=FieldProvenance(source="llm", checked=True),
                            )
                        self._auto_save()
                        st.rerun()
        with col_hint:
            st.markdown(
                "<div style='text-align:right;padding-top:12px;color:#888;font-size:12px'>Tab to move between fields</div>",
                unsafe_allow_html=True,
            )

        self._render_progress()

        # Completion state
        if annotated == total and total > 0:
            st.success("🎉 All fields annotated! Move to the next document.")
            st.markdown("---")

        # Separate scalar fields from array fields for cleaner layout
        scalar_fields = [f for f in self.fields if self._field_schemas.get(f, {}).get("type") != "array"]
        array_fields = [f for f in self.fields if self._field_schemas.get(f, {}).get("type") == "array"]

        # Scalar fields
        if scalar_fields:
            for field_name in scalar_fields:
                self._render_field(field_name, label=field_name)

        # Array fields in their own section
        if array_fields:
            st.markdown("---")
            st.markdown("##### Line Items")
            for field_name in array_fields:
                self._render_field(field_name, label=field_name)

    # ------------------------------------------------------------------
    # LLM Inference
    # ------------------------------------------------------------------

    def render_llm_inference(self) -> None:
        """Render LLM inference mode with pre-fill, batch ops, and editing."""
        annotated, total = self._annotated_count()

        col_title, col_hint = st.columns([3, 1])
        with col_title:
            st.subheader("LLM Inference")
        with col_hint:
            st.markdown(
                "<div style='text-align:right;padding-top:12px;color:#888;font-size:12px'>Tab to move between fields</div>",
                unsafe_allow_html=True,
            )

        self._render_progress()

        has_llm_fields = any(
            f in self.state.fields and self.state.fields[f].provenance.source == "llm"
            for f in self.fields
        )

        if not has_llm_fields:
            st.info("Click **Pre-fill** to send this document to the LLM for automatic field extraction.")
            if st.button("🤖 Pre-fill with LLM", key="llm_prefill_btn", type="primary"):
                if self.prefill_fn is None:
                    st.error("LLM backend not configured.")
                else:
                    with st.spinner("Extracting fields with LLM…"):
                        try:
                            predictions = self.prefill_fn(self.pdf_path, self.schema)
                        except Exception as exc:
                            st.error(f"LLM pre-fill failed: {exc}")
                            predictions = None
                    if predictions:
                        for field_name in self.fields:
                            raw_val = predictions.get(field_name)
                            is_none = raw_val is None
                            # For array fields, store list directly
                            field_schema = self._field_schemas.get(field_name, {})
                            if field_schema.get("type") == "array":
                                value = raw_val if isinstance(raw_val, list) else None
                                is_none = value is None
                                # Sync session_state array cache
                                items_key = f"array_items_{self._doc_key}_{field_name}"
                                st.session_state[items_key] = value or []
                            else:
                                value = None if is_none else str(raw_val) if raw_val is not None else None
                            self.state.fields[field_name] = FieldAnnotation(
                                value=value,
                                is_none=is_none,
                                provenance=FieldProvenance(source="llm", checked=False),
                            )
                        self._auto_save()
                        st.rerun()
            st.markdown("---")
        else:
            col_accept, col_reject, col_spacer = st.columns([1, 1, 2])
            with col_accept:
                if st.button("✅ Accept All", key="llm_accept_all", use_container_width=True):
                    for f in self.fields:
                        if f in self.state.fields and self.state.fields[f].provenance.source == "llm":
                            self.state.fields[f].provenance.checked = True
                    self._auto_save()
                    st.rerun()
            with col_reject:
                if st.button("❌ Reject All", key="llm_reject_all", use_container_width=True):
                    for f in self.fields:
                        if f in self.state.fields and self.state.fields[f].provenance.source == "llm":
                            self.state.fields[f] = FieldAnnotation(
                                value=None, is_none=False,
                                provenance=FieldProvenance(source="human", checked=False),
                            )
                    self._auto_save()
                    st.rerun()
            st.markdown("---")

        for field_name in self.fields:
            existing = self.state.fields.get(field_name)
            is_llm = existing is not None and existing.provenance.source == "llm"
            label = f"🤖 {field_name}" if is_llm else field_name
            self._render_field(field_name, label=label, source="human")

    # ------------------------------------------------------------------
    # HITL
    # ------------------------------------------------------------------

    def render_hitl(self) -> None:
        """Render HITL mode — one field at a time for accept/reject/edit."""
        st.subheader("Human-in-the-Loop Review")

        has_llm_fields = any(
            f in self.state.fields and self.state.fields[f].provenance.source == "llm"
            for f in self.fields
        )

        if not has_llm_fields:
            st.info("Click **Pre-fill** to send this document to the LLM, then review each prediction.")
            if st.button("🤖 Pre-fill with LLM", key="hitl_prefill_btn", type="primary"):
                st.info("LLM pre-fill will be wired in app.py.")
            self._render_progress()
            return

        llm_fields = [
            f for f in self.fields
            if f in self.state.fields and self.state.fields[f].provenance.source == "llm"
        ]
        reviewed = sum(1 for f in llm_fields if self.state.fields[f].provenance.checked)
        pct = reviewed / len(llm_fields) if llm_fields else 0
        st.progress(pct, text=f"**{reviewed} / {len(llm_fields)}** predictions reviewed")

        unreviewed = [f for f in llm_fields if not self.state.fields[f].provenance.checked]

        if not unreviewed:
            st.success("🎉 All predictions reviewed!")
            self._render_progress()
            st.markdown("---")
            for field_name in self.fields:
                existing = self.state.fields.get(field_name)
                if existing:
                    icon = "🤖" if existing.provenance.source == "llm" else "👤"
                    val = "—" if existing.is_none else str(existing.value or "")
                    st.markdown(f"{icon} **{field_name}:** {val}")
            return

        current_field = unreviewed[0]
        fa = self.state.fields[current_field]

        # Card-style review UI
        st.markdown(f"#### Reviewing field `{current_field}`")
        st.markdown(
            f"<div style='background:#f0f4ff;border-left:4px solid #4a90e2;padding:12px 16px;border-radius:4px;font-size:16px'>"
            f"🤖 <strong>LLM prediction:</strong> {fa.value or '—'}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        col_accept, col_reject, col_edit = st.columns(3)
        with col_accept:
            if st.button("✅ Accept", key=f"hitl_accept_{self._doc_key}_{current_field}", use_container_width=True, type="primary"):
                fa.provenance.checked = True
                self._auto_save()
                st.rerun()
        with col_reject:
            if st.button("❌ Reject", key=f"hitl_reject_{self._doc_key}_{current_field}", use_container_width=True):
                self.state.fields[current_field] = FieldAnnotation(
                    value=None, is_none=False,
                    provenance=FieldProvenance(source="human", checked=False),
                )
                self._auto_save()
                st.rerun()
        with col_edit:
            edit_key = f"hitl_editing_{self._doc_key}_{current_field}"
            if st.button("✏️ Edit", key=f"hitl_edit_btn_{self._doc_key}_{current_field}", use_container_width=True):
                st.session_state[edit_key] = True

        if st.session_state.get(f"hitl_editing_{self._doc_key}_{current_field}", False):
            st.markdown("---")
            col_input, col_none = st.columns([3, 1])
            with col_none:
                is_none = st.checkbox("N/A", value=False, key=f"hitl_none_{self._doc_key}_{current_field}")
            with col_input:
                new_val = st.text_input(
                    f"New value for {current_field}",
                    value=str(fa.value or ""),
                    disabled=is_none,
                    key=f"hitl_val_{self._doc_key}_{current_field}",
                )
            if st.button("Save", key=f"hitl_save_{self._doc_key}_{current_field}", type="primary"):
                final_value = None if is_none else (new_val if new_val != "" else None)
                self._update_field(current_field, final_value, is_none, source="human", checked=False)
                st.session_state[f"hitl_editing_{self._doc_key}_{current_field}"] = False
                st.rerun()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Dispatch to the mode-specific renderer."""
        if self.mode == AnnotationMode.ZERO_START:
            self.render_zero_start()
        elif self.mode == AnnotationMode.LLM_INFERENCE:
            self.render_llm_inference()
        elif self.mode == AnnotationMode.HITL:
            self.render_hitl()
        else:
            st.error(f"Unknown annotation mode: {self.mode}")
