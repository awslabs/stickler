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
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _field_is_annotated(field: FieldAnnotation) -> bool:
    """Return True if a field has a value or is explicitly marked None."""
    return field.is_none or (field.value is not None and field.value != "")


class AnnotationPanel:
    """Schema-driven annotation panel with mode-specific rendering.

    Takes the current ``AnnotationState`` and modifies it in place.
    Auto-saves after every field change.

    Args:
        schema: Raw JSON Schema dict (must have ``properties`` key).
        mode: The active annotation workflow.
        annotation_state: Mutable annotation state for the current document.
        pdf_path: Path to the source PDF (needed for auto-save).
    """

    def __init__(
        self,
        schema: dict,
        mode: AnnotationMode,
        annotation_state: AnnotationState,
        pdf_path: Path,
    ) -> None:
        self.schema = schema
        self.mode = mode
        self.state = annotation_state
        self.pdf_path = pdf_path
        self.fields = list(schema.get("properties", {}).keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        """Persist the current annotation state to disk."""
        self.state.updated_at = _now_iso()
        AnnotationSerializer.save(self.state, self.pdf_path)

    def _update_field(
        self,
        field_name: str,
        value: str | None,
        is_none: bool,
        source: str = "human",
        checked: bool = False,
    ) -> None:
        """Update a single field in the annotation state and auto-save."""
        self.state.fields[field_name] = FieldAnnotation(
            value=value,
            is_none=is_none,
            provenance=FieldProvenance(source=source, checked=checked),
        )
        self._auto_save()

    def _progress_text(self) -> str:
        """Return 'X of Y fields annotated' progress string."""
        total = len(self.fields)
        annotated = sum(
            1
            for f in self.fields
            if f in self.state.fields and _field_is_annotated(self.state.fields[f])
        )
        return f"{annotated} of {total} fields annotated"

    def _hitl_review_progress(self) -> str:
        """Return 'K of M fields reviewed' for HITL mode."""
        llm_fields = [
            f
            for f in self.fields
            if f in self.state.fields
            and self.state.fields[f].provenance.source == "llm"
        ]
        reviewed = sum(
            1 for f in llm_fields if self.state.fields[f].provenance.checked
        )
        return f"{reviewed} of {len(llm_fields)} fields reviewed"

    # ------------------------------------------------------------------
    # Zero Start
    # ------------------------------------------------------------------

    def render_zero_start(self) -> None:
        """Render all schema fields with empty inputs for manual entry.

        Each field gets a text input and a "Mark as None" checkbox.
        Provenance is always ``source="human", checked=False``.
        """
        st.subheader("Manual Annotation")
        st.caption(self._progress_text())

        for field_name in self.fields:
            existing = self.state.fields.get(field_name)
            current_value = ""
            current_is_none = False
            if existing is not None:
                current_is_none = existing.is_none
                current_value = "" if existing.is_none else (existing.value or "")

            col_input, col_none = st.columns([3, 1])

            with col_none:
                is_none = st.checkbox(
                    "None",
                    value=current_is_none,
                    key=f"zero_none_{field_name}",
                )

            with col_input:
                value = st.text_input(
                    field_name,
                    value="" if is_none else str(current_value),
                    disabled=is_none,
                    key=f"zero_val_{field_name}",
                )

            # Detect changes and update state
            new_value = None if is_none else (value if value != "" else None)
            new_is_none = is_none

            needs_update = False
            if existing is None:
                # Only save if user actually entered something
                if new_is_none or (new_value is not None and new_value != ""):
                    needs_update = True
            else:
                if existing.is_none != new_is_none or existing.value != new_value:
                    needs_update = True

            if needs_update:
                self._update_field(field_name, new_value, new_is_none, source="human")

    # ------------------------------------------------------------------
    # LLM Inference
    # ------------------------------------------------------------------

    def render_llm_inference(self) -> None:
        """Render LLM inference mode with pre-fill, batch ops, and editing.

        Shows a pre-fill button (wiring deferred to app.py). After pre-fill,
        fields display LLM values with 🤖 indicator. Batch accept/reject
        and individual editing are supported.
        """
        st.subheader("LLM Inference Mode")
        st.caption(self._progress_text())

        # Pre-fill button — actual LLM call wired in app.py
        has_llm_fields = any(
            f in self.state.fields
            and self.state.fields[f].provenance.source == "llm"
            for f in self.fields
        )

        if not has_llm_fields:
            if st.button("🤖 Pre-fill with LLM", key="llm_prefill_btn"):
                st.info("LLM pre-fill will be wired in app.py. Use session state to trigger.")
            st.markdown("---")

        # Batch operations (only when LLM fields exist)
        if has_llm_fields:
            col_accept, col_reject = st.columns(2)
            with col_accept:
                if st.button("✅ Accept All", key="llm_accept_all"):
                    for field_name in self.fields:
                        if field_name in self.state.fields:
                            fa = self.state.fields[field_name]
                            if fa.provenance.source == "llm":
                                fa.provenance.checked = True
                    self._auto_save()
                    st.rerun()
            with col_reject:
                if st.button("❌ Reject All", key="llm_reject_all"):
                    for field_name in self.fields:
                        if field_name in self.state.fields:
                            fa = self.state.fields[field_name]
                            if fa.provenance.source == "llm":
                                self.state.fields[field_name] = FieldAnnotation(
                                    value=None,
                                    is_none=False,
                                    provenance=FieldProvenance(
                                        source="human", checked=False
                                    ),
                                )
                    self._auto_save()
                    st.rerun()
            st.markdown("---")

        # Render each field
        for field_name in self.fields:
            existing = self.state.fields.get(field_name)
            is_llm = (
                existing is not None and existing.provenance.source == "llm"
            )

            # Visual distinction for LLM values
            label = f"🤖 {field_name}" if is_llm else field_name

            col_input, col_none = st.columns([3, 1])

            current_is_none = existing.is_none if existing else False
            current_value = ""
            if existing and not existing.is_none:
                current_value = existing.value or ""

            with col_none:
                is_none = st.checkbox(
                    "None",
                    value=current_is_none,
                    key=f"llm_none_{field_name}",
                )

            with col_input:
                value = st.text_input(
                    label,
                    value="" if is_none else str(current_value),
                    disabled=is_none,
                    key=f"llm_val_{field_name}",
                )

            # Detect changes
            new_value = None if is_none else (value if value != "" else None)
            new_is_none = is_none

            needs_update = False
            if existing is None:
                if new_is_none or (new_value is not None and new_value != ""):
                    needs_update = True
            else:
                if existing.is_none != new_is_none or existing.value != new_value:
                    needs_update = True

            if needs_update:
                # Editing an LLM value changes provenance to human
                self._update_field(
                    field_name, new_value, new_is_none, source="human"
                )

    # ------------------------------------------------------------------
    # HITL
    # ------------------------------------------------------------------

    def render_hitl(self) -> None:
        """Render HITL mode — one field at a time for accept/reject/edit.

        After pre-fill, presents each LLM-predicted field individually.
        Accept keeps the value and sets ``checked=True``. Reject clears
        the value for manual entry. Edit allows changing the value
        (provenance becomes human).
        """
        st.subheader("Human-in-the-Loop Review")

        has_llm_fields = any(
            f in self.state.fields
            and self.state.fields[f].provenance.source == "llm"
            for f in self.fields
        )

        if not has_llm_fields:
            if st.button("🤖 Pre-fill with LLM", key="hitl_prefill_btn"):
                st.info("LLM pre-fill will be wired in app.py. Use session state to trigger.")
            st.caption(self._progress_text())
            return

        # Show review progress
        st.caption(self._hitl_review_progress())

        # Find the next unreviewed LLM field
        unreviewed = [
            f
            for f in self.fields
            if f in self.state.fields
            and self.state.fields[f].provenance.source == "llm"
            and not self.state.fields[f].provenance.checked
        ]

        if not unreviewed:
            st.success("All LLM fields have been reviewed!")
            # Show summary of all fields for final edits
            st.caption(self._progress_text())
            for field_name in self.fields:
                existing = self.state.fields.get(field_name)
                display_val = ""
                if existing:
                    display_val = "None" if existing.is_none else str(existing.value or "")
                source_icon = "🤖" if (existing and existing.provenance.source == "llm") else "👤"
                st.text(f"{source_icon} {field_name}: {display_val}")
            return

        current_field = unreviewed[0]
        fa = self.state.fields[current_field]

        st.markdown(f"### Reviewing: `{current_field}`")
        st.markdown(f"**LLM prediction:** {fa.value}")

        col_accept, col_reject, col_edit = st.columns(3)

        with col_accept:
            if st.button("✅ Accept", key=f"hitl_accept_{current_field}"):
                fa.provenance.checked = True
                self._auto_save()
                st.rerun()

        with col_reject:
            if st.button("❌ Reject", key=f"hitl_reject_{current_field}"):
                self.state.fields[current_field] = FieldAnnotation(
                    value=None,
                    is_none=False,
                    provenance=FieldProvenance(source="human", checked=False),
                )
                self._auto_save()
                st.rerun()

        with col_edit:
            edit_key = f"hitl_editing_{current_field}"
            if st.button("✏️ Edit", key=f"hitl_edit_btn_{current_field}"):
                st.session_state[edit_key] = True

        # Show edit form if editing
        if st.session_state.get(f"hitl_editing_{current_field}", False):
            col_input, col_none = st.columns([3, 1])
            with col_none:
                is_none = st.checkbox(
                    "None",
                    value=False,
                    key=f"hitl_none_{current_field}",
                )
            with col_input:
                new_val = st.text_input(
                    f"New value for {current_field}",
                    value=str(fa.value or ""),
                    disabled=is_none,
                    key=f"hitl_val_{current_field}",
                )
            if st.button("Save Edit", key=f"hitl_save_{current_field}"):
                final_value = None if is_none else (new_val if new_val != "" else None)
                self._update_field(
                    current_field,
                    final_value,
                    is_none,
                    source="human",
                    checked=False,
                )
                st.session_state[f"hitl_editing_{current_field}"] = False
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
