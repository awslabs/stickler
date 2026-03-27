"""In-app schema builder for creating annotation schemas without writing code.

Provides a Streamlit UI for interactively building JSON Schema documents.
Users add/remove fields with types (string, number, integer, boolean),
nested objects, and arrays. The output is a valid JSON Schema compatible
with ``StructuredModel.from_json_schema()``.

Field state is maintained in ``st.session_state`` across Streamlit reruns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

# Primitive types supported by the schema builder.
PRIMITIVE_TYPES = ("string", "number", "integer", "boolean")

# All types including composite types.
ALL_TYPES = (*PRIMITIVE_TYPES, "object", "array")


def _default_fields_key() -> str:
    """Session-state key for the field list."""
    return "schema_builder_fields"


def _ensure_session_state() -> None:
    """Initialise the field list in session state if absent."""
    key = _default_fields_key()
    if key not in st.session_state:
        st.session_state[key] = []


def _get_fields() -> list[dict[str, Any]]:
    """Return the current field list from session state."""
    _ensure_session_state()
    return st.session_state[_default_fields_key()]


def _set_fields(fields: list[dict[str, Any]]) -> None:
    """Replace the field list in session state."""
    st.session_state[_default_fields_key()] = fields


def _build_field_schema(field: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal field dict to a JSON Schema property definition.

    Handles primitive types, nested objects (with sub-properties), and
    arrays (with an items type).
    """
    ftype = field["type"]

    if ftype == "object":
        props: dict[str, Any] = {}
        for sub in field.get("properties", []):
            props[sub["name"]] = _build_field_schema(sub)
        schema: dict[str, Any] = {"type": "object", "properties": props}
        if props:
            schema["required"] = list(props.keys())
        return schema

    if ftype == "array":
        items_type = field.get("items_type", "string")
        if items_type == "object":
            items_props: dict[str, Any] = {}
            for sub in field.get("items_properties", []):
                items_props[sub["name"]] = _build_field_schema(sub)
            items_schema: dict[str, Any] = {
                "type": "object",
                "properties": items_props,
            }
            if items_props:
                items_schema["required"] = list(items_props.keys())
            return {"type": "array", "items": items_schema}
        return {"type": "array", "items": {"type": items_type}}

    # Primitive type
    return {"type": ftype}


def _render_field_adder(
    label_prefix: str,
    key_prefix: str,
    target_list: list[dict[str, Any]],
    allow_composite: bool = True,
) -> None:
    """Render widgets for adding a new field to *target_list*.

    Args:
        label_prefix: Human-readable context shown in widget labels.
        key_prefix: Unique prefix for Streamlit widget keys.
        target_list: The list of field dicts to append to.
        allow_composite: Whether to offer object/array types.
    """
    available_types = ALL_TYPES if allow_composite else PRIMITIVE_TYPES
    cols = st.columns([2, 1, 1])
    with cols[0]:
        name = st.text_input("Field name", key=f"{key_prefix}_name")
    with cols[1]:
        ftype = st.selectbox("Type", available_types, key=f"{key_prefix}_type")
    with cols[2]:
        items_type: str | None = None
        if ftype == "array":
            items_type = st.selectbox(
                "Items type",
                (*PRIMITIVE_TYPES, "object"),
                key=f"{key_prefix}_items_type",
            )

    if st.button(f"Add {label_prefix} field", key=f"{key_prefix}_add"):
        clean_name = name.strip()
        if not clean_name:
            st.warning("Field name cannot be empty.")
        elif any(f["name"] == clean_name for f in target_list):
            st.warning(f"Field '{clean_name}' already exists.")
        else:
            new_field: dict[str, Any] = {"name": clean_name, "type": ftype}
            if ftype == "object":
                new_field["properties"] = []
            elif ftype == "array":
                new_field["items_type"] = items_type or "string"
                if items_type == "object":
                    new_field["items_properties"] = []
            target_list.append(new_field)
            st.rerun()


def _render_field_list(
    fields: list[dict[str, Any]],
    key_prefix: str,
    depth: int = 0,
) -> None:
    """Render the current field list with remove buttons and nested editors.

    Args:
        fields: List of field dicts to display.
        key_prefix: Unique prefix for widget keys.
        depth: Nesting depth (for indentation context).
    """
    indices_to_remove: list[int] = []
    for idx, field in enumerate(fields):
        fkey = f"{key_prefix}_{idx}"
        col1, col2 = st.columns([4, 1])
        with col1:
            type_label = field["type"]
            if field["type"] == "array":
                type_label = f"array[{field.get('items_type', 'string')}]"
            st.text(f"{'  ' * depth}{field['name']}  ({type_label})")
        with col2:
            if st.button("Remove", key=f"{fkey}_rm"):
                indices_to_remove.append(idx)

        # Nested object properties
        if field["type"] == "object":
            with st.expander(f"Properties of '{field['name']}'", expanded=False):
                _render_field_list(field["properties"], f"{fkey}_obj", depth + 1)
                _render_field_adder(
                    label_prefix=field["name"],
                    key_prefix=f"{fkey}_obj_add",
                    target_list=field["properties"],
                    allow_composite=True,
                )

        # Array with object items
        if field["type"] == "array" and field.get("items_type") == "object":
            with st.expander(
                f"Item properties of '{field['name']}'", expanded=False
            ):
                _render_field_list(
                    field["items_properties"], f"{fkey}_arr", depth + 1
                )
                _render_field_adder(
                    label_prefix=f"{field['name']} item",
                    key_prefix=f"{fkey}_arr_add",
                    target_list=field["items_properties"],
                    allow_composite=True,
                )

    # Apply removals in reverse order to keep indices stable
    for i in sorted(indices_to_remove, reverse=True):
        fields.pop(i)
    if indices_to_remove:
        st.rerun()


class SchemaBuilder:
    """Interactive JSON Schema builder rendered via Streamlit widgets.

    Maintains field definitions in ``st.session_state`` so they persist
    across Streamlit reruns. Call :meth:`render` inside a Streamlit app
    to display the builder UI.
    """

    def render(self) -> dict | None:
        """Render the schema builder UI.

        Returns the finalised JSON Schema dict when the user clicks
        *Finalize Schema*, or ``None`` while still editing.
        """
        st.subheader("Schema Builder")
        st.caption("🧪 Beta — for production use, bring a JSON Schema file.")
        _ensure_session_state()
        fields = _get_fields()

        # --- field list ---
        if fields:
            st.markdown("**Current fields**")
            _render_field_list(fields, "sb_root")
        else:
            st.info("No fields yet. Add one below.")

        # --- add top-level field ---
        st.markdown("---")
        _render_field_adder(
            label_prefix="top-level",
            key_prefix="sb_root_add",
            target_list=fields,
            allow_composite=True,
        )

        # --- finalize ---
        st.markdown("---")
        if st.button("Finalize Schema", key="sb_finalize"):
            if not fields:
                st.error("Add at least one field before finalizing.")
                return None
            schema = self._fields_to_schema(fields)
            return schema

        return None

    def export_to_file(self, schema: dict, path: str | Path) -> None:
        """Write *schema* to disk as indented JSON.

        Args:
            schema: A JSON Schema dict (typically from :meth:`render`).
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(schema, indent=4, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fields_to_schema(fields: list[dict[str, Any]]) -> dict:
        """Convert the internal field list to a JSON Schema document.

        The root schema always has ``"type": "object"`` with a
        ``"properties"`` dict — the format expected by
        ``StructuredModel.from_json_schema()``.
        """
        properties: dict[str, Any] = {}
        for field in fields:
            properties[field["name"]] = _build_field_schema(field)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if properties:
            schema["required"] = list(properties.keys())
        return schema