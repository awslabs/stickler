"""Shared Hypothesis strategies and pytest fixtures for annotator tests.

Provides reusable generators for property-based testing across all annotator
test modules. Strategies produce random but valid instances of the core domain
objects: JSON Schemas, AnnotationState, file paths, directory trees, etc.

**Import convention**: Hypothesis strategies are imported as ``st_hyp`` to
avoid collision with Streamlit's ``st`` alias::

    from hypothesis import strategies as st_hyp

Usage in test files::

    from tests.annotator.conftest import st_json_schema, st_annotation_state
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st_hyp

from stickler.annotator.models import (
    AnnotationState,
    FieldAnnotation,
    FieldProvenance,
)

# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------

# Valid field names: start with a letter, then alphanumeric + underscore, 1-30 chars
_FIELD_NAME_FIRST = st_hyp.sampled_from(list(string.ascii_lowercase))
_FIELD_NAME_REST = st_hyp.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=0,
    max_size=29,
)


@st_hyp.composite
def st_field_name(draw: st_hyp.DrawFn) -> str:
    """Generate a valid field name: starts with a letter, alphanumeric + underscore."""
    first = draw(_FIELD_NAME_FIRST)
    rest = draw(_FIELD_NAME_REST)
    return first + rest


def st_primitive_type() -> st_hyp.SearchStrategy[str]:
    """One of the four JSON Schema primitive types."""
    return st_hyp.sampled_from(["string", "number", "integer", "boolean"])


# ---------------------------------------------------------------------------
# JSON Schema strategies
# ---------------------------------------------------------------------------

# Stickler extension keys and sample values
_STICKLER_EXTENSIONS: dict[str, st_hyp.SearchStrategy[Any]] = {
    "x-aws-stickler-comparator": st_hyp.sampled_from([
        "LevenshteinComparator",
        "ExactMatchComparator",
        "NumericComparator",
    ]),
    "x-aws-stickler-threshold": st_hyp.floats(min_value=0.0, max_value=1.0),
    "x-aws-stickler-weight": st_hyp.floats(min_value=0.1, max_value=10.0),
}


@st_hyp.composite
def _st_stickler_extensions(draw: st_hyp.DrawFn) -> dict[str, Any]:
    """Optionally draw a subset of x-aws-stickler-* extensions."""
    extensions: dict[str, Any] = {}
    for key, value_strategy in _STICKLER_EXTENSIONS.items():
        if draw(st_hyp.booleans()):
            extensions[key] = draw(value_strategy)
    return extensions


@st_hyp.composite
def st_json_schema(
    draw: st_hyp.DrawFn,
    max_depth: int = 2,
    min_fields: int = 1,
    max_fields: int = 6,
    with_extensions: bool | None = None,
) -> dict[str, Any]:
    """Generate a valid JSON Schema with properties.

    Supports nested objects and arrays up to ``max_depth``. Optionally
    includes ``x-aws-stickler-*`` extension fields.

    Args:
        max_depth: Maximum nesting depth for object/array types.
        min_fields: Minimum number of top-level properties.
        max_fields: Maximum number of top-level properties.
        with_extensions: If True, always add extensions. If False, never.
            If None (default), randomly decide.
    """
    num_fields = draw(st_hyp.integers(min_value=min_fields, max_value=max_fields))
    names = draw(
        st_hyp.lists(
            st_field_name(),
            min_size=num_fields,
            max_size=num_fields,
            unique=True,
        )
    )

    properties: dict[str, Any] = {}
    for name in names:
        properties[name] = draw(_st_field_schema(max_depth))

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": names,
    }

    # Optionally add stickler extensions at the root level
    add_ext = draw(st_hyp.booleans()) if with_extensions is None else with_extensions
    if add_ext:
        schema.update(draw(_st_stickler_extensions()))

    return schema


@st_hyp.composite
def _st_field_schema(draw: st_hyp.DrawFn, max_depth: int) -> dict[str, Any]:
    """Generate a JSON Schema for a single field (possibly nested)."""
    if max_depth <= 0:
        # Only primitives at leaf level
        return {"type": draw(st_primitive_type())}

    kind = draw(st_hyp.sampled_from(["primitive", "object", "array"]))

    if kind == "primitive":
        field_schema: dict[str, Any] = {"type": draw(st_primitive_type())}
    elif kind == "object":
        num_sub = draw(st_hyp.integers(min_value=1, max_value=3))
        sub_names = draw(
            st_hyp.lists(
                st_field_name(), min_size=num_sub, max_size=num_sub, unique=True
            )
        )
        sub_props = {}
        for sn in sub_names:
            sub_props[sn] = draw(_st_field_schema(max_depth - 1))
        field_schema = {
            "type": "object",
            "properties": sub_props,
            "required": sub_names,
        }
    else:
        # array — items are either primitive or object
        items_kind = draw(st_hyp.sampled_from(["primitive", "object"]))
        if items_kind == "primitive":
            field_schema = {
                "type": "array",
                "items": {"type": draw(st_primitive_type())},
            }
        else:
            num_sub = draw(st_hyp.integers(min_value=1, max_value=3))
            sub_names = draw(
                st_hyp.lists(
                    st_field_name(), min_size=num_sub, max_size=num_sub, unique=True
                )
            )
            sub_props = {}
            for sn in sub_names:
                sub_props[sn] = draw(_st_field_schema(max_depth - 1))
            field_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": sub_props,
                    "required": sub_names,
                },
            }

    # Optionally add stickler extensions to this field
    if draw(st_hyp.booleans()):
        field_schema.update(draw(_st_stickler_extensions()))

    return field_schema


# ---------------------------------------------------------------------------
# Provenance and annotation strategies
# ---------------------------------------------------------------------------


@st_hyp.composite
def st_field_provenance(draw: st_hyp.DrawFn) -> FieldProvenance:
    """Generate a random FieldProvenance instance."""
    source = draw(st_hyp.sampled_from(["human", "llm"]))
    checked = draw(st_hyp.booleans())
    return FieldProvenance(source=source, checked=checked)


# Value generators per JSON Schema type
_VALUE_FOR_TYPE: dict[str, st_hyp.SearchStrategy[Any]] = {
    "string": st_hyp.text(min_size=0, max_size=50),
    "number": st_hyp.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
    "integer": st_hyp.integers(min_value=-10000, max_value=10000),
    "boolean": st_hyp.booleans(),
}


@st_hyp.composite
def st_field_annotation(draw: st_hyp.DrawFn) -> FieldAnnotation:
    """Generate a random FieldAnnotation with a value and provenance."""
    is_none = draw(st_hyp.booleans())
    if is_none:
        value = None
    else:
        value = draw(st_hyp.sampled_from(["string", "number", "integer", "boolean"]))
        value = draw(_VALUE_FOR_TYPE[value])
    provenance = draw(st_field_provenance())
    return FieldAnnotation(value=value, is_none=is_none, provenance=provenance)


@st_hyp.composite
def st_annotation_state(
    draw: st_hyp.DrawFn,
    schema_fields: list[str] | None = None,
) -> AnnotationState:
    """Generate a random AnnotationState.

    Args:
        schema_fields: If provided, use these as field names. Otherwise
            generate 1-6 random field names.
    """
    if schema_fields is None:
        num = draw(st_hyp.integers(min_value=1, max_value=6))
        schema_fields = draw(
            st_hyp.lists(st_field_name(), min_size=num, max_size=num, unique=True)
        )

    fields: dict[str, FieldAnnotation] = {}
    for name in schema_fields:
        fields[name] = draw(st_field_annotation())

    schema_hash = draw(st_hyp.text(alphabet=string.hexdigits.lower(), min_size=8, max_size=16))
    created = draw(st_hyp.sampled_from([
        "2025-01-01T00:00:00Z",
        "2025-03-15T10:30:00Z",
        "2025-06-20T18:45:00Z",
    ]))
    updated = draw(st_hyp.sampled_from([
        "2025-01-01T01:00:00Z",
        "2025-03-15T12:00:00Z",
        "2025-06-20T20:00:00Z",
    ]))

    return AnnotationState(
        schema_hash=schema_hash,
        fields=fields,
        created_at=created,
        updated_at=updated,
    )


# ---------------------------------------------------------------------------
# File path strategies
# ---------------------------------------------------------------------------


@st_hyp.composite
def st_pdf_filename(draw: st_hyp.DrawFn) -> str:
    """Generate a valid PDF filename (no path separators)."""
    stem = draw(
        st_hyp.text(
            alphabet=string.ascii_letters + string.digits + "_-",
            min_size=1,
            max_size=30,
        )
    )
    return stem + ".pdf"


def st_directory_tree(
    tmp_path: Path,
    min_pdfs: int = 1,
    max_pdfs: int = 8,
    min_others: int = 0,
    max_others: int = 5,
) -> st_hyp.SearchStrategy[tuple[Path, list[Path]]]:
    """Strategy that creates a temp directory with PDF and non-PDF files.

    Returns ``(dir_path, expected_pdf_paths)`` — the directory and the
    list of PDF file paths that were created.
    """

    @st_hyp.composite
    def _build(draw: st_hyp.DrawFn) -> tuple[Path, list[Path]]:
        num_pdfs = draw(st_hyp.integers(min_value=min_pdfs, max_value=max_pdfs))
        num_others = draw(st_hyp.integers(min_value=min_others, max_value=max_others))

        # Generate unique subdirectory names for variety
        subdirs = [""] + draw(
            st_hyp.lists(
                st_hyp.text(
                    alphabet=string.ascii_lowercase + string.digits,
                    min_size=1,
                    max_size=10,
                ),
                min_size=0,
                max_size=3,
                unique=True,
            )
        )

        pdf_paths: list[Path] = []

        # Create PDF files
        for i in range(num_pdfs):
            subdir = draw(st_hyp.sampled_from(subdirs))
            # Vary case of extension
            ext = draw(st_hyp.sampled_from([".pdf", ".PDF", ".Pdf"]))
            name = f"doc_{i}{ext}"
            dir_path = tmp_path / subdir if subdir else tmp_path
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / name
            file_path.write_bytes(b"%PDF-1.4")
            pdf_paths.append(file_path)

        # Create non-PDF files
        non_pdf_exts = [".txt", ".png", ".jpg", ".docx", ".csv", ".json"]
        for i in range(num_others):
            subdir = draw(st_hyp.sampled_from(subdirs))
            ext = draw(st_hyp.sampled_from(non_pdf_exts))
            name = f"other_{i}{ext}"
            dir_path = tmp_path / subdir if subdir else tmp_path
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / name).write_bytes(b"not a pdf")

        return tmp_path, pdf_paths

    return _build()


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_schema() -> dict[str, Any]:
    """A simple JSON Schema with a few fields for basic tests."""
    return {
        "type": "object",
        "properties": {
            "vendor_name": {"type": "string"},
            "invoice_number": {"type": "string"},
            "total_amount": {"type": "number"},
            "is_paid": {"type": "boolean"},
        },
        "required": ["vendor_name", "invoice_number", "total_amount", "is_paid"],
    }


@pytest.fixture()
def sample_annotation_state() -> AnnotationState:
    """A sample AnnotationState with mixed provenance for testing."""
    return AnnotationState(
        schema_hash="abc123def456",
        fields={
            "vendor_name": FieldAnnotation(
                value="Acme Corp",
                provenance=FieldProvenance(source="human", checked=False),
            ),
            "invoice_number": FieldAnnotation(
                value="INV-001",
                provenance=FieldProvenance(source="llm", checked=True),
            ),
            "total_amount": FieldAnnotation(
                value=1500.00,
                provenance=FieldProvenance(source="llm", checked=False),
            ),
            "is_paid": FieldAnnotation(
                value=None,
                is_none=True,
                provenance=FieldProvenance(source="human", checked=False),
            ),
        },
        created_at="2025-01-15T10:30:00Z",
        updated_at="2025-01-15T11:00:00Z",
    )
