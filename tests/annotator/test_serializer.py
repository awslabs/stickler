"""Unit tests for serializer.py — annotation file I/O."""

import json
from pathlib import Path

from stickler.annotator.models import (
    AnnotationState,
    FieldAnnotation,
    FieldProvenance,
)
from stickler.annotator.serializer import AnnotationSerializer


class TestAnnotationPathFor:
    """Tests for AnnotationSerializer.annotation_path_for()."""

    def test_replaces_pdf_extension_with_json(self):
        pdf = Path("/data/invoices/doc.pdf")
        assert AnnotationSerializer.annotation_path_for(pdf) == Path(
            "/data/invoices/doc.json"
        )

    def test_preserves_directory(self):
        pdf = Path("/some/deep/path/file.pdf")
        result = AnnotationSerializer.annotation_path_for(pdf)
        assert result.parent == pdf.parent

    def test_handles_uppercase_extension(self):
        pdf = Path("/data/DOC.PDF")
        result = AnnotationSerializer.annotation_path_for(pdf)
        assert result == Path("/data/DOC.json")

    def test_bijective_distinct_pdfs_produce_distinct_paths(self):
        a = AnnotationSerializer.annotation_path_for(Path("/dir/a.pdf"))
        b = AnnotationSerializer.annotation_path_for(Path("/dir/b.pdf"))
        assert a != b


class TestSave:
    """Tests for AnnotationSerializer.save()."""

    def _make_state(self) -> AnnotationState:
        return AnnotationState(
            schema_hash="hash123",
            fields={
                "vendor_name": FieldAnnotation(
                    value="Acme Corp",
                    provenance=FieldProvenance(source="human", checked=False),
                ),
                "amount": FieldAnnotation(
                    value=42.5,
                    provenance=FieldProvenance(source="llm", checked=True),
                ),
            },
            created_at="2025-01-15T10:30:00Z",
            updated_at="2025-01-15T11:00:00Z",
        )

    def test_creates_json_file(self, tmp_path: Path):
        state = self._make_state()
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        AnnotationSerializer.save(state, pdf)
        assert (tmp_path / "doc.json").exists()

    def test_data_section_contains_raw_values(self, tmp_path: Path):
        state = self._make_state()
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        AnnotationSerializer.save(state, pdf)
        loaded = json.loads((tmp_path / "doc.json").read_text())
        assert loaded["data"]["vendor_name"] == "Acme Corp"
        assert loaded["data"]["amount"] == 42.5

    def test_metadata_section_contains_provenance(self, tmp_path: Path):
        state = self._make_state()
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        AnnotationSerializer.save(state, pdf)
        loaded = json.loads((tmp_path / "doc.json").read_text())
        meta = loaded["metadata"]
        assert meta["schema_hash"] == "hash123"
        assert meta["created_at"] == "2025-01-15T10:30:00Z"
        assert meta["fields"]["vendor_name"] == {
            "source": "human",
            "checked": False,
        }
        assert meta["fields"]["amount"] == {
            "source": "llm",
            "checked": True,
        }

    def test_output_is_indented_json(self, tmp_path: Path):
        state = self._make_state()
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        AnnotationSerializer.save(state, pdf)
        text = (tmp_path / "doc.json").read_text()
        # 4-space indentation check
        assert "    " in text
        # Should be parseable
        json.loads(text)

    def test_none_value_stored_as_null(self, tmp_path: Path):
        state = AnnotationState(
            schema_hash="h",
            fields={
                "missing": FieldAnnotation(
                    value=None,
                    is_none=True,
                    provenance=FieldProvenance(source="human"),
                ),
            },
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        AnnotationSerializer.save(state, pdf)
        loaded = json.loads((tmp_path / "doc.json").read_text())
        assert loaded["data"]["missing"] is None


class TestLoad:
    """Tests for AnnotationSerializer.load()."""

    def test_returns_none_when_no_file(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        assert AnnotationSerializer.load(pdf) is None

    def test_loads_valid_annotation(self, tmp_path: Path):
        annotation = {
            "data": {"vendor_name": "Acme", "amount": 100},
            "metadata": {
                "schema_hash": "abc",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T01:00:00Z",
                "fields": {
                    "vendor_name": {"source": "human", "checked": False},
                    "amount": {"source": "llm", "checked": True},
                },
            },
        }
        (tmp_path / "doc.json").write_text(json.dumps(annotation))
        result = AnnotationSerializer.load(tmp_path / "doc.pdf")
        assert result is not None
        assert result.schema_hash == "abc"
        assert result.fields["vendor_name"].value == "Acme"
        assert result.fields["vendor_name"].provenance.source == "human"
        assert result.fields["amount"].provenance.checked is True

    def test_returns_none_for_corrupted_json(self, tmp_path: Path):
        (tmp_path / "doc.json").write_text("{invalid json!!!")
        result = AnnotationSerializer.load(tmp_path / "doc.pdf")
        assert result is None

    def test_returns_none_for_missing_keys(self, tmp_path: Path):
        (tmp_path / "doc.json").write_text(json.dumps({"unexpected": True}))
        result = AnnotationSerializer.load(tmp_path / "doc.pdf")
        assert result is None

    def test_defaults_provenance_when_missing(self, tmp_path: Path):
        """Fields without provenance metadata default to human/unchecked."""
        annotation = {
            "data": {"name": "Test"},
            "metadata": {
                "schema_hash": "h",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "fields": {},
            },
        }
        (tmp_path / "doc.json").write_text(json.dumps(annotation))
        result = AnnotationSerializer.load(tmp_path / "doc.pdf")
        assert result is not None
        assert result.fields["name"].provenance.source == "human"
        assert result.fields["name"].provenance.checked is False

    def test_is_none_set_for_null_values(self, tmp_path: Path):
        annotation = {
            "data": {"field": None},
            "metadata": {
                "schema_hash": "h",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "fields": {"field": {"source": "human", "checked": False}},
            },
        }
        (tmp_path / "doc.json").write_text(json.dumps(annotation))
        result = AnnotationSerializer.load(tmp_path / "doc.pdf")
        assert result is not None
        assert result.fields["field"].is_none is True
        assert result.fields["field"].value is None


class TestRoundTrip:
    """Tests for save → load round-trip integrity."""

    def test_round_trip_preserves_state(self, tmp_path: Path):
        original = AnnotationState(
            schema_hash="round_trip_hash",
            fields={
                "name": FieldAnnotation(
                    value="Test Corp",
                    provenance=FieldProvenance(source="human", checked=False),
                ),
                "total": FieldAnnotation(
                    value=999.99,
                    provenance=FieldProvenance(source="llm", checked=True),
                ),
            },
            created_at="2025-06-01T00:00:00Z",
            updated_at="2025-06-01T12:00:00Z",
        )
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        AnnotationSerializer.save(original, pdf)
        loaded = AnnotationSerializer.load(pdf)

        assert loaded is not None
        assert loaded.schema_hash == original.schema_hash
        assert loaded.created_at == original.created_at
        assert loaded.updated_at == original.updated_at
        for key in original.fields:
            assert loaded.fields[key].value == original.fields[key].value
            assert (
                loaded.fields[key].provenance.source
                == original.fields[key].provenance.source
            )
            assert (
                loaded.fields[key].provenance.checked
                == original.fields[key].provenance.checked
            )

    def test_round_trip_with_nested_data(self, tmp_path: Path):
        original = AnnotationState(
            schema_hash="nested_hash",
            fields={
                "line_items": FieldAnnotation(
                    value=[
                        {"description": "Widget", "amount": 10.0},
                        {"description": "Gadget", "amount": 20.0},
                    ],
                    provenance=FieldProvenance(source="llm", checked=False),
                ),
            },
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        AnnotationSerializer.save(original, pdf)
        loaded = AnnotationSerializer.load(pdf)

        assert loaded is not None
        assert loaded.fields["line_items"].value == [
            {"description": "Widget", "amount": 10.0},
            {"description": "Gadget", "amount": 20.0},
        ]
