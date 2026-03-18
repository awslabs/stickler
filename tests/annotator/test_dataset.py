"""Unit tests for dataset.py — PDF discovery and document status tracking."""

import json
from pathlib import Path

import pytest

from stickler.annotator.dataset import DatasetManager, PDFDocument
from stickler.annotator.models import DocumentStatus


class TestDatasetManagerInit:
    """Tests for DatasetManager construction and validation."""

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        """FileNotFoundError when directory does not exist."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DatasetManager(tmp_path / "nonexistent")

    def test_file_instead_of_directory_raises(self, tmp_path: Path):
        """NotADirectoryError when path is a file."""
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError, match="not a directory"):
            DatasetManager(f)

    def test_empty_directory_raises(self, tmp_path: Path):
        """ValueError when directory has no PDFs."""
        with pytest.raises(ValueError, match="No PDF files found"):
            DatasetManager(tmp_path)

    def test_directory_with_non_pdf_files_raises(self, tmp_path: Path):
        """ValueError when directory has files but none are PDFs."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")
        with pytest.raises(ValueError, match="No PDF files found"):
            DatasetManager(tmp_path)

    def test_valid_directory_with_pdf(self, tmp_path: Path):
        """Succeeds when directory contains at least one PDF."""
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        mgr = DatasetManager(tmp_path)
        assert mgr.dataset_dir == tmp_path

    def test_accepts_string_path(self, tmp_path: Path):
        """Accepts str path in addition to Path."""
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        mgr = DatasetManager(str(tmp_path))
        assert mgr.dataset_dir == tmp_path


class TestDiscover:
    """Tests for DatasetManager.discover()."""

    def test_discovers_single_pdf(self, tmp_path: Path):
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        mgr = DatasetManager(tmp_path)
        docs = mgr.discover()
        assert len(docs) == 1
        assert docs[0].path == tmp_path / "doc.pdf"
        assert docs[0].status == DocumentStatus.NOT_STARTED

    def test_discovers_pdfs_recursively(self, tmp_path: Path):
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "b.pdf").write_bytes(b"%PDF")
        mgr = DatasetManager(tmp_path)
        docs = mgr.discover()
        paths = [d.path for d in docs]
        assert tmp_path / "a.pdf" in paths
        assert sub / "b.pdf" in paths
        assert len(docs) == 2

    def test_case_insensitive_extension(self, tmp_path: Path):
        """Discovers .PDF, .Pdf, .pDf etc."""
        (tmp_path / "upper.PDF").write_bytes(b"%PDF")
        (tmp_path / "mixed.Pdf").write_bytes(b"%PDF")
        (tmp_path / "lower.pdf").write_bytes(b"%PDF")
        mgr = DatasetManager(tmp_path)
        docs = mgr.discover()
        assert len(docs) == 3

    def test_excludes_non_pdf_files(self, tmp_path: Path):
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        mgr = DatasetManager(tmp_path)
        docs = mgr.discover()
        assert len(docs) == 1
        assert docs[0].path.suffix == ".pdf"

    def test_results_sorted_by_path(self, tmp_path: Path):
        (tmp_path / "c.pdf").write_bytes(b"%PDF")
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "b.pdf").write_bytes(b"%PDF")
        mgr = DatasetManager(tmp_path)
        docs = mgr.discover()
        paths = [d.path for d in docs]
        assert paths == sorted(paths)

    def test_excludes_pdf_like_names(self, tmp_path: Path):
        """Files like 'pdf_notes.txt' or 'my.pdf.bak' are not included."""
        (tmp_path / "real.pdf").write_bytes(b"%PDF")
        (tmp_path / "my.pdf.bak").write_text("backup")
        (tmp_path / "pdf_notes.txt").write_text("notes")
        mgr = DatasetManager(tmp_path)
        docs = mgr.discover()
        assert len(docs) == 1


class TestGetStatus:
    """Tests for DatasetManager.get_status()."""

    def _make_manager(self, tmp_path: Path) -> DatasetManager:
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        return DatasetManager(tmp_path)

    def test_no_annotation_file_returns_not_started(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status(tmp_path / "doc.pdf", ["field_a", "field_b"])
        assert status == DocumentStatus.NOT_STARTED

    def test_all_fields_annotated_returns_complete(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        annotation = {
            "data": {"name": "Acme", "amount": 100},
            "metadata": {
                "schema_hash": "abc",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "fields": {
                    "name": {"source": "human", "checked": False},
                    "amount": {"source": "human", "checked": False},
                },
            },
        }
        ann_dir = tmp_path / ".annotations"
        ann_dir.mkdir()
        (ann_dir / "doc.json").write_text(json.dumps(annotation))
        status = mgr.get_status(tmp_path / "doc.pdf", ["name", "amount"])
        assert status == DocumentStatus.COMPLETE

    def test_some_fields_annotated_returns_in_progress(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        annotation = {
            "data": {"name": "Acme"},
            "metadata": {
                "schema_hash": "abc",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "fields": {
                    "name": {"source": "human", "checked": False},
                },
            },
        }
        ann_dir = tmp_path / ".annotations"
        ann_dir.mkdir()
        (ann_dir / "doc.json").write_text(json.dumps(annotation))
        status = mgr.get_status(tmp_path / "doc.pdf", ["name", "amount"])
        assert status == DocumentStatus.IN_PROGRESS

    def test_empty_schema_fields_returns_complete(self, tmp_path: Path):
        """With no schema fields, any annotation file means complete."""
        mgr = self._make_manager(tmp_path)
        annotation = {
            "data": {},
            "metadata": {
                "schema_hash": "abc",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "fields": {},
            },
        }
        ann_dir = tmp_path / ".annotations"
        ann_dir.mkdir()
        (ann_dir / "doc.json").write_text(json.dumps(annotation))
        status = mgr.get_status(tmp_path / "doc.pdf", [])
        assert status == DocumentStatus.COMPLETE

    def test_corrupted_json_returns_not_started(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        ann_dir = tmp_path / ".annotations"
        ann_dir.mkdir()
        (ann_dir / "doc.json").write_text("not valid json{{{")
        status = mgr.get_status(tmp_path / "doc.pdf", ["field_a"])
        assert status == DocumentStatus.NOT_STARTED

    def test_annotation_file_with_no_matching_fields_returns_not_started(
        self, tmp_path: Path
    ):
        mgr = self._make_manager(tmp_path)
        annotation = {
            "data": {"other_field": "value"},
            "metadata": {
                "schema_hash": "abc",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "fields": {
                    "other_field": {"source": "human", "checked": False},
                },
            },
        }
        ann_dir = tmp_path / ".annotations"
        ann_dir.mkdir()
        (ann_dir / "doc.json").write_text(json.dumps(annotation))
        status = mgr.get_status(tmp_path / "doc.pdf", ["expected_field"])
        assert status == DocumentStatus.NOT_STARTED
