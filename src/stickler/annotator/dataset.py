"""PDF dataset discovery and document status tracking.

Recursively discovers PDF files in a directory and derives each document's
completion status from its co-located annotation JSON file. No separate
status store — status is always computed from the annotation file state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .models import DocumentStatus
from .serializer import AnnotationSerializer

logger = logging.getLogger(__name__)


@dataclass
class PDFDocument:
    """A PDF document in the annotation queue.

    Attributes:
        path: Absolute path to the PDF file.
        status: Completion state derived from the annotation file.
    """

    path: Path
    status: DocumentStatus


class DatasetManager:
    """Discovers PDFs in a directory and tracks their annotation status.

    The manager validates the dataset directory on construction and provides
    methods to discover PDFs and derive their completion status from
    co-located annotation JSON files.

    Attributes:
        dataset_dir: Validated path to the dataset directory.
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        """Initialize with a dataset directory path.

        Args:
            dataset_dir: Path to the directory containing PDF files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If the directory contains no PDF files.
        """
        self.dataset_dir = Path(dataset_dir).resolve()

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.dataset_dir}"
            )

        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: {self.dataset_dir}"
            )

        # Check for at least one PDF file
        pdfs = list(self._find_pdfs())
        if not pdfs:
            raise ValueError(
                f"No PDF files found in dataset directory: {self.dataset_dir}"
            )

    def discover(self) -> list[PDFDocument]:
        """Recursively discover all PDF files and return them with status.

        Returns a sorted list of PDFDocument instances. Status defaults to
        NOT_STARTED — call with schema_fields via get_status() for accurate
        status derivation.

        Returns:
            List of PDFDocument sorted by path.
        """
        documents = []
        for pdf_path in self._find_pdfs():
            documents.append(
                PDFDocument(
                    path=pdf_path,
                    status=DocumentStatus.NOT_STARTED,
                )
            )
        return sorted(documents, key=lambda d: d.path)

    def get_status(
        self, pdf_path: Path, schema_fields: list[str], session=None
    ) -> DocumentStatus:
        """Derive document status from the annotation file on disk.

        Args:
            pdf_path: Path to the PDF file.
            schema_fields: List of field paths expected by the schema.
            session: Optional AnnotationSession for session-scoped lookup.

        Returns:
            The derived DocumentStatus.
        """
        annotation_path = AnnotationSerializer.annotation_path_for(
            pdf_path, session=session
        )

        if not annotation_path.exists():
            return DocumentStatus.NOT_STARTED

        try:
            with open(annotation_path, "r") as f:
                annotation_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read annotation file %s: %s", annotation_path, e)
            return DocumentStatus.NOT_STARTED

        if not schema_fields:
            return DocumentStatus.COMPLETE

        metadata = annotation_data.get("metadata", {})
        fields_metadata = metadata.get("fields", {})
        data = annotation_data.get("data", {})

        annotated_count = 0
        for field in schema_fields:
            if field in fields_metadata:
                annotated_count += 1
            elif field in data and data[field] is not None:
                annotated_count += 1

        if annotated_count == 0:
            return DocumentStatus.NOT_STARTED
        elif annotated_count >= len(schema_fields):
            return DocumentStatus.COMPLETE
        else:
            return DocumentStatus.IN_PROGRESS

    def _find_pdfs(self) -> list[Path]:
        """Recursively find all PDF files (case-insensitive extension).

        Returns:
            List of Path objects for discovered PDF files.
        """
        return [
            p
            for p in self.dataset_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() == ".pdf"
            and not any(
                part.startswith(".") for part in p.relative_to(self.dataset_dir).parts
            )
        ]
