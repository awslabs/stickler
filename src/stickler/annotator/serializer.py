"""Annotation file I/O — save and load annotation state as JSON.

Handles serialization of AnnotationState to the two-section JSON format
(``data`` + ``metadata``) and deserialization back. Annotation files are
co-located with their source PDFs using the same filename with a ``.json``
extension.

Key behaviours:
- ``annotation_path_for(pdf_path)`` replaces ``.pdf`` with ``.json`` (bijective).
- ``save()`` writes human-readable 4-space-indented JSON.
- ``load()`` returns ``None`` for missing or corrupted files (logs a warning).
- The ``data`` section is directly constructable via ``Model(**loaded_json["data"])``.
- The ``metadata`` section stores provenance per field without polluting data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import AnnotationState, FieldAnnotation, FieldProvenance

logger = logging.getLogger(__name__)


class AnnotationSerializer:
    """Read/write annotation JSON files co-located with PDFs."""

    @staticmethod
    def annotation_path_for(pdf_path: Path) -> Path:
        """Return the ``.json`` annotation path for a given PDF path.

        Replaces the file extension with ``.json``, keeping the same
        directory and stem. The mapping is bijective — distinct PDF paths
        always produce distinct annotation paths.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Path with ``.json`` extension in the same directory.
        """
        return pdf_path.with_suffix(".json")

    @staticmethod
    def save(annotation: AnnotationState, pdf_path: Path) -> None:
        """Write annotation state to a JSON file co-located with the PDF.

        The JSON file has two top-level sections:

        - ``data``: Raw field values keyed by field path. Nested structures
          (lists, dicts) are stored as-is so that
          ``Model(**loaded_json["data"])`` works directly.
        - ``metadata``: ``schema_hash``, ``created_at``, ``updated_at``, and
          a ``fields`` dict mapping each field path to its provenance
          (``source`` and ``checked``).

        Args:
            annotation: The annotation state to persist.
            pdf_path: Path to the source PDF (used to derive the JSON path).
        """
        data: dict[str, Any] = {}
        fields_metadata: dict[str, dict[str, Any]] = {}

        for field_path, field_annotation in annotation.fields.items():
            # data section: raw values
            data[field_path] = field_annotation.value

            # metadata.fields section: provenance info
            fields_metadata[field_path] = {
                "source": field_annotation.provenance.source,
                "checked": field_annotation.provenance.checked,
            }

        output = {
            "data": data,
            "metadata": {
                "schema_hash": annotation.schema_hash,
                "created_at": annotation.created_at,
                "updated_at": annotation.updated_at,
                "fields": fields_metadata,
            },
        }

        annotation_path = AnnotationSerializer.annotation_path_for(pdf_path)
        annotation_path.write_text(
            json.dumps(output, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load(pdf_path: Path) -> AnnotationState | None:
        """Load annotation state from the JSON file for a PDF.

        Returns ``None`` when the annotation file does not exist or cannot
        be parsed (corrupted JSON, missing keys, etc.). A warning is logged
        for corrupted files so the caller can surface it in the UI.

        Args:
            pdf_path: Path to the source PDF.

        Returns:
            Restored ``AnnotationState``, or ``None`` if unavailable.
        """
        annotation_path = AnnotationSerializer.annotation_path_for(pdf_path)

        if not annotation_path.exists():
            return None

        try:
            raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read annotation file %s: %s", annotation_path, exc
            )
            return None

        try:
            data: dict[str, Any] = raw["data"]
            metadata: dict[str, Any] = raw["metadata"]
            fields_provenance: dict[str, Any] = metadata.get("fields", {})

            fields: dict[str, FieldAnnotation] = {}
            for field_path, value in data.items():
                prov_info = fields_provenance.get(field_path, {})
                provenance = FieldProvenance(
                    source=prov_info.get("source", "human"),
                    checked=prov_info.get("checked", False),
                )
                fields[field_path] = FieldAnnotation(
                    value=value,
                    is_none=value is None,
                    provenance=provenance,
                )

            return AnnotationState(
                schema_hash=metadata["schema_hash"],
                fields=fields,
                created_at=metadata["created_at"],
                updated_at=metadata["updated_at"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Corrupted annotation file %s: %s", annotation_path, exc
            )
            return None
