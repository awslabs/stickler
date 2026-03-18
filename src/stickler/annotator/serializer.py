"""Annotation session storage — manifest + per-doc JSON files.

Storage layout::

    <dataset_dir>/
      .annotations/
        manifest.json          ← top-level index; embeds the full schema
        <session_guid>/
          <pdf_stem>.json      ← per-doc annotation (data + metadata)

The manifest is the single source of truth for:
- The JSON Schema used for all sessions in this dataset
- All session GUIDs with annotator, timestamps, and progress counts

A session is identified by a UUID4 generated at creation time.  The same
annotator using the same schema on the same dataset always creates a new
session (GUIDs are not reused), but can resume any previous session via
its GUID.

The deep link ``?dataset=./files&session=<guid>`` resumes a specific
session without needing a separate schema file — the schema is loaded
from the manifest.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AnnotationState, FieldAnnotation, FieldProvenance

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnnotationSession:
    """Represents one annotator's session for a dataset + schema combination.

    Attributes:
        dataset_dir: Root directory of the PDF dataset.
        session_id: UUID4 string identifying this session.
        annotator: Username of the annotator.
        schema: The full JSON Schema dict embedded in the manifest.
        schema_hash: MD5 hash of the schema for quick comparison.
    """

    def __init__(
        self,
        dataset_dir: Path,
        session_id: str,
        annotator: str,
        schema: dict,
        schema_hash: str,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.session_id = session_id
        self.annotator = annotator
        self.schema = schema
        self.schema_hash = schema_hash

    @property
    def session_dir(self) -> Path:
        return self.dataset_dir / ".annotations" / self.session_id

    def annotation_path_for(self, pdf_path: Path) -> Path:
        """Return the per-doc JSON path within this session."""
        return self.session_dir / pdf_path.with_suffix(".json").name

    def exists(self, pdf_path: Path) -> bool:
        return self.annotation_path_for(pdf_path).exists()

    def save(self, annotation: AnnotationState, pdf_path: Path) -> None:
        """Write annotation state for one PDF into this session."""
        data: dict[str, Any] = {}
        fields_metadata: dict[str, dict[str, Any]] = {}

        for field_path, field_annotation in annotation.fields.items():
            data[field_path] = field_annotation.value
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

        annotation_path = self.annotation_path_for(pdf_path)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps(output, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Update manifest progress counts
        AnnotationManifest(self.dataset_dir).update_session_progress(self.session_id)

    def load(self, pdf_path: Path) -> AnnotationState | None:
        """Load annotation state for one PDF from this session."""
        annotation_path = self.annotation_path_for(pdf_path)
        if not annotation_path.exists():
            return None

        try:
            raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read annotation file %s: %s", annotation_path, exc)
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
            logger.warning("Corrupted annotation file %s: %s", annotation_path, exc)
            return None


class AnnotationManifest:
    """Top-level manifest for a dataset's annotation sessions.

    The manifest lives at ``<dataset_dir>/.annotations/manifest.json`` and
    contains:

    - ``schema``: The full JSON Schema embedded at session creation time.
    - ``schema_hash``: MD5 of the schema for quick comparison.
    - ``sessions``: Dict of session_id → session metadata.

    This is the single file needed to reconstruct the full annotation
    history for a dataset, enabling merging, comparison, and eval loading.
    """

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = self.dataset_dir / ".annotations" / _MANIFEST_FILENAME

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """Load and return the raw manifest dict, or an empty manifest."""
        if not self.manifest_path.exists():
            return {"schema": None, "schema_hash": None, "sessions": {}}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read manifest %s: %s", self.manifest_path, exc)
            return {"schema": None, "schema_hash": None, "sessions": {}}

    def list_sessions(self) -> list[dict]:
        """Return all sessions sorted by updated_at descending."""
        manifest = self.load()
        sessions = []
        for sid, meta in manifest.get("sessions", {}).items():
            sessions.append({"session_id": sid, **meta})
        return sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True)

    def get_session(self, session_id: str) -> "AnnotationSession | None":
        """Load a specific session by GUID."""
        manifest = self.load()
        if session_id not in manifest.get("sessions", {}):
            return None
        schema = manifest.get("schema")
        schema_hash = manifest.get("schema_hash", "")
        meta = manifest["sessions"][session_id]
        return AnnotationSession(
            dataset_dir=self.dataset_dir,
            session_id=session_id,
            annotator=meta.get("annotator", ""),
            schema=schema,
            schema_hash=schema_hash,
        )

    def get_schema(self) -> dict | None:
        """Return the embedded schema, or None if no sessions exist yet."""
        return self.load().get("schema")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _save(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def create_session(
        self,
        schema: dict,
        schema_hash: str,
        annotator: str,
        doc_count: int = 0,
    ) -> "AnnotationSession":
        """Create a new session, embed the schema in the manifest, return session."""
        manifest = self.load()

        # Embed schema on first session (or update if schema changed)
        if manifest.get("schema") is None or manifest.get("schema_hash") != schema_hash:
            manifest["schema"] = schema
            manifest["schema_hash"] = schema_hash

        session_id = str(uuid.uuid4())
        now = _now_iso()
        manifest.setdefault("sessions", {})[session_id] = {
            "annotator": annotator,
            "created_at": now,
            "updated_at": now,
            "doc_count": doc_count,
            "completed_count": 0,
        }

        self._save(manifest)
        logger.info("Created annotation session %s for annotator %s", session_id, annotator)

        return AnnotationSession(
            dataset_dir=self.dataset_dir,
            session_id=session_id,
            annotator=annotator,
            schema=schema,
            schema_hash=schema_hash,
        )

    def update_session_progress(self, session_id: str) -> None:
        """Recount completed docs in the session dir and update manifest."""
        manifest = self.load()
        if session_id not in manifest.get("sessions", {}):
            return

        session_dir = self.dataset_dir / ".annotations" / session_id
        if not session_dir.exists():
            return

        completed = 0
        doc_count = 0
        for json_file in session_dir.glob("*.json"):
            doc_count += 1
            try:
                raw = json.loads(json_file.read_text(encoding="utf-8"))
                # A doc is complete if all data fields have non-null values
                data = raw.get("data", {})
                if data and all(v is not None for v in data.values()):
                    completed += 1
            except (json.JSONDecodeError, OSError):
                pass

        manifest["sessions"][session_id]["doc_count"] = doc_count
        manifest["sessions"][session_id]["completed_count"] = completed
        manifest["sessions"][session_id]["updated_at"] = _now_iso()
        self._save(manifest)


# ---------------------------------------------------------------------------
# Backwards-compatible shim — keeps existing call sites working while the
# new session-based API is wired in.  Will be removed once app.py is updated.
# ---------------------------------------------------------------------------

class AnnotationSerializer:
    """Thin shim that delegates to AnnotationSession.

    Existing callers pass a ``pdf_path`` and optionally a ``session``.
    If no session is provided, falls back to the old flat layout for
    backwards compatibility during migration.
    """

    @staticmethod
    def annotation_path_for(pdf_path: Path, session: "AnnotationSession | None" = None) -> Path:
        if session is not None:
            return session.annotation_path_for(pdf_path)
        # Legacy flat layout
        return pdf_path.parent / ".annotations" / pdf_path.with_suffix(".json").name

    @staticmethod
    def exists(pdf_path: Path, session: "AnnotationSession | None" = None) -> bool:
        if session is not None:
            return session.exists(pdf_path)
        return AnnotationSerializer.annotation_path_for(pdf_path).exists()

    @staticmethod
    def save(annotation: AnnotationState, pdf_path: Path, session: "AnnotationSession | None" = None) -> None:
        if session is not None:
            session.save(annotation, pdf_path)
            return
        # Legacy flat save
        data: dict[str, Any] = {}
        fields_metadata: dict[str, dict[str, Any]] = {}
        for field_path, field_annotation in annotation.fields.items():
            data[field_path] = field_annotation.value
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
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps(output, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load(pdf_path: Path, session: "AnnotationSession | None" = None) -> AnnotationState | None:
        if session is not None:
            return session.load(pdf_path)
        # Legacy flat load
        annotation_path = AnnotationSerializer.annotation_path_for(pdf_path)
        if not annotation_path.exists():
            return None
        try:
            raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read annotation file %s: %s", annotation_path, exc)
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
            logger.warning("Corrupted annotation file %s: %s", annotation_path, exc)
            return None
