"""Data models for annotation state, provenance, and field status.

Defines the core value objects used throughout the annotator:

- AnnotationMode: The three operating modes (zero_start, llm_inference, hitl)
- DocumentStatus: Completion state of a PDF document
- FieldProvenance: Tracks whether a value came from a human or LLM
- FieldAnnotation: A single field's value plus provenance metadata
- AnnotationState: Full annotation state for one document (all fields + timestamps)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class AnnotationMode(str, Enum):
    """Operating mode for the annotation tool.

    - ZERO_START: Manual annotation from scratch
    - LLM_INFERENCE: LLM pre-fills all fields, user reviews in batch
    - HITL: LLM pre-fills, user reviews field-by-field
    """

    ZERO_START = "zero_start"
    LLM_INFERENCE = "llm_inference"
    HITL = "hitl"


class DocumentStatus(str, Enum):
    """Completion state of a PDF document in the annotation queue.

    Derived from the annotation file state — not stored separately.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class FieldProvenance(BaseModel):
    """Tracks the origin and review status of a field annotation.

    Attributes:
        source: Who produced the value — "human" for manual entry, "llm" for
                Bedrock-generated predictions.
        checked: Whether an LLM-generated value has been reviewed by a human.
                 Defaults to False. Only meaningful when source is "llm".
    """

    source: Literal["human", "llm"]
    checked: bool = False


class FieldLocation(BaseModel):
    """Spatial location of a field value within a PDF page.

    Coordinates use [x1, y1, x2, y2] format scaled 0–1000 where
    (0,0) is top-left and (1000,1000) is bottom-right. Matches the
    format used by the AWS document localization reference implementation.

    Attributes:
        page: 1-indexed page number where the field was found.
        bbox: Bounding box as [x1, y1, x2, y2] scaled 0–1000.
    """

    page: int
    bbox: list[float]


class FieldAnnotation(BaseModel):
    """A single annotated field value with provenance metadata.

    Attributes:
        value: The annotation value. Can be str, int, float, bool, list, dict,
               or None.
        is_none: True when the user explicitly marks the field as having no
                 value (distinct from an unannotated field).
        provenance: Origin and review metadata for this value.
        location: Optional spatial location of the field in the PDF.
    """

    value: Any = None
    is_none: bool = False
    provenance: FieldProvenance
    location: FieldLocation | None = None


class AnnotationState(BaseModel):
    """Full annotation state for a single PDF document.

    Attributes:
        schema_hash: Hash of the JSON Schema used for annotation, enabling
                     detection of schema changes between sessions.
        fields: Mapping of field path (e.g. "vendor_name" or
                "line_items[0].amount") to its annotation.
        created_at: ISO 8601 timestamp of initial creation.
        updated_at: ISO 8601 timestamp of last modification.
    """

    schema_hash: str
    fields: dict[str, FieldAnnotation]
    created_at: str
    updated_at: str
