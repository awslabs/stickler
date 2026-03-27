"""KIE Annotation Tool — annotate PDF documents against a schema.

A Streamlit-based web application for annotating multi-page PDF documents
with structured key-value data. Supports three workflows:

- Zero Start: manual annotation from scratch
- LLM Inference: AWS Bedrock pre-fills all fields for batch review
- HITL: field-by-field review of LLM predictions

Annotations are output as stickler-compatible JSON files directly loadable
into StructuredModel instances.

Launch via::

    stickler-annotate
"""

from .annotation_panel import AnnotationPanel
from .config import ConfigResult, render_config_sidebar
from .dataset import DatasetManager, PDFDocument
from .llm_backend import BedrockLLMBackend
from .models import (
    AnnotationMode,
    AnnotationState,
    DocumentStatus,
    FieldAnnotation,
    FieldLocation,
    FieldProvenance,
)
from .pdf_viewer import PDFViewer
from .schema_builder import SchemaBuilder
from .schema_loader import SchemaLoader
from .serializer import AnnotationSerializer

__all__ = [
    "AnnotationMode",
    "AnnotationPanel",
    "AnnotationSerializer",
    "AnnotationState",
    "BedrockLLMBackend",
    "ConfigResult",
    "DatasetManager",
    "DocumentStatus",
    "FieldAnnotation",
    "FieldLocation",
    "FieldProvenance",
    "PDFDocument",
    "PDFViewer",
    "SchemaBuilder",
    "SchemaLoader",
    "render_config_sidebar",
]
