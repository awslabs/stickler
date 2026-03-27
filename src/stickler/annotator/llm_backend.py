"""AWS Bedrock + Strands agent for LLM-assisted annotation pre-filling.

PDF pages are rasterised to PNG and sent as image ContentBlocks (image
modality only). A structured extract_fields tool enforces the JSON Schema;
the Strands agent self-corrects on JSON/validation errors automatically.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_REGION = "us-east-1"
_MAX_PAGES = 100

# Models available for auto-annotate. Keys are display labels, values are
# cross-region model IDs (us. prefix for cross-region inference).
AVAILABLE_MODELS: dict[str, str] = {
    "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Claude Sonnet 4.6": "us.anthropic.claude-sonnet-4-6",
    "Claude Opus 4.6": "us.anthropic.claude-opus-4-6-v1",
    "Nova 2 Lite": "us.amazon.nova-2-lite-v1:0",
}

DEFAULT_MODEL_LABEL = "Claude Haiku 4.5"


def _pdf_to_image_bytes(pdf_path: Path, max_pages: int = _MAX_PAGES) -> list[bytes]:
    """Rasterise PDF pages to PNG bytes at 150 dpi."""
    from pdf2image import convert_from_path
    pages = convert_from_path(str(pdf_path), dpi=150, first_page=1, last_page=max_pages)
    result = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        result.append(buf.getvalue())
    return result


def _make_extract_tool(schema: dict):
    """Return a strands @tool that validates extracted JSON against schema."""
    from strands import tool

    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    @tool
    def extract_fields(fields_json: str) -> str:
        """Extract document fields according to the schema.

        Call this with a JSON object whose keys are the schema property names.
        Set any field not found in the document to null.

        Args:
            fields_json: JSON string mapping every schema property to its value.
        """
        try:
            data = json.loads(fields_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON - fix and retry. Error: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Must be a JSON object, not a list or scalar.")

        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required fields: {sorted(missing)}. "
                "Include every schema property (null for absent ones)."
            )

        result: dict = {}
        for field, prop_schema in properties.items():
            val = data.get(field)
            if val is not None:
                ftype = prop_schema.get("type")
                if ftype == "array" and not isinstance(val, list):
                    val = None
                elif ftype not in ("array", "object") and isinstance(val, (list, dict)):
                    val = None
            result[field] = val

        return json.dumps(result, ensure_ascii=False)

    return extract_fields


def _get_tool_result(agent) -> dict | None:
    """Return the last successful toolResult dict from agent messages."""
    for msg in reversed(agent.messages):
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            tr = block.get("toolResult")
            if tr and tr.get("status") == "success":
                for cb in tr.get("content", []):
                    text = cb.get("text", "")
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            continue
    return None


class BedrockLLMBackend:
    """Strands-based Bedrock client for pre-filling annotations via Claude.

    Sends PDF pages as PNG images (image modality only). The extract_fields
    tool enforces the JSON Schema and the agent self-corrects on errors.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        region: str = _DEFAULT_REGION,
    ) -> None:
        self.model_id = model_id
        self.region = region

    def prefill(self, pdf_path: Path, schema: dict) -> dict:
        """Rasterise PDF, send images to Claude, return extracted field values.

        Args:
            pdf_path: Path to the PDF file.
            schema: JSON Schema dict describing the fields to extract.

        Returns:
            Dict mapping every schema property to its extracted value (None if absent).

        Raises:
            FileNotFoundError: If pdf_path does not exist.
            RuntimeError: On Bedrock/agent errors.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        logger.info("Rasterising %s...", pdf_path.name)
        page_bytes = _pdf_to_image_bytes(pdf_path)
        logger.info("%d page(s) converted to PNG", len(page_bytes))

        from strands import Agent
        from strands.models import BedrockModel
        from strands.types.content import ContentBlock, ImageContent

        model = BedrockModel(model_id=self.model_id, region_name=self.region)
        agent = Agent(model=model, tools=[_make_extract_tool(schema)])

        field_names = list(schema.get("properties", {}).keys())
        schema_json = json.dumps(schema, indent=2)

        content: list[ContentBlock] = [
            ContentBlock(image=ImageContent(format="png", source={"bytes": page}))
            for page in page_bytes
        ]
        content.append(ContentBlock(text=(
            "You are a document data extraction assistant. "
            "The images above show pages of a document. "
            f"Extract these fields: {', '.join(field_names)}.\n\n"
            f"Schema:\n```json\n{schema_json}\n```\n\n"
            "Call `extract_fields` with a JSON object containing every schema "
            "property. Set fields not found in the document to null. "
            "Extract only values clearly visible in the document - do not guess."
        )))

        try:
            agent(content)
        except Exception as exc:
            raise RuntimeError(f"Bedrock agent error: {exc}") from exc

        result = _get_tool_result(agent)
        if result is None:
            logger.warning("No tool result found; returning empty extraction")
            result = {f: None for f in field_names}

        return result

    def estimate_cost(self, pdf_path: Path) -> float:
        """Rough cost estimate in USD (image tokens x Haiku pricing)."""
        if not pdf_path.exists():
            return 0.0
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_path))
            pages = min(info.get("Pages", 1), _MAX_PAGES)
        except Exception:
            pages = _MAX_PAGES
        return (pages * 1600 / 1000) * 0.00025
