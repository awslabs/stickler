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

# Models available for localization (bbox detection).
# All options use Haiku as orchestrator + the selected model via raw Converse.
LOCALIZATION_MODELS: dict[str, str] = {
    "Haiku → Nova Pro": "us.amazon.nova-pro-v1:0",
    "Haiku → Nova 2 Lite": "us.amazon.nova-2-lite-v1:0",
    "Haiku → Sonnet 4.6": "us.anthropic.claude-sonnet-4-6",
    "Haiku → Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

DEFAULT_LOCALIZATION_MODEL_LABEL = "Haiku → Nova Pro"


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

    def localize(
        self,
        pdf_path: Path,
        field_values: dict[str, object],
        model_id: str | None = None,
    ) -> dict[str, dict]:
        """Stage 2: locate extracted field values in the PDF pages.

        Uses a Claude agent that calls a `localize_single_field` tool for each
        field. The tool makes a raw Converse API call to the localization model
        (e.g. Nova Pro) — no tool use required on the localization model side.
        This allows using models that don't support tool use for bbox detection.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        page_bytes = _pdf_to_image_bytes(pdf_path)

        from strands import Agent
        from strands.models import BedrockModel
        from strands.types.content import ContentBlock, ImageContent

        mid = model_id or self.model_id
        loc_tool = _make_converse_localize_tool(page_bytes, mid, self.region)

        # Use a cheap/fast model for orchestration (it just calls the tool)
        orchestrator_model = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=self.region,
        )
        agent = Agent(model=orchestrator_model, tools=[loc_tool])

        fields_desc = "\n".join(
            f'- {name}: "{value}"'
            for name, value in field_values.items()
            if value is not None
        )

        prompt = (
            f"Localize these document fields by calling `localize_single_field` for each one:\n"
            f"{fields_desc}\n\n"
            "Call the tool once per field. After all calls complete, summarize the results."
        )

        try:
            agent(prompt)
        except Exception as exc:
            raise RuntimeError(f"Localization agent error: {exc}") from exc

        # Collect results from all tool calls
        validated: dict[str, dict] = {}
        for msg in agent.messages:
            if msg.get("role") != "user":
                continue
            for block in msg.get("content", []):
                tr = block.get("toolResult")
                if tr and tr.get("status") == "success":
                    for cb in tr.get("content", []):
                        text = cb.get("text", "")
                        if text:
                            try:
                                data = json.loads(text)
                                fname = data.get("field_name")
                                if fname and "page" in data and "bbox" in data:
                                    coords = [float(c) for c in data["bbox"]]
                                    logger.info(
                                        "Localized %s → page=%d bbox=[%.3f, %.3f, %.3f, %.3f]",
                                        fname, data["page"], *coords,
                                    )
                                    validated[fname] = {
                                        "page": data["page"],
                                        "bbox": coords,
                                    }
                            except (json.JSONDecodeError, TypeError, ValueError):
                                continue
        return validated


def _make_converse_localize_tool(page_bytes: list[bytes], model_id: str, region: str):
    """Return a strands @tool that calls Converse API directly for single-field localization."""
    import boto3
    from botocore.config import Config as BotoConfig
    from strands import tool

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(read_timeout=120, retries={"max_attempts": 3, "mode": "adaptive"}),
    )

    # Pre-build image content blocks for the Converse API
    image_blocks = []
    for idx, pb in enumerate(page_bytes, 1):
        image_blocks.append({"image": {"format": "png", "source": {"bytes": pb}}})
        image_blocks.append({"text": f"[Page {idx}]"})

    @tool
    def localize_single_field(field_name: str, field_value: str) -> str:
        """Locate a single field value in the document pages and return its bounding box.

        Args:
            field_name: The name of the field to locate.
            field_value: The text value to find in the document.
        """
        content = list(image_blocks)
        content.append({"text": (
            f"Locate the text \"{field_value}\" (field: {field_name}) in the document.\n\n"
            "Return ONLY a JSON object with this exact format:\n"
            '{"field_name": "' + field_name + '", "page": <page_number>, '
            '"bbox": [x1, y1, x2, y2]}\n\n'
            "Coordinates must be scaled between 0 and 1000 where "
            "(0, 0) is top-left and (1000, 1000) is bottom-right.\n"
            "Fit the bounding box tightly around the exact text. Return ONLY the JSON, nothing else."
        )})

        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
        )

        # Extract text from response
        resp_text = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                resp_text += block["text"]

        # Parse JSON from response (may be wrapped in markdown)
        resp_text = resp_text.strip()
        if resp_text.startswith("```"):
            resp_text = resp_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(resp_text)
            # Validate
            if "bbox" in data and "page" in data:
                data["field_name"] = field_name
                return json.dumps(data)
        except json.JSONDecodeError:
            pass

        return json.dumps({"field_name": field_name, "error": "Could not parse location"})

    return localize_single_field


