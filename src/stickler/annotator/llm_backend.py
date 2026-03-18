"""AWS Bedrock integration for LLM-assisted annotation pre-filling.

Sends PDF content (base64-encoded) and a JSON Schema to a Claude model via
the Bedrock ``invoke_model`` API. Parses the JSON response into field values
matching the schema structure.

Error handling:
- ``ClientError``, ``BotoCoreError``, ``EndpointConnectionError`` from boto3
  are caught and re-raised as ``RuntimeError`` with a descriptive message.
- Invalid JSON in the LLM response raises ``ValueError``.
- Partial results (response keys that are a subset of schema keys) are
  returned as-is — the caller decides how to handle missing fields.

All errors are designed to be catchable by the caller (``app.py``) for
graceful fallback to manual annotation.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Approximate tokens per byte for cost estimation.
# Claude tokenises at roughly 1 token per 4 bytes for English text.
# PDF base64 inflates size by ~1.37x, so we use a conservative factor.
_TOKENS_PER_BYTE = 0.25

# Rough cost per 1K input tokens (USD) for Claude Sonnet on Bedrock.
_INPUT_COST_PER_1K_TOKENS = 0.003

# Rough cost per 1K output tokens (USD).
_OUTPUT_COST_PER_1K_TOKENS = 0.015

# Assume output is ~5% of input size for cost estimation.
_OUTPUT_RATIO = 0.05

# Cost threshold (USD) above which a warning should be displayed.
COST_WARNING_THRESHOLD = 100.0


class BedrockLLMBackend:
    """AWS Bedrock client for pre-filling annotations via Claude models.

    Attributes:
        model_id: Bedrock model identifier (default: Claude Sonnet).
        region: AWS region for the Bedrock endpoint.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-sonnet-4-20250514",
        region: str = "us-east-1",
    ) -> None:
        self.model_id = model_id
        self.region = region
        self._client = None  # lazy-initialised

    def _get_client(self):
        """Lazily create the Bedrock runtime client.

        Raises ``RuntimeError`` if boto3 is unavailable or the service
        cannot be reached.
        """
        if self._client is not None:
            return self._client

        try:
            import boto3  # noqa: F811
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    retries={"max_attempts": 2, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=120,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Bedrock client in region {self.region}: {exc}"
            ) from exc

        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefill(self, pdf_path: Path, schema: dict) -> dict:
        """Send a PDF and schema to Bedrock and return predicted field values.

        The PDF is read as raw bytes, base64-encoded, and sent as a
        ``document`` content block alongside a text prompt that includes
        the JSON Schema. The model is instructed to return a JSON object
        whose keys match the schema's ``properties``.

        Args:
            pdf_path: Path to the PDF file.
            schema: JSON Schema dict describing the fields to extract.

        Returns:
            A dict of field values whose keys are a subset of the schema's
            property names.

        Raises:
            RuntimeError: On Bedrock API errors or unreachable service.
            ValueError: If the LLM response is not valid JSON.
            FileNotFoundError: If *pdf_path* does not exist.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_bytes = pdf_path.read_bytes()
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

        prompt = self._build_prompt(schema)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        response_text = self._invoke(body)
        return self._parse_response(response_text, schema)

    def estimate_cost(self, pdf_path: Path) -> float:
        """Estimate the cost in USD for processing a PDF.

        Uses a simple heuristic based on file size:
        - Convert file size to approximate token count.
        - Apply per-token pricing for input and estimated output.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Estimated cost in USD.
        """
        if not pdf_path.exists():
            return 0.0

        file_size = pdf_path.stat().st_size
        input_tokens = file_size * _TOKENS_PER_BYTE
        output_tokens = input_tokens * _OUTPUT_RATIO

        input_cost = (input_tokens / 1000) * _INPUT_COST_PER_1K_TOKENS
        output_cost = (output_tokens / 1000) * _OUTPUT_COST_PER_1K_TOKENS

        return input_cost + output_cost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(schema: dict) -> str:
        """Build the extraction prompt from a JSON Schema.

        Instructs the model to return a JSON object whose keys match the
        schema's ``properties`` and whose values are extracted from the
        PDF content.
        """
        schema_json = json.dumps(schema, indent=2)
        return (
            "You are a document data extraction assistant. "
            "Extract field values from the attached PDF document according "
            "to the following JSON Schema.\n\n"
            f"```json\n{schema_json}\n```\n\n"
            "Return ONLY a valid JSON object whose keys match the schema's "
            "\"properties\" and whose values are extracted from the document. "
            "If a field cannot be found in the document, set its value to null. "
            "Do not include any explanation or markdown formatting — output "
            "raw JSON only."
        )

    def _invoke(self, body: dict) -> str:
        """Call the Bedrock ``invoke_model`` API and return the response text.

        Raises ``RuntimeError`` on API errors or connectivity issues.
        """
        try:
            from botocore.exceptions import (
                BotoCoreError,
                ClientError,
                EndpointConnectionError,
            )
        except ImportError as exc:
            raise RuntimeError(
                "boto3/botocore is required for LLM pre-fill. "
                "Install the 'annotator' extra: pip install stickler[annotator]"
            ) from exc

        client = self._get_client()

        try:
            response = client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except EndpointConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Bedrock endpoint in region {self.region}. "
                f"Check your network connection and AWS configuration: {exc}"
            ) from exc
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            error_msg = exc.response.get("Error", {}).get("Message", str(exc))
            raise RuntimeError(
                f"Bedrock API error ({error_code}): {error_msg}"
            ) from exc
        except BotoCoreError as exc:
            raise RuntimeError(
                f"AWS SDK error while calling Bedrock: {exc}"
            ) from exc

        # Parse the Bedrock response envelope
        try:
            response_body = json.loads(response["body"].read())
        except (json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(
                f"Failed to parse Bedrock response envelope: {exc}"
            ) from exc

        # Extract text from the first content block
        content_blocks = response_body.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text":
                return block["text"]

        raise RuntimeError(
            "Bedrock response contained no text content block."
        )

    @staticmethod
    def _parse_response(response_text: str, schema: dict) -> dict:
        """Parse the LLM's text response into a dict of field values.

        Strips markdown code fences if present, then parses as JSON.
        Returns whatever keys match the schema — partial results are OK.

        Raises ``ValueError`` if the response is not valid JSON.
        """
        text = response_text.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response is not valid JSON: {exc}\n"
                f"Response text (first 500 chars): {response_text[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLM response is not a JSON object (got {type(parsed).__name__}). "
                f"Response text (first 500 chars): {response_text[:500]}"
            )

        return parsed
