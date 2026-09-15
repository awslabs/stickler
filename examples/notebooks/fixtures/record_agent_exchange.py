"""Record a live Bedrock exchange into strands_offline_agent.json.

Run once, with Bedrock credentials, to replace the committed fixture with a genuine capture
from your own agent:

    AWS_PROFILE=<profile> python record_agent_exchange.py

The recorder subclasses `Model` and wraps a real provider, copying the stream events through,
so the fragment boundaries and token usage in the fixture are the ones Bedrock actually
emitted. Nothing here runs during the notebook — the notebook only reads the JSON.

Exit codes, all of which leave the fixture untouched:

    1  strands-agents is not installed
    2  the agent call failed — bad or absent credentials, no model access, wrong region,
       a malformed tool payload, or any other runtime error
    3  the call succeeded but the model never invoked the structured-output tool

Absent credentials are not distinguishable from other call failures here: boto3 raises them
from inside the same `invoke_async`, so they share exit 2. The message prints the underlying
exception type, which is what tells them apart in practice.
"""

import asyncio
import datetime
import json
import pathlib
import sys
from typing import Any, AsyncGenerator, Optional

from pydantic import BaseModel

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
FIXTURE = pathlib.Path(__file__).with_name("strands_offline_agent.json")

DOCUMENT = """
INVOICE  #INV-2024-0042
Acme Corporation                    Date: 2024-03-15
PO: PO-88231

  2 x Wireless Mouse        (WM-100)  @  $29.99
  5 x USB-C Cable 1m        (UC-050)  @  $12.99
  1 x Mechanical Keyboard   (KB-200)  @  $89.99

Total: $1,247.50            Terms: Net 30
"""


class LineItem(BaseModel):
    sku: str
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    invoice_date: datetime.date
    total_amount: float
    po_number: Optional[str] = None
    line_items: list[LineItem] = []


def build_recorder(base: type) -> type:
    """Build a RecordingModel subclassing the SDK's `Model`.

    Subclassing matters: `Agent.__init__` reads `model.stateful`, a concrete property on the
    ABC, so a duck-typed wrapper raises `AttributeError` before the call is ever made.
    """

    class RecordingModel(base):  # type: ignore[misc,valid-type]
        """Delegates to a real provider and records the tool-use turns it streams back."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.turns: list[dict[str, Any]] = []

        def get_config(self) -> Any:
            return self._inner.get_config()

        def update_config(self, **kwargs: Any) -> None:
            self._inner.update_config(**kwargs)

        async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            async for event in self._inner.structured_output(*args, **kwargs):
                yield event

        async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            # Accumulate for the whole stream and commit once at the end. Providers are not
            # required to place `metadata` after `contentBlockStop`, and a message may carry
            # text blocks alongside the tool use, so neither event is a safe commit point.
            blocks: list[dict[str, Any]] = []
            current: Optional[dict[str, Any]] = None
            usage: dict[str, Any] = {}

            async for event in self._inner.stream(*args, **kwargs):
                start = event.get("contentBlockStart", {}).get("start", {})
                if "toolUse" in start:
                    current = {
                        "name": start["toolUse"]["name"],
                        "toolUseId": start["toolUse"]["toolUseId"],
                        "inputChunks": [],
                    }
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if current is not None and "toolUse" in delta:
                    current["inputChunks"].append(delta["toolUse"]["input"])
                if "contentBlockStop" in event and current is not None:
                    blocks.append(current)
                    current = None
                if "metadata" in event:
                    usage = event["metadata"].get("usage", {}) or usage
                yield event

            if not blocks:
                return
            if len(blocks) > 1:
                raise RuntimeError(
                    f"the model returned {len(blocks)} tool-use blocks in one message; this "
                    "fixture format records one per turn and the notebook replays it that way. "
                    "Re-record with a prompt that elicits a single structured-output call."
                )

            block = blocks[0]
            raw = "".join(block["inputChunks"])
            try:
                block["input"] = json.loads(raw)
            except ValueError as exc:
                raise RuntimeError(
                    f"the recorded tool input is not valid JSON ({exc}); the stream was "
                    f"truncated or malformed. First 200 chars: {raw[:200]!r}"
                ) from exc
            block["usage"] = usage
            self.turns.append({"toolUse": block})

    return RecordingModel


async def main() -> int:
    try:
        from strands import Agent
        from strands.models import BedrockModel, Model
    except ImportError as exc:
        print(f"strands-agents is not installed: {exc}", file=sys.stderr)
        return 1

    model = build_recorder(Model)(BedrockModel(model_id=MODEL_ID))

    try:
        agent = Agent(model=model, system_prompt="You extract invoice data.", callback_handler=None)
        result = await agent.invoke_async(
            f"Extract the invoice fields.\n\nDOCUMENT:\n{DOCUMENT}",
            structured_output_model=Invoice,
        )
    except Exception as exc:
        print(f"the agent call failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        print("check credentials, region and model access; the fixture was NOT changed.", file=sys.stderr)
        return 2

    if not model.turns:
        print("the model never called the structured-output tool; nothing to record.", file=sys.stderr)
        return 3

    fixture = {
        "provenance": {
            "model_id": MODEL_ID,
            "hand_authored": False,
            "recorded": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "note": f"Recorded from a live Bedrock call to {MODEL_ID}.",
        },
        "turns": model.turns,
    }
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"recorded {len(model.turns)} turn(s) to {FIXTURE.name}")
    print(f"stop_reason: {result.stop_reason}")
    print("review the payload before committing it:")
    print(json.dumps(model.turns[0]["toolUse"]["input"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
