"""Record a live Bedrock exchange into strands_offline_agent.json.

Run once, with Bedrock credentials, to replace the hand-authored fixture with a genuine
capture from your own agent:

    AWS_PROFILE=<profile> python record_agent_exchange.py

The recorder wraps a real model provider and copies the stream events through, so the
fragment boundaries and token usage in the fixture are the ones Bedrock actually emitted.
Nothing here runs during the notebook — the notebook only reads the JSON.

The script fails loudly. A missing install, absent credentials, no model access and a
model that never called the tool each report distinctly, so a real problem is never
written into the fixture as if it were an agent response.
"""

import asyncio
import datetime
import json
import pathlib
import sys
from typing import Any, Optional

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


class RecordingModel:
    """Delegates to a real provider and records the tool-use turns it streams back."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.turns: list[dict[str, Any]] = []

    def get_config(self) -> Any:
        return self._inner.get_config()

    def update_config(self, **kwargs: Any) -> None:
        self._inner.update_config(**kwargs)

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        async for event in self._inner.structured_output(*args, **kwargs):
            yield event

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        current: Optional[dict[str, Any]] = None
        async for event in self._inner.stream(*args, **kwargs):
            start = event.get("contentBlockStart", {}).get("start", {})
            if "toolUse" in start:
                current = {
                    "name": start["toolUse"]["name"],
                    "toolUseId": start["toolUse"]["toolUseId"],
                    "inputChunks": [],
                    "usage": {},
                }
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if current is not None and "toolUse" in delta:
                current["inputChunks"].append(delta["toolUse"]["input"])
            if "contentBlockStop" in event and current is not None:
                current["input"] = json.loads("".join(current["inputChunks"]))
                self.turns.append({"toolUse": current})
                current = None
            if "metadata" in event and self.turns:
                self.turns[-1]["toolUse"]["usage"] = event["metadata"].get("usage", {})
            yield event


async def main() -> int:
    try:
        from strands import Agent
        from strands.models import BedrockModel
    except ImportError as exc:
        print(f"strands-agents is not installed: {exc}", file=sys.stderr)
        return 1

    model = RecordingModel(BedrockModel(model_id=MODEL_ID))
    agent = Agent(model=model, system_prompt="You extract invoice data.", callback_handler=None)

    try:
        result = await agent.invoke_async(
            f"Extract the invoice fields.\n\nDOCUMENT:\n{DOCUMENT}",
            structured_output_model=Invoice,
        )
    except Exception as exc:
        print(f"the agent call failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        print("check credentials, region, and model access; the fixture was NOT changed.", file=sys.stderr)
        return 2

    if not model.turns:
        print("the model never called the structured-output tool; nothing to record.", file=sys.stderr)
        return 3

    fixture = {
        "provenance": {
            "model_id": MODEL_ID,
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
