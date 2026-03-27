# LLM Auto-Annotate

The annotation tool uses AWS Bedrock to pre-fill document fields from PDF page images. Extraction and localization are separate stages with independently selectable models.

## Architecture

```
PDF → pdf2image (150 DPI PNGs) → Strands Agent → extract_fields tool → field values
```

The Strands agent sends all PDF pages as image content blocks to the selected model. A structured `extract_fields` tool enforces the JSON Schema — the agent self-corrects on validation errors.

## Model Selection

Click the **⚙** popover next to the Auto-annotate button to select models:

| Stage | Models | Default |
|---|---|---|
| Extraction | Haiku 4.5, Sonnet 4.6, Opus 4.6, Nova 2 Lite | Haiku 4.5 |
| Localization | Haiku → Nova Pro, Haiku → Nova 2 Lite, Haiku → Sonnet 4.6, Haiku → Haiku 4.5 | Haiku → Nova Pro |

Extraction models are called via Strands agent with tool use. Localization models are called via raw Converse API (no tool use required), which enables Nova Pro support.

## Usage

1. Navigate to a document
2. Click **🤖 Auto-annotate** — the spinner shows which file is being processed
3. Field values populate in the annotation panel
4. Click **📍 Locate** to run bounding box detection (see [Field Localization](field-localization.md))

## Error Handling

If credentials are expired or missing, the tool shows a clear remediation message:

- **Expired credentials** — set `AWS_PROFILE` in `.env` and restart
- **Access denied** — ensure your role has `bedrock:InvokeModel` permission

## Configuration

Models are defined in `src/stickler/annotator/llm_backend.py`:

```python
AVAILABLE_MODELS = {
    "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Claude Sonnet 4.6": "us.anthropic.claude-sonnet-4-6",
    ...
}
```

The `us.` prefix enables cross-region inference. Add new models by appending to the dict.
