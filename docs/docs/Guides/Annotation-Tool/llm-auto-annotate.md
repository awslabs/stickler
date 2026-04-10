# LLM Auto-Annotate & Localization

The annotation tool uses AWS Bedrock to pre-fill document fields from PDF page images and optionally detect where each value appears on the page. Both features are optional — you can annotate entirely by hand.

## Setup

You need AWS credentials with `bedrock:InvokeModel` permission. Set them via `.env`:

```bash
AWS_PROFILE=your-profile
```

Or export `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` directly.

## Extraction

Click **🤖 Auto-annotate** to send all PDF pages as images to a Bedrock model. The model extracts field values according to your schema.

```
PDF → pdf2image (150 DPI PNGs) → Strands Agent → extract_fields tool → field values
```

The Strands agent enforces the JSON Schema via a structured tool — it self-corrects on validation errors.

### Extraction Models

Select via the **⚙** popover next to the Auto-annotate button:

| Model | Notes |
|---|---|
| Claude Haiku 4.5 | Fast, low cost (default) |
| Claude Sonnet 4.6 | Better accuracy |
| Claude Opus 4.6 | Highest accuracy, highest cost |
| Nova 2 Lite | Fast, lower cost |

Models use the `us.` prefix for cross-region inference. Add new models in `llm_backend.py`.

## Field Localization

After extraction (or manual annotation), click **📍 Locate** to detect where each field value appears in the PDF. Colored bounding boxes overlay the PDF viewer.

### How It Works

Localization uses a two-model approach so it can leverage models that don't support tool use (like Nova Pro) for spatial reasoning:

```
Haiku (orchestrator) → localize_single_field tool → Converse API → Nova Pro (localizer)
```

1. Haiku receives the list of field names and values
2. For each field, it calls the `localize_single_field` tool
3. The tool sends PDF page images + a focused prompt to the localizer via raw Converse API
4. The localizer returns `[x1, y1, x2, y2]` coordinates scaled 0–1000
5. Coordinates are stored in the annotation file and rendered on the PDF

### Localization Models

| Model | Notes |
|---|---|
| Haiku → Nova Pro | Only reliable option — best spatial reasoning accuracy |
| Haiku → Nova 2 Lite | Not recommended — poor localization accuracy |
| Haiku → Sonnet 4.6 | Not recommended — inconsistent bounding boxes |
| Haiku → Haiku 4.5 | Not recommended — unreliable coordinates |

In practice, Nova Pro is the only model that produces usable bounding boxes. The other options are available for experimentation but don't expect production-quality results.

### Coordinate Format

Bounding boxes use `[x1, y1, x2, y2]` scaled 0–1000, matching the [AWS document localization reference](https://github.com/aws-samples/sample-document-information-localization). The renderer converts to pixel coordinates based on the rendered image dimensions.

### Rendering

Boxes are drawn with PIL's `ImageDraw`:

- Semi-transparent fill so text stays readable
- 2px colored border with pill-style labels
- Smart label positioning (above, below, or inside the box)
- 10-color cycling palette

Click **🔄 Re-locate** to re-run after editing field values. Locations persist across sessions.

## Error Handling

If credentials are expired or missing, the tool shows remediation steps:

- **Expired credentials** — set `AWS_PROFILE` in `.env` and restart
- **Access denied** — ensure your role has `bedrock:InvokeModel` permission
