# Field Localization

Field localization is a second-stage inference that detects where each extracted field value appears in the PDF. Bounding boxes are drawn as colored overlays on the PDF viewer.

## Architecture

Localization uses a two-model approach:

```
Haiku (orchestrator) → localize_single_field tool → Converse API → Nova Pro (localizer)
```

1. A Haiku agent receives the list of extracted field names and values
2. For each field, it calls the `localize_single_field` tool
3. The tool sends the PDF page images + a focused prompt to the localization model via raw Converse API (no tool use)
4. The model returns `[x1, y1, x2, y2]` coordinates scaled 0–1000
5. Coordinates are stored in the annotation file and rendered on the PDF viewer

This design allows using models that don't support tool use (like Nova Pro) for the spatial reasoning task, while Haiku handles orchestration cheaply.

## Coordinate Format

Bounding boxes use `[x1, y1, x2, y2]` scaled 0–1000, matching the [AWS document localization reference implementation](https://github.com/aws-samples/sample-document-information-localization):

- `(0, 0)` = top-left corner
- `(1000, 1000)` = bottom-right corner
- `x1, y1` = top-left of the box
- `x2, y2` = bottom-right of the box

The renderer converts these to pixel coordinates based on the actual rendered image dimensions.

## Usage

1. Auto-annotate a document (or manually fill fields)
2. Click **📍 Locate** in the annotation panel header
3. Wait for localization to complete (one API call per field)
4. Bounding boxes appear on the PDF viewer with colored overlays and labels
5. Navigate pages — only boxes for the current page are shown
6. Click **🔄 Re-locate** to re-run after editing field values

## Rendering

Boxes are drawn using PIL's `ImageDraw` on a copy of the rendered page image:

- Semi-transparent colored fill (18/255 alpha) so text remains readable
- 2px solid colored border
- Pill-style labels with white text on colored background
- Smart label positioning to avoid overlap (tries above, below, then inside the box)
- 10-color cycling palette for distinct field colors

## Storage

Locations are stored in the annotation JSON alongside provenance metadata:

```json
{
  "metadata": {
    "fields": {
      "invoice_id": {
        "source": "llm",
        "checked": true,
        "location": {
          "page": 1,
          "bbox": [780, 55, 950, 80]
        }
      }
    }
  }
}
```

Locations persist across sessions and page refreshes. They are loaded from disk on startup and passed to the PDF viewer automatically.

## Model Selection

Select the localization model via the ⚙ popover. Options use the `Haiku → Model` naming convention to indicate the orchestrator + localizer pairing:

| Option | Localizer | Notes |
|---|---|---|
| Haiku → Nova Pro | Nova Pro | Best accuracy per AWS benchmarks |
| Haiku → Nova 2 Lite | Nova 2 Lite | Faster, lower cost |
| Haiku → Sonnet 4.6 | Sonnet 4.6 | Good accuracy, higher cost |
| Haiku → Haiku 4.5 | Haiku 4.5 | Fastest, lowest cost |
