# User Guide

This is the end-to-end walkthrough for annotating PDF documents with the KIE Annotation Tool. By the end you'll have a folder of structured JSON annotations ready for Stickler evaluation.

## Prerequisites

- Python 3.12+
- `poppler-utils` for PDF rendering: `brew install poppler` (macOS) or `apt-get install poppler-utils` (Linux)
- AWS credentials only if you want LLM auto-annotate (optional)

Install the annotator dependencies:

```bash
pip install -e ".[annotator,dev]"
```

## Step 1: Prepare Your Documents

Put your PDF files into a single folder. The tool discovers PDFs recursively, so subdirectories work too. Hidden directories (`.annotations/`, `.git/`, etc.) are automatically excluded.

```
my-dataset/
  invoice_001.pdf
  invoice_002.pdf
  invoice_003.pdf
```

That's it — no special structure required.

## Step 2: Launch the Tool

```bash
make annotate
```

This runs `streamlit run src/stickler/annotator/app.py` and opens the app in your browser. The landing page asks for a dataset directory.

## Step 3: Enter Your Dataset Path

Type the path to your PDF folder in the landing page input. The tool validates the directory exists and contains at least one PDF.

```
./my-dataset
```

If the folder has existing annotation sessions, they'll appear with resume buttons. Otherwise you'll see the configuration panel.

## Step 4: Configure Your Schema

The schema defines which fields you're annotating. You have three options:

### Option A: JSON Schema File (recommended)

Point to a `.json` file that describes your fields. If you have a sample JSON document but no schema yet, paste it into [JSONLint's Schema Generator](https://jsonlint.com/json-schema-generator) to get a starting point.

```json
{
  "type": "object",
  "properties": {
    "invoice_id": { "type": "string", "description": "Invoice number" },
    "vendor_name": { "type": "string" },
    "total_amount": { "type": "string" },
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": { "type": "string" },
          "amount": { "type": "string" }
        }
      }
    }
  }
}
```

Scalar fields render as text inputs. Array fields render as inline tables with add/remove row buttons.

### Option B: Schema Builder

Build a schema in the UI without writing JSON. Add fields, set types, and click Finalize. Good for quick experiments.

### Option C: Pydantic Import

Provide a dotted import path to a `StructuredModel` subclass: `mypackage.models.InvoiceModel`. The schema is derived from the model class.

See [Schema Configuration](schema-configuration.md) for details on each option.

## Step 5: Annotate Documents

Once configured, you'll see a side-by-side layout: PDF viewer on the left, annotation panel on the right.

**Navigate documents** using the Prev/Next buttons or the document picker (📋 Select).

**For each document:**

- Type field values into the text inputs
- Check **N/A** for fields that don't apply to this document
- Array fields (like line items) have ＋ Add row / ✕ Remove buttons

**Every change auto-saves** — you'll see a 💾 toast confirmation. The progress bar at the top tracks how many fields are filled.

### Speed Up with Auto-Annotate (Optional)

Click **🤖 Auto-annotate** to pre-fill all fields using an AWS Bedrock model. Review and correct the values — provenance metadata tracks which fields came from the LLM vs. human entry.

Select your preferred model via the ⚙ popover (Haiku 4.5, Sonnet 4.6, etc.). See [LLM Auto-Annotate](llm-auto-annotate.md) for setup.

### Locate Fields in the PDF (Optional)

After annotating, click **📍 Locate** to detect where each field value appears on the page. Colored bounding boxes overlay the PDF viewer.

## Step 6: Share and Resume Sessions

### Share with a Teammate

Your annotations live in `<dataset_dir>/.annotations/`. To share:

1. Zip the entire dataset folder (PDFs + `.annotations/`)
2. Send it to your teammate
3. They unzip, run `make annotate`, enter the folder path, and see your sessions with a Resume button

The manifest embeds the full schema, so no separate schema file is needed.

### Resume Your Own Session

The landing page shows all existing sessions with annotator name, progress, and a Resume button. Click it to pick up where you left off.

You can also bookmark the deep link shown in the header bar — it encodes the dataset, session, and current document so a page refresh restores your exact position.

### Deep Links

```
# Resume a specific session and document
http://localhost:8501/?dataset=./my-dataset&session=42fac85a-...&doc=invoice_002.pdf
```

## Step 7: Use Annotations for Evaluation

Load your annotations programmatically and compare against model predictions:

```python
from pathlib import Path
from stickler.annotator.serializer import AnnotationManifest
from stickler.annotator.schema_loader import SchemaLoader

# Load session
manifest = AnnotationManifest(Path("./my-dataset"))
session = manifest.get_session("42fac85a-...")

# Build model class from embedded schema
_, ModelClass = SchemaLoader.from_builder_schema(session.schema)

# Load ground truth for one document
state = session.load(Path("./my-dataset/invoice_001.pdf"))
ground_truth = ModelClass(**{k: v.value for k, v in state.fields.items()})

# Compare against your KIE model's output
prediction = ModelClass(**my_model.extract("./my-dataset/invoice_001.pdf"))
result = ground_truth.compare_with(prediction)
print(f"Score: {result['overall_score']:.3f}")
```

### Provenance Filtering

Only use human-verified annotations for high-confidence evaluation:

```python
verified_fields = {
    k: v.value for k, v in state.fields.items()
    if v.provenance.source == "human" or v.provenance.checked
}
```
