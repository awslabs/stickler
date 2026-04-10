# Schema Configuration

The schema defines which fields appear in the annotation panel. The tool supports three schema sources — all produce the same result internally.

## JSON Schema File

Point to a `.json` file on disk. This is the recommended approach for teams — check the schema into version control alongside your dataset.

If you have a sample JSON document but no schema yet, paste it into [JSONLint's Schema Generator](https://jsonlint.com/json-schema-generator) to get a starting point, then tweak the output.

```json
{
  "type": "object",
  "properties": {
    "invoice_id": { "type": "string", "description": "Unique invoice number" },
    "vendor_name": { "type": "string" },
    "total_amount": { "type": "string", "description": "Total including tax" },
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": { "type": "string" },
          "quantity": { "type": "string" },
          "unit_price": { "type": "string" }
        }
      }
    }
  }
}
```

**Field types:**

- `string`, `number`, `integer`, `boolean` — render as text inputs
- `array` with `items.type: "object"` — renders as an inline table (one row per item, add/remove with ＋/✕)
- `array` with primitive items — renders as a vertical list of inputs

The `description` field is shown as help text on hover.

The schema is embedded in the manifest on first use. Subsequent sessions load it from the manifest — the original file isn't needed again.

## Schema Builder

Build a schema interactively in the UI. Add fields with a name and type, then click **Finalize Schema**. The builder generates a JSON Schema dict internally.

Good for quick experiments or when you don't have a schema file yet. You can export the generated schema to a file for reuse.

## Pydantic Import

Provide a dotted import path to a `StructuredModel` subclass:

```
stickler.annotator.models_example.FccInvoiceModel
```

The tool imports the class and derives the JSON Schema from its Pydantic model definition. The class must be importable from the current Python environment.

To create your own `StructuredModel`, see the [Getting Started guide](../../Getting-Started/index.md). A model shipped with the annotator (`stickler.annotator.models_example.FccInvoiceModel`) works as a ready-made example for FCC political advertising invoices.
