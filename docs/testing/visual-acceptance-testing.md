# KIE Annotation Tool — Visual Acceptance Testing Log

Automated visual testing driven by Playwright MCP against the Streamlit UI at `localhost:8502`.

Each session documents: test flow, steps taken, observed behavior, bugs found, and fixes applied.

---

## Session 1 — 2026-03-18

**Tool:** Playwright MCP (Chromium)
**App:** `streamlit run src/stickler/annotator/app.py` on port 8502
**Tester:** Kiro (automated)

---

### Flow 1: Initial Page Load

**Steps:**
1. Navigate to `http://localhost:8502`
2. Wait 3s for Streamlit to render

**Result:** ✅ PASS
- Sidebar shows Configuration panel (Dataset directory, Schema source, Operating mode)
- Main area shows "Configure the tool in the sidebar to get started."

---

### Flow 2: Schema Builder — Select and Open

**Steps:**
1. Click "Schema Builder" radio in sidebar

**Result:** ✅ PASS
- Sidebar shows info: "Use the Schema Builder in the main panel to define fields."
- Schema file path input correctly disappears

---

### Flow 3: Schema Builder — Apply Before Finalizing

**Steps:**
1. Type dataset directory path
2. Click "Apply Configuration" before building schema

**Result:** ✅ PASS
- Warning: "Build and finalize a schema in the main panel first."
- Schema Builder UI appears in main panel

---

### Flow 4: Schema Builder — Add Fields

**Steps:**
1. Type "vendor_name", click "Add top-level field"
2. Type "invoice_number", click "Add top-level field"

**Result:** ✅ PASS
- Both fields appear in "Current fields" list with type labels and Remove buttons

---

### Flow 5: Schema Builder — Finalize Schema

**Steps:**
1. Click "Finalize Schema"

**Bug found (Issue 1):** Builder UI remained fully visible after clicking Finalize. User had to scroll past it to see the success prompt.

**Fix applied (`app.py`):** `_handle_schema_builder()` now calls `st.rerun()` after storing schema, setting `schema_just_finalized` flag. On next render only the success message shows.

**Result after fix:** ✅ Builder disappears, only "Schema finalized! Click Apply Configuration in the sidebar to start annotating." shown.

---

### Flow 6: Apply Configuration After Schema Finalization

**Steps:**
1. Click "Apply Configuration" in sidebar

**Result:** ✅ PASS
- Sidebar shows "Configuration applied."
- Main panel transitions to annotation workflow:
  - "✏️ Edit Schema" button
  - Document queue dropdown with status icons
  - PDF viewer (left) with page navigation
  - Manual Annotation panel (right) with field inputs and progress counter

---

### Flow 7: Manual Annotation — Enter Field Values + Save Indicator

**Steps:**
1. Type value into `vendor_name` field
2. Press Tab to commit

**Bug found (Issue 2):** No visual feedback that save occurred.

**Fix applied (`annotation_panel.py`):** `_auto_save()` now calls `st.toast("✓ Saved", icon="💾")` after every write.

**Result after fix:** ✅ `💾 ✓ Saved` toast appears in bottom-right after each field commit.

---

### Flow 8: Mark Field as None

**Steps:**
1. Click "None" checkbox for a field

**Result:** ✅ PASS
- Field input becomes disabled
- Save toast fires
- Progress counter updates

---

### Flow 9: PDF Page Navigation

**Steps:**
1. Click "Next ➡" button

**Result:** ✅ PASS
- Page counter updates (e.g. Page 2/17)
- "⬅ Prev" button becomes enabled

---

### Flow 10: Document Queue — Status Indicators

**Steps:**
1. Open document queue dropdown

**Result:** ✅ PASS
- 10 PDFs discovered
- Annotated docs show 🟢, unannotated show 🔴
- Status derived correctly from `.annotations/` folder

---

### Flow 11: Document Switch — Spurious Save Bug

**Steps:**
1. Switch from one document to another

**Bug found (Issue 3):** Save toast fired on document switch even though no field was changed. Widget keys were scoped by field name only, so Streamlit session state carried over previous document's values, triggering a false `needs_update`.

**Fix applied (`annotation_panel.py`):** All widget keys now include a short MD5 hash of the PDF path (`self._doc_key`), scoping them per document. Switching documents gets fresh widget state.

**Result after fix:** ✅ No spurious save on document switch.

---

### Flow 12: Error Handling — Bad Dataset Directory

**Steps:**
1. Enter `/nonexistent/path/to/nowhere` as dataset directory
2. Click "Apply Configuration"

**Result:** ✅ PASS
- Error: `"Dataset directory does not exist: /nonexistent/path/to/nowhere"`
- Config not applied

---

### Flow 13: Error Handling — Bad Schema File Path

**Steps:**
1. Enter valid dataset dir, enter `/also/fake/schema.json` as schema path
2. Click "Apply Configuration"

**Result:** ✅ PASS
- Error: `"Schema file not found: /also/fake/schema.json"`
- Config not applied

---

### Flow 14: Annotation Storage — .annotations/ Subfolder

**Context:** Previously annotations were saved co-located with PDFs (e.g. `samples/Nuveen.json`), polluting the source directory.

**Change implemented:**
- `serializer.py`: `annotation_path_for()` now returns `<pdf_dir>/.annotations/<stem>.json`
- `serializer.py`: `save()` creates `.annotations/` dir automatically
- `serializer.py`: Added `exists()` helper
- `dataset.py`: `get_status()` uses `AnnotationSerializer.annotation_path_for()` instead of hardcoded path
- Existing JSON files migrated to `.annotations/` folder

**Result:** ✅ Annotations stored cleanly in `.annotations/` subfolder, PDF directory stays clean.

---

### Flow 15: Resume / Start Fresh Prompt

**Steps:**
1. Open a document that has an existing annotation in `.annotations/`
2. Observe prompt
3. Click "▶ Resume" — verify previous annotation loads
4. Switch to same doc, click "🗑 Start Fresh" — verify blank state

**Result:** ✅ PASS
- Prompt shows: "An existing annotation was found for **Nuveen.pdf** (last saved: 2026-03-18 14:48:35 UTC)."
- Resume: loads previous annotation, progress counter reflects saved state
- Start Fresh: blank fields, "0 of N fields annotated"
- Choice persisted in session state per document; resets on document switch

---

### Flow 16: JSON Schema File — Pre-defined Schema Loading

**Steps:**
1. Select "JSON Schema file" (default) in sidebar
2. Enter `./files` as dataset directory
3. Enter `./files/fcc_invoice_schema.json` as schema path
4. Click "Apply Configuration"

**Schema used:** FCC Political Advertising Invoice schema with 15 scalar fields + 1 `line_items` array field with nested object items (air_date, program, spot_length, gross_rate, net_rate).

**Result:** ✅ PASS
- Config applied immediately — no schema builder needed
- All 15 scalar fields rendered as text inputs with None checkboxes
- `line_items` array field rendered with "＋ Add item" button
- "0 of 16 fields annotated" progress counter

---

### Flow 17: Array Field — Add and Fill Nested Object Items

**Steps:**
1. Click "＋ Add item to line_items"
2. Fill in all 5 sub-fields: air_date, program, spot_length, gross_rate, net_rate
3. Press Tab to commit

**Result:** ✅ PASS
- Item 1 expander appears with all 5 sub-fields
- Values persist correctly
- Save toast fires
- JSON saved to `.annotations/` with correct nested structure:
  ```json
  {"data": {"line_items": [{"air_date": "02/29/2016", "program": "Evening News", ...}]}}
  ```
- Document status shows 🟡 (in progress — scalar fields not yet filled)

**Bug found (Issue 6):** After clicking "＋ Add item", `st.rerun()` triggered the resume/start-fresh prompt to reappear because the session state reset logic was clearing the choice on every rerun.

**Fix applied (`app.py`):** Removed the `st.session_state.pop(f"resume_choice_{old}")` call from the document-switch handler. The resume choice now persists across reruns within the same document session.

---

### Flow 18: Session-Based Storage — Manifest + GUID Subdirectory

**Architecture implemented:**
```
dataset/
  .annotations/
    manifest.json          ← schema embedded + all session metadata
    <guid>/
      <pdf_stem>.json      ← per-doc annotations
```

**Manifest structure:**
- `schema`: Full JSON Schema embedded at session creation
- `schema_hash`: MD5 for quick comparison
- `sessions`: Dict of GUID → {annotator, created_at, updated_at, doc_count, completed_count}

**Result:** ✅ PASS
- Session created on first annotation
- Per-doc JSON saved to `files/.annotations/<guid>/001368e77...json`
- Manifest updated with session metadata

---

### Flow 19: Deep Link — New Session (schema file)

**URL:** `http://localhost:8501/?dataset=./files&schema=./files/fcc_invoice_schema.json&mode=zero_start`

**Result:** ✅ PASS
- Config auto-applied, no dialog needed
- New session created, GUID stored in session state
- Header shows: `📁 files · 📋 FCC Political Advertising Invoice · ⚡ Zero Start`
- Deep link in header updates to `?dataset=./files&session=<guid>`

---

### Flow 20: Deep Link — Resume Existing Session (GUID)

**URL:** `http://localhost:8501/?dataset=./files&session=<guid>`

**Result:** ✅ PASS
- Schema loaded from manifest (no schema file needed)
- Session resumed, annotations loaded from session subdir
- Document status correctly shows 🟡 for partially annotated doc
- `station_name` field shows `COX MEDIA - WEST` with ✅ status dot
- Progress bar shows `1 / 16 fields annotated`

**Bug fixed (Issue 5):** Deep link hardcoded `localhost:8502` — fixed to use correct port.
**Bug fixed (Issue 6):** `dataset.py` `get_status()` used old flat path — updated to accept `session` param for session-scoped lookup.

---

### Flow 21: Array Remove Items

**Steps:**
1. Resume session, scroll to Line Items section
2. Click "＋ Add row" twice — 2 rows appear
3. Fill row 1 with air_date, program, spot_length, gross_rate, net_rate
4. Click ✕ on row 1

**Result:** ✅ PASS
- Row removed, count drops from 2 to 1
- Save toast fires
- Remaining row stays intact

---

### Flow 22: Config Dialog — Reconfigure While Active

**Steps:**
1. Click ⚙️ gear while already configured
2. Observe dialog contents
3. Close dialog

**Result:** ✅ PASS
- Dialog opens with current values pre-filled (dataset dir, mode)
- Schema file path empty when session was resumed by GUID (minor — schema came from manifest)
- Close button works, annotation workflow unaffected

**Bug found (Issue 9):** Schema file path field empty in config dialog when session resumed via GUID. Low priority.

---

### Flow 23: Invalid Session GUID in URL

**URL:** `?dataset=./files&session=00000000-dead-beef-0000-000000000000`

**Result:** ✅ PASS
- Gracefully falls back to "Get started" prompt
- No crash, no error shown to user
- Gear icon available to configure manually

---

## Known Issues / Backlog

| # | Description | Severity | Status |
|---|---|---|---|
| 1 | Schema Builder visible after Finalize click | Medium | ✅ Fixed |
| 2 | No save feedback after field annotation | Medium | ✅ Fixed |
| 3 | Spurious save toast on document switch | Medium | ✅ Fixed |
| 4 | "Configure the tool" alert shows alongside Schema Builder | Low | Open |
| 5 | Deep link hardcoded wrong port | Medium | ✅ Fixed |
| 6 | Status tracking used old flat path, not session subdir | High | ✅ Fixed |
| 7 | Annotator name not shown/editable in config | Low | Open |
| 8 | No session picker UI for multiple sessions | Medium | Open |
| 9 | Schema file path empty in dialog when resumed by GUID | Low | Open |

---

## Test Coverage Checklist

| Area | Tested | Result |
|---|---|---|
| Initial page load | ✅ | Pass |
| Schema Builder flow | ✅ | Pass |
| JSON Schema file flow | ✅ | Pass |
| Pydantic import flow | ⬜ | Pending |
| Zero Start annotation | ✅ | Pass |
| Mark field as None | ✅ | Pass |
| Save indicator (toast) | ✅ | Pass |
| Progress counter accuracy | ✅ | Pass |
| Document queue switching | ✅ | Pass |
| PDF page navigation | ✅ | Pass |
| Document status (🔴🟡🟢) | ✅ | Pass |
| Spurious save on doc switch | ✅ | Pass |
| Annotations in session subdir | ✅ | Pass |
| Manifest schema embedding | ✅ | Pass |
| Resume existing annotation | ✅ | Pass |
| Start Fresh | ✅ | Pass |
| Array field rendering | ✅ | Pass |
| Array field — add nested object items | ✅ | Pass |
| Array field — remove items | ✅ | Pass |
| Array field — multiple items | ✅ | Pass |
| Deep link — new session | ✅ | Pass |
| Deep link — resume by GUID | ✅ | Pass |
| Deep link — invalid GUID | ✅ | Pass |
| Config dialog — reconfigure | ✅ | Pass |
| Edit Schema button | ⬜ | Pending |
| Error: bad dataset dir | ✅ | Pass |
| Error: bad schema file | ✅ | Pass |
| LLM Inference mode | ⬜ | Pending |
| HITL mode | ⬜ | Pending |

---

## Session 2 — 2026-03-18 (continued)

**Tool:** Playwright MCP (Chromium)
**App:** `streamlit run src/stickler/annotator/app.py` on port 8501
**Tester:** Kiro (automated)

---

### Flow 24: Progress Counter Bug — 0/16 After Start Fresh

**Steps:**
1. Navigate via deep link with schema file param (new session)
2. Click "Start Fresh" on existing annotation
3. Fill all 15 scalar fields via batch fill + mark 5 as N/A
4. Add 2 line item rows with all sub-fields

**Bug found (Issue 10):** Progress counter showed **0 / 16** and all status dots showed ⬜ even though all fields had values in the textboxes and data was correctly saved to disk. Root cause: `_resume_or_fresh_state` with `choice == "fresh"` always returned a blank `AnnotationState` on every Streamlit rerun, discarding the in-memory state that `_update_field` had populated.

**Fix applied (`app.py`):** `_resume_or_fresh_state` now caches the `AnnotationState` in `st.session_state` keyed by `annotation_state_{pdf_path}`. The cached state survives reruns triggered by field saves, keeping progress counter and status dots accurate. Cache is cleared when switching documents or clicking Resume/Start Fresh.

**Result after fix:** ✅ Progress counter and status dots update correctly in real time.

---

### Flow 25: Complete Full Document Annotation — 16/16

**Steps:**
1. Resume session via deep link `?dataset=./files&session=<guid>`
2. Click "▶ Resume"
3. Verify all 15 scalar fields show ✅ with correct values
4. Verify `line_items` shows ✅ with 2 rows
5. Verify progress bar shows 100%

**Result:** ✅ PASS
- **16 / 16 fields annotated** — progress bar at 100%
- 🎉 "All fields annotated! Move to the next document." banner shown
- All scalar fields: ✅ with values (COX MEDIA - WEST, INV-2016-0042, etc.)
- N/A fields (campaign_id, estimate_id, po_number, agency_commission, rep_firm_commission): ✅ with checkbox checked
- `line_items`: ✅ with 2 rows (Evening News :30, Morning Show :60)
- Document queue shows 🟢

---

### Flow 26: Multi-Document Workflow

**Steps:**
1. Switch to doc 2 (🔴 not started)
2. Verify blank state (0/16, all ⬜)
3. Type `station_name` = "WJLA-TV", press Enter
4. Switch back to doc 1
5. Verify doc 1 still shows 16/16 with all values intact

**Result:** ✅ PASS
- Doc 2 shows 0/16 on switch — no state bleed from doc 1
- Save toast fires on doc 2 field entry
- Doc 2 status updates to 🟡 in queue
- Doc 1 on return: 16/16, all ✅, both line items intact
- No spurious saves on document switch

---

### Flow 27: Stickler Integration — Model Instantiation + Round-Trip

**Steps:**
1. Load annotation JSON from session subdir
2. Instantiate `DynamicModel(**data)` via `SchemaLoader.from_json_schema_file`
3. Verify `model_dump()` round-trip matches original data
4. Verify manifest schema embedding and session progress counts

**Bug found (Issue 11):** `DynamicModel(**data)` raised `ValidationError` for 5 fields with `None` values (campaign_id, estimate_id, po_number, agency_commission, rep_firm_commission). Root cause: `JsonSchemaFieldConverter.convert_property_to_field` used `str` type for non-required fields instead of `Optional[str]`. Pydantic v2 rejects `None` for `str` fields even when `default=None`.

**Fix applied (`json_schema_field_converter.py`):**
- Added `Optional` import
- Non-required primitive fields wrapped: `field_type = Optional[field_type]`
- Non-required nested objects wrapped: `NestedModel = Optional[NestedModel]`
- Non-required arrays wrapped: `field_type = Optional[field_type]`

**Result after fix:** ✅
```
Model instantiation: SUCCESS
  station_name: COX MEDIA - WEST
  line_items count: 2
  line_items[0]: air_date='02/01/2016' program='Evening News' spot_length=':30' gross_rate='$750.00' net_rate='$554.55'
Round-trip dict comparison: PASS
Manifest sessions: 5
  doc_count: 2, completed_count: 1
Manifest integrity: PASS
```

---

### Flow 28: Test Suite — All Passing

**Steps:**
1. Updated `test_dataset.py` — `TestGetStatus` tests now write to `.annotations/` subfolder (matching current storage layout)
2. Updated `test_serializer.py` — `TestAnnotationPathFor`, `TestSave`, `TestLoad` tests updated for `.annotations/` layout

**Result:** ✅ 76/76 tests passing

---

## Known Issues / Backlog (updated)

| # | Description | Severity | Status |
|---|---|---|---|
| 1 | Schema Builder visible after Finalize click | Medium | ✅ Fixed |
| 2 | No save feedback after field annotation | Medium | ✅ Fixed |
| 3 | Spurious save toast on document switch | Medium | ✅ Fixed |
| 4 | "Configure the tool" alert shows alongside Schema Builder | Low | Open |
| 5 | Deep link hardcoded wrong port | Medium | ✅ Fixed |
| 6 | Status tracking used old flat path, not session subdir | High | ✅ Fixed |
| 7 | Annotator name not shown/editable in config | Low | Open |
| 8 | No session picker UI for multiple sessions | Medium | Open |
| 9 | Schema file path empty in dialog when resumed by GUID | Low | Open |
| 10 | Progress counter shows 0/N after Start Fresh (state cache bug) | High | ✅ Fixed |
| 11 | `DynamicModel` rejects None for non-required fields (Optional missing) | High | ✅ Fixed |

---

## Test Coverage Checklist (updated)

| Area | Tested | Result |
|---|---|---|
| Initial page load | ✅ | Pass |
| Schema Builder flow | ✅ | Pass |
| JSON Schema file flow | ✅ | Pass |
| Pydantic import flow | ⬜ | Pending |
| Zero Start annotation | ✅ | Pass |
| Mark field as None | ✅ | Pass |
| Save indicator (toast) | ✅ | Pass |
| Progress counter accuracy | ✅ | Pass |
| Progress counter after Start Fresh | ✅ | Pass |
| Document queue switching | ✅ | Pass |
| PDF page navigation | ✅ | Pass |
| Document status (🔴🟡🟢) | ✅ | Pass |
| Spurious save on doc switch | ✅ | Pass |
| Annotations in session subdir | ✅ | Pass |
| Manifest schema embedding | ✅ | Pass |
| Resume existing annotation | ✅ | Pass |
| Start Fresh | ✅ | Pass |
| Array field rendering | ✅ | Pass |
| Array field — add nested object items | ✅ | Pass |
| Array field — remove items | ✅ | Pass |
| Array field — multiple items | ✅ | Pass |
| Deep link — new session | ✅ | Pass |
| Deep link — resume by GUID | ✅ | Pass |
| Deep link — invalid GUID | ✅ | Pass |
| Config dialog — reconfigure | ✅ | Pass |
| Edit Schema button | ⬜ | Pending |
| Error: bad dataset dir | ✅ | Pass |
| Error: bad schema file | ✅ | Pass |
| Complete full document (16/16) | ✅ | Pass |
| Multi-document workflow | ✅ | Pass |
| Stickler model instantiation | ✅ | Pass |
| Round-trip annotation integrity | ✅ | Pass |
| Manifest session progress tracking | ✅ | Pass |
| Unit test suite (76 tests) | ✅ | Pass |
| LLM Inference mode | ⬜ | Pending |
| HITL mode | ⬜ | Pending |

---

## Session 3 — 2026-03-18 (Pydantic import flow)

**Tool:** Playwright MCP (Chromium)
**App:** `streamlit run src/stickler/annotator/app.py` on port 8501

---

### Flow 29: Pydantic Import Path — Valid Model

**Steps:**
1. Open ⚙️ config dialog
2. Select "Pydantic import path" radio
3. Enter dataset: `./files`, import path: `stickler.annotator.models_example.FccInvoiceModel`
4. Click "Apply Configuration"

**Setup:** Created `src/stickler/annotator/models_example.py` with `FccInvoiceModel` and `LineItemModel` as `StructuredModel` subclasses — demonstrates the Pydantic import path for users who prefer defining schemas as Python classes.

**Result:** ✅ PASS
- Config applied immediately
- All 16 fields rendered (same layout as JSON Schema path)
- `line_items` array section present with "＋ Add row"
- Save toast fires on field entry
- Session GUID created and shown in deep link

---

### Flow 30: Pydantic Import Path — Error Handling

**Steps:**
1. Enter `stickler.annotator.models_example.NonExistentModel` → Apply
2. Enter `totally.fake.module.SomeModel` → Apply

**Result:** ✅ PASS
- Bad class name: `"Invalid Pydantic import: Module 'stickler.annotator.models_example' has no attribute 'NonExistentModel'"`
- Bad module: `"Invalid Pydantic import: Could not import module 'totally.fake.module': No module named 'totally'"`
- Config not applied in either case, dialog stays open

---

## Test Coverage Checklist (final update)

| Area | Tested | Result |
|---|---|---|
| Pydantic import flow | ✅ | Pass |
| Pydantic import — error: bad class | ✅ | Pass |
| Pydantic import — error: bad module | ✅ | Pass |

---

## Session 4 — 2026-03-18 (LLM Inference mode)

**Tool:** Playwright MCP (Chromium) + Strands Agent + Bedrock Haiku 4.5

---

### Flow 31: LLM Inference — Pre-fill via Image Modality + Strands Agent

**Architecture:**
- PDF pages rasterised to PNG at 150dpi via `pdf2image`
- Pages sent as `image` ContentBlocks (image modality only — no base64 document)
- Strands `Agent` with `extract_fields` tool enforces JSON Schema
- Tool validates required fields, coerces types, raises `ValueError` on bad JSON so agent self-corrects
- Successful `toolResult` extracted from agent message history

**Steps:**
1. Navigate to `?dataset=./files&schema=./files/fcc_invoice_schema.json&mode=llm_inference`
2. Click "🤖 Pre-fill with LLM"
3. Wait for spinner "Extracting fields with LLM…"
4. Click "▶ Resume" on existing annotation prompt

**Result:** ✅ PASS
- **16 / 16 fields annotated** — 100% progress bar
- All scalar fields prefilled with 🤖 prefix: COX MEDIA - WEST, $2,249.00, ($337.35), etc.
- **54 line items** extracted and rendered in table (air_date, program, spot_length, gross_rate, net_rate)
- ✅ Accept All / ❌ Reject All batch controls visible
- Document queue shows 🟢
- Schema enforcement: `extract_fields` tool validated all required fields on first call

**Model:** `us.anthropic.claude-haiku-4-5-20251001-v1:0` (cross-region inference profile)

---

## Test Coverage Checklist (final)

| Area | Tested | Result |
|---|---|---|
| LLM Inference mode — pre-fill | ✅ | Pass |
| LLM Inference — image modality | ✅ | Pass |
| LLM Inference — schema enforcement via tool | ✅ | Pass |
| LLM Inference — line items array extraction | ✅ | Pass |
| LLM Inference — Accept All / Reject All | ⬜ | Pending |
| HITL mode | ⬜ | Pending |

---

## Session 5 — 2026-03-18 (Prev/Next document navigation)

### Flow 32: Prev/Next Document Buttons

**Change:** Added ◀ / ▶ buttons flanking the document dropdown for keyboard-free navigation.

**Steps:**
1. Load app — ◀ disabled (first doc), ▶ enabled
2. Click ▶ — navigates to doc 2, PDF viewer updates (Page 1/9), ◀ becomes enabled
3. Click ◀ — returns to doc 1 (Page 1/4), ◀ disabled again

**Result:** ✅ PASS
- Buttons render inline with dropdown in a single row
- Correct disabled state at boundaries (first/last doc)
- Tooltips: "Previous document" / "Next document"
- No state bleed between documents
- Dropdown still works independently and stays in sync
