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
