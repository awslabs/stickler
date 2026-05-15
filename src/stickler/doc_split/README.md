# doc_split — Document Packet Splitting Metrics

Evaluates how accurately a system splits a multi-page document packet into individual documents and classifies them.

Ported from the [GenAI IDP Accelerator](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws) with infrastructure dependencies removed.

## Reference

[DocSplit: A Comprehensive Benchmark Dataset and Evaluation Approach for Document Packet Recognition and Splitting](https://arxiv.org/abs/2602.15958) (arXiv:2602.15958)

## Two Metric Sets

### 1. Classical Metrics (`DocSplitClassificationMetrics`)

Binary exact-match metrics operating on section-level data.

| Metric | What it measures |
|--------|-----------------|
| Page Level Accuracy | Per-page classification correctness |
| Split Accuracy (Without Order) | Correct page grouping + class, ignoring page order |
| Split Accuracy (With Order) | Correct page grouping + class + exact page order |

### 2. Proposed Metrics (`evaluate_packet`)

Continuous clustering + ordering metrics from the DocSplit paper.

| Metric | What it measures |
|--------|-----------------|
| V-measure | Homogeneity × completeness of page clustering |
| Rand Index | Pairwise clustering similarity |
| Kendall's Tau | Page ordering correlation per document group |
| Packet Score | α · S_clustering + β · S_ordering |

---

## Input Formats

### Classical Metrics: Section-Level Dicts

`DocSplitClassificationMetrics.load_sections()` takes two lists: ground truth sections and predicted sections. Each section is a dict representing one document within the packet.

#### Required keys

| Key | Type | Description |
|-----|------|-------------|
| `section_id` | `str` | Unique identifier for this section. Optional — auto-generated if missing. |
| `document_class` | `str` or `dict` | Document type label. Can be a plain string (`"invoice"`) or a dict with a `"type"` key (`{"type": "invoice"}`). |
| `page_indices` | `list[int]` | Zero-based page indices belonging to this section. Can also be nested under `split_document.page_indices`. |

#### Example: flat format

```python
gt_sections = [
    {
        "section_id": "s1",
        "document_class": "invoice",
        "page_indices": [0, 1, 2]
    },
    {
        "section_id": "s2",
        "document_class": "form",
        "page_indices": [3, 4]
    }
]

pred_sections = [
    {
        "section_id": "p1",
        "document_class": "invoice",
        "page_indices": [0, 1, 2]
    },
    {
        "section_id": "p2",
        "document_class": "form",
        "page_indices": [3, 4]
    }
]
```

#### Example: accelerator nested format

Also supported — `document_class` as a dict and `page_indices` nested under `split_document`:

```python
{
    "section_id": "s1",
    "document_class": {"type": "invoice"},
    "split_document": {"page_indices": [0, 1, 2]}
}
```

#### Rules

- `page_indices` are zero-based integers.
- Pages can be non-sequential (e.g., `[0, 2, 4]` for interleaved documents).
- A page index should appear in exactly one section per list.
- The two lists (gt and pred) do not need the same number of sections.
- `section_id` values do not need to match between gt and pred — matching is done by page set and class.

---

### Proposed Metrics: Page-Level Data

`evaluate_packet()` takes page-level data where each row represents one page in the packet. Accepts three input types:

1. `List[Dict]` — list of page dicts (recommended for programmatic use)
2. `str` — path to a CSV file
3. `pd.DataFrame` — pandas DataFrame (pass-through)

#### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | `str` or `int` | Ground truth document group this page belongs to. |
| `group_id_predicted` | `str` or `int` | Predicted document group. |
| `page_number` | `int` | Ground truth page order within its document (1-based). |
| `page_number_predicted` | `int` | Predicted page order within its document. |

#### Additional columns for strict mode

When `strict_clustering=True`, these are also required:

| Column | Type | Description |
|--------|------|-------------|
| `class_label` | `str` | Ground truth document type (e.g., `"invoice"`, `"form"`). |
| `class_label_predicted` | `str` | Predicted document type. |

#### Example: list of dicts

```python
pages = [
    # 3-page invoice (document group "inv-01")
    {"group_id": "inv-01", "group_id_predicted": "inv-01",
     "page_number": 1, "page_number_predicted": 1,
     "class_label": "invoice", "class_label_predicted": "invoice"},
    {"group_id": "inv-01", "group_id_predicted": "inv-01",
     "page_number": 2, "page_number_predicted": 2,
     "class_label": "invoice", "class_label_predicted": "invoice"},
    {"group_id": "inv-01", "group_id_predicted": "inv-01",
     "page_number": 3, "page_number_predicted": 3,
     "class_label": "invoice", "class_label_predicted": "invoice"},
    # 2-page form (document group "form-01")
    {"group_id": "form-01", "group_id_predicted": "form-01",
     "page_number": 4, "page_number_predicted": 4,
     "class_label": "form", "class_label_predicted": "form"},
    {"group_id": "form-01", "group_id_predicted": "form-01",
     "page_number": 5, "page_number_predicted": 5,
     "class_label": "form", "class_label_predicted": "form"},
]
```

#### Example: CSV file

`evaluate_packet()` can read directly from a CSV file. The CSV must have a header row with the column names listed above.

```csv
group_id,group_id_predicted,page_number,page_number_predicted,class_label,class_label_predicted
inv-01,inv-01,1,1,invoice,invoice
inv-01,inv-01,2,2,invoice,invoice
inv-01,inv-01,3,3,invoice,invoice
form-01,form-01,4,4,form,form
form-01,form-01,5,5,form,form
```

```python
results = evaluate_packet("path/to/data.csv", strict_clustering=True)
```

The `class_label` and `class_label_predicted` columns can be omitted from the CSV if you're not using `strict_clustering=True`.

#### From pandas DataFrame

If you already have your data in a DataFrame, pass it directly:

```python
import pandas as pd
from stickler.doc_split import evaluate_packet

df = pd.read_csv("evaluation_data.csv")
results = evaluate_packet(df, strict_clustering=True)
```

Column names must match exactly (case-sensitive). Types are flexible: `group_id` / `group_id_predicted` can be strings or integers, `page_number` / `page_number_predicted` can be any numeric type.

Full column reference:

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `group_id` | Always | str or int | GT document group. Pages with the same value belong to the same document. |
| `group_id_predicted` | Always | str or int | Predicted document group. Labels don't need to match GT — only partition structure matters. |
| `page_number` | Always | int | GT page position within its document. Used for Kendall's Tau ordering evaluation. |
| `page_number_predicted` | Always | int | Predicted page position within its document. |
| `class_label` | Strict mode | str | GT document type (e.g., `"invoice"`, `"form"`). |
| `class_label_predicted` | Strict mode | str | Predicted document type. |

#### Rules

- One row per page in the packet.
- `group_id` values do not need to match `group_id_predicted` values — clustering metrics compare the partition structure, not the labels.
- `page_number` represents page ordering within a document. Both within-document numbering (1, 2, 3 per document) and global-sequential numbering (e.g., form-01 uses 4, 5 if it starts at the 4th page) are valid — Kendall's Tau only compares relative ordering.
- Single-page groups receive a perfect ordering score of 1.0 (a single page is trivially in the correct order).
- `strict_clustering=True` breaks clustering credit for misclassified pages by assigning them unique error IDs internally.

---

## Usage

### Classical Metrics

```python
from stickler.doc_split import DocSplitClassificationMetrics

m = DocSplitClassificationMetrics()
m.load_sections(gt_sections, pred_sections)
results = m.calculate_all_metrics()
report = m.generate_markdown_report(results)
```

### Proposed Metrics

```python
from stickler.doc_split import evaluate_packet

# From list of dicts
results = evaluate_packet(pages, strict_clustering=True)

# From CSV
results = evaluate_packet("path/to/data.csv", strict_clustering=True)

# Access individual scores
print(results["final_score"])        # Combined packet score
print(results["clustering_score"])   # V-measure + Rand Index
print(results["avg_ordering_score"]) # Mean Kendall's Tau
```
