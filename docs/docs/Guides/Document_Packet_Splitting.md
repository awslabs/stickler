# Document Packet Splitting Evaluation

Stickler provides two complementary metric sets for evaluating document packet splitting — the task of separating a multi-page document packet into individual documents, classifying them, and ordering their pages.

Both metric sets are based on the [DocSplit paper](https://arxiv.org/abs/2602.15958).

## When to Use Which

| Metric Set | Best For |
|-----------|----------|
| Classical (`DocSplitClassificationMetrics`) | Quick pass/fail evaluation, section-level ground truth |
| Proposed (`evaluate_packet`) | Nuanced evaluation with partial credit, page-level ground truth |

The classical metrics are binary (exact match or miss). The proposed metrics provide continuous scores that distinguish between near-correct and completely wrong predictions.

---

## Classical Metrics

### What They Measure

| Metric | Description |
|--------|-------------|
| Page Level Accuracy | What fraction of pages have the correct document class? |
| Split Accuracy (Without Order) | What fraction of GT sections have a predicted section with the same page set and class? |
| Split Accuracy (With Order) | Same as above, but page order must also match exactly. |

### Input Format

`load_sections()` takes two lists of section dicts — one for ground truth, one for predictions.

Each section dict represents one document within the packet:

```python
{
    "section_id": "s1",           # Unique ID (optional, auto-generated if missing)
    "document_class": "invoice",  # Document type label
    "page_indices": [0, 1, 2]    # Zero-based page indices in this document
}
```

The `document_class` field also accepts a dict: `{"type": "invoice"}`.

The `page_indices` field can also be nested: `{"split_document": {"page_indices": [0, 1, 2]}}`.

**Rules:**

- `page_indices` are zero-based integers
- Pages can be non-sequential (e.g., `[0, 2, 4]` for interleaved documents)
- A page index should appear in exactly one section per list
- GT and predicted lists don't need the same number of sections
- `section_id` values don't need to match between GT and pred — matching is by page set + class

### Example

```python
from stickler.doc_split import DocSplitClassificationMetrics

gt = [
    {"section_id": "s1", "document_class": "invoice", "page_indices": [0, 1, 2]},
    {"section_id": "s2", "document_class": "form", "page_indices": [3, 4]},
]

pred = [
    {"section_id": "p1", "document_class": "invoice", "page_indices": [0, 1, 2]},
    {"section_id": "p2", "document_class": "form", "page_indices": [3, 4]},
]

m = DocSplitClassificationMetrics()
m.load_sections(gt, pred)
results = m.calculate_all_metrics()

print(results["page_level_accuracy"]["accuracy"])           # 1.0
print(results["split_accuracy_without_order"]["accuracy"])   # 1.0
print(results["split_accuracy_with_order"]["accuracy"])      # 1.0

# Generate a markdown report
report = m.generate_markdown_report(results)
```

---

## Proposed Metrics (Packet Score)

### What They Measure

| Metric | Range | Description |
|--------|-------|-------------|
| V-measure | [0, 1] | Are predicted clusters homogeneous and complete? |
| Rand Index | [0, 1] | Pairwise clustering agreement |
| Clustering Score | [0, 1] | `w × V-measure + (1-w) × Rand Index` |
| Kendall's Tau | [-1, 1] | Page ordering correlation per document |
| Ordering Score | [-1, 1] | Mean Tau across multi-page documents |
| Packet Score | [-0.5, 1] | `α × Clustering + β × Ordering` |

### Why These Are Better Than Classical

The paper demonstrates several failure modes of classical metrics:

- **No partial credit**: A single boundary error cascades to 0% on all affected sections
- **No severity distinction**: Over-segmentation and under-segmentation score identically
- **No ordering direction**: Reversed pages and scrambled pages score the same

The proposed metrics address all of these with continuous scoring.

### Input Format

`evaluate_packet()` accepts page-level data in three formats:

1. **`List[Dict]`** — list of page dicts (recommended)
2. **`str`** — path to a CSV file
3. **`pd.DataFrame`** — pandas DataFrame

Each row represents one page in the packet:

```python
{
    "group_id": "inv-01",                # GT document group
    "group_id_predicted": "inv-01",      # Predicted document group
    "page_number": 1,                    # GT page order within document
    "page_number_predicted": 1,          # Predicted page order
    "class_label": "invoice",            # GT document type (strict mode only)
    "class_label_predicted": "invoice",  # Predicted document type (strict mode only)
}
```

**Required columns (always):**

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | str/int | Ground truth document group |
| `group_id_predicted` | str/int | Predicted document group |
| `page_number` | int | GT page order within its document |
| `page_number_predicted` | int | Predicted page order |

**Additional columns (strict mode):**

| Column | Type | Description |
|--------|------|-------------|
| `class_label` | str | GT document type |
| `class_label_predicted` | str | Predicted document type |

**Rules:**

- One row per page in the packet
- `group_id` values don't need to match `group_id_predicted` — clustering metrics compare partition structure, not labels
- `page_number` represents page ordering within a document. Both within-document numbering (1, 2, 3 per document) and global-sequential numbering (e.g., form-01 uses 4, 5 if it starts at the 4th page) are valid — Kendall's Tau only compares relative ordering
- Single-page groups are excluded from ordering score
- `strict_clustering=True` penalizes misclassification by breaking clustering credit for misclassified pages

### Example

```python
from stickler.doc_split import evaluate_packet

pages = [
    {"group_id": "inv-01", "group_id_predicted": "inv-01",
     "page_number": 1, "page_number_predicted": 1,
     "class_label": "invoice", "class_label_predicted": "invoice"},
    {"group_id": "inv-01", "group_id_predicted": "inv-01",
     "page_number": 2, "page_number_predicted": 2,
     "class_label": "invoice", "class_label_predicted": "invoice"},
    {"group_id": "inv-01", "group_id_predicted": "inv-01",
     "page_number": 3, "page_number_predicted": 3,
     "class_label": "invoice", "class_label_predicted": "invoice"},
    {"group_id": "form-01", "group_id_predicted": "form-01",
     "page_number": 4, "page_number_predicted": 4,
     "class_label": "form", "class_label_predicted": "form"},
    {"group_id": "form-01", "group_id_predicted": "form-01",
     "page_number": 5, "page_number_predicted": 5,
     "class_label": "form", "class_label_predicted": "form"},
]

results = evaluate_packet(pages, strict_clustering=True)

print(f"Packet Score:    {results['final_score']:.4f}")
print(f"Clustering:      {results['clustering_score']:.4f}")
print(f"V-measure:       {results['v_measure']:.4f}")
print(f"Rand Index:      {results['rand_index']:.4f}")
print(f"Ordering:        {results['avg_ordering_score']:.4f}")
```

### From CSV

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
results = evaluate_packet("path/to/evaluation_data.csv", strict_clustering=True)
```

The `class_label` and `class_label_predicted` columns can be omitted from the CSV if you're not using `strict_clustering=True`.

### From pandas DataFrame

If you already have your data in a DataFrame (common when working with evaluation pipelines), pass it directly:

```python
import pandas as pd
from stickler.doc_split import evaluate_packet

df = pd.read_csv("evaluation_data.csv")

# The DataFrame must have these columns:
#   group_id, group_id_predicted, page_number, page_number_predicted
# And optionally (for strict_clustering=True):
#   class_label, class_label_predicted

results = evaluate_packet(df, strict_clustering=True)
```

The DataFrame column names must match exactly — they are case-sensitive. The column types are flexible: `group_id` and `group_id_predicted` can be strings or integers, and `page_number` / `page_number_predicted` can be any numeric type.

Here's the full column reference for the DataFrame:

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `group_id` | Always | str or int | Ground truth document group identifier. Pages with the same `group_id` belong to the same document. |
| `group_id_predicted` | Always | str or int | Model's predicted document group. Does not need to use the same labels as `group_id` — only the partition structure matters. |
| `page_number` | Always | int | Ground truth page position within its document. Used for ordering evaluation via Kendall's Tau. |
| `page_number_predicted` | Always | int | Model's predicted page position within its document. |
| `class_label` | Strict mode | str | Ground truth document type (e.g., `"invoice"`, `"form"`, `"letter"`). |
| `class_label_predicted` | Strict mode | str | Model's predicted document type. |

### Tuning Weights

```python
# Emphasize clustering over ordering
results = evaluate_packet(pages, alpha=0.7, beta=0.3)

# Emphasize V-measure over Rand Index in clustering
results = evaluate_packet(pages, v_measure_weight=0.8)
```

---

## API Reference

::: stickler.doc_split.DocSplitClassificationMetrics

::: stickler.doc_split.packet_evaluation_metrics.evaluate_packet
