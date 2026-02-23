"""Document split classification metrics module.

Provides evaluation metrics for document packet splitting tasks, measuring
how accurately a system splits a multi-page document packet into individual
documents and classifies them.

Three metric types are supported (classical metrics):
1. Page Level Accuracy: Per-page classification correctness
2. Split Accuracy (Without Order): Correct page grouping regardless of order
3. Split Accuracy (With Order): Correct page grouping with exact order

These implement the "classical" metrics from the DocSplit paper
(arXiv:2602.15958). The paper also proposes clustering-based metrics
(V-measure, Rand Index, Kendall's Tau) which are planned for future addition.

Usage:
    from stickler.doc_split import DocSplitClassificationMetrics

    metrics = DocSplitClassificationMetrics()
    metrics.load_sections(ground_truth_sections, predicted_sections)
    results = metrics.calculate_all_metrics()
"""

from .doc_split_classification_metrics import DocSplitClassificationMetrics

__all__ = ["DocSplitClassificationMetrics"]
