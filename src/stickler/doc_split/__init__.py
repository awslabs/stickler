"""Document split classification and packet evaluation metrics.

Two complementary metric sets for document packet splitting tasks:

1. Classical metrics (DocSplitClassificationMetrics):
   - Page Level Accuracy, Split Accuracy (unordered/ordered)

2. Proposed metrics (evaluate_packet):
   - V-measure + Rand Index (clustering), Kendall's Tau (ordering),
     combined Packet Score

Reference: "DocSplit: A Comprehensive Benchmark Dataset and Evaluation
Approach for Document Packet Recognition and Splitting" (arXiv:2602.15958)
"""

from .doc_split_classification_metrics import DocSplitClassificationMetrics
from .packet_evaluation_metrics import evaluate_packet

__all__ = ["DocSplitClassificationMetrics", "evaluate_packet"]
