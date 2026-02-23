#!/usr/bin/env python3
"""Demo: Document packet splitting evaluation with stickler.

Evaluates a simulated 10-page lending packet containing a 1099 form,
W-2, and pay stub. Shows perfect prediction vs. a prediction with a
boundary error.
"""

from stickler.doc_split import DocSplitClassificationMetrics


def main():
    # Ground truth: 10-page lending packet
    gt_sections = [
        {"section_id": "1099", "document_class": "1099", "page_indices": [0, 1, 2]},
        {"section_id": "w2", "document_class": "w2", "page_indices": [3, 4, 5, 6]},
        {"section_id": "paystub", "document_class": "pay_stub", "page_indices": [7, 8, 9]},
    ]

    # --- Scenario 1: Perfect prediction ---
    print("=" * 60)
    print("Scenario 1: Perfect prediction")
    print("=" * 60)

    pred_perfect = [
        {"section_id": "p1", "document_class": "1099", "page_indices": [0, 1, 2]},
        {"section_id": "p2", "document_class": "w2", "page_indices": [3, 4, 5, 6]},
        {"section_id": "p3", "document_class": "pay_stub", "page_indices": [7, 8, 9]},
    ]

    m = DocSplitClassificationMetrics()
    m.load_sections(gt_sections, pred_perfect)
    results = m.calculate_all_metrics()

    print(f"Page Level Accuracy:          {results['page_level_accuracy']['accuracy']:.2%}")
    print(f"Split Accuracy (no order):    {results['split_accuracy_without_order']['accuracy']:.2%}")
    print(f"Split Accuracy (with order):  {results['split_accuracy_with_order']['accuracy']:.2%}")

    # --- Scenario 2: Boundary error (page 6 assigned to wrong section) ---
    print()
    print("=" * 60)
    print("Scenario 2: Boundary error on page 6")
    print("=" * 60)

    pred_boundary_error = [
        {"section_id": "p1", "document_class": "1099", "page_indices": [0, 1, 2]},
        {"section_id": "p2", "document_class": "w2", "page_indices": [3, 4, 5]},
        {"section_id": "p3", "document_class": "pay_stub", "page_indices": [6, 7, 8, 9]},
    ]

    m2 = DocSplitClassificationMetrics()
    m2.load_sections(gt_sections, pred_boundary_error)
    results2 = m2.calculate_all_metrics()

    print(f"Page Level Accuracy:          {results2['page_level_accuracy']['accuracy']:.2%}")
    print(f"Split Accuracy (no order):    {results2['split_accuracy_without_order']['accuracy']:.2%}")
    print(f"Split Accuracy (with order):  {results2['split_accuracy_with_order']['accuracy']:.2%}")

    # --- Markdown report for scenario 2 ---
    print()
    print("=" * 60)
    print("Markdown Report (Scenario 2)")
    print("=" * 60)
    print(m2.generate_markdown_report(results2))


if __name__ == "__main__":
    main()
