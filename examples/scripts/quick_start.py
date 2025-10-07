"""
Quick Start Example for StructuredObjectEvaluator

This example demonstrates the core functionality of the stickler library:
1. Comparing individual structured objects
2. Comparing sets/lists of objects (the main strength!)
3. Understanding evaluation metrics

Usage:
    python quick_start.py
"""

from typing import List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


# Step 1: Define your data structures
class Product(StructuredModel):
    """Product model for demonstration."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=2.0,  # Name is important
    )

    price: float = ComparableField(
        threshold=0.95,  # Very strict for prices
        weight=1.5,
    )

    category: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )


class Order(StructuredModel):
    """Order containing multiple products."""

    order_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )

    customer: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    # This is where the magic happens - list comparison!
    products: List[Product] = ComparableField(
        weight=3.0  # Most important field
    )


def demo_individual_comparison():
    """Demonstrate comparing individual objects."""
    print("🔍 INDIVIDUAL OBJECT COMPARISON")
    print("=" * 50)

    # Create two similar products
    gt_product = Product(name="Wireless Mouse", price=25.99, category="Electronics")

    pred_product = Product(
        name="Wireless Mouse",  # Exact match
        price=25.95,  # Close but not exact
        category="Electronics",  # Exact match
    )

    print(
        f"Ground Truth: {gt_product.name} | ${gt_product.price} | {gt_product.category}"
    )
    print(
        f"Prediction:   {pred_product.name} | ${pred_product.price} | {pred_product.category}"
    )

    # Compare them
    result = gt_product.compare_with(pred_product)

    print(f"\n📊 Results:")
    print(f"  Overall Score: {result['overall_score']:.3f}")
    print(f"  All Fields Match: {result['all_fields_matched']}")

    print(f"\n📋 Field Scores:")
    for field, score in result["field_scores"].items():
        print(f"  {field:10}: {score:.3f}")

    return result["overall_score"]


def demo_list_comparison():
    """Demonstrate the real power - comparing lists of objects."""
    print("\n🚀 LIST COMPARISON - THE REAL POWER!")
    print("=" * 50)

    # Ground truth order
    gt_order = Order(
        order_id="ORD-2024-001",
        customer="John Smith",
        products=[
            Product(name="Wireless Mouse", price=25.99, category="Electronics"),
            Product(name="USB Cable", price=12.99, category="Electronics"),
            Product(name="Notebook", price=8.50, category="Office Supplies"),
        ],
    )

    # Prediction with some differences
    pred_order = Order(
        order_id="ORD-2024-001",  # Perfect match
        customer="John Smith",  # Perfect match
        products=[
            Product(
                name="Wireless Mouse", price=25.99, category="Electronics"
            ),  # Perfect match
            Product(
                name="USB Cord", price=12.99, category="Electronics"
            ),  # Name variation
            Product(
                name="Legal Pad", price=9.99, category="Office Supplies"
            ),  # Different product
        ],
    )

    print("Ground Truth Order:")
    print(f"  ID: {gt_order.order_id} | Customer: {gt_order.customer}")
    print("  Products:")
    for i, p in enumerate(gt_order.products, 1):
        print(f"    {i}. {p.name} | ${p.price} | {p.category}")

    print("\nPredicted Order:")
    print(f"  ID: {pred_order.order_id} | Customer: {pred_order.customer}")
    print("  Products:")
    for i, p in enumerate(pred_order.products, 1):
        print(f"    {i}. {p.name} | ${p.price} | {p.category}")

    # Compare the orders
    result = gt_order.compare_with(pred_order, include_confusion_matrix=True)

    print(f"\n📊 Comparison Results:")
    print(f"  Overall Score: {result['overall_score']:.3f}")
    print(f"  All Fields Match: {result['all_fields_matched']}")

    print(f"\n📋 Field Scores:")
    for field, score in result["field_scores"].items():
        print(f"  {field:10}: {score:.3f}")

    # Show confusion matrix for the products list
    if "confusion_matrix" in result:
        cm = result["confusion_matrix"]
        if "fields" in cm and "products" in cm["fields"]:
            prod_metrics = cm["fields"]["products"]["overall"]

            print(f"\n🎯 Products List Analysis:")
            print(f"  True Positives:  {prod_metrics.get('tp', 0)} (correct matches)")
            print(f"  False Positives: {prod_metrics.get('fp', 0)} (incorrect/extra)")
            print(f"  False Negatives: {prod_metrics.get('fn', 0)} (missed)")

            print(f"\n📈 List Metrics:")
            print(
                f"  Precision: {prod_metrics.get('derived', {}).get('cm_precision', 0):.3f}"
            )
            print(
                f"  Recall:    {prod_metrics.get('derived', {}).get('cm_recall', 0):.3f}"
            )
            print(f"  F1 Score:  {prod_metrics.get('derived', {}).get('cm_f1', 0):.3f}")

    return result["overall_score"]


def demo_evaluator_detailed_analysis():
    """Show how to use the evaluator for detailed analysis."""
    print("\n📈 DETAILED ANALYSIS WITH EVALUATOR")
    print("=" * 50)

    # Simple case for clear demonstration
    gt_order = Order(
        order_id="ORD-123",
        customer="Alice Johnson",
        products=[
            Product(name="Laptop", price=999.99, category="Electronics"),
            Product(name="Mouse", price=29.99, category="Electronics"),
        ],
    )

    pred_order = Order(
        order_id="ORD-123",
        customer="Alice Johnson",
        products=[
            Product(
                name="Laptop Computer", price=999.99, category="Electronics"
            ),  # Name variation
            Product(
                name="Wireless Mouse", price=29.99, category="Electronics"
            ),  # Name variation
        ],
    )

    print("Evaluating similar but not identical orders...")

    evaluator = StructuredModelEvaluator()
    result = evaluator.evaluate(gt_order, pred_order)

    print(f"\n📊 Overall Metrics:")
    print(f"  Precision: {result['overall']['precision']:.3f}")
    print(f"  Recall:    {result['overall']['recall']:.3f}")
    print(f"  F1 Score:  {result['overall']['f1']:.3f}")
    print(f"  ANLS:      {result['overall']['anls_score']:.3f}")

    print(f"\n📋 Field Analysis:")
    for field, metrics in result["fields"].items():
        if isinstance(metrics, dict):
            if "anls_score" in metrics:
                print(f"  {field:10}: {metrics['anls_score']:.3f}")
            elif "overall" in metrics and "anls_score" in metrics["overall"]:
                print(f"  {field:10}: {metrics['overall']['anls_score']:.3f} (list)")


def main():
    """Run all demonstrations."""
    print("🚀 STICKLER LIBRARY QUICK START")
    print("=" * 50)
    print("Demonstrating structured object comparison & evaluation")
    print("=" * 50)

    # Demo 1: Individual object comparison
    individual_score = demo_individual_comparison()

    # Demo 2: List comparison (the main strength)
    list_score = demo_list_comparison()

    # Demo 3: Detailed evaluator analysis
    demo_evaluator_detailed_analysis()

    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 50)
    print(f"✅ Individual object comparison works great!")
    print(f"✅ List comparison uses Hungarian algorithm for optimal matching")
    print(f"✅ Get detailed metrics: precision, recall, F1, confusion matrices")
    print(f"✅ Configure thresholds and weights per field")

    print(f"\n🚀 Perfect Use Cases:")
    print("  • Document extraction evaluation (invoices, forms, receipts)")
    print("  • OCR quality assessment")
    print("  • Entity extraction validation")
    print("  • ML model output evaluation")
    print("  • Structured data quality measurement")

    print(f"\n📚 Next Steps:")
    print("  1. Try with your own data structures")
    print("  2. Experiment with different comparators and thresholds")
    print("  3. Run the non_match_analysis_demo.py for debugging")
    print("  4. Check out the Jupyter notebook for interactive examples")

    print(f"\n✨ Key Insight:")
    print("  The Hungarian algorithm matching in list comparison is what")
    print("  makes this library special - it finds optimal pairings even")
    print("  when objects are in different orders or partially missing!")


if __name__ == "__main__":
    main()
