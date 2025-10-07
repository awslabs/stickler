#!/usr/bin/env python3

"""
Print Results Demo - Beautiful Output for Evaluation Results

This example demonstrates the various pretty printing methods available
for displaying evaluation results in a beautiful, readable format.

Usage:
    python print_results_demo.py
"""

from typing import List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

# Import the beautiful print functions
from stickler.structured_object_evaluator.utils.pretty_print import (
    print_confusion_matrix,
)


# Define sample models
class Invoice(StructuredModel):
    """Invoice model for demonstration."""

    invoice_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )

    vendor: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.5
    )

    amount: float = ComparableField(threshold=0.95, weight=2.0)


class Product(StructuredModel):
    """Product model for list comparison demo."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )

    price: float = ComparableField(threshold=0.95, weight=1.5)


class Order(StructuredModel):
    """Order containing products."""

    order_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )

    products: List[Product] = ComparableField(weight=3.0)


def demo_individual_model_printing():
    """Demo 1: Print results from individual model comparison."""
    print("🎨 DEMO 1: Individual Model Comparison Results")
    print("=" * 60)

    # Create sample models
    gt_invoice = Invoice(
        invoice_id="INV-2024-001", vendor="ABC Corporation", amount=1500.00
    )

    pred_invoice = Invoice(
        invoice_id="INV-2024-001",  # Perfect match
        vendor="ABC Corp",  # Close match
        amount=1499.95,  # Close match
    )

    print("Comparing invoices...")
    print(f"GT:   {gt_invoice.invoice_id} | {gt_invoice.vendor} | ${gt_invoice.amount}")
    print(
        f"Pred: {pred_invoice.invoice_id} | {pred_invoice.vendor} | ${pred_invoice.amount}"
    )

    # Compare using compare_with method
    result = gt_invoice.compare_with(pred_invoice, include_confusion_matrix=True)

    print("\n🎯 Using print_confusion_matrix():")
    print_confusion_matrix(result, show_details=True)


def demo_evaluator_printing():
    """Demo 2: Print results from StructuredModelEvaluator."""
    print("\n🎨 DEMO 2: StructuredModelEvaluator Results")
    print("=" * 60)

    # Create sample order with products
    gt_order = Order(
        order_id="ORD-123",
        products=[
            Product(name="Laptop", price=999.99),
            Product(name="Mouse", price=29.99),
            Product(name="Keyboard", price=79.99),
        ],
    )

    pred_order = Order(
        order_id="ORD-123",
        products=[
            Product(name="Laptop Computer", price=999.99),  # Name variation
            Product(name="Wireless Mouse", price=29.99),  # Name variation
            Product(name="Mechanical Keyboard", price=85.00),  # Name + price variation
        ],
    )

    print("Comparing orders with product lists...")

    # Use evaluator
    evaluator = StructuredModelEvaluator()
    result = evaluator.evaluate(gt_order, pred_order)

    print("\n🎯 Using print_confusion_matrix() with evaluator results:")
    print_confusion_matrix(result, show_details=True)


def demo_bulk_evaluator_printing():
    """Demo 3: Print results from BulkStructuredModelEvaluator."""
    print("\n🎨 DEMO 3: BulkStructuredModelEvaluator Results")
    print("=" * 60)

    # Create bulk evaluator
    bulk_evaluator = BulkStructuredModelEvaluator(target_schema=Invoice, verbose=False)

    # Generate sample data
    sample_data = [
        (
            Invoice(invoice_id="INV-001", vendor="Company A", amount=1000.00),
            Invoice(
                invoice_id="INV-001", vendor="Company A Inc", amount=1000.00
            ),  # Close match
            "doc_1",
        ),
        (
            Invoice(invoice_id="INV-002", vendor="Company B", amount=2000.00),
            Invoice(
                invoice_id="INV-002", vendor="Company B", amount=2000.00
            ),  # Perfect match
            "doc_2",
        ),
        (
            Invoice(invoice_id="INV-003", vendor="Company C", amount=1500.00),
            Invoice(
                invoice_id="INV-999", vendor="Company X", amount=9999.99
            ),  # Poor match
            "doc_3",
        ),
    ]

    print("Processing sample invoices with bulk evaluator...")

    # Process documents
    for gt_invoice, pred_invoice, doc_id in sample_data:
        bulk_evaluator.update(gt_invoice, pred_invoice, doc_id)

    # Get results
    result = bulk_evaluator.compute()

    print("\n🎯 Method 1: Using print_confusion_matrix() with bulk results:")
    print_confusion_matrix(result, show_details=True)

    print("\n🎯 Method 2: Using bulk_evaluator.pretty_print_metrics():")
    bulk_evaluator.pretty_print_metrics()


def demo_advanced_print_options():
    """Demo 4: Advanced printing options and filtering."""
    print("\n🎨 DEMO 4: Advanced Print Options")
    print("=" * 60)

    # Create a more complex example
    gt_order = Order(
        order_id="ORD-COMPLEX-001",
        products=[
            Product(name="Gaming Laptop", price=1899.99),
            Product(name="Wireless Mouse", price=89.99),
            Product(name="Gaming Headset", price=159.99),
            Product(name="USB-C Hub", price=49.99),
        ],
    )

    pred_order = Order(
        order_id="ORD-COMPLEX-001",
        products=[
            Product(name="Gaming Laptop Pro", price=1899.99),  # Name variation
            Product(name="Gaming Mouse", price=89.99),  # Name variation
            Product(name="Audio Headset", price=159.99),  # Name variation
            Product(name="Extra Monitor", price=299.99),  # Different product
        ],
    )

    # Use evaluator for detailed analysis
    evaluator = StructuredModelEvaluator()
    result = evaluator.evaluate(gt_order, pred_order)

    print("\n🎯 Standard output:")
    print_confusion_matrix(result, show_details=True)

    print("\n🎯 Filtered by products field only:")
    print_confusion_matrix(result, field_filter="products", show_details=True)

    print("\n🎯 Sorted by F1 score:")
    print_confusion_matrix(result, sort_by="f1", show_details=True)

    print("\n🎯 Without colors (for file output):")
    print_confusion_matrix(result, use_color=False, show_details=True)


def main():
    """Run all printing demonstrations."""
    print("🎨 BEAUTIFUL RESULTS PRINTING DEMO")
    print("=" * 70)
    print("Demonstrating the various pretty print methods available")
    print("=" * 70)

    # Demo 1: Individual model comparison
    demo_individual_model_printing()

    # Demo 2: Evaluator results
    demo_evaluator_printing()

    # Demo 3: Bulk evaluator results
    demo_bulk_evaluator_printing()

    # Demo 4: Advanced options
    demo_advanced_print_options()

    # Summary
    print("\n🎯 SUMMARY OF PRINT METHODS")
    print("=" * 50)
    print("✅ print_confusion_matrix() - Universal pretty printer")
    print("   • Works with: compare_with(), evaluator.evaluate(), bulk results")
    print("   • Features: Colors, bars, filtering, sorting")
    print(
        "   • Import: from stickler.structured_object_evaluator.utils.pretty_print import print_confusion_matrix"
    )

    print("\n✅ bulk_evaluator.pretty_print_metrics() - Bulk-specific printer")
    print("   • Works with: BulkStructuredModelEvaluator results only")
    print("   • Features: Processing stats, field performance breakdown")
    print("   • Usage: bulk_evaluator.pretty_print_metrics()")

    print("\n🚀 Key Features:")
    print("   • Automatic format detection (works with any result type)")
    print("   • Colored terminal output with visual progress bars")
    print("   • Hierarchical field display for nested structures")
    print("   • Field filtering with regex patterns")
    print("   • Sorting options (name, precision, recall, f1)")
    print("   • File output support")
    print("   • Confusion matrix visualization")

    print("\n📚 Quick Usage:")
    print("   # For any evaluation result:")
    print(
        "   from stickler.structured_object_evaluator.utils.pretty_print import print_confusion_matrix"
    )
    print("   result = model1.compare_with(model2, include_confusion_matrix=True)")
    print("   print_confusion_matrix(result)")

    print("\n   # For bulk evaluator:")
    print("   bulk_evaluator.pretty_print_metrics()  # Built-in method")
    print("   # OR")
    print("   result = bulk_evaluator.compute()")
    print("   print_confusion_matrix(result)  # Universal method")


if __name__ == "__main__":
    main()
