"""
Demo script showing how to use the EvaluationHTMLReporter with bulk evaluation.
"""

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import BulkStructuredModelEvaluator
from stickler.reporting.html import EvaluationHTMLReporter, ReportConfig


# Define a simple model for testing
class Product(StructuredModel):
    """Product model for demonstration."""
    
    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.8,  # High threshold for product names
        weight=2.0
    )
    
    price: float = ComparableField(
        threshold=0.95,  # Very strict for prices
        weight=1.5
    )
    
    category: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,  # Moderate threshold for categories
        weight=1.0
    )


def demo_html_report():
    """Demonstrate HTML report generation with bulk evaluation."""
    print("🎯 HTML Report Generation Demo - Bulk Evaluation")
    print("=" * 60)
    
    # Initialize bulk evaluator with individual results logging
    individual_results_path = "reports/individual_results.jsonl"
    bulk_evaluator = BulkStructuredModelEvaluator(
        target_schema=Product,
        verbose=True,
        document_non_matches=True,
        individual_results_jsonl=individual_results_path
    )
    
    # Document 1: Mostly accurate with minor differences
    print("\n📄 Document 1 - High Quality Prediction")
    gt_product1 = Product(
        name="Wireless Mouse",
        price=25.99,
        category="Electronics"
    )
    
    pred_product1 = Product(
        name="Wireless Mouse",  # Exact match
        price=25.95,  # Close but not exact (should pass threshold)
        category="Electronic Devices"  # Similar but different
    )
    
    print("Ground Truth 1:", gt_product1.model_dump())
    print("Prediction 1:  ", pred_product1.model_dump())
    
    # Document 2: More significant mismatches
    print("\n📄 Document 2 - Lower Quality Prediction")
    gt_product2 = Product(
        name="Gaming Keyboard",
        price=89.99,
        category="Computer Accessories"
    )
    
    pred_product2 = Product(
        name="Mechanical Keyboard",  # Similar but different
        price=85.50,  # Different price
        category="Gaming Equipment"  # Different category
    )
    
    print("Ground Truth 2:", gt_product2.model_dump())
    print("Prediction 2:  ", pred_product2.model_dump())
    
    # Process both documents using bulk evaluator
    print("\n🔄 Processing documents with bulk evaluator...")
    bulk_evaluator.update(gt_product1, pred_product1, doc_id="product_001")
    bulk_evaluator.update(gt_product2, pred_product2, doc_id="product_002")
    
    # Compute final results
    results = bulk_evaluator.compute()
    print(results)
    
    # Display bulk evaluation metrics using built-in pretty print
    bulk_evaluator.pretty_print_metrics()
    
    # Generate HTML report using ProcessEvaluation object
    reporter = EvaluationHTMLReporter()

    # Example with document files
    # document_files = {
    #     "product_001": "examples/scripts/images/xx",
    #     "product_002": "examples/scripts/images/xx"
    # }
    
    # Create report configuration
    config = ReportConfig(
        include_executive_summary=True,
        include_field_analysis=True,
        include_confusion_matrix=True,
        include_non_matches=True,
        # document_file_type='pdf', # or 'image'
        max_non_matches_displayed=50
    )
    
    # Generate the report - HTML reporter automatically detects bulk evaluation
    report_result = reporter.generate_report(
        evaluation_results=results,  
        output_path="reports/bulk_product_evaluation_report.html",
        config=config,
        model_schema=Product,
        individual_results_jsonl_path=individual_results_path,
        # document_files=document_files
    )
    
    print("\n📄 HTML Report Generated!")
    print(f"Path: {report_result.output_path}")
    print(f"Generation Time: {report_result.generation_time_seconds:.2f}s")
    print(f"Sections: {', '.join(report_result.sections_included)}")
    print(f"Document Count: {report_result.metadata.get('document_count', 'N/A')}")
    
    if report_result.success:
        print(f"\n✅ {report_result.summary()}")
        print(f"\nOpen the report: file://{os.path.abspath(report_result.output_path)}")
    else:
        print("\n❌ Report generation failed:")
        for error in report_result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("reports", exist_ok=True)
    
    # Run the demo
    demo_html_report()
    
    print("\n🎉 Demo completed!")
    print("Check the generated HTML report for a comprehensive view of evaluation results.")
