"""
Bulk Evaluation Demo for Large Datasets

This example demonstrates how to efficiently evaluate large datasets using
the BulkStructuredModelEvaluator for memory-efficient, scalable evaluation.

Usage:
    python bulk_evaluation_demo.py
"""

from typing import List
import time
import json
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
import random


# Define a simple document model for demonstration
class Document(StructuredModel):
    """Simple document model for bulk evaluation demo."""

    doc_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )

    title: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )

    author: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )

    content: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.6,  # More lenient for content
        weight=1.0,
    )


def generate_sample_documents(count: int) -> List[tuple]:
    """Generate sample ground truth and prediction document pairs."""

    print(f"📋 Generating {count:,} sample document pairs...")

    # Sample data templates
    titles = [
        "Introduction to Machine Learning",
        "Advanced Python Programming",
        "Data Science Fundamentals",
        "Web Development with React",
        "Database Design Principles",
    ]

    authors = [
        "Dr. Alice Smith",
        "Prof. Bob Johnson",
        "Sarah Williams",
        "Michael Chen",
        "Jennifer Davis",
    ]

    content_templates = [
        "This document covers the fundamental concepts and practical applications",
        "A comprehensive guide to advanced techniques and methodologies",
        "Learn the essential principles and best practices for",
        "Explore the latest developments and trends in",
        "Master the core concepts through hands-on examples",
    ]

    documents = []

    for i in range(count):
        # Create ground truth
        doc_id = f"DOC-{i + 1:05d}"
        title = random.choice(titles)
        author = random.choice(authors)
        content = f"{random.choice(content_templates)} {title.lower()}."

        gt_doc = Document(doc_id=doc_id, title=title, author=author, content=content)

        # Create prediction with some variations
        # 80% chance of correct doc_id
        pred_doc_id = doc_id if random.random() < 0.8 else f"DOC-{i + 2:05d}"

        # 70% chance of exact title, 20% abbreviated, 10% wrong
        rand = random.random()
        if rand < 0.7:
            pred_title = title
        elif rand < 0.9:
            # Abbreviated title
            pred_title = title.split()[0] + " " + title.split()[-1]
        else:
            # Wrong title
            pred_title = random.choice([t for t in titles if t != title])

        # 85% chance of correct author, 15% variation
        if random.random() < 0.85:
            pred_author = author
        else:
            # Name variation (e.g., "Dr. Alice Smith" -> "Alice Smith")
            pred_author = author.replace("Dr. ", "").replace("Prof. ", "")

        # 60% chance of exact content, 40% with minor changes
        if random.random() < 0.6:
            pred_content = content
        else:
            # Minor content variation
            pred_content = content.replace("fundamental", "basic").replace(
                "essential", "key"
            )

        pred_doc = Document(
            doc_id=pred_doc_id,
            title=pred_title,
            author=pred_author,
            content=pred_content,
        )

        documents.append((gt_doc, pred_doc, f"doc_{i + 1}"))

    return documents


def demo_basic_bulk_evaluation():
    """Demonstrate basic bulk evaluation."""
    print("🚀 BASIC BULK EVALUATION")
    print("=" * 50)

    # Generate sample data
    sample_size = 1000
    documents = generate_sample_documents(sample_size)

    print(f"📊 Evaluating {len(documents):,} document pairs...")

    # Create bulk evaluator
    evaluator = BulkStructuredModelEvaluator(
        target_schema=Document,
        verbose=True,  # Show progress
        document_non_matches=False,  # Skip for speed
        elide_errors=False,  # Count errors in metrics
    )

    # Process documents one by one (simulating streaming)
    start_time = time.time()

    for gt_doc, pred_doc, doc_id in documents:
        evaluator.update(gt_doc, pred_doc, doc_id)

    # Get final results
    result = evaluator.compute()
    elapsed = time.time() - start_time

    print(f"\n📊 Bulk Evaluation Results:")
    print(f"  Documents Processed: {len(documents):,}")
    print(f"  Processing Time: {elapsed:.2f}s")
    print(f"  Rate: {len(documents) / elapsed:.1f} docs/sec")

    print(f"\n📈 Overall Metrics:")
    metrics = result.metrics or {}
    derived = metrics.get("derived", {}) or {}
    print(f"  Precision: {derived.get('cm_precision', 0):.3f}")
    print(f"  Recall:    {derived.get('cm_recall', 0):.3f}")
    print(f"  F1 Score:  {derived.get('cm_f1', 0):.3f}")
    print(f"  Accuracy:  {derived.get('cm_accuracy', 0):.3f}")

    print(f"\n📋 Field-Level Performance:")
    field_metrics = result.field_metrics or {}
    for field, metrics in field_metrics.items():
        field_derived = (metrics or {}).get("derived", {}) or {}
        precision = field_derived.get("cm_precision", 0)
        recall = field_derived.get("cm_recall", 0)
        f1 = field_derived.get("cm_f1", 0)
        print(f"  {field:10}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    return evaluator


def demo_batch_processing():
    """Demonstrate batch processing for efficiency."""
    print("\n🔥 BATCH PROCESSING DEMO")
    print("=" * 50)

    # Generate larger dataset
    sample_size = 2500
    documents = generate_sample_documents(sample_size)

    print(f"📊 Processing {len(documents):,} documents in batches...")

    # Create bulk evaluator
    evaluator = BulkStructuredModelEvaluator(target_schema=Document, verbose=True)

    # Process in batches for efficiency
    batch_size = 500
    start_time = time.time()

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        evaluator.update_batch(batch)

        # Show intermediate progress
        current_metrics = evaluator.get_current_metrics()
        processed = evaluator._processed_count
        current_metrics_dict = current_metrics.metrics or {}
        current_derived = current_metrics_dict.get("derived", {}) or {}
        f1_score = current_derived.get("cm_f1", 0)
        print(f"  Processed {processed:,} docs, Current F1: {f1_score:.3f}")

    # Final results
    final_result = evaluator.compute()
    elapsed = time.time() - start_time

    print(f"\n📊 Final Batch Results:")
    print(f"  Total Documents: {len(documents):,}")
    print(f"  Total Time: {elapsed:.2f}s")
    print(f"  Rate: {len(documents) / elapsed:.1f} docs/sec")
    final_metrics = final_result.metrics or {}
    final_derived = final_metrics.get("derived", {}) or {}
    print(f"  Final F1 Score: {final_derived.get('cm_f1', 0):.3f}")


def demo_evaluation_with_output():
    """Demonstrate saving individual results and metrics."""
    print("\n💾 EVALUATION WITH OUTPUT SAVING")
    print("=" * 50)

    # Smaller dataset for demo
    sample_size = 500
    documents = generate_sample_documents(sample_size)

    # Create evaluator with output file
    output_file = "individual_results.jsonl"
    metrics_file = "evaluation_metrics.json"

    evaluator = BulkStructuredModelEvaluator(
        target_schema=Document, verbose=True, individual_results_jsonl=output_file
    )

    print(f"📊 Processing {len(documents):,} documents with output saving...")

    # Process all documents
    for gt_doc, pred_doc, doc_id in documents:
        evaluator.update(gt_doc, pred_doc, doc_id)

    # Get results and save metrics
    result = evaluator.compute()
    evaluator.save_metrics(metrics_file)

    print(f"\n💾 Output Files Created:")
    print(f"  Individual Results: {output_file}")
    print(f"  Metrics Summary: {metrics_file}")

    # Show sample of individual results
    print(f"\n📋 Sample Individual Results:")
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 results
                    break
                record = json.loads(line.strip())
                doc_id = record["doc_id"]
                overall_score = record["comparison_result"]["overall_score"]
                print(f"  {doc_id}: Overall Score = {overall_score:.3f}")
    except FileNotFoundError:
        print("  (Output file not found - check file permissions)")

    # Pretty print final metrics
    evaluator.pretty_print_metrics()

    return output_file, metrics_file


def demo_performance_comparison():
    """Compare bulk evaluator performance with individual evaluation."""
    print("\n⚡ PERFORMANCE COMPARISON")
    print("=" * 50)

    # Generate test dataset
    sample_size = 1000
    documents = generate_sample_documents(sample_size)

    print(f"📊 Comparing evaluation methods on {sample_size:,} documents...")

    # Method 1: Bulk evaluator (efficient)
    print("\n🚀 Method 1: Bulk Evaluator")
    bulk_evaluator = BulkStructuredModelEvaluator(target_schema=Document, verbose=False)

    start_time = time.time()
    for gt_doc, pred_doc, doc_id in documents:
        bulk_evaluator.update(gt_doc, pred_doc, doc_id)
    bulk_result = bulk_evaluator.compute()
    bulk_time = time.time() - start_time

    print(f"  Time: {bulk_time:.2f}s")
    print(f"  Rate: {len(documents) / bulk_time:.1f} docs/sec")
    bulk_metrics = bulk_result.metrics or {}
    bulk_derived = bulk_metrics.get("derived", {}) or {}
    print(f"  F1 Score: {bulk_derived.get('cm_f1', 0):.3f}")

    # Method 2: Individual comparisons (for comparison)
    print("\n🐌 Method 2: Individual Comparisons")
    start_time = time.time()
    individual_f1_scores = []

    for gt_doc, pred_doc, doc_id in documents[:100]:  # Sample only for speed
        result = gt_doc.compare_with(pred_doc, include_confusion_matrix=True)
        # Extract F1 from confusion matrix if available
        cm = result.get("confusion_matrix", {})
        overall = cm.get("overall", {})
        f1 = overall.get("derived", {}).get("cm_f1", 0)
        individual_f1_scores.append(f1)

    individual_time = time.time() - start_time
    avg_f1 = (
        sum(individual_f1_scores) / len(individual_f1_scores)
        if individual_f1_scores
        else 0
    )

    print(f"  Time (100 docs): {individual_time:.2f}s")
    print(f"  Estimated rate: {100 / individual_time:.1f} docs/sec")
    print(f"  Average F1: {avg_f1:.3f}")

    # Show performance improvement
    speed_improvement = (100 / individual_time) / (len(documents) / bulk_time)
    print(f"\n⚡ Performance Improvement:")
    print(f"  Bulk evaluator is {speed_improvement:.1f}x faster!")


def main():
    """Run all bulk evaluation demonstrations."""
    print("🚀 BULK EVALUATION DEMO")
    print("=" * 60)
    print("Demonstrating efficient evaluation of large datasets")
    print("=" * 60)

    # Demo 1: Basic bulk evaluation
    evaluator = demo_basic_bulk_evaluation()

    # Demo 2: Batch processing
    demo_batch_processing()

    # Demo 3: Output saving
    output_file, metrics_file = demo_evaluation_with_output()

    # Demo 4: Performance comparison
    demo_performance_comparison()

    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 50)
    print("✅ Bulk evaluation handles large datasets efficiently")
    print("✅ Batch processing provides memory management")
    print("✅ Individual results can be saved for analysis")
    print("✅ Significantly faster than individual comparisons")
    print("✅ Stateful design allows streaming processing")

    print(f"\n🚀 Key Benefits:")
    print("  • Memory-efficient: Process datasets larger than RAM")
    print("  • Scalable: Linear performance with dataset size")
    print("  • Flexible: Stream processing or batch processing")
    print("  • Observable: Progress tracking and intermediate metrics")
    print("  • Persistent: Save results and metrics for later analysis")

    print(f"\n📚 Perfect For:")
    print("  • Large-scale model evaluation")
    print("  • Production ML pipeline assessment")
    print("  • Batch document processing evaluation")
    print("  • A/B testing of extraction models")
    print("  • Quality monitoring of live systems")

    print(f"\n🗂️  Generated Files:")
    print(f"  • {output_file} - Individual comparison results")
    print(f"  • {metrics_file} - Aggregated evaluation metrics")


if __name__ == "__main__":
    main()
