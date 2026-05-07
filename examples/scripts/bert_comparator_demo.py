#!/usr/bin/env python3
"""
BERTComparator Demo Script

Demonstrates using BERTComparator to evaluate GenAI document extraction outputs.
The scenario: an LLM extracts structured fields from invoices and receipts, and
we compare its output against human-labeled ground truth. BERTComparator handles
the inevitable paraphrasing, abbreviation, and rewording that LLMs produce.

BERTScore works at the token level - it embeds each token individually and
greedily matches tokens between strings. This makes it robust to word reordering
and synonym substitution, but produces scores in a compressed range (typically
0.75-1.0), so threshold tuning is critical.

Requirements:
- Install bert extras: uv sync --extra bert (or pip install stickler-eval[bert])
- This installs: torch, bert-score, evaluate

Note on first run:
- Downloads the BERTScore model from HuggingFace Hub (~1.4 GB for roberta-large)
- First compute() call takes several seconds on CPU
- Set HF_TOKEN env var to silence unauthenticated-requests warnings
- You will see RobertaModel weight-load diagnostics in stderr (normal)
"""

try:
    from stickler.comparators.bert import BERTComparator
except ImportError as e:
    print(f"Missing dependency: {e}")
    print(
        "Install bert extras with: uv sync --extra bert "
        "(or pip install stickler-eval[bert])"
    )
    raise SystemExit(1)

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print(f"{'=' * 60}")


def demo_extraction_evaluation():
    """Evaluate an LLM's extraction of invoice line item descriptions.

    Ground truth is a human-labeled extraction. Predictions are what the
    LLM actually produced. BERTComparator captures that "Q4 2024 consulting
    services" means the same as "Professional consulting services - Q4 2024"
    - something ExactComparator and FuzzyComparator would both fail on.
    """
    print_section_header("INVOICE DESCRIPTION EXTRACTION EVALUATION")

    comparator = BERTComparator(threshold=0.85)

    # Each pair: (ground_truth, llm_prediction, field_name)
    extractions = [
        (
            "Professional consulting services - Q4 2024",
            "Q4 2024 consulting services",
            "line_item_1",
        ),
        (
            # Word reordering + synonym substitution (no adversarial punctuation)
            "Overnight express shipping and handling",
            "Express overnight delivery and handling",
            "line_item_2",
        ),
        (
            "Annual software license renewal fee",
            "Software license - annual renewal",
            "line_item_3",
        ),
        (
            "Office supplies: printer paper and toner cartridges",
            "Printer paper & toner cartridges (office supplies)",
            "line_item_4",
        ),
        (
            "Organic free-range chicken breast",
            "Free range organic chicken breast",
            "line_item_5",
        ),
    ]

    print(f"\nThreshold: {comparator.threshold}")
    print("Evaluating LLM extraction against human-labeled ground truth:\n")
    print(f"{'Field':<14} {'Score':>6}  {'Pass?':>5}")
    print("-" * 60)

    for gt, pred, field in extractions:
        score = comparator.compare(gt, pred)
        passed = "✅" if score >= comparator.threshold else "❌"
        print(f"{field:<14} {score:>6.3f}  {passed:>5}")
        print(f'    GT:  "{gt}"')
        print(f'    LLM: "{pred}"')
        print()


def demo_threshold_tuning():
    """Show why threshold tuning matters for extraction evaluation.

    The same extracted description may pass or fail depending on the
    threshold. Invoice descriptions are often loosely reworded by LLMs,
    so a lower threshold (0.80-0.85) is appropriate. Exact fields like
    invoice numbers should use ExactComparator instead.
    """
    print_section_header("THRESHOLD TUNING FOR INVOICE DESCRIPTIONS")

    gt = "Monthly maintenance and technical support contract"
    pred = "Technical support & maintenance - monthly"

    print(f'  GT:  "{gt}"')
    print(f'  LLM: "{pred}"')
    print()
    print("  How strict should the threshold be?")
    print()

    thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    for threshold in thresholds:
        comparator = BERTComparator(threshold=threshold)
        score = comparator.compare(gt, pred)
        passed = "✅ PASS" if score >= threshold else "❌ FAIL"
        print(f"  Threshold {threshold:.2f} -> Score {score:.3f} -> {passed}")

    print()
    print("  💡 Recommendation: start at 0.85, tighten for critical fields")


def demo_structured_model_integration():
    """Full evaluation pipeline: LLM extracts a receipt.

    This is the real-world pattern - a StructuredModel defines which
    comparator to use per field. Exact fields (merchant name, total) use
    ExactComparator. Free-text fields (item description) use BERTComparator
    since the LLM will inevitably reword them.
    """
    print_section_header("RECEIPT EXTRACTION PIPELINE")

    class ExtractedReceipt(StructuredModel):
        """A receipt extracted by a document processing LLM."""

        merchant_name: str = ComparableField(
            comparator=ExactComparator(),
            threshold=0.9,
            weight=1.0,
        )
        item_description: str = ComparableField(
            # ComparableField.threshold is what gates the pipeline;
            # comparator.threshold is only used when .compare() is called directly
            comparator=BERTComparator(threshold=0.85),
            threshold=0.85,
            weight=2.0,
        )
        total: str = ComparableField(
            comparator=ExactComparator(),
            threshold=1.0,
            weight=1.5,
        )

    print("Schema: ExtractedReceipt")
    print("  merchant_name:    ExactComparator (threshold=0.9, weight=1.0)")
    print("  item_description: BERTComparator  (threshold=0.85, weight=2.0)")
    print("  total:            ExactComparator (threshold=1.0, weight=1.5)")

    # Human-labeled ground truth
    gt = ExtractedReceipt(
        merchant_name="Whole Foods Market",
        item_description="Organic free-range chicken breast",
        total="$24.99",
    )

    # What the LLM actually extracted
    pred = ExtractedReceipt(
        merchant_name="Whole Foods Market",
        item_description="Free range organic chicken breast",
        total="$24.99",
    )

    print("\nGround Truth (human-labeled):")
    print(f'  merchant_name:    "{gt.merchant_name}"')
    print(f'  item_description: "{gt.item_description}"')
    print(f'  total:            "{gt.total}"')

    print("\nLLM Prediction:")
    print(f'  merchant_name:    "{pred.merchant_name}"')
    print(f'  item_description: "{pred.item_description}"')
    print(f'  total:            "{pred.total}"')

    result = gt.compare_with(pred)

    print("\n📊 Results:")
    print(f"  Overall Score:    {result['overall_score']:.3f}")
    print(f"  All Fields Match: {result['all_fields_matched']}")
    print("\n📋 Field Scores:")
    for field, score in result["field_scores"].items():
        print(f"  {field:<20}: {score:.3f}")


def main():
    """Run all BERTComparator demonstrations."""
    print("🚀 BERT COMPARATOR DEMO - Document Extraction Evaluation")
    print("=" * 60)
    print("Scenario: Evaluating LLM-extracted fields against ground truth.")
    print("BERTComparator handles the paraphrasing LLMs inevitably produce.")
    print("=" * 60)

    try:
        demo_extraction_evaluation()
        demo_threshold_tuning()
        demo_structured_model_integration()

        print_section_header("DEMO COMPLETE")
        print(
            "✅ All demos ran without error."
            " (Note: some comparisons intentionally fail their thresholds"
            " to illustrate sensitivity.)"
        )
        print("\n💡 Key Takeaways:")
        print("   • BERTComparator handles LLM rewording that Exact/Fuzzy can't")
        print("   • Threshold 0.85 is a good starting point for extraction eval")
        print("   • Use ExactComparator for IDs, dates, amounts (must be verbatim)")
        print("   • Use BERTComparator for descriptions, notes, free-text fields")
        print("   • StructuredModel lets you mix comparators per field type")

        print("\n🔧 Comparator selection guide for document extraction:")
        print("   • ExactComparator  - invoice numbers, dates, currency amounts")
        print("   • FuzzyComparator  - OCR errors, minor typos in names")
        print("   • BERTComparator   - descriptions, line items, paraphrased content")
        print("   • LLMComparator    - nuanced judgements requiring reasoning")

    except Exception:
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
