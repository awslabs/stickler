#!/usr/bin/env python3
"""
LLM Comparator Demo Script

This script demonstrates the LLMComparator functionality for semantic comparison
of values using Large Language Models. The LLMComparator leverages AWS Bedrock
models through the strands-agents library to perform intelligent comparisons
that go beyond simple string matching.

Requirements:
- AWS credentials configured for Bedrock access
- Environment variables for model configuration (optional)
"""
from stickler.comparators.llm import LLMComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print(f"{'=' * 60}")


def demo_structured_model_integration():
    """Demonstrate LLM comparator integration with StructuredModel."""
    print_section_header("STRUCTURED MODEL INTEGRATION")
    
    # Define a customer model with mixed comparators
    class CustomerAddress(StructuredModel):
        street: str = ComparableField(
            comparator=LLMComparator(
                model="us.amazon.nova-lite-v1:0",
                eval_guidelines="Consider street abbreviations equivalent (St=Street, Ave=Avenue, etc.)"
            ),
            threshold=0.8,
            weight=1.0
        )
        city: str = ComparableField(
            comparator=LevenshteinComparator(),
            threshold=0.9,
            weight=1.0
        )
        zip_code: str = ComparableField(
            comparator=ExactComparator(),
            threshold=1.0,
            weight=1.0
        )
    
    class Customer(StructuredModel):
        name: str = ComparableField(
            comparator=ExactComparator(),
            threshold=0.8,
            weight=1.0
        )
        email: str = ComparableField(
            comparator=ExactComparator(),
            threshold=1.0,
            weight=1.0
        )
        address: CustomerAddress = ComparableField(
            comparator=ExactComparator(),
            threshold=1.0,
            weight=1.0
        )
    
    print("Comparing customer records with mixed comparator types...")
    
    # Ground truth customer
    gt_customer = Customer(
        name="Robert Johnson",
        email="robert.johnson@email.com",
        address=CustomerAddress(
            street="123 Main Street",
            city="Seattle",
            zip_code="98101"
        )
    )
    
    # Predicted customer with variations
    pred_customer = Customer(
        name="Robert Johnson", 
        email="robert.johnson@email.com",
        address=CustomerAddress(
            street="123 Main St",  # Street abbreviation
            city="Seattle",
            zip_code="98101"
        )
    )
    
    # Compare the customers
    result = gt_customer.compare_with(pred_customer, include_confusion_matrix=True)
        
    # Show field-level results
    print("\nField-level comparison results:")
    cm = result['confusion_matrix']
    for field_name, field_data in cm['fields'].items():
        field_result = field_data['overall']
        print(f"   {field_name}: {field_result}")


def main():
    """Run all demonstration functions."""
    print("🚀 LLM COMPARATOR COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases the LLMComparator functionality for")
    print("semantic comparison using Large Language Models.")
    
    # Check for required environment setup
    print("\n📋 Environment Check:")
    
    try:
  
        demo_structured_model_integration()
     
        print_section_header("DEMO COMPLETE")
        print("✅ All demonstrations completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • LLMComparator provides semantic comparison beyond string matching")
        print("   • Integrates seamlessly with StructuredModel for complex objects")
        
        print("\n🔧 Best Practices:")
        print("   • Use specific guidelines for better accuracy")
        print("   • Choose appropriate models for your use case")
        print("   • Handle None values and edge cases")
        print("   • Monitor API costs and latency")
        print("   • Test with representative data")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your AWS credentials and model access.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
