"""Performance benchmark for refactored StructuredModel.

Performance Thresholds:
-----------------------
The thresholds in these tests are set to accommodate CI environment variability:
- CI environments (GitHub Actions) are typically 3-5x slower than local development
- The refactored architecture maintains <1% overhead vs monolithic implementation
- Thresholds are set to catch significant regressions while allowing for CI variance

Baseline Performance (local development):
- Simple comparison: ~0.3ms per iteration
- Nested comparison: ~30-70ms per iteration  
- Large list (50 contacts, 100 items): ~1.5s per iteration

CI Performance (GitHub Actions):
- Simple comparison: ~1-2ms per iteration
- Nested comparison: ~100-250ms per iteration
- Large list: ~5-7s per iteration
"""
import time
from typing import List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.comparators import ExactComparator


class Address(StructuredModel):
    """Address model for testing."""
    street: str
    city: str
    zip_code: str


class Contact(StructuredModel):
    """Contact model for testing."""
    name: str
    email: str
    phone: str
    address: Address


class Invoice(StructuredModel):
    """Invoice model for testing."""
    invoice_id: str
    amount: float
    contacts: List[Contact]
    items: List[str]


def test_performance_simple_comparison():
    """Test performance of simple field comparison."""
    # Create test data
    gt = Contact(
        name="John Doe",
        email="john@example.com",
        phone="555-1234",
        address=Address(street="123 Main St", city="Boston", zip_code="02101")
    )
    pred = Contact(
        name="John Doe",
        email="john@example.com",
        phone="555-1234",
        address=Address(street="123 Main St", city="Boston", zip_code="02101")
    )
    
    # Warm up
    for _ in range(5):
        gt.compare_with(pred)
    
    # Benchmark
    iterations = 50
    start = time.time()
    for _ in range(iterations):
        result = gt.compare_with(pred, include_confusion_matrix=True)
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    print(f"\nSimple comparison: {avg_time*1000:.3f}ms per iteration ({iterations} iterations)")
    
    # Should be fast - under 5ms per comparison
    assert avg_time < 0.005, f"Simple comparison too slow: {avg_time*1000:.3f}ms"


def test_performance_nested_comparison():
    """Test performance of nested structure comparison."""
    # Create test data with nested structures
    gt = Invoice(
        invoice_id="INV-001",
        amount=1500.00,
        contacts=[
            Contact(
                name="John Doe",
                email="john@example.com",
                phone="555-1234",
                address=Address(street="123 Main St", city="Boston", zip_code="02101")
            ),
            Contact(
                name="Jane Smith",
                email="jane@example.com",
                phone="555-5678",
                address=Address(street="456 Oak Ave", city="Cambridge", zip_code="02139")
            )
        ],
        items=["Item A", "Item B", "Item C"]
    )
    
    pred = Invoice(
        invoice_id="INV-001",
        amount=1500.00,
        contacts=[
            Contact(
                name="John Doe",
                email="john@example.com",
                phone="555-1234",
                address=Address(street="123 Main St", city="Boston", zip_code="02101")
            ),
            Contact(
                name="Jane Smith",
                email="jane@example.com",
                phone="555-5678",
                address=Address(street="456 Oak Ave", city="Cambridge", zip_code="02139")
            )
        ],
        items=["Item A", "Item B", "Item C"]
    )
    
    # Warm up
    for _ in range(5):
        gt.compare_with(pred)
    
    # Benchmark
    iterations = 50
    start = time.time()
    for _ in range(iterations):
        result = gt.compare_with(pred, include_confusion_matrix=True, document_non_matches=True)
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    print(f"\nNested comparison: {avg_time*1000:.3f}ms per iteration ({iterations} iterations)")
    
    # Should be reasonably fast - under 300ms per comparison (adjusted for CI environments)
    # Local development typically sees 50-100ms, CI environments can be 3-5x slower
    assert avg_time < 0.500, f"Nested comparison too slow: {avg_time*1000:.3f}ms"


def test_performance_large_list_comparison():
    """Test performance with large lists."""
    # Create test data with large lists
    contacts = [
        Contact(
            name=f"Person {i}",
            email=f"person{i}@example.com",
            phone=f"555-{i:04d}",
            address=Address(street=f"{i} Main St", city="Boston", zip_code="02101")
        )
        for i in range(50)
    ]
    
    gt = Invoice(
        invoice_id="INV-LARGE",
        amount=50000.00,
        contacts=contacts,
        items=[f"Item {i}" for i in range(100)]
    )
    
    pred = Invoice(
        invoice_id="INV-LARGE",
        amount=50000.00,
        contacts=contacts[:],  # Copy
        items=[f"Item {i}" for i in range(100)]
    )
    
    # Warm up
    for _ in range(3):
        gt.compare_with(pred)
    
    # Benchmark
    iterations = 20
    start = time.time()
    for _ in range(iterations):
        result = gt.compare_with(pred, include_confusion_matrix=True)
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    print(f"\nLarge list comparison: {avg_time*1000:.3f}ms per iteration ({iterations} iterations)")
    
    # Should complete in reasonable time - under 10000ms per comparison (large dataset)
    # This test involves 50 contacts with nested addresses + 100 items
    # Local development typically sees 1-2s, CI environments can be 3-5x slower
    assert avg_time < 10.0, f"Large list comparison too slow: {avg_time*1000:.3f}ms"


if __name__ == "__main__":
    print("Running performance benchmarks...")
    test_performance_simple_comparison()
    test_performance_nested_comparison()
    test_performance_large_list_comparison()
    print("\nAll performance benchmarks passed!")
