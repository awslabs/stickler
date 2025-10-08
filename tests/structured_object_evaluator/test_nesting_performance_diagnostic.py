"""
Performance diagnostic test for nesting levels.

This test gradually increases nesting depth to identify where performance issues begin.
"""

import time
from typing import List, Optional

from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    StructuredModelEvaluator,
)
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


# Level 1: Simple model
class SimpleItem(StructuredModel):
    """Level 1: Simple model with basic fields."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    value: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


# Level 2: Contains Level 1
class Container(StructuredModel):
    """Level 2: Contains SimpleItem."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    items: List[SimpleItem] = ComparableField(weight=1.0)
    main_item: Optional[SimpleItem] = ComparableField(threshold=0.8, weight=1.0)


# Level 3: Contains Level 2
class Group(StructuredModel):
    """Level 3: Contains Container."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    containers: List[Container] = ComparableField(weight=1.0)
    primary_container: Optional[Container] = ComparableField(threshold=0.8, weight=1.0)


# Level 4: Contains Level 3
class Department(StructuredModel):
    """Level 4: Contains Group."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    groups: List[Group] = ComparableField(weight=1.0)
    main_group: Optional[Group] = ComparableField(threshold=0.8, weight=1.0)


def create_simple_item(base_id: int, variation: str = "base") -> SimpleItem:
    """Create a simple item."""
    if variation == "diff":
        return SimpleItem(
            id=f"SI{base_id:03d}", name=f"Different Item {base_id}", value=base_id + 10
        )
    return SimpleItem(
        id=f"SI{base_id:03d}", name=f"Simple Item {base_id}", value=base_id
    )


def create_container(base_id: int, variation: str = "base") -> Container:
    """Create a container with 2 simple items."""
    items = [
        create_simple_item(base_id * 10 + 1, variation),
        create_simple_item(base_id * 10 + 2, variation),
    ]

    if variation == "diff":
        return Container(
            id=f"C{base_id:03d}",
            name=f"Different Container {base_id}",
            items=items,
            main_item=items[0],
        )
    return Container(
        id=f"C{base_id:03d}",
        name=f"Container {base_id}",
        items=items,
        main_item=items[0],
    )


def create_group(base_id: int, variation: str = "base") -> Group:
    """Create a group with 2 containers."""
    containers = [
        create_container(base_id * 10 + 1, variation),
        create_container(base_id * 10 + 2, variation),
    ]

    if variation == "diff":
        return Group(
            id=f"G{base_id:03d}",
            name=f"Different Group {base_id}",
            containers=containers,
            primary_container=containers[0],
        )
    return Group(
        id=f"G{base_id:03d}",
        name=f"Group {base_id}",
        containers=containers,
        primary_container=containers[0],
    )


def create_department(base_id: int, variation: str = "base") -> Department:
    """Create a department with 2 groups."""
    groups = [
        create_group(base_id * 10 + 1, variation),
        create_group(base_id * 10 + 2, variation),
    ]

    if variation == "diff":
        return Department(
            id=f"D{base_id:03d}",
            name=f"Different Department {base_id}",
            groups=groups,
            main_group=groups[0],
        )
    return Department(
        id=f"D{base_id:03d}",
        name=f"Department {base_id}",
        groups=groups,
        main_group=groups[0],
    )


def test_performance_by_level():
    """Test performance at each nesting level."""
    evaluator = StructuredModelEvaluator(threshold=0.7, document_non_matches=False)

    print("🔍 Performance Diagnostic Test")
    print("=" * 50)

    # Level 1: Simple Item
    print("\n📊 Level 1: SimpleItem")
    start_time = time.time()
    item1 = create_simple_item(1, "base")
    item2 = create_simple_item(1, "diff")
    result = evaluator.evaluate(item1, item2)
    duration = time.time() - start_time
    print(f"   Time: {duration:.3f}s | Score: {result['overall']['anls_score']:.3f}")

    if duration > 5.0:
        print("❌ PERFORMANCE ISSUE: Level 1 took too long!")
        return

    # Level 2: Container
    print("\n📊 Level 2: Container (with lists)")
    start_time = time.time()
    container1 = create_container(1, "base")
    container2 = create_container(1, "diff")
    result = evaluator.evaluate(container1, container2)
    duration = time.time() - start_time
    print(f"   Time: {duration:.3f}s | Score: {result['overall']['anls_score']:.3f}")

    if duration > 10.0:
        print("❌ PERFORMANCE ISSUE: Level 2 took too long!")
        return

    # Level 3: Group
    print("\n📊 Level 3: Group (lists of containers)")
    start_time = time.time()
    group1 = create_group(1, "base")
    group2 = create_group(1, "diff")
    result = evaluator.evaluate(group1, group2)
    duration = time.time() - start_time
    print(f"   Time: {duration:.3f}s | Score: {result['overall']['anls_score']:.3f}")

    if duration > 15.0:
        print("❌ PERFORMANCE ISSUE: Level 3 took too long!")
        return

    # Level 4: Department
    print("\n📊 Level 4: Department (lists of groups)")
    start_time = time.time()
    dept1 = create_department(1, "base")
    dept2 = create_department(1, "diff")
    result = evaluator.evaluate(dept1, dept2)
    duration = time.time() - start_time
    print(f"   Time: {duration:.3f}s | Score: {result['overall']['anls_score']:.3f}")

    if duration > 30.0:
        print("❌ PERFORMANCE ISSUE: Level 4 took too long!")
        return

    print("\n✅ All levels completed successfully!")
    print("\n🎯 Performance Summary:")
    print("   Level 1 (SimpleItem): < 5s")
    print("   Level 2 (Container): < 10s")
    print("   Level 3 (Group): < 15s")
    print("   Level 4 (Department): < 30s")


def test_with_non_match_documentation():
    """Test performance with non-match documentation enabled."""
    print("\n🔍 Testing with Non-Match Documentation")
    print("=" * 50)

    evaluator = StructuredModelEvaluator(threshold=0.7, document_non_matches=True)

    # Test Level 2 with non-match docs
    print("\n📊 Level 2 with Non-Match Docs")
    start_time = time.time()
    container1 = create_container(1, "base")
    container2 = create_container(1, "diff")
    result = evaluator.evaluate(container1, container2)
    duration = time.time() - start_time
    print(f"   Time: {duration:.3f}s | Score: {result['overall']['anls_score']:.3f}")
    print(f"   Non-matches: {len(result.get('non_matches', []))}")

    if duration > 15.0:
        print("❌ PERFORMANCE ISSUE: Non-match documentation is too slow!")
        return

    print("✅ Non-match documentation performance acceptable")


if __name__ == "__main__":
    try:
        test_performance_by_level()
        test_with_non_match_documentation()
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
