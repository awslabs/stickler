"""
Realistic test for 6-level deep nesting with performance safeguards.

This test uses realistic enterprise data structures with sensible complexity
to verify deep nesting works without performance issues.

Realistic Hierarchy (Linear, not exponential):
Level 1: Organization (1 instance)
├── Level 2: Division (1 instance)
    ├── Level 3: Department (1 instance)
        ├── Level 4: Team (1 instance)
            ├── Level 5: Project (1 instance + small list)
                └── Level 6: Task (1 instance + small list)

Total objects: ~10 (not 64!), testing both single nesting and small lists.
"""

import pytest
import time
import signal
from typing import List, Optional

from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    StructuredModelEvaluator,
)
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


# Timeout decorator for performance safety
def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded maximum execution time")


def with_timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except TimeoutError:
                signal.alarm(0)
                raise

        return wrapper

    return decorator


# Level 6: Task (deepest level)
class Task(StructuredModel):
    """Level 6: Deepest nested model - Task."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    status: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    priority: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    estimated_hours: float = ComparableField(
        comparator=ExactComparator(), threshold=0.95, weight=1.0
    )


# Level 5: Project
class Project(StructuredModel):
    """Level 5: Project containing Tasks."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    budget: float = ComparableField(
        comparator=ExactComparator(), threshold=0.95, weight=1.0
    )
    status: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )

    # Single nested task (most common case)
    main_task: Task = ComparableField(threshold=0.8, weight=2.0)

    # Small list of tasks (realistic: 2-3 items max)
    tasks: List[Task] = ComparableField(weight=1.0)


# Level 4: Team
class Team(StructuredModel):
    """Level 4: Team containing Projects."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    size: int = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    lead: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    # Single nested project (most common case)
    current_project: Project = ComparableField(threshold=0.8, weight=2.0)


# Level 3: Department
class Department(StructuredModel):
    """Level 3: Department containing Teams."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    budget: float = ComparableField(
        comparator=ExactComparator(), threshold=0.95, weight=1.0
    )
    head: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    # Single nested team (most common case)
    primary_team: Team = ComparableField(threshold=0.8, weight=2.0)


# Level 2: Division
class Division(StructuredModel):
    """Level 2: Division containing Departments."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    region: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    # Single nested department (most common case)
    main_department: Department = ComparableField(threshold=0.8, weight=2.0)


# Level 1: Organization (top level)
class Organization(StructuredModel):
    """Level 1: Top-level Organization containing Divisions."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    founded_year: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    ceo: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    # Single nested division (most common case)
    main_division: Division = ComparableField(threshold=0.8, weight=2.0)


class TestRealisticDeepNesting:
    """Test suite for realistic 6-level deep nesting with performance safeguards."""

    def setUp(self):
        """Set up test evaluator."""
        self.evaluator = StructuredModelEvaluator(
            threshold=0.7, document_non_matches=True
        )

    def create_test_organization(self, variation: str = "base") -> Organization:
        """Create test organization with realistic 6-level hierarchy.

        Args:
            variation: Type of test data variation

        Returns:
            Organization with realistic complexity (~10 total objects)
        """
        # Level 6: Tasks (small list: 2 items)
        task1 = Task(
            id="T001",
            name="Feature Development" if variation != "level6_diff" else "Bug Fix",
            status="in-progress",
            priority=1 if variation != "level6_diff" else 2,
            estimated_hours=40.0,
        )

        task2 = Task(
            id="T002",
            name="Code Review",
            status="pending",
            priority=2,
            estimated_hours=8.0,
        )

        # Level 5: Project
        project = Project(
            id="P001",
            name="Alpha Release" if variation != "level5_diff" else "Beta Release",
            budget=100000.0,
            status="active",
            main_task=task1,
            tasks=[task1, task2],
        )

        # Level 4: Team
        team = Team(
            id="TM001",
            name="Backend Team" if variation != "level4_diff" else "Frontend Team",
            size=5,
            lead="John Doe",
            current_project=project,
        )

        # Level 3: Department
        department = Department(
            id="D001",
            name="Engineering" if variation != "level3_diff" else "Product",
            budget=500000.0,
            head="Jane Smith",
            primary_team=team,
        )

        # Level 2: Division
        division = Division(
            id="DIV001",
            name="Technology" if variation != "level2_diff" else "Operations",
            region="North America",
            main_department=department,
        )

        # Level 1: Organization
        organization = Organization(
            id="ORG001",
            name="TechCorp Inc" if variation != "level1_diff" else "InnovCorp Inc",
            founded_year=2020,
            ceo="Steve Jobs",
            main_division=division,
        )

        return organization

    @with_timeout(10)  # 10 second timeout
    def test_perfect_match_6_levels_with_timeout(self):
        """Test perfect match across 6 levels with performance safeguard."""
        self.setUp()

        start_time = time.time()

        # Create identical organizations
        gt_org = self.create_test_organization("base")
        pred_org = self.create_test_organization("base")

        # Evaluate
        result = self.evaluator.evaluate(gt_org, pred_org)

        duration = time.time() - start_time

        # Performance assertions
        assert duration < 5.0, f"Perfect match took too long: {duration:.2f}s"

        # Functionality assertions
        assert result["overall"]["anls_score"] == 1.0, (
            f"Expected perfect match, got {result['overall']['anls_score']}"
        )
        assert result["overall"]["precision"] == 1.0
        assert result["overall"]["recall"] == 1.0

        print(f"✅ Perfect match test passed: {duration:.3f}s")

    @with_timeout(15)  # 15 second timeout
    def test_differences_at_each_level_with_timeout(self):
        """Test differences at each level with performance safeguards."""
        self.setUp()

        base_org = self.create_test_organization("base")

        test_variations = [
            ("level1_diff", "Level 1 (Organization)"),
            ("level2_diff", "Level 2 (Division)"),
            ("level3_diff", "Level 3 (Department)"),
            ("level4_diff", "Level 4 (Team)"),
            ("level5_diff", "Level 5 (Project)"),
            ("level6_diff", "Level 6 (Task)"),
        ]

        for variation, level_name in test_variations:
            start_time = time.time()

            modified_org = self.create_test_organization(variation)

            # Clear non-matches from previous test
            self.evaluator.clear_non_match_documents()

            result = self.evaluator.evaluate(base_org, modified_org)

            duration = time.time() - start_time

            # Performance assertions
            assert duration < 3.0, f"{level_name} took too long: {duration:.2f}s"

            # Functionality assertions
            assert 0.0 < result["overall"]["anls_score"] < 1.0, (
                f"{level_name}: Expected partial match, got {result['overall']['anls_score']}"
            )

            # Should have non-matches documented
            assert len(result["non_matches"]) > 0, f"{level_name}: Expected non-matches"

            print(
                f"✅ {level_name}: {duration:.3f}s, Score: {result['overall']['anls_score']:.3f}"
            )

    @with_timeout(10)  # 10 second timeout
    def test_deep_field_paths_with_timeout(self):
        """Test 6-level deep field path generation with performance safeguard."""
        self.setUp()

        start_time = time.time()

        base_org = self.create_test_organization("base")
        level6_diff_org = self.create_test_organization("level6_diff")

        result = self.evaluator.evaluate(base_org, level6_diff_org)

        duration = time.time() - start_time

        # Performance assertion
        assert duration < 5.0, f"Deep field path test took too long: {duration:.2f}s"

        # Should have non-matches with deep field paths
        non_matches = result["non_matches"]
        assert len(non_matches) > 0, "Expected non-matches to be documented"

        # Look for 6-level deep field paths
        deep_paths = [nm for nm in non_matches if nm["field_path"].count(".") >= 5]

        # Should find paths like: main_division.main_department.primary_team.current_project.main_task.name
        expected_deep_pattern = (
            "main_division.main_department.primary_team.current_project.main_task"
        )

        found_deep_paths = [
            nm["field_path"]
            for nm in non_matches
            if expected_deep_pattern in nm["field_path"]
        ]

        assert len(found_deep_paths) > 0, (
            f"Expected deep paths with pattern {expected_deep_pattern}"
        )

        print(f"✅ Deep field paths test passed: {duration:.3f}s")
        print(f"   Found {len(deep_paths)} deep paths")
        print(f"   Sample: {found_deep_paths[0] if found_deep_paths else 'None'}")

    @with_timeout(10)  # 10 second timeout
    def test_confusion_matrix_6_levels_with_timeout(self):
        """Test confusion matrix aggregation across 6 levels with performance safeguard."""
        self.setUp()

        start_time = time.time()

        base_org = self.create_test_organization("base")
        diff_org = self.create_test_organization("level3_diff")

        result = self.evaluator.evaluate(base_org, diff_org)

        duration = time.time() - start_time

        # Performance assertion
        assert duration < 5.0, f"Confusion matrix test took too long: {duration:.2f}s"

        # Should have confusion matrix data
        assert "confusion_matrix" in result, "Expected confusion matrix in result"
        cm = result["confusion_matrix"]

        # Should have overall aggregated metrics
        overall = cm["overall"]
        assert overall["tp"] + overall["fp"] + overall["tn"] + overall["fn"] > 0, (
            "Expected non-zero confusion matrix counts"
        )

        # Should have field-level metrics for nested structures
        fields = cm["fields"]

        # Check that we have the main nested field
        assert "main_division" in fields, (
            f"Expected main_division field, got: {list(fields.keys())}"
        )

        # Note: Nested dot-notation fields (like main_division.main_department.name)
        # are a potential future enhancement. Current implementation tracks top-level fields.
        nested_fields = [field for field in fields.keys() if "." in field]

        print(f"✅ Confusion matrix test passed: {duration:.3f}s")
        print(f"   Nested field entries: {len(nested_fields)} (dot-notation fields)")
        print(f"   Top-level fields: {list(fields.keys())}")
        print(f"   Overall: TP={overall['tp']}, FP={overall['fp']}, FN={overall['fn']}")

    @with_timeout(20)  # 20 second timeout for performance test
    def test_performance_stress_test_with_timeout(self):
        """Stress test with multiple evaluations to ensure no performance degradation."""
        self.setUp()

        start_time = time.time()

        # Create multiple organizations for stress testing
        organizations = []
        for i in range(5):  # 5 organizations (reasonable for stress test)
            org = self.create_test_organization("base")
            organizations.append(org)

        # Evaluate each against a modified version
        results = []
        for i, org in enumerate(organizations):
            modified = self.create_test_organization(
                "level6_diff" if i % 2 == 0 else "level3_diff"
            )
            result = self.evaluator.evaluate(org, modified)
            results.append(result)

        duration = time.time() - start_time

        # Performance assertions
        assert duration < 15.0, f"Stress test took too long: {duration:.2f}s"

        # Verify all evaluations completed successfully
        assert len(results) == 5, "Not all evaluations completed"
        assert all("overall" in result for result in results), "Some evaluations failed"

        avg_time = duration / 5

        print(f"✅ Stress test passed: {duration:.2f}s total")
        print(f"   Average per evaluation: {avg_time:.3f}s")
        print(f"   All {len(results)} evaluations completed successfully")


def test_integration():
    """Quick integration test."""
    test_instance = TestRealisticDeepNesting()

    # Run key tests
    test_instance.test_perfect_match_6_levels_with_timeout()
    test_instance.test_deep_field_paths_with_timeout()

    print("✅ Realistic deep nesting integration test passed!")


if __name__ == "__main__":
    test_integration()
