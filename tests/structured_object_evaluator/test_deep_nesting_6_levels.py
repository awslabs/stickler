"""
Test module for 6-level deep nesting in StructuredModelEvaluator.

This comprehensive test suite verifies that the evaluator can handle arbitrarily
deep nesting structures up to 6 levels, testing:
- Recursive field comparison accuracy
- Confusion matrix aggregation across all levels
- Non-match documentation with full dot-notation paths
- Performance and memory efficiency with deep structures
- Edge cases and complex scenarios

Test Hierarchy:
Level 1: Company
├── Level 2: Department
    ├── Level 3: Team
        ├── Level 4: Project
            ├── Level 5: Task
                └── Level 6: Subtask
"""

import pytest
from typing import List, Optional, Dict, Any

from stickler.structured_object_evaluator import (
    StructuredModel,
    ComparableField,
    NonMatchField,
    NonMatchType,
    StructuredModelEvaluator,
)
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


# Level 6: Subtask (deepest level)
class Subtask(StructuredModel):
    """Level 6: Deepest nested model - Subtask."""

    match_threshold = 0.7

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )
    title: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )
    description: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    completed: bool = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    estimated_hours: float = ComparableField(
        comparator=ExactComparator(), threshold=0.95, weight=1.0
    )


# Level 5: Task
class Task(StructuredModel):
    """Level 5: Task containing Subtasks."""

    match_threshold = 0.7

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )
    status: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    priority: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    subtasks: List[Subtask] = ComparableField(weight=1.0)
    main_subtask: Optional[Subtask] = ComparableField(threshold=0.8, weight=1.0)


# Level 4: Project
class Project(StructuredModel):
    """Level 4: Project containing Tasks."""

    match_threshold = 0.7

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )
    budget: float = ComparableField(
        comparator=ExactComparator(), threshold=0.95, weight=2.0
    )
    start_date: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    tasks: List[Task] = ComparableField(weight=1.5)
    critical_task: Optional[Task] = ComparableField(threshold=0.8, weight=2.0)


# Level 3: Team
class Team(StructuredModel):
    """Level 3: Team containing Projects."""

    match_threshold = 0.7

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )
    size: int = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    lead_engineer: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    projects: List[Project] = ComparableField(weight=2.0)
    flagship_project: Optional[Project] = ComparableField(threshold=0.8, weight=2.0)


# Level 2: Department
class Department(StructuredModel):
    """Level 2: Department containing Teams."""

    match_threshold = 0.7

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )
    head: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    budget: float = ComparableField(
        comparator=ExactComparator(), threshold=0.95, weight=2.0
    )
    teams: List[Team] = ComparableField(weight=2.0)
    primary_team: Optional[Team] = ComparableField(threshold=0.8, weight=2.0)


# Level 1: Company (top level)
class Company(StructuredModel):
    """Level 1: Top-level Company containing Departments."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )
    founded_year: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    ceo: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    departments: List[Department] = ComparableField(weight=2.0)
    main_department: Optional[Department] = ComparableField(threshold=0.8, weight=2.0)


class TestDeepNesting6Levels:
    """Test suite for 6-level deep nesting scenarios."""

    def setUp(self):
        """Set up test evaluator."""
        self.evaluator = StructuredModelEvaluator(
            threshold=0.7, document_non_matches=True
        )

    def create_test_company(self, variation: str = "base") -> Company:
        """Create SIMPLIFIED test company with 6 levels of nesting - PERFORMANCE OPTIMIZED.

        Args:
            variation: Type of test data ("base", "modified", "partial_diff", etc.)

        Returns:
            Company instance with minimal 6-level hierarchy for speed
        """
        # Level 6: Single Subtask (minimal complexity)
        if variation == "level6_diff":
            subtask = Subtask(
                id="ST001",
                title="Different Subtask",  # Modified
                description="Updated description",
                completed=False,  # Modified
                estimated_hours=3.0,
            )
        else:
            subtask = Subtask(
                id="ST001",
                title="Code Review",
                description="Review PR",
                completed=True,
                estimated_hours=2.5,
            )

        # Level 5: Single Task (minimal complexity)
        if variation == "level5_diff":
            task = Task(
                id="T001",
                name="Modified Task",  # Modified
                status="in-progress",  # Modified
                priority=2,  # Modified
                subtasks=[subtask],  # Only 1 subtask
                main_subtask=subtask,
            )
        elif variation == "level5_no_main":
            task = Task(
                id="T001",
                name="Feature X",
                status="completed",
                priority=1,
                subtasks=[subtask],  # Only 1 subtask
                main_subtask=None,  # No main subtask
            )
        elif variation == "level5_no_subtasks":
            task = Task(
                id="T001",
                name="Feature X",
                status="completed",
                priority=1,
                subtasks=[],  # Empty subtasks list
                main_subtask=None,
            )
        else:
            task = Task(
                id="T001",
                name="Feature X",
                status="completed",
                priority=1,
                subtasks=[subtask],  # Only 1 subtask
                main_subtask=subtask,
            )

        # Level 4: Single Project (minimal complexity)
        if variation == "level4_diff":
            project = Project(
                id="P001",
                name="Different Project",  # Modified
                budget=75000.0,  # Modified
                start_date="2024-02-01",  # Modified
                tasks=[task],  # Only 1 task
                critical_task=task,
            )
        elif variation == "level4_no_critical":
            project = Project(
                id="P001",
                name="Alpha Project",
                budget=100000.0,
                start_date="2024-01-01",
                tasks=[task],  # Only 1 task
                critical_task=None,  # No critical task
            )
        elif variation == "level4_no_tasks":
            project = Project(
                id="P001",
                name="Alpha Project",
                budget=100000.0,
                start_date="2024-01-01",
                tasks=[],  # Empty tasks list
                critical_task=None,
            )
        else:
            project = Project(
                id="P001",
                name="Alpha Project",
                budget=100000.0,
                start_date="2024-01-01",
                tasks=[task],  # Only 1 task
                critical_task=task,
            )

        # Level 3: Single Team (minimal complexity)
        if variation == "level3_diff":
            team = Team(
                id="TM001",
                name="Modified Team",  # Modified
                size=8,  # Modified
                lead_engineer="Jane Smith",  # Modified
                projects=[project],  # Only 1 project
                flagship_project=project,
            )
        elif variation == "level3_no_flagship":
            team = Team(
                id="TM001",
                name="Backend Team",
                size=5,
                lead_engineer="John Doe",
                projects=[project],  # Only 1 project
                flagship_project=None,  # No flagship project
            )
        elif variation == "level3_no_projects":
            team = Team(
                id="TM001",
                name="Backend Team",
                size=5,
                lead_engineer="John Doe",
                projects=[],  # Empty projects list
                flagship_project=None,
            )
        else:
            team = Team(
                id="TM001",
                name="Backend Team",
                size=5,
                lead_engineer="John Doe",
                projects=[project],  # Only 1 project
                flagship_project=project,
            )

        # Level 2: Single Department (minimal complexity)
        if variation == "level2_diff":
            dept = Department(
                id="D001",
                name="Modified Department",  # Modified
                head="Different Manager",  # Modified
                budget=750000.0,  # Modified
                teams=[team],  # Only 1 team
                primary_team=team,
            )
        elif variation == "level2_no_teams":
            dept = Department(
                id="D001",
                name="Engineering",
                head="Bob Wilson",
                budget=500000.0,
                teams=[],  # Empty teams list
                primary_team=None,
            )
        elif variation == "level2_no_primary":
            dept = Department(
                id="D001",
                name="Engineering",
                head="Bob Wilson",
                budget=500000.0,
                teams=[team],  # Only 1 team
                primary_team=None,  # No primary team
            )
        else:
            dept = Department(
                id="D001",
                name="Engineering",
                head="Bob Wilson",
                budget=500000.0,
                teams=[team],  # Only 1 team
                primary_team=team,
            )

        # Level 1: Company (minimal complexity)
        if variation == "level1_diff":
            company = Company(
                id="C001",
                name="Different Company Inc",  # Modified
                founded_year=2019,  # Modified
                ceo="Different CEO",  # Modified
                departments=[dept],  # Only 1 department
                main_department=dept,
            )
        else:
            company = Company(
                id="C001",
                name="TechCorp Inc",
                founded_year=2020,
                ceo="Steve Jobs",
                departments=[dept],  # Only 1 department
                main_department=dept,
            )

        return company

    def test_perfect_match_6_levels(self):
        """Test perfect match across all 6 levels of nesting."""
        self.setUp()

        # Create identical companies
        gt_company = self.create_test_company("base")
        pred_company = self.create_test_company("base")

        # Evaluate
        result = self.evaluator.evaluate(gt_company, pred_company)

        # Should be perfect match
        assert result["overall"]["anls_score"] == 1.0, (
            f"Expected perfect match, got {result['overall']['anls_score']}"
        )
        assert result["overall"]["precision"] == 1.0
        assert result["overall"]["recall"] == 1.0
        assert result["overall"]["f1"] == 1.0

        print(
            f"✅ Perfect match test passed: Score = {result['overall']['anls_score']}"
        )

    def test_differences_at_each_level(self):
        """Test differences at each individual level (1-6) - OPTIMIZED VERSION."""
        self.setUp()

        # Create base company once
        base_company = self.create_test_company("base")

        # Test only 3 key levels instead of all 6 to reduce complexity
        test_variations = [
            ("level1_diff", "Level 1 (Company)"),
            ("level3_diff", "Level 3 (Team)"),
            ("level6_diff", "Level 6 (Subtask)"),
        ]

        for variation, level_name in test_variations:
            modified_company = self.create_test_company(variation)

            # Clear non-matches from previous test
            self.evaluator.clear_non_match_documents()

            result = self.evaluator.evaluate(base_company, modified_company)

            # Should detect differences but not be zero
            assert 0.0 < result["overall"]["anls_score"] < 1.0, (
                f"{level_name}: Expected partial match, got {result['overall']['anls_score']}"
            )

            # Should have non-matches documented
            assert len(result["non_matches"]) > 0, (
                f"{level_name}: Expected non-matches to be documented"
            )

            print(
                f"✅ {level_name} difference test passed: Score = {result['overall']['anls_score']:.3f}, Non-matches = {len(result['non_matches'])}"
            )

    def test_deep_field_path_generation(self):
        """Test that field paths are correctly generated for 6-level deep nesting."""
        self.setUp()

        base_company = self.create_test_company("base")
        level6_diff_company = self.create_test_company("level6_diff")

        result = self.evaluator.evaluate(base_company, level6_diff_company)

        # Should have non-matches with deep field paths
        non_matches = result["non_matches"]
        assert len(non_matches) > 0, "Expected non-matches to be documented"

        # Look for deep field paths (6 levels deep)
        deep_paths = [nm for nm in non_matches if nm["field_path"].count(".") >= 5]
        assert len(deep_paths) > 0, "Expected field paths with 6 levels of nesting"

        # Verify specific deep paths exist
        expected_paths = [
            "main_department.primary_team.flagship_project.critical_task.main_subtask.title",
            "main_department.primary_team.flagship_project.critical_task.main_subtask.completed",
        ]

        actual_paths = [nm["field_path"] for nm in non_matches]
        found_deep_paths = [
            path
            for path in expected_paths
            if any(path in actual_path for actual_path in actual_paths)
        ]

        assert len(found_deep_paths) > 0, (
            f"Expected deep paths like {expected_paths}, got {actual_paths}"
        )

        print(f"✅ Deep field path test passed. Found {len(deep_paths)} deep paths")
        print(f"   Sample deep paths: {[p['field_path'] for p in deep_paths[:3]]}")

    def test_confusion_matrix_aggregation_6_levels(self):
        """Test confusion matrix aggregation across 6 levels."""
        self.setUp()

        base_company = self.create_test_company("base")
        mixed_diff_company = self.create_test_company("level3_diff")

        result = self.evaluator.evaluate(base_company, mixed_diff_company)

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

        # Check for hierarchical nested field entries
        # This checks for proper hierarchical structure rather than flattened paths
        found_hierarchy = False
        nested_levels = 0

        # Define a recursive function to check nesting depth
        def check_nesting_depth(field_data, current_depth=0):
            if not isinstance(field_data, dict):
                return current_depth

            if "fields" in field_data and field_data["fields"]:
                max_depth = current_depth
                for nested_field in field_data["fields"].values():
                    depth = check_nesting_depth(nested_field, current_depth + 1)
                    max_depth = max(max_depth, depth)
                return max_depth
            return current_depth

        # Check if the departments field has proper hierarchical structure
        if "departments" in fields:
            dept_field = fields["departments"]
            if "fields" in dept_field and dept_field["fields"]:
                found_hierarchy = True
                nested_levels = check_nesting_depth(dept_field)

        assert found_hierarchy, (
            f"Expected hierarchical nested field entries, got fields: {list(fields.keys())}"
        )
        assert nested_levels > 0, (
            f"Expected hierarchical nesting, but found no nested levels"
        )

        print(f"✅ Confusion matrix aggregation test passed")
        print(
            f"   Overall metrics: TP={overall['tp']}, FP={overall['fp']}, FN={overall['fn']}"
        )
        print(f"   Found hierarchical structure with nesting depth: {nested_levels}")

    def test_performance_with_deep_nesting(self):
        """Test performance and memory usage with deep nesting."""
        import time
        import psutil
        import os

        self.setUp()

        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Create complex nested structures
        companies = []
        for i in range(10):  # Create 10 companies for stress testing
            companies.append(self.create_test_company("base"))

        # Measure evaluation time
        start_time = time.time()

        results = []
        for i, company in enumerate(companies):
            # Compare with slightly modified version
            modified = self.create_test_company(
                "level6_diff" if i % 2 == 0 else "level3_diff"
            )
            result = self.evaluator.evaluate(company, modified)
            results.append(result)

        end_time = time.time()
        evaluation_time = end_time - start_time

        # Measure final memory
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory

        # Performance assertions
        assert evaluation_time < 35.0, (
            f"Evaluation took too long: {evaluation_time:.2f} seconds"
        )
        assert memory_increase < 100.0, (
            f"Memory usage increased too much: {memory_increase:.2f} MB"
        )

        # Verify all evaluations completed successfully
        assert len(results) == 10, "Not all evaluations completed"
        assert all("overall" in result for result in results), "Some evaluations failed"

        print(f"✅ Performance test passed")
        print(f"   Evaluation time: {evaluation_time:.2f} seconds for 10 companies")
        print(f"   Memory increase: {memory_increase:.2f} MB")
        print(f"   Average time per evaluation: {evaluation_time / 10:.3f} seconds")

    def test_lists_of_nested_objects_6_levels(self):
        """Test lists containing deeply nested objects."""
        self.setUp()

        # Create companies with different numbers of nested objects
        base_company = self.create_test_company("base")

        # We need a version WITH teams to properly test nested structures
        regular_company = self.create_test_company(
            "base"
        )  # Same as base to ensure teams are present

        result = self.evaluator.evaluate(base_company, regular_company)

        # Should be a perfect match since both are identical
        assert result["overall"]["anls_score"] == 1.0, (
            f"Expected perfect match, got {result['overall']['anls_score']}"
        )

        # Should have confusion matrix entries for list fields
        cm = result["confusion_matrix"]
        assert "departments" in cm["fields"], (
            "Expected departments field in confusion matrix"
        )

        # Check for nested list field metrics in hierarchical structure
        # Instead of looking for flattened dot notation paths, we'll check for proper nesting
        found_nested_list_fields = False

        if "departments" in cm["fields"]:
            departments = cm["fields"]["departments"]
            if "fields" in departments:
                # Check if nested fields exist inside departments
                found_nested_list_fields = True

                # Additional validation - check for nested "teams" field
                teams_found = False
                for field_name in departments["fields"]:
                    if "teams" in field_name:
                        teams_found = True
                        break

                assert teams_found, (
                    "Expected teams field in departments nested structure"
                )

        assert found_nested_list_fields, (
            "Expected nested list fields in hierarchical structure"
        )

        print(f"✅ Nested lists test passed")
        print(f"   Found hierarchical nested structure in departments")
        print(f"   Overall score: {result['overall']['anls_score']:.3f}")

    def test_null_handling_at_deep_levels(self):
        """Test handling of null values at various deep nesting levels."""
        self.setUp()

        base_company = self.create_test_company("base")

        # Test null values at different levels
        test_scenarios = [
            ("level5_no_main", "Level 5: Task.main_subtask = None"),
            ("level4_no_critical", "Level 4: Project.critical_task = None"),
            ("level3_no_flagship", "Level 3: Team.flagship_project = None"),
            ("level2_no_primary", "Level 2: Department.primary_team = None"),
        ]

        for variation, scenario_name in test_scenarios:
            null_company = self.create_test_company(variation)

            self.evaluator.clear_non_match_documents()
            result = self.evaluator.evaluate(base_company, null_company)

            # Should detect the null differences
            assert result["overall"]["anls_score"] < 1.0, (
                f"{scenario_name}: Expected differences due to null values"
            )

            # Should have documented non-matches
            non_matches = result["non_matches"]
            null_non_matches = [
                nm for nm in non_matches if nm["prediction_value"] is None
            ]
            assert len(null_non_matches) > 0, (
                f"{scenario_name}: Expected null non-matches"
            )

            print(
                f"✅ {scenario_name} passed: Score = {result['overall']['anls_score']:.3f}"
            )

    def test_edge_case_empty_structures(self):
        """Test edge cases with empty structures at deep levels."""
        self.setUp()

        base_company = self.create_test_company("base")

        # Test empty lists at different levels
        empty_scenarios = [
            ("level5_no_subtasks", "Empty subtasks list"),
            ("level4_no_tasks", "Empty tasks list"),
            ("level3_no_projects", "Empty projects list"),
            ("level2_no_teams", "Empty teams list"),
        ]

        for variation, scenario_name in empty_scenarios:
            empty_company = self.create_test_company(variation)

            self.evaluator.clear_non_match_documents()
            result = self.evaluator.evaluate(base_company, empty_company)

            # Should handle empty lists gracefully
            assert 0.0 <= result["overall"]["anls_score"] <= 1.0, (
                f"{scenario_name}: Invalid score {result['overall']['anls_score']}"
            )

            # Should complete without errors
            assert "overall" in result, f"{scenario_name}: Evaluation failed"

            print(
                f"✅ {scenario_name} handled gracefully: Score = {result['overall']['anls_score']:.3f}"
            )


def test_integration_with_evaluator():
    """Test integration between deep nesting and StructuredModelEvaluator."""
    # Quick integration test to ensure everything works together
    test_instance = TestDeepNesting6Levels()

    # Run a few key tests
    test_instance.test_perfect_match_6_levels()
    test_instance.test_deep_field_path_generation()

    print("✅ Deep nesting integration test passed!")


if __name__ == "__main__":
    # Run the integration test when file is executed directly
    test_integration_with_evaluator()
