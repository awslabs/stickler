"""Tests for the key_scores utility module."""

from stickler.structured_object_evaluator.utils.key_scores import (
    ScoreNode,
    construct_nested_dict,
    merge_and_calculate_mean,
)


class TestScoreNode:
    """Test cases for the ScoreNode class."""

    def test_score_node_initialization(self):
        """Test initializing a ScoreNode."""
        node = ScoreNode(name="test", score=0.5)
        assert node.name == "test"
        assert node.score == 0.5
        assert node.children == {}

        # Test with children
        child = ScoreNode(name="child", score=0.8)
        node = ScoreNode(name="parent", score=0.5, children={"child": child})
        assert node.name == "parent"
        assert node.score == 0.5
        assert len(node.children) == 1
        assert node.children["child"].name == "child"
        assert node.children["child"].score == 0.8


class TestConstructNestedDict:
    """Test cases for the construct_nested_dict function."""

    def test_empty_list(self):
        """Test with an empty list."""
        result = construct_nested_dict([])
        assert result == {}

    def test_single_level(self):
        """Test with single-level keys."""
        input_list = [{("a",): 1.0}, {("b",): 2.0}, {("c",): 3.0}]
        result = construct_nested_dict(input_list)

        assert len(result) == 3
        assert result["a"].name == "a"
        assert result["a"].score == 1.0
        assert result["b"].name == "b"
        assert result["b"].score == 2.0
        assert result["c"].name == "c"
        assert result["c"].score == 3.0

    def test_nested_keys(self):
        """Test with nested keys."""
        input_list = [{("a", "b", "c"): 1.0}, {("a", "b", "d"): 2.0}, {("a", "e"): 3.0}]
        result = construct_nested_dict(input_list)

        assert len(result) == 1
        assert result["a"].name == "a"
        assert result["a"].score is None  # No direct score for "a"

        # Check "a.b" node
        assert result["a"].children["b"].name == "b"
        assert result["a"].children["b"].score is None  # No direct score for "a.b"

        # Check "a.b.c" and "a.b.d" nodes
        assert result["a"].children["b"].children["c"].name == "c"
        assert result["a"].children["b"].children["c"].score == 1.0
        assert result["a"].children["b"].children["d"].name == "d"
        assert result["a"].children["b"].children["d"].score == 2.0

        # Check "a.e" node
        assert result["a"].children["e"].name == "e"
        assert result["a"].children["e"].score == 3.0

    def test_duplicate_keys(self):
        """Test with duplicate keys (last value should be used)."""
        input_list = [
            {("a",): 1.0},
            {("a",): 2.0},  # This should override the previous value
        ]
        result = construct_nested_dict(input_list)

        assert len(result) == 1
        assert result["a"].name == "a"
        assert result["a"].score == 2.0  # Last value should be used


class TestMergeAndCalculateMean:
    """Test cases for the merge_and_calculate_mean function."""

    def test_empty_list(self):
        """Test with an empty list."""
        result = merge_and_calculate_mean([])
        assert result == []

    def test_single_dict(self):
        """Test with a single dictionary."""
        input_list = [{("a",): 1.0, ("b",): 2.0}]
        result = merge_and_calculate_mean(input_list)

        # Convert result to a more easily testable format
        result_dict = {}
        for item in result:
            for k, v in item.items():
                result_dict[k] = v

        assert len(result_dict) == 2
        assert result_dict[("a",)] == 1.0
        assert result_dict[("b",)] == 2.0

    def test_multiple_dicts_with_common_keys(self):
        """Test with multiple dictionaries with common keys."""
        input_list = [{("a",): 1.0, ("b",): 2.0}, {("a",): 3.0, ("c",): 4.0}]
        result = merge_and_calculate_mean(input_list)

        # Convert result to a more easily testable format
        result_dict = {}
        for item in result:
            for k, v in item.items():
                result_dict[k] = v

        assert len(result_dict) == 3
        assert result_dict[("a",)] == 2.0  # Mean of 1.0 and 3.0
        assert result_dict[("b",)] == 2.0
        assert result_dict[("c",)] == 4.0

    def test_multiple_occurrences(self):
        """Test with a key appearing in more than two dictionaries."""
        input_list = [{("a",): 1.0}, {("a",): 2.0}, {("a",): 3.0}]
        result = merge_and_calculate_mean(input_list)

        # Convert result to a more easily testable format
        result_dict = {}
        for item in result:
            for k, v in item.items():
                result_dict[k] = v

        assert len(result_dict) == 1
        assert result_dict[("a",)] == 2.0  # Mean of 1.0, 2.0, and 3.0
