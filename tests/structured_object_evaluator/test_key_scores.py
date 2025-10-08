"""Tests for the key_scores utility module."""

import unittest
from typing import Dict, List, Tuple

from stickler.structured_object_evaluator.utils.key_scores import (
    ScoreNode,
    construct_nested_dict,
    merge_and_calculate_mean,
)


class TestScoreNode(unittest.TestCase):
    """Test cases for the ScoreNode class."""

    def test_score_node_initialization(self):
        """Test initializing a ScoreNode."""
        node = ScoreNode(name="test", score=0.5)
        self.assertEqual(node.name, "test")
        self.assertEqual(node.score, 0.5)
        self.assertEqual(node.children, {})

        # Test with children
        child = ScoreNode(name="child", score=0.8)
        node = ScoreNode(name="parent", score=0.5, children={"child": child})
        self.assertEqual(node.name, "parent")
        self.assertEqual(node.score, 0.5)
        self.assertEqual(len(node.children), 1)
        self.assertEqual(node.children["child"].name, "child")
        self.assertEqual(node.children["child"].score, 0.8)


class TestConstructNestedDict(unittest.TestCase):
    """Test cases for the construct_nested_dict function."""

    def test_empty_list(self):
        """Test with an empty list."""
        result = construct_nested_dict([])
        self.assertEqual(result, {})

    def test_single_level(self):
        """Test with single-level keys."""
        input_list = [{("a",): 1.0}, {("b",): 2.0}, {("c",): 3.0}]
        result = construct_nested_dict(input_list)

        self.assertEqual(len(result), 3)
        self.assertEqual(result["a"].name, "a")
        self.assertEqual(result["a"].score, 1.0)
        self.assertEqual(result["b"].name, "b")
        self.assertEqual(result["b"].score, 2.0)
        self.assertEqual(result["c"].name, "c")
        self.assertEqual(result["c"].score, 3.0)

    def test_nested_keys(self):
        """Test with nested keys."""
        input_list = [{("a", "b", "c"): 1.0}, {("a", "b", "d"): 2.0}, {("a", "e"): 3.0}]
        result = construct_nested_dict(input_list)

        self.assertEqual(len(result), 1)
        self.assertEqual(result["a"].name, "a")
        self.assertIsNone(result["a"].score)  # No direct score for "a"

        # Check "a.b" node
        self.assertEqual(result["a"].children["b"].name, "b")
        self.assertIsNone(result["a"].children["b"].score)  # No direct score for "a.b"

        # Check "a.b.c" and "a.b.d" nodes
        self.assertEqual(result["a"].children["b"].children["c"].name, "c")
        self.assertEqual(result["a"].children["b"].children["c"].score, 1.0)
        self.assertEqual(result["a"].children["b"].children["d"].name, "d")
        self.assertEqual(result["a"].children["b"].children["d"].score, 2.0)

        # Check "a.e" node
        self.assertEqual(result["a"].children["e"].name, "e")
        self.assertEqual(result["a"].children["e"].score, 3.0)

    def test_duplicate_keys(self):
        """Test with duplicate keys (last value should be used)."""
        input_list = [
            {("a",): 1.0},
            {("a",): 2.0},  # This should override the previous value
        ]
        result = construct_nested_dict(input_list)

        self.assertEqual(len(result), 1)
        self.assertEqual(result["a"].name, "a")
        self.assertEqual(result["a"].score, 2.0)  # Last value should be used


class TestMergeAndCalculateMean(unittest.TestCase):
    """Test cases for the merge_and_calculate_mean function."""

    def test_empty_list(self):
        """Test with an empty list."""
        result = merge_and_calculate_mean([])
        self.assertEqual(result, [])

    def test_single_dict(self):
        """Test with a single dictionary."""
        input_list = [{("a",): 1.0, ("b",): 2.0}]
        result = merge_and_calculate_mean(input_list)

        # Convert result to a more easily testable format
        result_dict = {}
        for item in result:
            for k, v in item.items():
                result_dict[k] = v

        self.assertEqual(len(result_dict), 2)
        self.assertEqual(result_dict[("a",)], 1.0)
        self.assertEqual(result_dict[("b",)], 2.0)

    def test_multiple_dicts_with_common_keys(self):
        """Test with multiple dictionaries with common keys."""
        input_list = [{("a",): 1.0, ("b",): 2.0}, {("a",): 3.0, ("c",): 4.0}]
        result = merge_and_calculate_mean(input_list)

        # Convert result to a more easily testable format
        result_dict = {}
        for item in result:
            for k, v in item.items():
                result_dict[k] = v

        self.assertEqual(len(result_dict), 3)
        self.assertEqual(result_dict[("a",)], 2.0)  # Mean of 1.0 and 3.0
        self.assertEqual(result_dict[("b",)], 2.0)
        self.assertEqual(result_dict[("c",)], 4.0)

    def test_multiple_occurrences(self):
        """Test with a key appearing in more than two dictionaries."""
        input_list = [{("a",): 1.0}, {("a",): 2.0}, {("a",): 3.0}]
        result = merge_and_calculate_mean(input_list)

        # Convert result to a more easily testable format
        result_dict = {}
        for item in result:
            for k, v in item.items():
                result_dict[k] = v

        self.assertEqual(len(result_dict), 1)
        self.assertEqual(result_dict[("a",)], 2.0)  # Mean of 1.0, 2.0, and 3.0


if __name__ == "__main__":
    unittest.main()
