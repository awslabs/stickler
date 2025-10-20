"""Tests for the tree-based structured object evaluation."""

import unittest

from stickler.structured_object_evaluator.trees import (
    ANLSTree,
    ANLSDict,
    ANLSLeaf,
    ANLSList,
    ANLSNone,
    ANLSTuple,
)
from stickler.comparators.levenshtein import LevenshteinComparator


class TestANLSTree(unittest.TestCase):
    """Test cases for the ANLSTree base class and factory methods."""

    def test_make_tree_leaf_types(self):
        """Test creating leaf nodes for primitive types."""
        # Test string
        tree = ANLSTree.make_tree("hello", is_gt=True)
        self.assertIsInstance(tree, ANLSLeaf)
        self.assertEqual(tree.obj, "hello")

        # Test number
        tree = ANLSTree.make_tree(42, is_gt=True)
        self.assertIsInstance(tree, ANLSLeaf)
        self.assertEqual(tree.obj, 42)

        # Test boolean
        tree = ANLSTree.make_tree(True, is_gt=True)
        self.assertIsInstance(tree, ANLSLeaf)
        self.assertEqual(tree.obj, True)

        # Test None
        tree = ANLSTree.make_tree(None, is_gt=True)
        self.assertIsInstance(tree, ANLSNone)
        self.assertEqual(tree.obj, None)

    def test_make_tree_container_types(self):
        """Test creating tree nodes for container types."""
        # Test list
        tree = ANLSTree.make_tree(["a", "b", "c"], is_gt=True)
        self.assertIsInstance(tree, ANLSList)
        self.assertEqual(tree.obj, ["a", "b", "c"])

        # Test dict
        tree = ANLSTree.make_tree({"a": 1, "b": 2}, is_gt=True)
        self.assertIsInstance(tree, ANLSDict)
        self.assertEqual(tree.obj, {"a": 1, "b": 2})

        # Test tuple (only allowed for ground truth)
        tree = ANLSTree.make_tree(("a", "b"), is_gt=True)
        self.assertIsInstance(tree, ANLSTuple)
        self.assertEqual(tree.obj, ("a", "b"))

    def test_tuple_not_allowed_in_prediction(self):
        """Test that tuples are not allowed in predictions."""
        with self.assertRaises(ValueError):
            ANLSTree.make_tree(("a", "b"), is_gt=False)

    def test_empty_tuple_not_allowed(self):
        """Test that empty tuples are not allowed."""
        with self.assertRaises(ValueError):
            ANLSTree.make_tree((), is_gt=True)


class TestANLSLeaf(unittest.TestCase):
    """Test cases for the ANLSLeaf class."""

    def test_leaf_comparison_exact_match(self):
        """Test comparing leaf nodes with exact matches."""
        leaf1 = ANLSLeaf("hello", comparator=LevenshteinComparator())
        leaf2 = ANLSLeaf("hello", comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = leaf1.nls_list(leaf2, (), [])
        self.assertEqual(nls_list, [1.0])
        self.assertEqual(closest_gt, "hello")

    def test_leaf_comparison_similar(self):
        """Test comparing leaf nodes with similar values."""
        leaf1 = ANLSLeaf("hello", comparator=LevenshteinComparator())
        leaf2 = ANLSLeaf("helo", comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = leaf1.nls_list(leaf2, (), [])
        # With default threshold of 0.5, this should pass
        self.assertGreater(nls_list[0], 0.0)
        self.assertEqual(closest_gt, "hello")

    def test_leaf_comparison_different(self):
        """Test comparing leaf nodes with different values."""
        leaf1 = ANLSLeaf("hello", comparator=LevenshteinComparator())
        leaf2 = ANLSLeaf("world", comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = leaf1.nls_list(leaf2, (), [])
        # With default threshold of 0.5, this should fail
        self.assertEqual(nls_list, [0.0])
        self.assertEqual(closest_gt, "hello")

    def test_leaf_comparison_type_mismatch(self):
        """Test comparing a leaf node with a non-leaf node."""
        leaf = ANLSLeaf("hello", comparator=LevenshteinComparator())
        list_node = ANLSList(["hello"], is_gt=True, comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = leaf.nls_list(list_node, (), [])
        self.assertEqual(nls_list, [0.0])
        self.assertEqual(closest_gt, "hello")


class TestANLSNone(unittest.TestCase):
    """Test cases for the ANLSNone class."""

    def test_none_comparison_exact_match(self):
        """Test comparing None nodes with exact matches."""
        none1 = ANLSNone(comparator=LevenshteinComparator())
        none2 = ANLSNone(comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = none1.nls_list(none2, (), [])
        self.assertEqual(nls_list, [1.0])
        self.assertEqual(closest_gt, None)

    def test_none_comparison_with_empty_containers(self):
        """Test comparing None with empty containers."""
        none = ANLSNone(comparator=LevenshteinComparator())

        # Empty list
        empty_list = ANLSList([], is_gt=False, comparator=LevenshteinComparator())
        nls_list, closest_gt, key_scores = none.nls_list(empty_list, (), [])
        self.assertEqual(nls_list, [1.0])

        # Empty dict
        empty_dict = ANLSDict({}, is_gt=False, comparator=LevenshteinComparator())
        nls_list, closest_gt, key_scores = none.nls_list(empty_dict, (), [])
        self.assertEqual(nls_list, [1.0])

        # Empty string
        empty_string = ANLSLeaf("", comparator=LevenshteinComparator())
        nls_list, closest_gt, key_scores = none.nls_list(empty_string, (), [])
        self.assertEqual(nls_list, [1.0])

    def test_none_comparison_with_non_empty(self):
        """Test comparing None with non-empty values."""
        none = ANLSNone(comparator=LevenshteinComparator())

        # Non-empty list
        non_empty_list = ANLSList(
            ["a"], is_gt=False, comparator=LevenshteinComparator()
        )
        nls_list, closest_gt, key_scores = none.nls_list(non_empty_list, (), [])
        self.assertEqual(nls_list, [0.0])

        # Non-empty string
        non_empty_string = ANLSLeaf("hello", comparator=LevenshteinComparator())
        nls_list, closest_gt, key_scores = none.nls_list(non_empty_string, (), [])
        self.assertEqual(nls_list, [0.0])


class TestANLSDict(unittest.TestCase):
    """Test cases for the ANLSDict class."""

    def test_dict_comparison_exact_match(self):
        """Test comparing dictionary nodes with exact matches."""
        dict1 = ANLSDict(
            {"a": 1, "b": "hello"}, is_gt=True, comparator=LevenshteinComparator()
        )
        dict2 = ANLSDict(
            {"a": 1, "b": "hello"}, is_gt=False, comparator=LevenshteinComparator()
        )

        nls_list, closest_gt, key_scores = dict1.nls_list(dict2, (), [])
        self.assertEqual(sum(nls_list) / len(nls_list), 1.0)
        self.assertEqual(closest_gt, {"a": 1, "b": "hello"})

    def test_dict_comparison_missing_key(self):
        """Test comparing dictionaries with a missing key."""
        dict1 = ANLSDict(
            {"a": 1, "b": "hello"}, is_gt=True, comparator=LevenshteinComparator()
        )
        dict2 = ANLSDict({"a": 1}, is_gt=False, comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = dict1.nls_list(dict2, (), [])
        # One key matches, one is missing
        self.assertEqual(len(nls_list), 2)
        self.assertEqual(nls_list.count(1.0), 1)  # One perfect match
        self.assertEqual(nls_list.count(0.0), 1)  # One missing key
        self.assertEqual(closest_gt, {"a": 1, "b": "hello"})

    def test_dict_comparison_extra_key(self):
        """Test comparing dictionaries with an extra key."""
        dict1 = ANLSDict({"a": 1}, is_gt=True, comparator=LevenshteinComparator())
        dict2 = ANLSDict(
            {"a": 1, "b": "hello"}, is_gt=False, comparator=LevenshteinComparator()
        )

        nls_list, closest_gt, key_scores = dict1.nls_list(dict2, (), [])
        # One key matches, one is extra
        self.assertEqual(len(nls_list), 2)
        self.assertEqual(nls_list.count(1.0), 1)  # One perfect match
        self.assertEqual(nls_list.count(0.0), 1)  # One extra key
        # The closest GT should include the extra key with None value
        self.assertEqual(closest_gt, {"a": 1, "b": None})

    def test_dict_comparison_different_values(self):
        """Test comparing dictionaries with different values."""
        dict1 = ANLSDict(
            {"a": 1, "b": "hello"}, is_gt=True, comparator=LevenshteinComparator()
        )
        dict2 = ANLSDict(
            {"a": 2, "b": "hello"}, is_gt=False, comparator=LevenshteinComparator()
        )

        nls_list, closest_gt, key_scores = dict1.nls_list(dict2, (), [])
        # One key matches, one has different value
        self.assertEqual(len(nls_list), 2)
        self.assertEqual(nls_list.count(1.0), 1)  # One perfect match
        self.assertEqual(nls_list.count(0.0), 1)  # One different value
        self.assertEqual(closest_gt, {"a": 1, "b": "hello"})


class TestANLSList(unittest.TestCase):
    """Test cases for the ANLSList class."""

    def test_list_comparison_exact_match(self):
        """Test comparing list nodes with exact matches."""
        list1 = ANLSList(
            ["a", "b", "c"], is_gt=True, comparator=LevenshteinComparator()
        )
        list2 = ANLSList(
            ["a", "b", "c"], is_gt=False, comparator=LevenshteinComparator()
        )

        nls_list, closest_gt, key_scores = list1.nls_list(list2, (), [])
        self.assertEqual(len(nls_list), 3)  # Three items
        self.assertEqual(nls_list.count(1.0), 3)  # All perfect matches
        self.assertEqual(closest_gt, ["a", "b", "c"])

    def test_list_comparison_permutation(self):
        """Test comparing lists with permuted elements."""
        list1 = ANLSList(
            ["a", "b", "c"], is_gt=True, comparator=LevenshteinComparator()
        )
        list2 = ANLSList(
            ["c", "a", "b"], is_gt=False, comparator=LevenshteinComparator()
        )

        nls_list, closest_gt, key_scores = list1.nls_list(list2, (), [])
        self.assertEqual(len(nls_list), 3)  # Three items
        self.assertEqual(nls_list.count(1.0), 3)  # All perfect matches
        # The closest GT should match the order of the prediction
        self.assertEqual(closest_gt, ["c", "a", "b"])

    def test_list_comparison_missing_element(self):
        """Test comparing lists with a missing element."""
        list1 = ANLSList(
            ["a", "b", "c"], is_gt=True, comparator=LevenshteinComparator()
        )
        list2 = ANLSList(["a", "b"], is_gt=False, comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = list1.nls_list(list2, (), [])
        self.assertEqual(len(nls_list), 2)  # Two matched items
        self.assertEqual(nls_list.count(1.0), 2)  # Two perfect matches
        # The closest GT should include the matched elements plus the missing one
        self.assertEqual(closest_gt, ["a", "b", "c"])

    def test_list_comparison_extra_element(self):
        """Test comparing lists with an extra element."""
        list1 = ANLSList(["a", "b"], is_gt=True, comparator=LevenshteinComparator())
        list2 = ANLSList(
            ["a", "b", "c"], is_gt=False, comparator=LevenshteinComparator()
        )

        nls_list, closest_gt, key_scores = list1.nls_list(list2, (), [])
        self.assertEqual(len(nls_list), 2)  # Two matched items
        self.assertEqual(nls_list.count(1.0), 2)  # Two perfect matches
        # The closest GT should include just the matched elements
        self.assertEqual(closest_gt, ["a", "b"])


class TestANLSTuple(unittest.TestCase):
    """Test cases for the ANLSTuple class."""

    def test_tuple_comparison_exact_match(self):
        """Test comparing tuple nodes with exact matches."""
        tuple_node = ANLSTuple(
            ("hello", "world"), is_gt=True, comparator=LevenshteinComparator()
        )
        leaf_node = ANLSLeaf("hello", comparator=LevenshteinComparator())

        nls_list, closest_gt, key_scores = tuple_node.nls_list(leaf_node, (), [])
        self.assertEqual(nls_list, [1.0])
        self.assertEqual(closest_gt, "hello")

    def test_tuple_comparison_best_match(self):
        """Test that tuple comparison selects the best matching option."""
        tuple_node = ANLSTuple(
            ("hello", "world"), is_gt=True, comparator=LevenshteinComparator()
        )

        # Test with exact match to second option
        leaf_node1 = ANLSLeaf("world", comparator=LevenshteinComparator())
        nls_list1, closest_gt1, key_scores1 = tuple_node.nls_list(leaf_node1, (), [])
        self.assertEqual(nls_list1, [1.0])
        self.assertEqual(closest_gt1, "world")

        # Test with similar match to first option
        leaf_node2 = ANLSLeaf("helo", comparator=LevenshteinComparator())
        nls_list2, closest_gt2, key_scores2 = tuple_node.nls_list(leaf_node2, (), [])
        # Should match "hello" with some similarity score
        self.assertGreater(nls_list2[0], 0.0)
        self.assertEqual(closest_gt2, "hello")

        # Test with no good match
        leaf_node3 = ANLSLeaf("xyz", comparator=LevenshteinComparator())
        nls_list3, closest_gt3, key_scores3 = tuple_node.nls_list(leaf_node3, (), [])
        # Should return the best option even if it's a poor match
        self.assertEqual(nls_list3, [0.0])
        # The closest GT could be either option since both are equally bad matches
        self.assertIn(closest_gt3, ("hello", "world"))


if __name__ == "__main__":
    unittest.main()
