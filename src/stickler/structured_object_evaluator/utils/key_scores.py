"""Utility functions for handling key scores in structured object evaluation."""

from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ScoreNode:
    """Node in a score tree representing scores for hierarchical structures.

    Attributes:
        name: The name of this node (key in the hierarchy).
        score: The score for this node, or None if this is an intermediate node.
        children: A dictionary mapping child keys to their ScoreNode objects.
    """

    name: str = ""
    score: Optional[float] = None
    children: Dict[str, Any] = field(default_factory=dict)

    # For backward compatibility
    @property
    def anls_score(self):
        """Alias for score to maintain backward compatibility."""
        return self.score

    @anls_score.setter
    def anls_score(self, value):
        """Setter for anls_score that updates the score attribute."""
        self.score = value


def construct_nested_dict(
    list_of_dicts: List[Dict[Tuple[str, ...], float]],
) -> Dict[str, ScoreNode]:
    """Construct a nested dictionary from a list of dictionaries with nested keys.

    This function transforms a flat list of dictionaries with tuple keys into a
    hierarchical structure of ScoreNode objects. This is useful for representing
    and analyzing scores for nested data structures like dictionaries and lists.

    Note: If there are duplicates of keys in the list of dictionaries, the last value will be used.

    Args:
        list_of_dicts: A list of dictionaries with nested keys.

    Returns:
        A nested dictionary of ScoreNode objects.

    Example:
        >>> list_of_dicts = [
                {("a",): 3},
                {("a", "b", "c"): 1},
                {("a", "b", "d"): 2},
                {("a", "c", "e"): 3},
            ],
        >>> construct_nested_dict(list_of_dicts)
            {
                "a": ScoreNode(
                    anls_score=3,
                    children={
                        "b": ScoreNode(
                            children={
                                "c": ScoreNode(anls_score=1),
                                "d": ScoreNode(anls_score=2),
                            }
                        ),
                        "c": ScoreNode(children={"e": ScoreNode(anls_score=3)}),
                    },
                )
            },
    """
    nested_dict: Dict[str, ScoreNode] = {}

    if len(list_of_dicts) == 0:
        return nested_dict

    for entry in list_of_dicts:
        for key_tuple, value in entry.items():
            current_dict: Dict[str, ScoreNode] = nested_dict
            # Traverse and build nested dict, except for last entry
            for key in key_tuple[:-1]:
                if key not in current_dict:
                    current_dict[key] = ScoreNode(name=key)
                current_dict = current_dict[key].children

            # Set the value for the final key
            final_key = key_tuple[-1]
            if final_key not in current_dict:
                current_dict[final_key] = ScoreNode(name=final_key)
            current_dict[final_key].score = value

    return nested_dict


def merge_and_calculate_mean(
    list_of_dicts: List[Dict[Tuple[str, ...], float]],
) -> List[Dict[Tuple[str, ...], float]]:
    """
    Merge a list of dictionaries and calculate the mean value for each key.

    This function takes a list of dictionaries where keys are tuples of strings and
    values are floats. It combines the dictionaries and calculates the mean value
    for each unique key across all the dictionaries.

    Args:
        list_of_dicts: A list of dictionaries with tuple keys and float values.

    Returns:
        A list of dictionaries, each containing a single key-value pair where
        values are the mean of the original values for the corresponding key.

    Example:
        >>> list_of_dicts = [
                {('a', 'b'): 10.0, ('c', 'd'): 20.0},
                {('a', 'b'): 30.0, ('e', 'f'): 40.0}
            ]
        >>> merge_and_calculate_mean(list_of_dicts)
            [{('a', 'b'): 20.0}, {('c', 'd'): 20.0}, {('e', 'f'): 40.0}]
    """
    combined_scores: Dict[Tuple[str, ...], float] = {}
    count_dict: Dict[Tuple[str, ...], int] = {}

    # Combine scores for the same keys
    for d in list_of_dicts:
        for k, v in d.items():
            if k not in combined_scores:
                combined_scores[k] = 0
                count_dict[k] = 0
            combined_scores[k] += v
            count_dict[k] += 1

    # Calculate the mean for each key
    for k in combined_scores.keys():
        combined_scores[k] /= count_dict[k]

    # Convert back to a list of dictionaries
    list_combined_scores = [{k: v} for k, v in combined_scores.items()]

    return list_combined_scores
