"""LLM-based comparator for semantic equivalence."""

from typing import Any

from stickler.comparators.base import BaseComparator
from stickler.utils.time_util import sleep


class LLMComparator(BaseComparator):
    """Comparator that uses LLM to determine semantic equivalence.

    This comparator uses an LLM to determine if two values are semantically
    equivalent, returning 1.0 if True and 0.0 if False.

    Attributes:
        prompt: Prompt template to use for comparison
        model_id: Model ID to use for LLM
        temp: Temperature for LLM inference
        system_prompt: System prompt for the LLM
    """

    def __init__(
        self, prompt: str, model_id: str, temp: float = 0.5, threshold: float = 0.5
    ):
        """Initialize the LLMComparator.

        Args:
            prompt: Prompt template to use for comparison
            model_id: Model ID to use for LLM
            temp: Temperature for LLM inference
            threshold: Similarity threshold (0.0-1.0)
        """
        super().__init__(threshold=threshold)
        self.prompt = prompt
        self.temp = temp
        self.model_id = model_id
        self.system_prompt = "You are an evaluation assistant. Carefully decide if the two values are same or not. Respond only with 'TRUE' or 'FALSE', nothing else."

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two values using LLM.

        Args:
            str1: First value
            str2: Second value

        Returns:
            1.0 if LLM determines values are equivalent, 0.0 otherwise
        """
        if str1 is None or str2 is None:
            return 0.0
        raise Exception("This implementation is not working yet!")
        ci = ClaudeInvoker(
            self.prompt,
            self.model_id,
            system_prompt=self.system_prompt,
            temperature=self.temp,
        )

        kwargs = {"value1": str1, "value2": str2}

        try:
            response = ci.inference(kwargs)
        except Exception as e:
            print(f"LLM error: {str(e)}")
            sleep(2)
            response = ci.inference(kwargs)

        result = response == "TRUE"
        if result:
            print(
                "WARNING: LLM evaluation returned True. Please refine the prompt or review the result."
            )
            return 1.0
        else:
            return 0.0
