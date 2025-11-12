"""LLM-based semantic comparison comparator.

This module provides the LLMComparator class, which uses Large Language Models
to perform semantic comparisons between values. Unlike traditional string-based
comparators, the LLMComparator can understand context, synonyms, abbreviations,
and other semantic relationships to determine if two values are equivalent.

The comparator integrates with AWS Bedrock models through the strands-agents
library and supports customizable evaluation guidelines for domain-specific
comparison logic.

Example:
    Integration with StructuredModel:
        >>> from stickler.structured_object_evaluator.models.comparable_field import ComparableField
        >>> 
        >>> class Address(StructuredModel):
        ...     street: str = ComparableField(
        ...         comparator=LLMComparator(eval_guidelines="Consider street abbreviations"),
        ...         threshold=0.8
        ...     )
"""
from strands.models import Model
from strands import Agent
import html
from typing import Any, Dict, Union
from stickler.comparators.base import BaseComparator
from jinja2 import Template
from botocore.exceptions import NoCredentialsError


class LLMComparator(BaseComparator):
    """Large Language Model-based semantic comparator.
    
    This comparator uses LLMs to perform intelligent semantic comparisons that go
    beyond simple string matching. It can understand context, handle abbreviations,
    recognize synonyms, and apply domain-specific comparison logic through custom
    evaluation guidelines.
    
    The comparator returns binary similarity scores (0.0 or 1.0) based on whether
    the LLM determines the values are semantically equivalent. It handles edge cases
    like None values and provides detailed comparison information for debugging.
    
    Attributes:
        model (Union[Model, str]): The LLM model identifier or Model instance.
        eval_guidelines (str, optional): Custom guidelines for comparison logic.
        system_prompt (str): The system prompt used to instruct the LLM.
        prompt_template (Template): Jinja2 template for formatting comparison prompts.
        agent (Agent): The strands Agent instance for LLM interactions.
        threshold (float): Inherited from BaseComparator, used for binary decisions.
    
    Note:
        This comparator requires AWS Bedrock access and proper authentication.
        API calls incur costs and latency, so consider caching for repeated comparisons.
    """
    def __init__(
        self,
        model: Union[Model, str] = None,
        eval_guidelines: str = None,
    ):
        """Initialize the LLM comparator.
        
        Args:
            model: The LLM model to use for comparisons. Can be a model identifier
                string (e.g., "us.anthropic.claude-3-haiku-20240307-v1:0") or a
                strands Model instance. Defaults to Claude 3 Haiku.
            eval_guidelines: Optional custom guidelines to include in the comparison
                prompt. These guidelines help the LLM understand domain-specific
                comparison rules (e.g., "Consider abbreviations equivalent").
        
        Raises:
            Exception: If the model cannot be initialized or AWS credentials are invalid.
        
        Example:
            >>> # Basic initialization
            >>> comparator = LLMComparator()
            
            >>> # With custom model and guidelines
            >>> comparator = LLMComparator(
            ...     model="us.amazon.nova-lite-v1:0",
            ...     eval_guidelines="Consider street abbreviations equivalent"
            ... )
        """
        super().__init__()
        if model is None:
            raise ValueError("Model must be provided for LLMComparator.")
        self.model = model
        self.system_prompt = self._default_system_prompt()
        self.prompt_template = self._default_prompt_template()
        if eval_guidelines is not None:
            self.eval_guidelines = html.escape(eval_guidelines)
        else:
            self.eval_guidelines = eval_guidelines
        
        # Initialize Agent
        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            callback_handler=None
        )

    def _default_system_prompt(self) -> str:
        """Generate the default system prompt for the LLM.
        
        Returns:
            str: System prompt instructing the LLM to perform binary comparisons.
        """
        return "You are a helpful assistant that compares two values and determines if they are equivalent. Only return one word: 'true' or 'false'."
    
    def _default_prompt_template(self) -> Template:
        """Generate the default Jinja2 template for comparison prompts.
        
        Returns:
            Template: Jinja2 template that formats comparison prompts with values
                and optional evaluation guidelines.
        """
        prompt_template = """
            Compare these two values and determine if they are equivalent:

            Value 1: {{ value1 }}
            Value 2: {{ value2 }}

            {% if eval_guidelines is not none %}
            <guidelines>
            Here are some guidelines to follow for the comparison:
            {{ eval_guidelines }}
            </guidelines>
            {% endif %}

            If the values are equivalent, return 'true'. If not, return 'false'. Only return one word: 'true' or 'false'.
            """

        template = Template(prompt_template)
        return template
                    
    def _invoke_agent(self, prompt: str) -> str:
        """Invoke the LLM agent with a formatted prompt.
        
        Args:
            prompt: The formatted prompt string to send to the LLM.
        
        Returns:
            str: The text response from the LLM.
        
        Raises:
            Exception: If the agent call fails or response format is unexpected.
        """
        result = self.agent(prompt)
        return result.message["content"][0]["text"]
    
    def compare(self, value1: Any, value2: Any) -> float:
        """Compare two values using LLM-based semantic analysis.
        
        This method converts both values to strings and uses the configured LLM
        to determine if they are semantically equivalent. The comparison considers
        context, abbreviations, synonyms, and any provided evaluation guidelines.
        
        Args:
            value1: First value to compare. Can be any type that converts to string.
            value2: Second value to compare. Can be any type that converts to string.
        
        Returns:
            float: Binary similarity score:
                - 1.0 if the LLM determines the values are equivalent
                - 0.0 if the LLM determines the values are not equivalent
                - 0.0 if an error occurs during comparison
        
        Note:
            - None values: Returns 1.0 if both are None, 0.0 if only one is None
            - Error handling: Returns 0.0 for any exceptions during LLM calls
            - Cost consideration: Each call incurs API costs and latency
        
        Example:
            >>> comparator = LLMComparator()
            >>> comparator.compare("St. John's Street", "Saint John's St")
            1.0
            >>> comparator.compare("apple", "orange")
            0.0
            >>> comparator.compare(None, None)
            1.0
        """
        # Handle None values
        if value1 is None and value2 is None:
            return 1.0
        elif value1 is None or value2 is None:
            return 0.0

        # Format the prompt with your values
        formatted_prompt = self.prompt_template.render(
            value1=html.escape(str(value1)),
            value2=html.escape(str(value2)),
            eval_guidelines=self.eval_guidelines
        )
        
        try:
            # Get LLM response
            response = self._invoke_agent(formatted_prompt)
            # Parse response to boolean
            response_lower = response.strip().lower()
            if 'true' in response_lower:
                return 1.0
            else:
                return 0.0
            
        except NoCredentialsError:
            print(f"Error: AWS credentials not found.")
            raise 

        except Exception as e:
            print(f"Error during LLM call: {e}")
            raise

    
    def get_comparison_details(self, value1: Any, value2: Any) -> Dict[str, Any]:
        """Get detailed information about a comparison operation.
        
        This method provides comprehensive details about the comparison process,
        including the formatted prompt, LLM response, model information, and
        final comparison result. Useful for debugging, auditing, and understanding
        how the LLM made its decision.
        
        Args:
            value1: First value to compare. Can be any type that converts to string.
            value2: Second value to compare. Can be any type that converts to string.
        
        Returns:
            Dict[str, Any]: Dictionary containing comparison details:
                - 'prompt' (str): The formatted prompt sent to the LLM
                - 'llm_response' (str): Raw response from the LLM
                - 'model_id' (Union[Model, str]): The model used (string ID or Model instance)
                - 'comparison_result' (float): Final similarity score (0.0 or 1.0)
                
                On error:
                - 'error' (str): Error message describing what went wrong
                - 'comparison_result' (bool): False to indicate failure
        
        Example:
            >>> comparator = LLMComparator(eval_guidelines="Consider abbreviations")
            >>> details = comparator.get_comparison_details("St. John", "Saint John")
            >>> print(details['llm_response'])
            'true'
            >>> print(details['comparison_result'])
            1.0
            >>> print('guidelines' in details['prompt'])
            True
        """
        formatted_prompt = self.prompt_template.render(
            value1=html.escape(str(value1)),
            value2=html.escape(str(value2)),
            eval_guidelines=self.eval_guidelines
        )
        
        try:
            response = self._invoke_agent(formatted_prompt)
            return {
                "prompt": formatted_prompt,
                "llm_response": response,
                "model_id": self.model,
                "comparison_result": self.compare(value1, value2)
            }
        except Exception as e:
            return {
                "error": str(e),
                "comparison_result": False
            }
