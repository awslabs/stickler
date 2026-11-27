"""LLM-based semantic comparison comparator.

This module provides the LLMComparator class, which uses Large Language Models
to perform semantic comparisons between values. Unlike traditional string-based
comparators, the LLMComparator can understand context, synonyms, abbreviations,
and other semantic relationships to determine if two values are equivalent.

The comparator integrates with AWS Bedrock models through the strands-agents
library and supports customizable evaluation guidelines for domain-specific
comparison logic.

Note:
    This comparator requires the optional 'llm' dependencies. Install with:
    pip install "stickler-eval[llm]"

Example:
    Integration with StructuredModel:
        >>> from stickler import ComparableField
        >>>
        >>> class Address(StructuredModel):
        ...     street: str = ComparableField(
        ...         comparator=LLMComparator(eval_guidelines="Consider street abbreviations"),
        ...         threshold=0.8
        ...     )
"""

import html
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from stickler.comparators.base import BaseComparator, _is_missing

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jinja2 import Template

try:
    from botocore.exceptions import NoCredentialsError
    from strands import Agent
    from strands.models import Model

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

    # Create mock classes for when strands is not available
    class Model:
        pass

    class Agent:
        pass

    class NoCredentialsError(Exception):
        pass


#: Quote, backtick and markdown-emphasis characters stripped from both ends of
#: a reply, along with whitespace, before it is read as a verdict.
_VERDICT_WRAPPERS = "\"'`*_“”‘’"
#: Characters additionally stripped from the end, e.g. ``true.``.
_VERDICT_TRAILING_PUNCTUATION = ".,;:!?"
#: Longest reply shown in the unparseable-verdict warning.
_REPLY_PREVIEW_CHARS = 80


def _parse_verdict(response: str) -> Optional[bool]:
    """Read an LLM reply as a ``true``/``false`` verdict (#506).

    Repeatedly strips surrounding whitespace (``str.strip()``, so non-breaking
    spaces too), quotes, backticks and markdown emphasis from both ends, and
    punctuation from the end, then lower-cases. Returns ``True`` only for
    exactly ``true``, ``False`` for exactly ``false``, and ``None`` for anything
    else: a substring test would score "untrue" or "not true" as a match.
    """
    verdict = response
    while True:
        stripped = (
            verdict.strip()
            .strip(_VERDICT_WRAPPERS)
            .rstrip(_VERDICT_TRAILING_PUNCTUATION)
        )
        if stripped == verdict:
            break
        verdict = stripped
    verdict = verdict.lower()
    if verdict == "true":
        return True
    if verdict == "false":
        return False
    return None


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
            ImportError: If strands-agents is not installed.
            ValueError: If the model parameter is not provided.

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

        # Check if strands is available
        if not STRANDS_AVAILABLE:
            raise ImportError(
                "LLMComparator requires the 'strands-agents' package. "
                'Install it with: pip install "stickler-eval[llm]"'
            )

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
            model=self.model, system_prompt=self.system_prompt, callback_handler=None
        )

    def _default_system_prompt(self) -> str:
        """Generate the default system prompt for the LLM.

        Returns:
            str: System prompt instructing the LLM to perform binary comparisons.
        """
        return "You are a helpful assistant that compares two values and determines if they are equivalent. Only return one word: 'true' or 'false'."

    def _default_prompt_template(self) -> "Template":
        """Generate the default Jinja2 template for comparison prompts.

        Returns:
            Template: Jinja2 template that formats comparison prompts with values
                and optional evaluation guidelines.

        Raises:
            ImportError: If jinja2 is not installed.
        """
        # Imported here rather than at module scope: jinja2 is only needed to
        # render this prompt, and a module-level import would put it on the
        # `import stickler` path via the comparator registry.
        try:
            from jinja2 import Template
        except ImportError as exc:  # pragma: no cover - exercised by the extras gate
            raise ImportError(
                "LLMComparator requires jinja2. Install it with: "
                'pip install "stickler-eval[llm]"'
            ) from exc

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

    def _compare(self, value1: Any, value2: Any) -> float:
        """Compare two values using LLM-based semantic analysis.

        This method converts both values to strings and uses the configured LLM
        to determine if they are semantically equivalent. The comparison considers
        context, abbreviations, synonyms, and any provided evaluation guidelines.

        Args:
            value1: First value to compare. Can be any type that converts to string.
            value2: Second value to compare. Can be any type that converts to string.

        Returns:
            float: 1.0 if the normalized reply is exactly ``true``, otherwise 0.0.
                Normalizing strips surrounding whitespace, quotes, backticks and
                markdown emphasis, strips trailing ``.,;:!?``, and ignores case. A
                reply that is neither ``true`` nor ``false`` also emits a
                ``UserWarning``.

        Raises:
            NoCredentialsError: If AWS credentials are not found.
            Exception: Any other error from the LLM call is re-raised.

        Note:
            Cost consideration: Each call incurs API costs and latency.

        Example:
            >>> comparator = LLMComparator()
            >>> comparator.compare("St. John's Street", "Saint John's St")
            1.0
            >>> comparator.compare("apple", "orange")
            0.0
        """
        try:
            response = self._invoke_agent(self._render_prompt(value1, value2))
        except NoCredentialsError:
            print("Error: AWS credentials not found.")
            raise
        except Exception as e:
            print(f"Error during LLM call: {e}")
            raise

        # _score_reply (1) -> _compare (2) -> BaseComparator.compare (3) ->
        # compare()'s caller (4). That is user code only for a direct
        # compare(); via __call__, binary_compare or compare_with it lands
        # inside stickler.
        return self._score_reply(response, stacklevel=4)

    def _render_prompt(self, value1: Any, value2: Any) -> str:
        """Format the comparison prompt for two values."""
        return self.prompt_template.render(
            value1=html.escape(str(value1)),
            value2=html.escape(str(value2)),
            eval_guidelines=self.eval_guidelines,
        )

    def _score_reply(self, response: str, stacklevel: int) -> float:
        """Score a model reply: 1.0 for ``true``, otherwise 0.0.

        A reply that is neither ``true`` nor ``false`` also emits a
        ``UserWarning``. ``stacklevel`` goes straight to ``warnings.warn``, so
        each caller can attribute the warning to its own caller.
        """
        verdict = _parse_verdict(response)
        if verdict is None:
            preview = response
            if len(preview) > _REPLY_PREVIEW_CHARS:
                preview = preview[:_REPLY_PREVIEW_CHARS] + "..."
            warnings.warn(
                f"LLMComparator could not read the model's reply as 'true' or "
                f"'false', so it scored 0.0. Reply: {preview!r}",
                UserWarning,
                stacklevel=stacklevel,
            )
            return 0.0
        return 1.0 if verdict else 0.0

    def get_comparison_details(self, value1: Any, value2: Any) -> Dict[str, Any]:
        """Get detailed information about a comparison operation.

        This method provides comprehensive details about the comparison process,
        including the formatted prompt, LLM response, model information, and
        final comparison result. Useful for debugging, auditing, and understanding
        how the LLM made its decision.

        The model is called once, and ``comparison_result`` scores that same
        reply, as :meth:`compare` would. If either value is ``None`` or NaN, the
        model is not called: the result follows the ``None`` policy of
        :meth:`compare`, and ``prompt`` and ``llm_response`` are ``None``.

        Args:
            value1: First value to compare. Can be any type that converts to string.
            value2: Second value to compare. Can be any type that converts to string.

        Returns:
            Dict[str, Any]: Dictionary containing comparison details:
                - 'prompt' (str or None): The formatted prompt sent to the LLM
                - 'llm_response' (str or None): Raw response from the LLM
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
        if _is_missing(value1) or _is_missing(value2):
            # Scored by the shared None policy, exactly as compare() would,
            # so the model is not asked.
            return {
                "prompt": None,
                "llm_response": None,
                "model_id": self.model,
                "comparison_result": self.compare(value1, value2),
            }

        formatted_prompt = self._render_prompt(value1, value2)
        try:
            response = self._invoke_agent(formatted_prompt)
            return {
                "prompt": formatted_prompt,
                "llm_response": response,
                "model_id": self.model,
                # Scores the reply shown above, without a second call (#589).
                # _score_reply (1) -> get_comparison_details (2) -> caller (3).
                "comparison_result": self._score_reply(response, stacklevel=3),
            }
        except Exception as e:
            return {"error": str(e), "comparison_result": False}
