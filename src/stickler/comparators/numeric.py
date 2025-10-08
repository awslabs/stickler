"""Numeric comparison comparator."""

from typing import Any, Union, Optional
import re
from decimal import Decimal, InvalidOperation

from stickler.comparators.base import BaseComparator


class NumericComparator(BaseComparator):
    """Comparator for numeric values with configurable tolerance.

    This comparator extracts and compares numeric values from strings or numbers.
    It supports relative and absolute tolerance for comparison.

    Example:
        ```python
        # Default exact matching
        exact = NumericComparator()
        exact.compare("123", "123.0")  # Returns 1.0
        exact.compare("123", "124")    # Returns 0.0

        # With tolerance
        approx = NumericComparator(relative_tolerance=0.1)  # 10% tolerance
        approx.compare("100", "109")   # Returns 1.0 (within 10%)
        approx.compare("100", "111")   # Returns 0.0 (beyond 10%)
        ```
    """

    def __init__(
        self,
        threshold: float = 1.0,
        relative_tolerance: float = 0.0,
        absolute_tolerance: float = 0.0,
        tolerance: Optional[float] = None,
    ):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0)
            relative_tolerance: Relative tolerance for comparison (default 0.0)
            absolute_tolerance: Absolute tolerance for comparison (default 0.0)
            tolerance: Alias for absolute_tolerance (for backward compatibility)
        """
        super().__init__(threshold=threshold)
        self.relative_tolerance = relative_tolerance

        # Handle tolerance alias for backward compatibility
        if tolerance is not None:
            if absolute_tolerance != 0.0:
                raise ValueError(
                    "Cannot specify both 'tolerance' and 'absolute_tolerance'. Use 'absolute_tolerance'."
                )
            self.absolute_tolerance = tolerance
        else:
            self.absolute_tolerance = absolute_tolerance

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two values numerically.

        Args:
            str1: First value
            str2: Second value

        Returns:
            1.0 if the numbers match within tolerance, 0.0 otherwise
        """
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0

        # Extract numeric values
        num1 = self._extract_number(str1)
        num2 = self._extract_number(str2)

        if num1 is None or num2 is None:
            return 0.0

        # Check equality with tolerance
        if self._numbers_equal(num1, num2):
            return 1.0

        return 0.0

    def _extract_number(self, value: Any) -> Union[Decimal, None]:
        """Extract a numeric value from a string or number.

        Args:
            value: Value to extract a number from

        Returns:
            Decimal value or None if no valid number could be extracted
        """
        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if not isinstance(value, str):
            value = str(value)

        # Check for accounting notation: (123) means -123
        is_negative = False
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1]  # Remove the parentheses
            is_negative = True

        # Remove common currency symbols and other non-numeric characters
        value = re.sub(r"[^0-9.-]", "", value)

        # Handle empty string
        if not value:
            return None

        # Try to convert to Decimal
        try:
            decimal_value = Decimal(value)
            # Apply negative sign if accounting notation was used
            if is_negative:
                decimal_value = -decimal_value
            return decimal_value
        except InvalidOperation:
            return None

    def _numbers_equal(self, num1: Decimal, num2: Decimal) -> bool:
        """Check if two numbers are equal within tolerance.

        Args:
            num1: First number
            num2: Second number

        Returns:
            True if numbers are equal within tolerance, False otherwise
        """
        if num1 == num2:
            return True

        # Check with relative tolerance
        if self.relative_tolerance > 0:
            # Handle zero case
            if num1 == 0:
                return abs(num2) <= self.relative_tolerance

            # Calculate relative difference using num1 as base
            relative_diff = abs(num1 - num2) / abs(num1)
            if relative_diff <= self.relative_tolerance:
                return True

        # Check with absolute tolerance
        if self.absolute_tolerance > 0:
            if abs(num1 - num2) <= self.absolute_tolerance:
                return True

        return False


# Class alias for backward compatibility with NumericExactC() in evaluation metrics
NumericExactC = NumericComparator
