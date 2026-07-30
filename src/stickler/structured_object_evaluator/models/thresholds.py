"""Threshold checking for StructuredModel comparisons."""


def is_above_threshold(score: float, threshold: float) -> bool:
    """Check if a score is above threshold with floating point precision handling.

    Uses a small epsilon tolerance (1e-10) so that scores which are
    mathematically equal to the threshold but differ by floating-point
    rounding error are still treated as a match.

    Args:
        score: The similarity score to check
        threshold: The threshold value

    Returns:
        True if score is above or equal to threshold (considering floating point precision)
    """
    return score >= threshold or abs(score - threshold) < 1e-10
