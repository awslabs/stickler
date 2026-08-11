"""Threshold checking helper for StructuredModel comparisons."""

from typing import Any, Optional

from stickler.utils.deprecation import warn_once

#: Where the `>=` cliff and what to use instead are explained.
THRESHOLD_DOCS_URL = "https://github.com/awslabs/stickler/issues/234"

_FIELD_CONSEQUENCE = (
    "every compared pair is counted as a true positive, so a wholly incorrect "
    "prediction reports perfect precision"
)

# `match_threshold` is read only when the model is a `List[StructuredModel]`
# element (see structured_list_comparator.py). On a standalone model the value
# changes nothing, and the hook cannot know at class-definition time which the
# class will be -- so this states the condition rather than asserting an outcome
# that may not apply.
_MATCH_CONSEQUENCE = (
    "if this model is compared as a list element, every paired object is "
    "counted as a true positive, so wholly incorrect objects report perfect "
    "precision"
)


def warn_if_threshold_is_zero(
    value: Optional[float],
    context: str,
    parameter: str,
    dedup_key: Optional[str] = None,
) -> None:
    """Warn when a threshold is exactly ``0.0``, which disables classification.

    The threshold test is ``>=``, so ``0.0`` is satisfied by every score
    including ``0.0`` itself. Everything the algorithm compares is then a true
    positive -- the hardest failure direction to notice, because nothing errors
    and the numbers look ideal.

    Only exactly ``0.0`` is flagged. ``0.01`` already classifies correctly, so
    this is a single misbehaving value rather than a "low thresholds are risky"
    heuristic; warning on low-but-positive values would fire on legitimate
    configuration.

    The message claims precision only. Recall survives unmatched extras: 2
    ground-truth objects against 1 prediction at ``match_threshold=0.0`` gives
    precision ``1.0`` but recall ``0.5``, since the unpaired item is still an
    FN. Claiming both would make the warning false for unequal-length lists.

    Args:
        value: The configured threshold, or None if unset.
        context: Where it was set, e.g. ``"Invoice.vendor"``. Appears in the
            message.
        parameter: The parameter name to name in the message.
        dedup_key: Overrides ``context`` for warn-once bookkeeping. Use when
            ``context`` is not unique per configuration site -- dynamically
            built models share the default name ``"DynamicModel"``, so keying
            on it would report the first anonymous config and silence the rest.
            Kept separate from ``context`` so an internal identity does not leak
            into user-visible text.
    """
    # `bool` is an `int`, and `False == 0.0`, so an unguarded numeric check
    # reports `match_threshold = False` as "sets match_threshold=0.0" -- a value
    # the user never wrote. A wrong type is not this function's business.
    if isinstance(value, bool):
        return
    if value is None or not isinstance(value, (int, float)):
        return
    if value != 0.0:
        return

    consequence = (
        _MATCH_CONSEQUENCE if parameter == "match_threshold" else _FIELD_CONSEQUENCE
    )

    warn_once(
        "threshold-zero",
        f"{dedup_key or context}.{parameter}",
        f"{context} sets {parameter}=0.0. The threshold test is `>=`, so every "
        f"score satisfies it: {consequence}. Use a small positive value (for "
        f"example 0.01) to accept weak matches. See {THRESHOLD_DOCS_URL}",
        category=UserWarning,
        # stacklevel=1 attributes the warning to this module. The call sites sit
        # behind __init_subclass__ / ABCMeta.__new__ / ModelMetaclass.__new__ and
        # a C frame, at differing depths, so no single value reaches user code --
        # 4 landed inside `<frozen abc>`, which is worse than honest. The message
        # names the exact site instead.
        stacklevel=1,
    )


class ThresholdHelper:
    """Helper class for consistent threshold checking with floating point precision handling."""

    @staticmethod
    def is_above_threshold(score: float, threshold: float) -> bool:
        """Check if a score is above threshold with floating point precision handling.

        Args:
            score: The similarity score to check
            threshold: The threshold value

        Returns:
            True if score is above or equal to threshold (considering floating point precision)
        """
        return score >= threshold or abs(score - threshold) < 1e-10

    @staticmethod
    def is_below_threshold(score: float, threshold: float) -> bool:
        """Check if a score is below threshold with floating point precision handling.

        Args:
            score: The similarity score to check
            threshold: The threshold value

        Returns:
            True if score is below threshold (considering floating point precision)
        """
        return score < threshold and abs(score - threshold) >= 1e-10

    @staticmethod
    def classify_match(score: float, threshold: float) -> str:
        """Classify a match based on threshold.

        Args:
            score: The similarity score
            threshold: The threshold value

        Returns:
            "TP" if above threshold, "FD" if below threshold
        """
        if ThresholdHelper.is_above_threshold(score, threshold):
            return "TP"
        else:
            return "FD"

    @staticmethod
    def get_match_threshold(obj: Any, default: float = 0.7) -> float:
        """Get the match threshold from an object or return default.

        Args:
            obj: Object to get threshold from (should have match_threshold attribute)
            default: Default threshold if object doesn't have match_threshold

        Returns:
            The match threshold value
        """
        if (
            obj
            and hasattr(obj, "__class__")
            and hasattr(obj.__class__, "match_threshold")
        ):
            return obj.__class__.match_threshold
        return default

    @staticmethod
    def apply_threshold_logic(matched_pairs, threshold: float) -> tuple[int, int]:
        """Apply threshold logic to matched pairs to get TP and FD counts.

        Args:
            matched_pairs: List of (gt_idx, pred_idx, similarity_score) tuples
            threshold: The threshold value

        Returns:
            Tuple of (tp_count, fd_count)
        """
        tp = 0
        fd = 0

        for i, j, score in matched_pairs:
            if ThresholdHelper.is_above_threshold(score, threshold):
                tp += 1
            else:
                fd += 1

        return tp, fd

    @staticmethod
    def format_threshold_reason(score: float, threshold: float) -> str:
        """Format a threshold-related reason string.

        Args:
            score: The similarity score
            threshold: The threshold value

        Returns:
            Formatted reason string
        """
        return f"below threshold ({score:.3f} < {threshold})"
