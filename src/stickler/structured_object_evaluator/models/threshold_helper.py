"""Threshold checking helper for StructuredModel comparisons."""

from typing import Any, Optional

from stickler.utils.deprecation import warn_once

#: Where the `>=` cliff and what to use instead are explained. Points at the
#: docs rather than issue #234, which was the interim target while the section
#: did not exist yet ([#235](https://github.com/awslabs/stickler/issues/235)).
#: The anchor is load-bearing: `tests/.../test_threshold_zero_warning.py`
#: asserts the literal, and `docs/.../thresholds-and-metrics.md` must keep a
#: heading that slugifies to `the-zero-threshold-trap`.
THRESHOLD_DOCS_URL = (
    "https://awslabs.github.io/stickler/Getting-Started/"
    "thresholds-and-metrics/#the-zero-threshold-trap"
)

# What a zero threshold actually guarantees: nothing the comparison touches can
# be reported as a false discovery, because FD means "compared and scored below
# threshold" and no score is below 0.0. Measured exhaustively at 0.0 -- 256
# scalar-value combinations and 36 list-length combinations -- FD was 0 in every
# single one.
#
# Deliberately does NOT claim perfect precision or recall. Both are false in
# reachable cases, symmetrically: an unmatched prediction is still an FA (2
# ground-truth objects vs 3 predictions gives precision 0.667) and an unmatched
# ground-truth item is still an FN (2 vs 1 gives recall 0.5). Only the pairs the
# algorithm actually compares are forced to TP; unmatched items are not subject
# to any threshold. Claiming a metric outcome would make the warning false for
# unequal-length lists, and a user who saw imperfect precision would conclude
# the warning did not describe their situation and keep the broken threshold.
_FIELD_CONSEQUENCE = (
    "every value it compares is counted as a true positive and nothing can be "
    "reported as a false discovery, however wrong the prediction is"
)

# `match_threshold` reaches the comparison two ways, which is why this states
# the consequence unconditionally rather than hedging on list membership:
#
#   1. as the object-matching threshold for a `List[StructuredModel]` element
#      (structured_list_comparator.py), and
#   2. as the *default field threshold* for any field with no explicit
#      comparison config -- `ConfigurationHelper.get_comparison_info` falls back
#      to `getattr(cls, "match_threshold", 0.5)`, so a plainly annotated
#      `name: str` inherits it.
#
# Route 2 is easy to miss because a field declared with `ComparableField()`
# takes an earlier branch and gets a hardcoded 0.5 instead, so probing with
# ComparableField makes the value look inert. It is not: a plain-annotated model
# at 0.0 reports precision and recall 1.0 for a wholly wrong prediction, with no
# list anywhere. An earlier version of this message said "if this model is
# compared as a list element", which told exactly the users who were affected
# that it did not apply to them (#237).
_MATCH_CONSEQUENCE = (
    "every value it compares is counted as a true positive and nothing can be "
    "reported as a false discovery, however wrong the prediction is. This "
    "applies both to object matching when the model is a list element, and to "
    "any field with no explicit threshold of its own, which inherits this value"
)



def model_identity(model_name: str, field_names: Any = ()) -> str:
    """Name a model so that distinct configurations are distinguishable.

    Dynamically built models all default to ``"DynamicModel"``, so the name
    alone identifies nothing. That matters twice over: it is unhelpful to a
    reader, and Python's ``__warningregistry__`` is keyed on the message text,
    so two configs producing identical text mean the interpreter prints only the
    first under its default "once per location" action -- silently swallowing
    the second misconfiguration even though ``warn_once`` approved it.

    Object identity would distinguish them but is unusable as a key. It is
    unstable (a new class per call, so repeated loads of one config each warn --
    measured 192 warnings for 200 loads, with the process-global ``_warned`` set
    growing without bound) and it is *reused*: CPython recycles the address once
    a class is collected, so a batch loop that builds, uses and drops each model
    collides on the same key and drops most of its warnings.

    Field names are stable across repeated loads of one config, distinct between
    different configs, and cannot be recycled. Named models are left alone,
    since their own name already distinguishes them.
    """
    if model_name != "DynamicModel":
        return model_name
    joined = ",".join(sorted(field_names))
    return f"{model_name}(fields: {joined})" if joined else model_name


def warn_if_threshold_is_zero(
    value: Optional[float], context: str, parameter: str
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

    The message names the invariant -- no false discovery is possible -- rather
    than a metric outcome. Perfect precision and perfect recall are both false
    in reachable cases, symmetrically: unmatched predictions are still FAs (2
    ground-truth objects vs 3 predictions gives precision ``0.667``) and
    unmatched ground-truth items are still FNs (2 vs 1 gives recall ``0.5``),
    because unmatched items are not subject to any threshold. Verified at 0.0
    over 36 list-length and 256 scalar-value combinations: FD was ``0`` in all
    of them, while precision was ``1.0`` in only 20 of the 36.

    Args:
        value: The configured threshold, or None if unset.
        context: Where it was set, e.g. ``"Invoice.vendor"``. Appears in the
            message and keys the warn-once bookkeeping, so it must distinguish
            configuration sites -- see ``ModelFactory._config_identity`` for why
            a bare ``"DynamicModel"`` does not.
        parameter: The parameter name to name in the message.
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
        f"{context}.{parameter}",
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
