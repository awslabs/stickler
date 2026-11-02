"""Operation-local scoring data shared with report collectors, never serialized."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportingContext:
    """Side maps keyed by field name at each level of the model tree.

    List pair results hold child contexts only for accepted pairs. Keeping these
    separate from metric dictionaries makes every comparison result safe to
    return directly, including the deprecated field-dispatch entry point.
    """

    lists: dict[str, dict[str, Any]] = field(default_factory=dict)
    children: dict[str, "ReportingContext"] = field(default_factory=dict)
    recursive_result: dict[str, Any] = field(default_factory=dict)
    field_scores: dict[str, float] = field(default_factory=dict)
