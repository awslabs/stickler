"""Process-scoped warn-once helper for deprecation messages.

Bulk evaluation can fan a single deprecated call out to thousands of
documents. Without suppression, each one re-emits the same warning,
which floods stderr and turns ``-W error::DeprecationWarning`` into a
sea of unrelated failures. This module keys warnings by
``(deprecation_id, context)`` so each unique site fires once per
process while still preserving the per-field-path signal.

Tests that need to assert the warning fires can clear ``_warned``
directly via the autouse fixture in ``tests/.../conftest.py``.
"""

import warnings
from typing import Set, Tuple, Type

_warned: Set[Tuple[str, str]] = set()


def warn_once(
    deprecation_id: str,
    context: str,
    message: str,
    category: Type[Warning] = DeprecationWarning,
    stacklevel: int = 3,
) -> None:
    """Emit ``message`` only the first time this ``(deprecation_id, context)`` is seen.

    The default ``stacklevel=3`` points at the caller of the function
    that wraps ``warn_once`` (one frame for ``warn_once`` itself, one
    for the wrapping call, one to land on user code).
    """
    key = (deprecation_id, context)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(message, category, stacklevel=stacklevel)
