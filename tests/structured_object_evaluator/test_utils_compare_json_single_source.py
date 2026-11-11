"""``compare_json`` must have exactly one definition and one identity.

``utils/anls_score.py`` used to define its own copy of ``compare_json``,
identical to the one in ``utils/compare_json.py``. Editing one and not the
other gave you a silently different function depending on the import path.

It now imports from ``utils/compare_json.py``. These tests pin that the
re-export stays a re-export: importing the submodule must not turn
``utils.compare_json`` into a module, and every public path must hand back the
same function object.
"""

import stickler
import stickler.structured_object_evaluator.utils as utils_pkg
import stickler.structured_object_evaluator.utils.compare_json  # noqa: F401


def test_utils_compare_json_is_still_the_function():
    """Importing the submodule leaves the package attribute as the function.

    ``import a.b.c`` normally rebinds ``a.b.c`` to the module. Here the package
    already exposes the function under that name, so the attribute stays
    callable — the one place where that shadowing is what we want.
    """
    assert callable(utils_pkg.compare_json)


def test_every_public_path_returns_the_same_function():
    from stickler.structured_object_evaluator.utils.anls_score import (
        compare_json as via_anls,
    )
    from stickler.structured_object_evaluator.utils.compare_json import (
        compare_json as definition,
    )

    assert stickler.compare_json is definition
    assert via_anls is definition
    assert utils_pkg.compare_json is definition
