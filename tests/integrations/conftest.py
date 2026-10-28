import inspect
from pathlib import Path

import pytest

import stickler

CHECKOUT_SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="session", autouse=True)
def _stickler_is_this_checkout():
    """Fail loudly unless the stickler under test is this checkout (AGENTS.md).

    An editable install pointing at another worktree, or strands-agents-evals
    installed with its `stickler` extra (which pulls a PyPI stickler-eval),
    would make every number in these tests measure the wrong tree.
    """
    imported = Path(inspect.getfile(stickler)).resolve()
    if not imported.is_relative_to(CHECKOUT_SRC.resolve()):
        pytest.fail(
            f"stickler imported from {imported}, not this checkout ({CHECKOUT_SRC}). "
            "Install strands-agents-evals without the [stickler] extra.",
            pytrace=False,
        )
