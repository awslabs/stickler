import inspect
from pathlib import Path

import pytest

import stickler

CHECKOUT_SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="session", autouse=True)
def _stickler_is_this_checkout():
    """Stop the run unless the stickler under test is this checkout (AGENTS.md).

    An editable install pointing at another worktree, or strands-agents-evals
    installed with its `stickler` extra (which pulls a PyPI stickler-eval),
    would make every number in these tests measure the wrong tree.

    This exits the session instead of failing the fixture: pytest reports a
    fixture failure in an `xfail` test as "xfailed", so a single xfail test
    run against the wrong tree would otherwise pass.
    """
    imported = Path(inspect.getfile(stickler)).resolve()
    if not imported.is_relative_to(CHECKOUT_SRC.resolve()):
        pytest.exit(
            reason=f"stickler imported from {imported}, not this checkout ({CHECKOUT_SRC}). "
            "Run uv sync --frozen in this checkout, or set PYTHONPATH to its src/ "
            "directory. Install strands-agents-evals without the [stickler] extra.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
