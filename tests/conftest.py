from pathlib import Path

import pytest

import stickler

CHECKOUT_SRC = Path(__file__).resolve().parents[1] / "src"


def pytest_sessionstart(session):
    """Stop the run unless the stickler under test is this checkout (AGENTS.md).

    An editable install pointing at another worktree, or strands-agents-evals
    installed with its `stickler` extra (which pulls a PyPI stickler-eval),
    would make every number in these tests measure the wrong tree.

    Check before xdist starts workers. Exiting from a worker fixture produces
    an internal error, and failing a fixture can be swallowed by an xfail.
    """
    module_file = getattr(stickler, "__file__", None)
    imported = Path(module_file).resolve() if module_file else None
    if imported is None or not imported.is_relative_to(CHECKOUT_SRC.resolve()):
        origin = imported if imported is not None else "a package without __file__"
        pytest.exit(
            reason=f"stickler imported from {origin}, not this checkout ({CHECKOUT_SRC}). "
            "Run uv sync --frozen in this checkout, or set PYTHONPATH to its src/ "
            "directory. Install strands-agents-evals without the [stickler] extra.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
