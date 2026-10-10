"""Build docs, failing on logged MkDocs diagnostics except nine known warnings.

Python warnings (such as DeprecationWarning) are not routed into this gate.
"""

import argparse
import logging
import re
from pathlib import Path

from mkdocs.commands.build import build
from mkdocs.config import load_config
from mkdocs.exceptions import MkDocsException

# Issue #355's existing annotation warnings. Keep messages and source locations
# explicit: a changed diagnostic needs review, not a blanket plugin exemption.
_GRIFFE_WARNINGS = {
    "griffe: src/stickler/structured_object_evaluator/" + warning
    for warning in (
        "bulk_structured_model_evaluator.py:965: No type or annotation for parameter 'df'",
        "models/structured_model.py:1842: No type or annotation for parameter '**kwargs'",
        "models/structured_model.py:1845: No type or annotation for returned value 1",
        "models/comparable_field.py:102: No type or annotation for parameter '**field_kwargs'",
        "models/comparable_field.py:105: No type or annotation for returned value 1",
        "models/field.py:51: No type or annotation for parameter '**kwargs'",
    )
}
_AUTOREFS_WARNING = re.compile(
    r"mkdocs_autorefs: API-Reference/models\.md: from (?:.*?/)?"
    r"src/stickler/structured_object_evaluator/models/structured_model\.py:262: "
    r"\(stickler\.structured_object_evaluator\.models\.structured_model\.StructuredModel\) "
    r"Could not find cross-reference target ''(?:tp|fd|derived)''"
)


class _LinkWarnings(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        message = record.getMessage().replace("\\", "/")
        if record.levelno == logging.WARNING:
            if record.name == "mkdocs.plugins.griffe":
                # griffe uses absolute paths when invoked from docs/ (make check).
                message = re.sub(
                    r"^griffe: (?:.*?/)?src/stickler/",
                    "griffe: src/stickler/",
                    message,
                )
                if message in _GRIFFE_WARNINGS:
                    return
            if (
                record.name == "mkdocs.plugins.mkdocs_autorefs._internal.plugin"
                and _AUTOREFS_WARNING.fullmatch(message)
            ):
                return
        self.count += 1


def check_links(config_file: str, site_dir: str | None = None) -> int:
    """Build a site and return nonzero for broken links; build errors propagate."""
    counter = _LinkWarnings()
    logger = logging.getLogger("mkdocs")
    logger.addHandler(counter)
    try:
        # Configuration and plugin loading can emit warnings before build().
        config = load_config(config_file=config_file, site_dir=site_dir)
        config.plugins.on_startup(command="build", dirty=False)
        try:
            build(config)
        finally:
            config.plugins.on_shutdown()
    finally:
        logger.removeHandler(counter)
        counter.close()
    if counter.count:
        logger.error(
            "Documentation check failed: %d warning(s) or error(s).",
            counter.count,
        )
        return 1
    return 0


def main() -> int:
    """Run the same build-only check locally and in pull-request CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f", "--config-file", default=str(Path(__file__).with_name("mkdocs.yml"))
    )
    parser.add_argument("-d", "--site-dir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        return check_links(args.config_file, args.site_dir)
    except MkDocsException as error:
        logging.getLogger("mkdocs").error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
