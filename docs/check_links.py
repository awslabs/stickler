"""Build docs, failing on MkDocs diagnostics except known API warnings."""

import argparse
import logging
from pathlib import Path

from mkdocs.commands.build import build
from mkdocs.config import load_config
from mkdocs.exceptions import Abort


class _LinkWarnings(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        # These sources emit the site's existing API annotation/cross-reference
        # warnings. Errors remain fatal, including errors from these plugins.
        if record.levelno == logging.WARNING and any(
            record.name == name or record.name.startswith(name + ".")
            for name in (
                "mkdocs.plugins.griffe",
                "mkdocs.plugins.mkdocs_autorefs",
            )
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
    except Abort as error:
        logging.getLogger("mkdocs").error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
