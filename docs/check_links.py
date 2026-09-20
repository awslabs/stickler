"""Build docs, failing on MkDocs page/navigation warnings, not plugin warnings."""

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
        self.count += 1


def check_links(config_file: str, site_dir: str | None = None) -> int:
    """Build a site and return nonzero for broken links; build errors propagate."""
    config = load_config(config_file=config_file, site_dir=site_dir)
    # Anchors default to INFO, which even a strict build does not reject.
    config.validation.links.anchors = logging.WARNING
    config.validation.links.not_found = logging.WARNING
    config.validation.nav.not_found = logging.WARNING
    counter = _LinkWarnings()
    loggers = [
        logging.getLogger(f"mkdocs.structure.{name}") for name in ("pages", "nav")
    ]
    for logger in loggers:
        logger.addHandler(counter)
    try:
        config.plugins.on_startup(command="build", dirty=False)
        try:
            build(config)
        finally:
            config.plugins.on_shutdown()
    finally:
        for logger in loggers:
            logger.removeHandler(counter)
        counter.close()
    if counter.count:
        logging.getLogger("mkdocs").error(
            "Documentation link check failed: %d page/navigation warning(s).",
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
