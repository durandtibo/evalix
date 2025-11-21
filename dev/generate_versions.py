# noqa: INP001
r"""Script to create or update the package versions."""

from __future__ import annotations

import logging
from pathlib import Path

from feu.utils.io import save_json
from feu.version import get_latest_minor_versions, get_versions

logger = logging.getLogger(__name__)


def get_package_versions() -> dict[str, list[str]]:
    r"""Get the versions for each package.

    Returns:
        A dictionary with the versions for each package.
    """
    return {
        "coola": list(get_versions("coola", lower="0.9.1")),
        "matplotlib": list(get_latest_minor_versions("matplotlib", lower="3.8")),
        "numpy": list(get_latest_minor_versions("numpy", lower="2.0")),
        "scikit-learn": list(get_latest_minor_versions("scikit-learn", lower="1.5")),
        # Optional dependencies
        "colorlog": list(get_latest_minor_versions("colorlog", lower="6.7")),
        "hya": list(get_versions("hya", lower="0.3.1")),
        "hydra-core": list(get_latest_minor_versions("hydra-core", lower="1.3")),
        "markdown": list(get_latest_minor_versions("markdown", lower="3.4")),
        "objectory": list(get_versions("objectory", lower="0.2.1")),
        "omegaconf": list(get_latest_minor_versions("omegaconf", lower="2.1")),
        "scipy": list(get_latest_minor_versions("scipy", lower="1.10")),
        "tqdm": list(get_latest_minor_versions("tqdm", lower="4.65")),
    }


def main() -> None:
    r"""Generate the package versions and save them in a JSON file."""
    versions = get_package_versions()
    logger.info(f"{versions=}")
    path = Path(__file__).parent.parent.joinpath("dev/config").joinpath("package_versions.json")
    logger.info(f"Saving package versions to {path}")
    save_json(versions, path, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
