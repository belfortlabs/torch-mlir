"""Script to calculate the version for the belfort-torch-mlir Python package.

Adapted from the HEIR release tooling. Produces a date-based version
(``YYYY.MM.DD``) for tagged/scheduled releases and an auto-incrementing
``YYYY.MM.DD.devN`` for development releases off ``main``. The version is fed to
``setup.py`` via the ``TORCH_MLIR_PYTHON_PACKAGE_VERSION`` environment variable.
"""

import argparse
import datetime
import os
import pathlib
import re
import sys
import tomllib

import requests

# Branch that development (.devN) releases are cut from.
DEV_BRANCH_REF = "refs/heads/main"

# Fallback package name if pyproject.toml has no [project] name.
DEFAULT_PACKAGE = "belfort-torch-mlir"


def get_package_name():
    """Read the package name from pyproject.toml, falling back to the default."""
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with open(pyproject, "rb") as f:
            return tomllib.load(f)["project"]["name"]
    except (KeyError, FileNotFoundError):
        return DEFAULT_PACKAGE


def get_pypi_versions(package_name):
    """Fetch all published versions for a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return list(data.get("releases", {}).keys())
    except Exception as e:
        print(f"Error fetching from PyPI: {e}", file=sys.stderr)
    return []


def get_next_dev_version(package_name):
    """Calculate the next .devN version for today's date."""
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y.%m.%d")
    versions = get_pypi_versions(package_name)

    # Find versions matching today's date and the .dev suffix
    pattern = re.compile(rf"^{re.escape(today)}\.dev(\d+)$")
    max_dev = -1

    for v in versions:
        match = pattern.match(v)
        if match:
            max_dev = max(max_dev, int(match.group(1)))

    next_dev = max_dev + 1
    return f"{today}.dev{next_dev}"


def calculate_version(event, ref, tag, package):
    version = "0.0.0"
    should_publish = "false"

    # NOTE: if/elif rather than match/case so the repo's pinned (py38-target)
    # black can parse this file.
    if event == "workflow_dispatch":
        if tag:
            # Manual release of an existing tag; use for example when the release
            # workflow fails to trigger the wheel upload.
            version = tag.lstrip("v")
            should_publish = "true"
        elif ref == DEV_BRANCH_REF:
            # For dev releases
            version = get_next_dev_version(package)
            should_publish = "true"

    elif event == "schedule":
        if tag:
            version = tag.lstrip("v")
        else:
            version = datetime.datetime.now(datetime.timezone.utc).strftime("%Y.%m.%d")
        should_publish = "true"

    elif event == "pull_request":
        version = "0.0.0"
        should_publish = "false"

    return version, should_publish


def main():
    parser = argparse.ArgumentParser(
        description="Calculate belfort-torch-mlir package version."
    )
    parser.add_argument(
        "--event",
        default="workflow_dispatch",
        help="GitHub event name (e.g., schedule, workflow_dispatch)",
    )
    parser.add_argument(
        "--ref",
        default="refs/heads/main",
        help="GitHub ref (e.g., refs/heads/main)",
    )
    parser.add_argument("--tag", help="Release tag name")
    parser.add_argument(
        "--package",
        default=None,
        help="PyPI package name (defaults to the name in pyproject.toml)",
    )
    parser.add_argument("--gha", action="store_true", help="Output for GitHub Actions")

    args = parser.parse_args()

    package = args.package or get_package_name()
    version, should_publish = calculate_version(args.event, args.ref, args.tag, package)

    if args.gha:
        # Writing to GITHUB_OUTPUT if available
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"version={version}\n")
                f.write(f"should_publish={should_publish}\n")

    print(f"version={version}")
    print(f"should_publish={should_publish}")


if __name__ == "__main__":
    main()
