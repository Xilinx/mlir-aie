#!/usr/bin/env python3

# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bump the pinned llvm-aie (Peano) nightly in utils/peano-version.txt.

CI installs llvm-aie from the Xilinx/llvm-aie `nightly` find-links channel, which
dependabot cannot track (it is a rolling GitHub-release tag, not a PyPI package).
This script resolves the newest nightly from that channel and rewrites the pin so
the update-peano workflow can open a PR; the PR's on-device NPU tests are what
actually gate a bad nightly out of `main`.
"""

import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version

NIGHTLY_INDEX_URL = (
    "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
)
REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "utils" / "peano-version.txt"

# Wheel filenames look like: llvm_aie-21.0.0.2026062501+c83e305a-py3-none-...whl
WHEEL_VERSION_RE = re.compile(r"llvm_aie-([0-9][0-9.]*\+[0-9a-f]+)-")


def get_request(url):
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=60)


def fetch_latest_nightly():
    with get_request(NIGHTLY_INDEX_URL) as response:
        html = response.read().decode("utf-8")
    versions = set(WHEEL_VERSION_RE.findall(html))
    if not versions:
        sys.exit(
            f"error: no llvm-aie wheels found at {NIGHTLY_INDEX_URL}; "
            "the index format may have changed."
        )
    return max(versions, key=Version)


def read_current():
    return VERSION_FILE.read_text().strip()


def write_output(**kwargs):
    """Emit key=value pairs to $GITHUB_OUTPUT (no-op when run locally)."""
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identify-only",
        action="store_true",
        help="Resolve the newest nightly and report changes without editing the file.",
    )
    parser.add_argument(
        "--peano-version",
        default="",
        help="Pin this exact version instead of resolving the newest nightly.",
    )
    args = parser.parse_args()

    current = read_current()
    target = args.peano_version.strip() or fetch_latest_nightly()

    print(f"current: {current}")
    print(f"target:  {target}")

    if target == current:
        print("Already up to date.")
        write_output(target_version=target, changes="false")
        return

    write_output(
        target_version=target,
        changes="true",
        bump_reason=f"{current} -> {target}",
    )

    if args.identify_only:
        return

    VERSION_FILE.write_text(f"{target}\n")
    print(f"Wrote {VERSION_FILE.relative_to(REPO_ROOT)}: {target}")


if __name__ == "__main__":
    main()
