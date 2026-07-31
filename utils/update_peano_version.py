#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bump the pinned llvm-aie (Peano) nightly in utils/peano-requirements.txt.

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
REQUIREMENTS_FILE = REPO_ROOT / "utils" / "peano-requirements.txt"

# Wheel filenames look like:
#   llvm_aie-21.0.0.2026062501+c83e305a-py3-none-manylinux_2_27_x86_64...whl
#   llvm_aie-21.0.0.2026062501+c83e305a-py3-none-win_amd64.whl
# Capture the version and the platform tag separately: a nightly is only usable
# once it has published a wheel for every platform CI builds on.
WHEEL_RE = re.compile(r"llvm_aie-([0-9][0-9.]*\+[0-9a-f]+)-[^\"]*?-([^\"-]+)\.whl")

# Platform tags CI needs. The Linux runners install the manylinux wheel and the
# Windows runners the win_amd64 one, so a nightly missing either breaks half of
# CI at IRON setup -- with a confusing error, because pip's "from versions:"
# list is filtered to the current platform and so appears to omit the pin
# entirely rather than reporting a platform mismatch.
#
# Match the architecture too, not just the OS: every runner is x86_64, so a
# nightly that shipped only win_arm64 or only linux_aarch64 would satisfy a
# looser OS-only check while still being uninstallable on the machines that
# actually run CI. Add an entry here if CI grows a non-x86_64 runner.
REQUIRED_PLATFORMS = {
    "linux-x86_64": lambda tag: "linux" in tag and tag.endswith("x86_64"),
    "windows-x86_64": lambda tag: tag == "win_amd64",
}

# The pin line in peano-requirements.txt, e.g. llvm-aie==21.0.0.2026062501+c83e305a
PIN_RE = re.compile(r"^(llvm-aie==)(\S+)[^\S\n]*$", re.MULTILINE)


def get_request(url):
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=60)


def fetch_nightly_platforms():
    """Map each nightly version to the set of REQUIRED_PLATFORMS it ships."""
    with get_request(NIGHTLY_INDEX_URL) as response:
        html = response.read().decode("utf-8")
    found = {}
    for version, platform_tag in WHEEL_RE.findall(html):
        names = {
            name
            for name, matches in REQUIRED_PLATFORMS.items()
            if matches(platform_tag)
        }
        if names:
            found.setdefault(version, set()).update(names)
    if not found:
        sys.exit(
            f"error: no llvm-aie wheels found at {NIGHTLY_INDEX_URL}; "
            "the index format may have changed."
        )
    return found


def fetch_latest_nightly():
    """Newest nightly that has a wheel for every platform CI builds on.

    Publishing is not atomic across platforms -- a nightly can ship its Linux
    wheel while the Windows one is delayed or never appears (2026072101 through
    2026072901 had no win_amd64 wheel at all). Taking the newest version
    outright pins something half the CI fleet cannot install, so skip any
    nightly that is not complete.
    """
    found = fetch_nightly_platforms()
    complete = [v for v, plats in found.items() if plats == set(REQUIRED_PLATFORMS)]
    if not complete:
        newest = max(found, key=Version)
        sys.exit(
            f"error: no llvm-aie nightly at {NIGHTLY_INDEX_URL} has wheels for "
            f"all required platforms ({', '.join(sorted(REQUIRED_PLATFORMS))}). "
            f"Newest available is {newest} with: "
            f"{', '.join(sorted(found[newest])) or 'none'}."
        )

    latest = max(complete, key=Version)
    newest_any = max(found, key=Version)
    if Version(newest_any) > Version(latest):
        missing = sorted(set(REQUIRED_PLATFORMS) - found[newest_any])
        print(
            f"note: skipping {newest_any} and newer; no wheel for "
            f"{', '.join(missing)}. Falling back to {latest}."
        )
    return latest


def read_current():
    text = REQUIREMENTS_FILE.read_text()
    match = PIN_RE.search(text)
    if not match:
        sys.exit(
            f"error: no 'llvm-aie==<version>' pin found in "
            f"{REQUIREMENTS_FILE.relative_to(REPO_ROOT)}."
        )
    return match.group(2)


def write_current(target):
    text = REQUIREMENTS_FILE.read_text()
    new_text, count = PIN_RE.subn(rf"\g<1>{target}", text)
    if count != 1:
        sys.exit(
            f"error: expected exactly one 'llvm-aie==' pin in "
            f"{REQUIREMENTS_FILE.relative_to(REPO_ROOT)}, found {count}."
        )
    REQUIREMENTS_FILE.write_text(new_text)


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
    explicit = args.peano_version.strip()
    if explicit:
        # Check the hand-picked version too: pinning one that is missing a
        # platform is exactly the failure this script exists to prevent.
        available = fetch_nightly_platforms().get(explicit, set())
        missing = sorted(set(REQUIRED_PLATFORMS) - available)
        if missing:
            sys.exit(
                f"error: llvm-aie {explicit} has no wheel for "
                f"{', '.join(missing)} at {NIGHTLY_INDEX_URL}."
            )
        target = explicit
    else:
        target = fetch_latest_nightly()

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

    write_current(target)
    print(f"Wrote {REQUIREMENTS_FILE.relative_to(REPO_ROOT)}: {target}")


if __name__ == "__main__":
    main()
