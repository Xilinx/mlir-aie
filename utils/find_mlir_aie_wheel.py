#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Find the published mlir_aie wheel built from a specific git commit.

The rolling "latest-wheels*" GitHub releases each accumulate many wheels. Every
wheel encodes the commit it was built from in its setuptools_scm local version,
e.g. `mlir_aie-1.3.2.dev115+g2401e53-cp312-...whl` where `2401e53` is the
abbreviated commit sha. Given a commit sha and a list of release channels, this
scans each channel's `expanded_assets` page (the same listing pip consumes via
`-f`; unlike the GitHub API it is not rate limited) for a wheel whose embedded
short sha is a prefix of the commit.

On a match it prints `<channel> <version>` and exits 0. If no channel has a
matching wheel it prints nothing and exits 1.

Usage:
    find_mlir_aie_wheel.py <commit-sha>
"""

import argparse
import re
import sys
import urllib.request

EXPANDED_ASSETS = "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/"
WHEEL_RE = re.compile(r"mlir_aie-[A-Za-z0-9_.+-]*\.whl")

# Rolling channels published as GitHub releases, searched newest first.
CHANNELS = ["latest-wheels-4", "latest-wheels-3", "latest-wheels-2", "latest-wheels"]


def find_wheel(commit, channels):
    """Return (channel, version) for the wheel built from commit, or None."""
    for channel in channels:
        try:
            req = urllib.request.Request(
                EXPANDED_ASSETS + channel, headers={"User-Agent": "pip"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", "replace")
        except Exception:
            continue
        for name in WHEEL_RE.findall(html):
            # name: mlir_aie-<version>-<py>-<abi>-<platform>.whl
            version = name.split("-")[1]
            if "+" not in version:
                continue
            # Local version is e.g. "g2401e53" (optionally ".dirty"/".dYYYYMMDD").
            local = version.split("+", 1)[1].split(".")[0]
            if local[:1] == "g":
                local = local[1:]
            if len(local) >= 7 and commit.startswith(local):
                return channel, version
    return None


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("commit", help="full git commit sha to match")
    args = parser.parse_args(argv)

    match = find_wheel(args.commit, CHANNELS)
    if match is None:
        return 1
    print("%s %s" % match)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
