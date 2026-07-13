#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fail if third_party/aie_api is pinned to a commit that upstream doesn't know
about.

third_party/aie_api tracks stock upstream Xilinx/aie_api directly (no fork,
see #3262) after the local ADF-header patch was replaced by a compile-time
define. The pinned commit has since been silently reverted to a stale,
pre-#3262 commit twice (#3292, #3330), both times by a merge/rebase from a
branch based before the fix landed. Guard against a third recurrence: the
pin must be reachable from upstream's default branch.
"""

import subprocess
import sys

SUBMODULE_PATH = "third_party/aie_api"
DEFAULT_BRANCH = "main"


def run(*args):
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout.strip()


def main():
    url = run(
        "git", "config", "-f", ".gitmodules", "--get",
        f"submodule.{SUBMODULE_PATH}.url",
    )
    pinned = run("git", "ls-tree", "HEAD", "--", SUBMODULE_PATH).split()[2]

    run("git", "fetch", "--quiet", "--force", url, f"{pinned}:refs/tmp/aie-api-pinned")
    run("git", "fetch", "--quiet", "--force", url, f"{DEFAULT_BRANCH}:refs/tmp/aie-api-main")

    upstream_head = run("git", "rev-parse", "refs/tmp/aie-api-main")

    is_ancestor = subprocess.run(
        [
            "git", "merge-base", "--is-ancestor",
            "refs/tmp/aie-api-pinned", "refs/tmp/aie-api-main",
        ]
    ).returncode == 0

    if not is_ancestor:
        print(
            f"::error::{SUBMODULE_PATH} is pinned to {pinned}, which is not "
            f"reachable from {url}@{DEFAULT_BRANCH} ({upstream_head}). This "
            "usually means a merge/rebase silently reverted the submodule "
            "pointer to a stale, pre-defork commit (see #3262, #3292, #3330). "
            "Re-point it to a commit that is an ancestor of upstream "
            f"{DEFAULT_BRANCH}."
        )
        sys.exit(1)

    print(f"OK: {SUBMODULE_PATH} pin ({pinned}) is an ancestor of upstream {DEFAULT_BRANCH}.")


if __name__ == "__main__":
    main()
