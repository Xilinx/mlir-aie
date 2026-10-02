#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject submodule pointer changes that are not fast-forward bumps.

A stale submodule checkout picked up by `git commit -a` or `git add .` silently
moves the pointer back to whatever commit happens to be checked out. A pointer
change is allowed only when the new commit descends from the old one.

Usage:
    python utils/check_submodule_regression.py                # staged (pre-commit)
    python utils/check_submodule_regression.py --from-ref origin/main --to-ref HEAD
"""

import argparse
import os
import subprocess
import sys

GITLINK_MODE = "160000"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def changed_gitlinks(from_ref, to_ref):
    """Yield (path, old_sha, new_sha) for every modified submodule pointer."""
    cmd = ["diff", "--raw", "--no-abbrev", "--no-renames", "--ignore-submodules=none"]
    if from_ref:
        cmd.append(f"{from_ref}...{to_ref}")
    else:
        cmd.append("--cached")
    out = git(*cmd)
    if out.returncode:
        sys.exit(out.stderr.strip())
    for line in out.stdout.splitlines():
        meta, path = line.split("\t", 1)
        old_mode, new_mode, old, new, _status = meta.lstrip(":").split()
        if old_mode == new_mode == GITLINK_MODE and old != new:
            yield path, old, new


def check(path, old, new):
    """Return a problem description, or None if old -> new is a bump."""
    span = f"{path}: {old[:12]} -> {new[:12]}"
    top = git("rev-parse", "--show-toplevel", cwd=path) if os.path.isdir(path) else None
    if not top or top.returncode or not os.path.samefile(top.stdout.strip(), path):
        return f"{span}: submodule not initialized; run `git submodule update --init {path}`"
    for sha in (old, new):
        if git("cat-file", "-e", f"{sha}^{{commit}}", cwd=path).returncode:
            return f"{span}: commit {sha[:12]} not found; run `git -C {path} fetch`"
    if git("merge-base", "--is-ancestor", old, new, cwd=path).returncode == 0:
        return None
    if git("merge-base", "--is-ancestor", new, old, cwd=path).returncode == 0:
        return f"{span}: rolls the submodule back"
    return f"{span}: new commit does not descend from the old one"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-ref")
    p.add_argument("--to-ref", default="HEAD")
    p.add_argument("filenames", nargs="*", help="ignored; pre-commit passes these")
    args = p.parse_args()

    # Set by pre-commit on pre-push and when run with --from-ref/--to-ref.
    from_ref = args.from_ref or os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF") or args.to_ref

    os.chdir(git("rev-parse", "--show-toplevel").stdout.strip())
    problems = [
        msg
        for path, old, new in changed_gitlinks(from_ref, to_ref)
        if (msg := check(path, old, new))
    ]
    for msg in problems:
        print(msg)
    if problems:
        print(
            "\nSubmodule pointers may only move forward. If a stale checkout was "
            "staged, run\n`git submodule update` and restage. For an intentional "
            "downgrade, rerun with\nSKIP=check-submodule-regression."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
