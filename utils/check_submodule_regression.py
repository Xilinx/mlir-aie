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
import shlex
import subprocess
import sys

GITLINK_MODE = "160000"


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="surrogateescape",
    )


def changed_gitlinks(diff_args):
    """Yield (path, old_sha, new_sha) for every modified submodule pointer."""
    out = git(
        "diff",
        "-z",
        "--raw",
        "--no-abbrev",
        "--no-renames",
        "--ignore-submodules=none",
        *diff_args,
    )
    if out.returncode:
        sys.exit(out.stderr.strip())
    fields = out.stdout.split("\0")
    for meta, path in zip(fields[0::2], fields[1::2]):
        old_mode, new_mode, old, new, _status = meta.lstrip(":").split()
        if old_mode == new_mode == GITLINK_MODE and old != new:
            yield path, old, new


def check(path, old, new):
    """Return (problem, regressed), or None if old -> new is a bump."""
    span = f"{path}: {old[:12]} -> {new[:12]}"
    top = git("rev-parse", "--show-toplevel", cwd=path) if os.path.isdir(path) else None
    if not top or top.returncode or not os.path.samefile(top.stdout.strip(), path):
        hint = f"git submodule update --init -- {shlex.quote(path)}"
        return f"{span}: submodule not initialized; run `{hint}`", False
    for sha in (old, new):
        if git("cat-file", "-e", f"{sha}^{{commit}}", cwd=path).returncode:
            hint = f"git -C {shlex.quote(path)} fetch"
            return f"{span}: commit {sha[:12]} not found; run `{hint}`", False
    if git("merge-base", "--is-ancestor", old, new, cwd=path).returncode == 0:
        return None
    if git("merge-base", "--is-ancestor", new, old, cwd=path).returncode == 0:
        return f"{span}: rolls the submodule back", True
    return f"{span}: new commit does not descend from the old one", True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-ref")
    p.add_argument("--to-ref", default="HEAD")
    p.add_argument("filenames", nargs="*", help="ignored; pre-commit passes these")
    args = p.parse_args()

    # Set by pre-commit on pre-push and when run with --from-ref/--to-ref.
    from_ref = args.from_ref or os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF") or args.to_ref

    if not from_ref:
        diff_args = ["--cached"]
    elif os.environ.get("PRE_COMMIT_REMOTE_BRANCH"):
        # Pre-push: from_ref is the remote tip, so a rewind only shows tip to tip.
        diff_args = [from_ref, to_ref]
    else:
        diff_args = [f"{from_ref}...{to_ref}"]

    os.chdir(git("rev-parse", "--show-toplevel").stdout.strip())
    problems = []
    regressed = []
    for path, old, new in changed_gitlinks(diff_args):
        if result := check(path, old, new):
            problems.append(result[0])
            if result[1]:
                regressed.append(shlex.quote(path))
    if not problems:
        return 0

    print("\n".join(problems))
    print("\nSubmodule pointers may only move forward.")
    if regressed and not from_ref:
        paths = " ".join(regressed)
        print(
            "To unstage a stale checkout, run\n"
            f"  git restore --staged -- {paths} && git submodule update -- {paths}"
        )
    elif regressed:
        print("Fix the commit that moved the pointer back before pushing.")
    print("For an intentional downgrade, rerun with SKIP=check-submodule-regression.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
