#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fail if any submodule's pin moves backward relative to origin/main.

Submodule gitlinks are merged like ordinary blobs: a PR branch based on an
old commit of main that never rebases can silently reset a submodule pin to
whatever it was on that old base, undoing a bump that has since landed on
main. third_party/aie_api hit exactly this twice (#3292, then again via
#3296, fixed by #3330).

This check is deliberately generic across every submodule in .gitmodules,
not just aie_api, and it does not need to know which branch/tag each one is
meant to track (cmake/modulesXilinx, platforms/boards, third_party/aie-rt,
and third_party/bootgen all track different release branches/tags of their
own). It only cares whether, for each path, HEAD's pin is equal to or a
descendant of origin/main's pin -- which is exactly what a silent revert
from a stale-branch merge is not.

A submodule whose .gitmodules url also changed in this diff is still held
to the same standard: the ancestry check runs against the *new* url instead
of being skipped, since a bad merge could just as easily regress the url
back to an old fork alongside the pin (that combination must not silently
pass just because "the url changed"). A deliberate remote switch (see
#3262) can legitimately be a non-fast-forward change -- when it is, this
script fails and says so, rather than silently waving it through; landing
one is a manual, reviewed action (e.g. an admin merge), not something this
check should rubber-stamp.
"""

import subprocess
import sys

BASE_REF = "origin/main"

# Ancestry checks only need commit/tree objects, not blobs or tags.
FETCH_FLAGS = ["--quiet", "--force", "--filter=blob:none", "--no-tags"]


def run(*args):
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout.strip()


def try_run(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip()


def submodule_paths_and_urls(rev):
    ok, paths_out = try_run(
        "git",
        "config",
        f"--blob={rev}:.gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
    )
    if not ok:
        return {}

    urls = dict(
        line.split(" ", 1)
        for line in run(
            "git",
            "config",
            f"--blob={rev}:.gitmodules",
            "--get-regexp",
            r"^submodule\..*\.url$",
        ).splitlines()
    )

    path_by_url_key = {}
    for line in paths_out.splitlines():
        key, path = line.split(" ", 1)
        url_key = key[: -len(".path")] + ".url"
        path_by_url_key[path] = urls.get(url_key)

    return path_by_url_key


def pinned_commit(rev, path):
    ok, out = try_run("git", "ls-tree", rev, "--", path)
    if not ok or not out.strip():
        return None
    return out.split()[2]


def fetch_into(url, commit, ref):
    return try_run("git", "fetch", *FETCH_FLAGS, url, f"{commit}:{ref}")[0]


def is_forward_move(path, url, old_pin, new_pin):
    """Returns True if new_pin is reachable from old_pin at url, False if
    it provably isn't, or None if that couldn't be determined (e.g. old_pin
    no longer exists on any ref reachable from url)."""
    old_ref = f"refs/tmp/submodule-check-old-{abs(hash(path))}"
    new_ref = f"refs/tmp/submodule-check-new-{abs(hash(path))}"

    if not fetch_into(url, old_pin, old_ref) or not fetch_into(url, new_pin, new_ref):
        return None

    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", old_ref, new_ref]
        ).returncode
        == 0
    )


def main():
    if not try_run("git", "fetch", "--quiet", "origin", "main")[0]:
        print(f"::error::Could not fetch {BASE_REF} from origin to compare against.")
        sys.exit(1)

    head_submodules = submodule_paths_and_urls("HEAD")
    base_submodules = submodule_paths_and_urls(BASE_REF)

    failures = []
    for path, url in head_submodules.items():
        new_pin = pinned_commit("HEAD", path)
        old_pin = pinned_commit(BASE_REF, path)

        if new_pin is None or old_pin is None:
            print(f"OK: {path} added or removed on this branch, nothing to compare.")
            continue

        if old_pin == new_pin:
            print(f"OK: {path} unchanged ({new_pin}).")
            continue

        url_changed = base_submodules.get(path) != url
        result = is_forward_move(path, url, old_pin, new_pin)

        if result is True:
            note = " (url also changed)" if url_changed else ""
            print(f"OK: {path} moved forward {old_pin} -> {new_pin}{note}")
        else:
            failures.append((path, old_pin, new_pin, url, url_changed, result))

    if failures:
        for path, old_pin, new_pin, url, url_changed, result in failures:
            reason = (
                f"{old_pin} is not reachable from {url} at all"
                if result is None
                else f"{new_pin} is not a descendant of it at {url}"
            )
            url_note = (
                " Its .gitmodules url also changed in this diff -- a "
                "deliberate remote switch can legitimately be non-fast-forward, "
                "but that requires a human to say so explicitly (e.g. an admin "
                "merge), not a silently passing check."
                if url_changed
                else ""
            )
            print(
                f"::error::{path} moved from {old_pin} (on {BASE_REF}) to "
                f"{new_pin}: {reason}. This is what a silent regression from a "
                "stale-branch merge looks like (see #3292, #3330)."
                f"{url_note} If this is an intentional downgrade, rebase onto "
                "the latest main before merging so the pin doesn't silently "
                "clobber a bump that landed after this branch was created."
            )
        sys.exit(1)

    print("All submodule pins are unchanged or moved forward relative to origin/main.")


if __name__ == "__main__":
    main()
