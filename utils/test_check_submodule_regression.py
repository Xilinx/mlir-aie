#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for utils/check_submodule_regression.py."""

import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "check_submodule_regression.py"
)

ENV = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT_"))}
ENV.update(
    GIT_CONFIG_GLOBAL=os.devnull,
    GIT_CONFIG_NOSYSTEM="1",
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@example.com",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@example.com",
)


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


class SubmoduleRegressionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.sub = os.path.join(self.root, "sub")
        os.mkdir(self.sub)
        git(self.root, "init", "-q", "-b", "main")
        git(self.sub, "init", "-q", "-b", "main")

        # sub history: a -- b, plus d forked from a.
        self.a = self.sub_commit("a")
        self.b = self.sub_commit("b")
        git(self.sub, "checkout", "-q", "-b", "fork", self.a)
        self.d = self.sub_commit("d")

        self.point_at(self.a)
        self.commit("init")

    def sub_commit(self, msg):
        git(self.sub, "commit", "-q", "--allow-empty", "-m", msg)
        return git(self.sub, "rev-parse", "HEAD")

    def point_at(self, sha, path="sub"):
        git(self.root, "update-index", "--add", "--cacheinfo", f"160000,{sha},{path}")

    def commit(self, msg):
        git(self.root, "commit", "-q", "--allow-empty", "-m", msg)

    def run_check(self, *args, env=None):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            cwd=self.root,
            env={**ENV, **(env or {})},
            capture_output=True,
            text=True,
        )

    def assertPasses(self, *args, **kw):
        r = self.run_check(*args, **kw)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def assertFails(self, needle, *args, **kw):
        r = self.run_check(*args, **kw)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(needle, r.stdout)

    def test_bump_passes(self):
        self.point_at(self.b)
        self.assertPasses()

    def test_rollback_fails(self):
        self.point_at(self.b)
        self.commit("bump")
        self.point_at(self.a)
        self.assertFails("rolls the submodule back")

    def test_suggested_fix_clears_a_staged_rollback(self):
        self.point_at(self.b)
        self.commit("bump")
        self.point_at(self.a)
        self.assertFails("git restore --staged -- sub")
        git(self.root, "restore", "--staged", "--", "sub")
        self.assertPasses()

    def test_path_needing_quotes_is_checked(self):
        path = os.path.join(self.root, "sub\tü")
        os.mkdir(path)
        git(path, "init", "-q")
        git(path, "commit", "-q", "--allow-empty", "-m", "x")
        x = git(path, "rev-parse", "HEAD")
        git(path, "commit", "-q", "--allow-empty", "-m", "y")
        y = git(path, "rev-parse", "HEAD")
        self.point_at(x, "sub\tü")
        self.commit("add")
        self.point_at(y, "sub\tü")
        self.assertPasses()
        self.commit("bump")
        self.point_at(x, "sub\tü")
        self.assertFails("rolls the submodule back")

    def test_move_to_unrelated_commit_fails(self):
        self.point_at(self.b)
        self.commit("bump")
        self.point_at(self.d)
        self.assertFails("does not descend")

    def test_unknown_commit_fails(self):
        self.point_at("1" * 40)
        self.assertFails("not found")

    def test_uninitialized_submodule_fails(self):
        os.mkdir(os.path.join(self.root, "ghost"))
        self.point_at(self.a, "ghost")
        self.commit("add ghost")
        self.point_at(self.b, "ghost")
        self.assertFails("not initialized")

    def test_other_changes_pass(self):
        with open(os.path.join(self.root, "f.txt"), "w") as f:
            f.write("x\n")
        git(self.root, "add", "f.txt")
        self.assertPasses()

    def test_added_and_removed_submodules_pass(self):
        self.point_at(self.b, "other")
        git(self.root, "rm", "-q", "--cached", "sub")
        self.assertPasses()

    def test_range_rollback_fails(self):
        self.point_at(self.b)
        self.commit("bump")
        git(self.root, "checkout", "-q", "-b", "feature")
        self.point_at(self.a)
        self.commit("stale checkout")
        self.assertFails("rolls the submodule back", "--from-ref", "main")

    def test_range_is_taken_from_the_merge_base(self):
        git(self.root, "branch", "feature")
        self.point_at(self.b)
        self.commit("bump on main")
        git(self.root, "checkout", "-q", "feature")
        self.commit("unrelated work")
        self.assertPasses(
            env={"PRE_COMMIT_FROM_REF": "main", "PRE_COMMIT_TO_REF": "HEAD"}
        )

    def test_push_rewinding_a_branch_fails(self):
        self.point_at(self.b)
        self.commit("bump")
        tip = git(self.root, "rev-parse", "HEAD")
        self.assertFails(
            "rolls the submodule back",
            env={
                "PRE_COMMIT_FROM_REF": tip,
                "PRE_COMMIT_TO_REF": f"{tip}~1",
                "PRE_COMMIT_REMOTE_BRANCH": "refs/heads/main",
            },
        )


if __name__ == "__main__":
    unittest.main()
