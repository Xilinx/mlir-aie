#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Regression tests for keeping CI lint tooling aligned with pre-commit."""

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
DEPENDABOT_CONFIG = REPO_ROOT / ".github" / "dependabot.yml"
LINT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "lintAndFormat.yml"


def require(pattern, text, msg, flags=re.MULTILINE):
    match = re.search(pattern, text, flags)
    if not match:
        raise AssertionError(msg)
    return match


class LintDependencySyncTests(unittest.TestCase):
    def setUp(self):
        self.pre_commit = PRE_COMMIT_CONFIG.read_text()
        self.dependabot = DEPENDABOT_CONFIG.read_text()
        self.workflow = LINT_WORKFLOW.read_text()

    def test_ci_ruff_lint_runs_the_pre_commit_hook(self):
        require(
            r"^\s*- name: Lint Python \(ruff\)\n\s+run: pre-commit run ruff-check --hook-stage pre-push --all-files --color never$",
            self.workflow,
            "lintAndFormat.yml should lint Python via the pre-commit ruff hook",
        )

    def test_dependabot_updates_the_pre_commit_config(self):
        require(
            r'- package-ecosystem: "pre-commit"\n\s+directory: "/"',
            self.dependabot,
            "dependabot.yml should track pre-commit updates for the repository root",
        )

    def test_clang_format_hook_covers_td_files(self):
        self.assertIn(
            "- repo: https://github.com/pre-commit/mirrors-clang-format",
            self.pre_commit,
        )
        self.assertIn(r"files: \.(c|cc|cpp|cxx|h|hpp|td)$", self.pre_commit)

    def test_ci_clang_format_runs_the_pre_commit_hook(self):
        self.assertIn("pre-commit run clang-format", self.workflow)
        self.assertIn("--from-ref origin/main", self.workflow)
        self.assertIn("--to-ref HEAD", self.workflow)
        self.assertNotIn("git clang-format origin/main", self.workflow)
        self.assertNotIn("clangformat: ${{", self.workflow)

    def test_pre_commit_pin_remains_the_clang_format_source_of_truth(self):
        self.assertIsNotNone(
            re.search(
                r"repo: https://github\.com/pre-commit/mirrors-clang-format\n\s+rev: .*?# frozen: v([0-9][^\s]*)",
                self.pre_commit,
                re.MULTILINE,
            ),
            ".pre-commit-config.yaml should pin clang-format with a frozen version comment",
        )


if __name__ == "__main__":
    unittest.main()
