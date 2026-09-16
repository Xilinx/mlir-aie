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

    def test_ci_derives_clang_format_version_from_pre_commit(self):
        pre_commit_version = require(
            r"repo: https://github\.com/pre-commit/mirrors-clang-format\n\s+rev: .*?# frozen: v([0-9][^\s]*)",
            self.pre_commit,
            ".pre-commit-config.yaml should pin clang-format with a frozen version comment",
        ).group(1)

        require(
            r"^\s*- name: Read clang-format version from pre-commit config\n\s+id: clang-format-version$",
            self.workflow,
            "lintAndFormat.yml should define a step that reads clang-format's version from pre-commit",
        )
        self.assertIn('.pre-commit-config.yaml").read_text()', self.workflow)
        self.assertIn('>> "$GITHUB_OUTPUT"', self.workflow)
        self.assertIn(
            "clangformat: ${{ steps.clang-format-version.outputs.version }}",
            self.workflow,
        )
        self.assertIn(f"# frozen: v{pre_commit_version}", self.pre_commit)


if __name__ == "__main__":
    unittest.main()
