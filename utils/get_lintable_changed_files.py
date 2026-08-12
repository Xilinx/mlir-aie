#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# clang-tidy-diff hard-fails (and takes the whole lint job down) on any
# changed file absent from compile_commands.json: instead of skipping the
# file, it falls back to a bare, flag-less compile that can't find its own
# includes. That's the normal case for anything needing headers this lint
# build doesn't have (XRT, Peano, ...), e.g. test/npu-xrt/**/test.cpp --
# regardless of which PR touches them. Scope the diff clang-tidy-diff sees to
# files it can actually compile.

import json
import shlex
import subprocess
import sys
from pathlib import Path

compile_commands_path = Path(
    sys.argv[1] if len(sys.argv) > 1 else "build/compile_commands.json"
)
with open(compile_commands_path) as f:
    compilable = {entry["file"] for entry in json.load(f)}

repo_root = (
    subprocess.check_output(shlex.split("git rev-parse --show-toplevel"))
    .decode()
    .strip()
)

changed_files = (
    subprocess.check_output(shlex.split("git diff --name-only origin/main"))
    .decode()
    .split()
)

for f in changed_files:
    if f"{repo_root}/{f}" in compilable:
        print(f)
