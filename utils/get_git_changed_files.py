#!/usr/bin/env python3

# Copyright (C) 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import shlex
import subprocess
from pathlib import Path

changed_files = (
    subprocess.check_output(shlex.split("git diff --name-only origin/main HEAD"))
    .decode()
    .split()
)
cov_files = list(
    filter(lambda f: re.search(r"(\.cpp|\.c|\.h|\.hpp)$", f), changed_files)
)
# Only files under these directories are actually compiled into the
# instrumented binaries (aie-opt/aie-translate/aiecc). A file outside them
# (e.g. a kernel source under programming_examples/) will never appear in
# the coverage mapping, and handing llvm-cov a --restrict path that matches
# nothing makes it silently report on every file instead.
cov_files = [f for f in cov_files if f.startswith(("lib/", "include/", "tools/"))]
# CMakeLists.txt's default INSTRUMENTED_COVERAGE_FILES also lists python/,
# but coverage cannot be attributed reliably to python.exe, so exclude any
# file under a directory literally named python/ that slips into the three
# dirs above too. (A path segment match, not a substring match, so e.g.
# lib/Foo/PythonLikeThing.cpp is unaffected.)
cov_files = [
    f for f in cov_files if "python" not in (p.lower() for p in Path(f).parts[:-1])
]
print(
    ";".join(
        [
            os.environ.get(
                "GITHUB_WORKSPACE", str(Path(__file__).parent.parent.absolute())
            )
            + "/"
            + c
            for c in cov_files
        ]
    )
)
