#!/usr/bin/env python3

# Copyright (C) 2018-2026 Advanced Micro Devices, Inc.
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
# Can't track back to python.exe for some reason so don't bother.
cov_files = [f for f in cov_files if "python" not in f]
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
