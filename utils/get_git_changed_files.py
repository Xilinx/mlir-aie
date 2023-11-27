#!/usr/bin/env python3
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
