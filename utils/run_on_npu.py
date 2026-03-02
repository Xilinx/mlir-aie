#!/usr/bin/env python3

# (c) Copyright 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_on_npu.py <npu-kind> <command> [args...]", file=sys.stderr)
        return 2

    # Mimic the existing %run_on_npu*% bash functionality, broadening OS support.
    # "NPU kind" is currently informational only.
    _npu_kind = sys.argv[1]
    command = sys.argv[2:]

    if os.name == "nt":
        completed = subprocess.run(command)
        return completed.returncode

    xrt_dir = os.environ.get("XRT_DIR") or os.environ.get("XRT_ROOT") or "/opt/xilinx/xrt"
    setup_script = os.path.join(xrt_dir, "setup.sh")
    if not os.path.isfile(setup_script):
        completed = subprocess.run(command)
        return completed.returncode

    bash_command = [
        "bash",
        "-lc",
        'source "$1" >/dev/null 2>&1; shift; exec "$@"',
        "run_on_npu",
        setup_script,
        *command,
    ]
    completed = subprocess.run(bash_command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
