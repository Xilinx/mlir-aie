#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys


def pytest_arguments(arguments: list[str]) -> list[str]:
    runtime = os.environ.get("NPU_RUNTIME")
    if os.environ.get("MLIR_AIE_NPU_TEST") or runtime in {"hrx", "hsa"}:
        return [
            "-n1",
            "--reruns",
            "1",
            "--reruns-delay",
            "3",
            "--rerun-show-tracebacks",
            *arguments,
        ]
    return arguments


def main() -> int:
    command = [sys.executable, "-m", "pytest", *pytest_arguments(sys.argv[1:])]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
