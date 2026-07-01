#!/usr/bin/env python3
# (c) Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import difflib
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare generated trace JSON with the expected output."
    )
    parser.add_argument("--actual", required=True, help="Generated trace JSON")
    parser.add_argument("--expected", required=True, help="Expected trace JSON")
    args = parser.parse_args()

    actual_path = Path(args.actual)
    expected_path = Path(args.expected)
    actual = actual_path.read_text(encoding="utf-8").splitlines()
    expected = expected_path.read_text(encoding="utf-8").splitlines()

    if actual == expected:
        return 0

    diff = difflib.unified_diff(
        expected,
        actual,
        fromfile=str(expected_path),
        tofile=str(actual_path),
        lineterm="",
    )
    sys.stdout.write("\n".join(diff))
    sys.stdout.write("\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
