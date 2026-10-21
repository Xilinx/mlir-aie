#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def default_cache_home() -> str:
    return str(Path.home() / ".npu" / "cache")


def base_cache_home() -> str:
    return os.environ.get("NPU_CACHE_HOME") or os.environ.get(
        "IRON_CACHE_HOME", default_cache_home()
    )


def cache_home_for_test(base_cache_home: str, test_key: str) -> str:
    digest = hashlib.sha256(test_key.encode("utf-8")).hexdigest()
    return os.path.join(os.path.abspath(base_cache_home), "lit", digest)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(
            "usage: run_with_test_cache.py <test-key> <command> [args...]",
            file=sys.stderr,
        )
        return 1

    test_key, command = argv[0], argv[1:]
    test_cache_home = cache_home_for_test(base_cache_home(), test_key)
    os.makedirs(test_cache_home, exist_ok=True)
    os.environ["NPU_CACHE_HOME"] = test_cache_home
    os.environ["IRON_CACHE_HOME"] = test_cache_home
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
