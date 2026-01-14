# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable, NoReturn
from pathlib import Path

__all__ = [
    "MLIR_AIE_BIN_DIR",
    "aie_lsp_server",
    "aie_opt",
    "aie_reset",
    "aie_translate",
    "aie_visualize",
    "bootgen",
    "xchesscc_wrapper",
]


def __dir__() -> list[str]:
    return __all__


MLIR_AIE_BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "bin")
assert os.path.isdir(MLIR_AIE_BIN_DIR), "MLIR-AIE tools directory does not exist"


def _program(name: str, args: Iterable[str]) -> int:
    return subprocess.call(
        [os.path.join(MLIR_AIE_BIN_DIR, name), *args], close_fds=False
    )


def _program_exit(name: str, *args: str) -> NoReturn:
    if sys.platform.startswith("win"):
        raise SystemExit(_program(name, args))
    cmake_exe = os.path.join(MLIR_AIE_BIN_DIR, name)
    os.execl(cmake_exe, cmake_exe, *args)


def aie_lsp_server() -> NoReturn:
    _program_exit("aie-lsp-server", *sys.argv[1:])


def aie_opt() -> NoReturn:
    _program_exit("aie-opt", *sys.argv[1:])


def aie_reset() -> NoReturn:
    _program_exit("aie-reset", *sys.argv[1:])


def aie_translate() -> NoReturn:
    _program_exit("aie-translate", *sys.argv[1:])


def aie_visualize() -> NoReturn:
    _program_exit("aie-visualize", *sys.argv[1:])


def bootgen() -> NoReturn:
    _program_exit("bootgen", *sys.argv[1:])


def xchesscc_wrapper() -> NoReturn:
    _program_exit("xchesscc_wrapper", *sys.argv[1:])
