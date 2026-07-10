# __init__.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compilation utilities: MLIR module compilation, kernel linking, and cache management."""

import os
from pathlib import Path

from .utils import (
    compile_cxx_core_function,
    compile_mlir_module,
    compile_external_kernel,
    resolve_target_arch,
)

# Compiled kernels are cached inside the `NPU_CACHE_HOME` directory.
NPU_CACHE_HOME = Path(
    os.environ.get("NPU_CACHE_HOME", Path.home() / ".npu" / "cache")
).resolve()
