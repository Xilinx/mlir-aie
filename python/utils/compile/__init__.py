# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
from pathlib import Path

from .link import merge_object_files
from .utils import (
    compile_cxx_core_function,
    compile_mlir_module,
    compile_external_kernel,
)

# Compiled kernels are cached inside the `NPU_CACHE_HOME` directory.
NPU_CACHE_HOME = Path(
    os.environ.get("NPU_CACHE_HOME", Path.home() / ".npu" / "cache")
).resolve()
