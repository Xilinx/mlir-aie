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

# The `iron.compiledesign` decorator below caches compiled kenrels inside the `IRON_CACHE_HOME` directory.
# Kernels are cached based on their hash value of the MLIR module string. If during compilation,
# we hit in the cache, the `iron.jit` will load the xclbin and instruction binary files from the cache.
IRON_CACHE_HOME = os.environ.get("IRON_CACHE_HOME", Path.home() / ".iron" / "cache")
