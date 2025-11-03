# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import functools

from .callabledesign import CallableDesign
from ..compile.cache import CircularCache

# Global cache for compiled kernels at the function level
# Key: (function_name, args_signature) -> NPUKernel instance
# There is a limit on the number of kernels we have in cache
_compiled_kernels = CircularCache(max_size=1)


def jit(mlir_generator=None, **kwargs):
    """
    Decorator to JIT compile and run an IRON kernel on the NPU.
    """

    if mlir_generator is None:
        return functools.partial(jit, **kwargs)

    return CallableDesign(mlir_generator, **kwargs)
