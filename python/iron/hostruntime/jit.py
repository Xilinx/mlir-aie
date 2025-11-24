# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import functools

from ..hostruntime.callabledesign import CallableDesign


def jit(mlir_generator=None, **kwargs):
    """
    Decorator to JIT compile and run an IRON kernel on the NPU.
    """

    if mlir_generator is None:
        return functools.partial(jit, **kwargs)

    return CallableDesign(mlir_generator, **kwargs)
