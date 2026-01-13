# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

import functools

from .compilable import Compilable
from .npucallable import NPUCallable


def _unwrap(function):
    """
    Unwrap NPUCallable or Compilable to get the original function.
    This allows stacking decorators where the outer one takes precedence.
    """
    if isinstance(function, NPUCallable) and function.compilable:
        return _unwrap(function.compilable.function)
    if isinstance(function, Compilable):
        return _unwrap(function.function)
    return function


def compileconfig(function=None, is_placed=True, use_cache=True):
    """
    Decorator to create a Compilable object from a function.

    Args:
        function (callable, optional): The function to compile.
        is_placed (bool, optional): Whether the kernel is using explicit or implicit placement. Defaults to True.
        use_cache (bool, optional): Use cached MLIR module if available. Defaults to True.

    Returns:
        Compilable: The compilable object.
    """
    if function is None:
        return functools.partial(
            compileconfig, is_placed=is_placed, use_cache=use_cache
        )

    function = _unwrap(function)
    return Compilable(function, is_placed=is_placed, use_cache=use_cache)


def jit(function=None, is_placed=True, use_cache=True):
    """
    Decorator to create a lazy NPUCallable from a function.
    The function will be compiled on the first call using the arguments provided.

    Args:
        function (callable, optional): The function to compile.
        is_placed (bool, optional): Whether the kernel is using explicit or implicit placement. Defaults to True.
        use_cache (bool, optional): Use cached MLIR module if available. Defaults to True.

    Returns:
        NPUCallable: The lazy callable.
    """
    if function is None:
        return functools.partial(jit, is_placed=is_placed, use_cache=use_cache)

    function = _unwrap(function)
    compilable = Compilable(function, is_placed=is_placed, use_cache=use_cache)
    return NPUCallable(compilable=compilable)
