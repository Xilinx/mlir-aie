# algorithms.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

from typing import Optional, Callable
from .tensor import Tensor
from .graph import is_graph_capture_enabled
from .detail.matmul import matmul_impl, matmul_graph_capture_impl
from .detail.for_each import for_each_impl, for_each_graph_capture_impl

def matmul(input: Tensor, other: Tensor, out: Optional[Tensor] = None, async_mode: bool = True) -> Tensor:
    """
    Matrix product of two tensors, similar to torch.matmul.

    This function switches between regular implementation and graph capture
    implementation based on the current graph capture mode.

    Args:
        input: First input tensor
        other: Second input tensor
        out: Optional output tensor
        async_mode: Whether to use asynchronous execution

    Returns:
        Tensor: The matrix product of the input tensors
    """
    if is_graph_capture_enabled():
        return matmul_graph_capture_impl(input, other, out, async_mode)
    else:
        return matmul_impl(input, other, out, async_mode)


def for_each(input: Tensor, func: Callable, async_mode: bool = True) -> Tensor:
    """
    Apply a function to each element of a tensor in-place.

    This is an in-place operation that modifies the input tensor.
    This function switches between regular implementation and graph capture
    implementation based on the current graph capture mode.

    Args:
        input: Input tensor to modify in-place
        func: Function to apply to each element
        async_mode: Whether to use asynchronous execution

    Returns:
        Tensor: The same input tensor (modified in-place)
    """
    if is_graph_capture_enabled():
        return for_each_graph_capture_impl(input, func, async_mode)
    else:
        return for_each_impl(input, func, async_mode)