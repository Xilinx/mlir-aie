# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validation and stride helpers shared by the access-pattern algebra."""

from typing import Sequence

from ._symbolic import IntLike, is_sym, require, show, sprod


def validate_and_clean_sizes_strides(
    sizes: Sequence[IntLike], strides: Sequence[IntLike]
) -> tuple[list[IntLike], list[IntLike]]:
    """Validate sizes and strides, and zero the strides of leading unit dimensions.

    A check on a staged value becomes a dispatch-time guard.

    Args:
        sizes (Sequence[IntLike]): Extent of each dimension, outermost first.
        strides (Sequence[IntLike]): Element step of each dimension, outermost first.

    Returns:
        tuple[list[IntLike], list[IntLike]]: The sizes and the cleaned strides.

    Raises:
        ValueError: If the lists are empty or differ in length, a size is
            below 1 or a stride below 0.
    """
    sizes, strides = list(sizes), list(strides)
    if not sizes:
        raise ValueError("len(sizes) must be >0")
    if len(strides) != len(sizes):
        raise ValueError(f"len(sizes) ({len(sizes)}) != len(strides) ({len(strides)})")
    for s in sizes:
        require(s >= 1, f"All sizes must be >= 1, but got {show(sizes)}")
    for s in strides:
        require(s >= 0, f"All strides must be >= 0, but got {show(strides)}")
    return sizes, zero_leading_unit_strides(sizes, strides)


def zero_leading_unit_strides(
    sizes: Sequence, strides: Sequence, start: int = 0
) -> list:
    """Zero the stride of each unit dimension from `start` up to the first that steps.

    A unit dimension never steps, so this makes equal walks compare equal.
    The innermost stride is left as is. Rank and unit-ness are structural,
    so a staged size ends the scan.

    Args:
        sizes (Sequence): Extent of each dimension, outermost first.
        strides (Sequence): Element step of each dimension, outermost first.
        start (int, optional): The first dimension to consider. Defaults to 0.

    Returns:
        list: The strides, with those of the leading unit dimensions zeroed.
    """
    strides = list(strides)
    for i in range(start, len(sizes) - 1):
        if is_sym(sizes[i]) or sizes[i] != 1:
            break
        strides[i] = 0
    return strides


def validate_tensor_dims(tensor_dims: Sequence[IntLike]) -> list[IntLike]:
    """Check that a tensor has at least one dimension and every dimension is >= 1.

    Args:
        tensor_dims (Sequence[IntLike]): Tensor dimensions to check.

    Returns:
        list[IntLike]: The tensor dimensions.

    Raises:
        ValueError: If there are no dimensions or a dimension is below 1.
    """
    tensor_dims = list(tensor_dims)
    if not tensor_dims:
        raise ValueError(
            f"Number of tensor dimensions must be >= 1 (dimensions={show(tensor_dims)})"
        )
    for d in tensor_dims:
        require(
            d >= 1,
            f"Each tensor dimension must be >= 1 (dimensions={show(tensor_dims)})",
        )
    return tensor_dims


def validate_offset(offset: IntLike, tensor_dims: Sequence[IntLike]) -> IntLike:
    """Check that `offset` is an element index into a tensor of shape `tensor_dims`.

    Args:
        offset (IntLike): The offset to check.
        tensor_dims (Sequence[IntLike]): Shape of the tensor.

    Returns:
        IntLike: The offset.

    Raises:
        ValueError: If the offset is negative or past the last element.
    """
    require(offset >= 0, f"Offset must be >= 0 (offset={show(offset)})")
    numel = sprod(tensor_dims)
    require(
        offset < numel,
        f"Offset too large: {show(offset)}. Max value allowed for tensor: {show(numel)}",
    )
    return offset


def row_major_strides(dims: Sequence) -> list:
    """Row-major (C-order) element strides for a tensor of shape `dims`.

    Args:
        dims (Sequence): Tensor dimensions; entries may be staged values.

    Returns:
        list: One stride per dimension, outermost first.
    """
    strides: list = [1] * len(dims)
    for axis in range(len(dims) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * dims[axis + 1]
    return strides


def validate_permutation(axes: Sequence[int], rank: int, what: str) -> tuple[int, ...]:
    """Check that `axes` is a permutation of `range(rank)`.

    Args:
        axes (Sequence[int]): The permutation to check.
        rank (int): Number of dimensions permuted.
        what (str): Name of the argument, for the error message.

    Returns:
        tuple[int, ...]: The permutation as a tuple of ints.

    Raises:
        ValueError: If `axes` is not a permutation of `range(rank)`.
    """
    axes = tuple(int(a) for a in axes)
    if sorted(axes) != list(range(rank)):
        raise ValueError(f"{what} must be a permutation of range({rank}), got {axes}")
    return axes
