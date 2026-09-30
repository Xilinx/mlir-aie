# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import numpy as np

from .symbolic import is_sym, require, show, sprod, sym_any


def validate_and_clean_sizes_strides(
    sizes: Sequence[int] | None,
    strides: Sequence[int] | None,
    allow_none: bool = False,
    expected_dims: int | None = None,
) -> tuple[Sequence[int] | None, Sequence[int] | None]:
    """Validate sizes and strides, and remove any unused values from upper dimensions if possible.

    Args:
        sizes (Sequence[int] | None): The transformation strides, or None
        strides (Sequence[int] | None): The transformation sizes, or None
        allow_none (bool, optional): Allow sizes and/or strides to be None. Defaults to False.
        expected_dims (int | None, optional): Number of dimensions expected for both sizes and strides. Defaults to None.

    Raises:
        ValueError: Validate sizes and strides

    Returns:
        tuple[Sequence[int] | None, Sequence[int] | None]: The 'cleaned' sizes and strides.
    """
    if not allow_none:
        if sizes is None:
            raise ValueError("Sizes is None, but expected Sequence[int]")
        if strides is None:
            raise ValueError("Strides is None, but expected Sequence[int]")
    # After this point can assume None is ok for sizes/strides

    if expected_dims is not None:
        if expected_dims < 1:
            raise ValueError(f"Expected dimensions ({expected_dims}) should be >= 1")

    if sizes is None and strides is None:
        # nothing to do
        return None, None

    # Validate dimensions
    if (sizes is not None) and len(sizes) == 0:
        raise ValueError("len(sizes) must be >0")
    if (strides is not None) and len(strides) == 0:
        raise ValueError("len(strides) must be >0")

    if sizes and strides:
        if expected_dims:
            if len(sizes) != expected_dims:
                raise ValueError(
                    f"Num dimensions of sizes ({show(sizes)}) is not expected number of dimensions ({expected_dims})"
                )
            if len(strides) != expected_dims:
                raise ValueError(
                    f"Num dimensions of strides ({show(strides)}) is not expected number of dimensions ({expected_dims})"
                )
        elif len(strides) != len(sizes):
            raise ValueError(
                f"len(sizes) ({len(sizes)}) != len(strides) ({len(strides)})"
            )
    if strides:
        num_dims = len(strides)
    else:
        assert sizes is not None
        num_dims = len(sizes)

    # Validate sizes/strides values. A staged (runtime) value becomes a
    # dispatch-time guard instead of a generation-time check.
    if sizes:
        sizes = list(sizes)
        for s in sizes:
            require(s >= 1, f"All sizes must be >= 1, but got {show(sizes)}")
    if strides:
        strides = list(strides)
        for s in strides:
            require(s >= 0, f"All strides must be >= 0, but got {show(strides)}")

    # Clean (set size=1, stride=0 for as many dims as possible). Rank and
    # unit-ness are structural, so a staged size stops the scan.
    if sizes and strides:
        strides = list(strides)
        # Leave last dimension strides as whatever it happens to be
        for i in range(num_dims - 1):
            if is_sym(sizes[i]) or sizes[i] != 1:
                break
            strides[i] = 0
    return sizes, strides


def validate_tensor_dims(
    tensor_dims: Sequence[int], expected_dims: int | None = None
) -> Sequence[int]:
    """Validate dimensions of tensors by ensuring each dimension is > 0 and the dimensionality is as expected.

    Args:
        tensor_dims (Sequence[int]): Tensor dimensions to check
        expected_dims (int | None, optional): Expected number of dimensions. Defaults to None.

    Raises:
        ValueError: Validate the tensor dimensions

    Returns:
        Sequence[int]: The validated tensor dimensions.
    """
    if expected_dims is not None:
        if expected_dims < 1:
            raise ValueError(f"Expected dimensions ({expected_dims}) should be >= 1")
    tensor_dims = list(tensor_dims)

    # Validate tensor dims and offset, then set
    if len(tensor_dims) == 0:
        raise ValueError(
            f"Number of tensor dimensions must be >= 1 (dimensions={show(tensor_dims)})"
        )
    for d in tensor_dims:
        require(
            d >= 1,
            f"Each tensor dimension must be >= 1 (dimensions={show(tensor_dims)})",
        )

    # We can treat a 1-dimensional tensor as a 2-dimensional tensor,
    if len(tensor_dims) == 1:
        tensor_dims = [1, tensor_dims[0]]

    if expected_dims is not None and len(tensor_dims) != expected_dims:
        raise ValueError(
            f"Tensor dimension ({show(tensor_dims)}) does not match expected dimension ({expected_dims})"
        )

    return tensor_dims


def validate_offset(offset: int, tensor_dims: Sequence[int] | None) -> int:
    """Validate an offset into the tensor.

    Primarily checks to see if the offset is a valid index to the tensor.

    Args:
        offset (int): The offset to check.
        tensor_dims (Sequence[int] | None): The dimensions of the tensor the offset corresponds to.

    Raises:
        ValueError: Validate the offset.

    Returns:
        int: The validated offset.
    """
    require(offset >= 0, f"Offset must be >= 0 (offset={show(offset)})")
    if tensor_dims:
        numel = (
            sprod(tensor_dims) if sym_any(tensor_dims) else int(np.prod(tensor_dims))
        )
        require(
            offset < numel,
            f"Offset too large: {show(offset)}. Max value allowed for tensor: {numel}",
        )
    return offset


def row_major_strides(dims: Sequence) -> list:
    """Row-major (C-order) element strides for a tensor of shape ``dims``.

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
    """Check that ``axes`` is a permutation of ``range(rank)``.

    Args:
        axes (Sequence[int]): The permutation to check.
        rank (int): Number of dimensions permuted.
        what (str): Name of the argument, for the error message.

    Raises:
        ValueError: If ``axes`` is not a permutation of ``range(rank)``.

    Returns:
        tuple[int, ...]: The permutation as a tuple of ints.
    """
    axes = tuple(int(a) for a in axes)
    if sorted(axes) != list(range(rank)):
        raise ValueError(f"{what} must be a permutation of range({rank}), got {axes}")
    return axes
