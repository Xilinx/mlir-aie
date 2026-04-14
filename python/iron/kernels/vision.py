# kernels/vision.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Vision kernel factories: color conversion, threshold, filter2d, add_weighted."""

import numpy as np

from aie.iron.kernel import ExternalFunction

from ._common import _detect_arch, _dtype_to_bit_width, _include_dirs, _kernel_source


def _color_convert_kernel(
    func_name: str, filename: str, in_size: int, out_size: int
) -> ExternalFunction:
    """Shared implementation for color-space conversion line kernels."""
    arch = _detect_arch()
    in_ty = np.ndarray[(in_size,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]

    source = _kernel_source(arch, arch, filename)
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=_include_dirs(),
    )


def _bitwise_kernel(op: str, line_width: int, dtype) -> ExternalFunction:
    """Shared implementation for :func:`bitwise_or` and :func:`bitwise_and`."""
    bit_width = _dtype_to_bit_width(dtype, factory_name=f"bitwise{op}")

    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]

    source = _kernel_source(arch, arch, f"bitwise{op}.cc")
    return ExternalFunction(
        f"bitwise{op}Line",
        source_file=str(source),
        arg_types=[line_ty, line_ty, line_ty, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


def rgba2hue(line_width: int = 1920) -> ExternalFunction:
    """Converts a line of RGBA pixels to hue values.

    Args:
        line_width: Number of pixels per line.

    Returns:
        ExternalFunction configured for ``rgba2hueLine``.
    """
    return _color_convert_kernel(
        "rgba2hueLine", "rgba2hue.cc", line_width * 4, line_width
    )


def threshold(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Applies a threshold operation to a line of pixels.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).

    Returns:
        ExternalFunction configured for ``thresholdLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="threshold")
    scalar_ty = np.int32 if bit_width == 32 else np.int16

    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]

    source = _kernel_source(arch, arch, "threshold.cc")
    return ExternalFunction(
        "thresholdLine",
        source_file=str(source),
        arg_types=[
            line_ty,
            line_ty,
            np.int32,
            scalar_ty,
            scalar_ty,
            np.int8,
        ],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


def bitwise_or(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Element-wise bitwise OR of two lines.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).

    Returns:
        ExternalFunction configured for ``bitwiseORLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    return _bitwise_kernel("OR", line_width, dtype)


def bitwise_and(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Element-wise bitwise AND of two lines.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).

    Returns:
        ExternalFunction configured for ``bitwiseANDLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    return _bitwise_kernel("AND", line_width, dtype)


def gray2rgba(line_width: int = 1920) -> ExternalFunction:
    """Converts a grayscale line to RGBA.

    Args:
        line_width: Number of pixels per line.

    Returns:
        ExternalFunction configured for ``gray2rgbaLine``.
    """
    return _color_convert_kernel(
        "gray2rgbaLine", "gray2rgba.cc", line_width, line_width * 4
    )


def rgba2gray(line_width: int = 1920) -> ExternalFunction:
    """Converts an RGBA line to grayscale.

    Args:
        line_width: Number of pixels per line.

    Returns:
        ExternalFunction configured for ``rgba2grayLine``.
    """
    return _color_convert_kernel(
        "rgba2grayLine", "rgba2gray.cc", line_width * 4, line_width
    )


def filter2d(line_width: int = 1920) -> ExternalFunction:
    """Applies a 3x3 2D convolution filter across three input lines.

    Args:
        line_width: Number of pixels per line.

    Returns:
        ExternalFunction configured for ``filter2dLine``.
    """
    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
    kernel_ty = np.ndarray[(3, 3), np.dtype[np.int16]]

    source = _kernel_source(arch, arch, "filter2d.cc")
    return ExternalFunction(
        "filter2dLine",
        source_file=str(source),
        arg_types=[line_ty, line_ty, line_ty, line_ty, np.int32, kernel_ty],
        include_dirs=_include_dirs(),
    )


def add_weighted(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Weighted addition of two lines with a gamma offset.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).

    Returns:
        ExternalFunction configured for ``addWeightedLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="add_weighted")
    _gamma_types = {8: np.int8, 16: np.int16, 32: np.int32}
    gamma_ty = _gamma_types[bit_width]

    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]

    source = _kernel_source(arch, arch, "addWeighted.cc")
    return ExternalFunction(
        "addWeightedLine",
        source_file=str(source),
        arg_types=[
            line_ty,
            line_ty,
            line_ty,
            np.int32,
            np.int16,
            np.int16,
            gamma_ty,
        ],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )
