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

from ._common import (
    _default_source_path,
    _dtype_to_bit_width,
    _make_extern,
)


def _color_convert_kernel(
    func_name: str, filename: str, in_size: int, out_size: int, use_chess: bool = False
) -> ExternalFunction:
    """Shared implementation for color-space conversion line kernels."""
    in_ty = np.ndarray[(in_size,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]
    return _make_extern(
        func_name,
        _default_source_path(filename),
        [in_ty, out_ty, np.int32],
        use_chess=use_chess,
    )


def _bitwise_kernel(
    op: str, line_width: int, dtype, use_chess: bool = False
) -> ExternalFunction:
    """Shared implementation for :func:`bitwise_or` and :func:`bitwise_and`."""
    bit_width = _dtype_to_bit_width(dtype, factory_name=f"bitwise{op}")
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        f"bitwise{op}Line",
        _default_source_path(f"bitwise{op}.cc"),
        [line_ty, line_ty, line_ty, np.int32],
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
        use_chess=use_chess,
    )


def rgba2hue(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Converts a line of RGBA pixels to hue values."""
    return _color_convert_kernel(
        "rgba2hueLine", "rgba2hue.cc", line_width * 4, line_width, use_chess=use_chess
    )


def threshold(
    line_width: int = 1920, dtype=np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Applies a threshold operation to a line of pixels.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).
        use_chess: When ``True``, build the .o with ``xchesscc_wrapper``
            instead of Peano.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="threshold")
    scalar_ty = np.int32 if bit_width == 32 else np.int16
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        "thresholdLine",
        _default_source_path("threshold.cc"),
        [line_ty, line_ty, np.int32, scalar_ty, scalar_ty, np.int8],
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
        use_chess=use_chess,
    )


def bitwise_or(
    line_width: int = 1920, dtype=np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Element-wise bitwise OR of two lines."""
    return _bitwise_kernel("OR", line_width, dtype, use_chess=use_chess)


def bitwise_and(
    line_width: int = 1920, dtype=np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Element-wise bitwise AND of two lines."""
    return _bitwise_kernel("AND", line_width, dtype, use_chess=use_chess)


def gray2rgba(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Converts a grayscale line to RGBA."""
    return _color_convert_kernel(
        "gray2rgbaLine", "gray2rgba.cc", line_width, line_width * 4, use_chess=use_chess
    )


def rgba2gray(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Converts an RGBA line to grayscale."""
    return _color_convert_kernel(
        "rgba2grayLine", "rgba2gray.cc", line_width * 4, line_width, use_chess=use_chess
    )


def filter2d(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Applies a 3x3 2D convolution filter across three input lines."""
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
    kernel_ty = np.ndarray[(3, 3), np.dtype[np.int16]]
    return _make_extern(
        "filter2dLine",
        _default_source_path("filter2d.cc"),
        [line_ty, line_ty, line_ty, line_ty, np.int32, kernel_ty],
        use_chess=use_chess,
    )


def add_weighted(
    line_width: int = 1920, dtype=np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Weighted addition of two lines with a gamma offset.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).
        use_chess: When ``True``, build the .o with ``xchesscc_wrapper``
            instead of Peano.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="add_weighted")
    gamma_ty = {8: np.int8, 16: np.int16, 32: np.int32}[bit_width]
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        "addWeightedLine",
        _default_source_path("addWeighted.cc"),
        [line_ty, line_ty, line_ty, np.int32, np.int16, np.int16, gamma_ty],
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
        use_chess=use_chess,
    )
