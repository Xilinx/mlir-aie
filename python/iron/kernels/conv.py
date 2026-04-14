# kernels/conv.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Convolution kernel factories: conv2dk1/3/14, bottleneck (bn_*) variants."""

import numpy as np

from aie.iron.kernel import ExternalFunction

from ._common import (
    _conv_act_dtype_info,
    _detect_arch,
    _include_dirs,
    _kernel_source,
)


def conv2dk1(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """1x1 convolution kernel.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).

    Returns:
        ExternalFunction configured for the conv2dk1 kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1", act_dtype, factory_name="conv2dk1"
    )

    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[act_dtype]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]

    arch = _detect_arch()
    source = _kernel_source(arch, arch, "conv2dk1.cc")
    arg_types = [in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32]
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


def conv2dk3(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """3x3 convolution kernel.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).

    Returns:
        ExternalFunction configured for the conv2dk3 kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk3", act_dtype, factory_name="conv2dk3"
    )

    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[act_dtype]]
    wt_ty = np.ndarray[(3 * 3 * input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]

    arch = _detect_arch()
    source = _kernel_source(arch, arch, "conv2dk3.cc")
    arg_types = [
        line_ty,
        line_ty,
        line_ty,
        wt_ty,
        out_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ]
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


def conv2dk1_skip(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """1x1 convolution kernel with skip (residual) connection.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).

    Returns:
        ExternalFunction configured for the conv2dk1_skip kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1_skip", act_dtype, factory_name="conv2dk1_skip"
    )

    half_ch = input_channels // 2
    in0_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    in1_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]
    skip_ty = np.ndarray[(input_width * output_channels,), np.dtype[act_dtype]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "conv2dk1_skip.cc")
    arg_types = [
        in0_ty,
        in1_ty,
        wt_ty,
        out_ty,
        skip_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ]
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


def conv2dk1_i8(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """1x1 convolution kernel with int8 activations/weights/output.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        ExternalFunction configured for the conv2dk1_i8 kernel.
    """
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.int8]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.int8]]

    arch = _detect_arch()
    source = _kernel_source(arch, arch, "conv2dk1_i8.cc")
    arg_types = [in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32]
    return ExternalFunction(
        "conv2dk1_i8",
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=["-DINT8_ACT"],
    )


def conv2dk14(
    input_width: int = 224,
    input_channels: int = 16,
    output_channels: int = 16,
    kernel_width: int = 14,
) -> ExternalFunction:
    """14x14 convolution kernel (aie2p only).

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        kernel_width: Width (and height) of the convolution kernel.

    Returns:
        ExternalFunction configured for the conv2dk14 kernel.
    """
    tiles = input_width // kernel_width
    pixels = kernel_width * kernel_width
    _RGBA = 4
    _ACC_FACTOR = 8
    in_ty = np.ndarray[(tiles * pixels * _RGBA,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(output_channels * pixels * _RGBA,), np.dtype[np.int8]]
    out_ty = np.ndarray[(output_channels * tiles * _ACC_FACTOR,), np.dtype[np.int8]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2p", "conv2dk14.cc")
    arg_types = [
        in_ty,
        wt_ty,
        out_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ]
    return ExternalFunction(
        "conv2dk14_i8",
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
    )


def conv2dk1_skip_init(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """1x1 convolution kernel with skip-init connection.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).

    Returns:
        ExternalFunction configured for the conv2dk1_skip_init kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1_skip_init", act_dtype, factory_name="conv2dk1_skip_init"
    )

    half_ch = input_channels // 2
    in0_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    in1_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]
    skip_ty = np.ndarray[(input_width * output_channels,), np.dtype[act_dtype]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "conv2dk1_skip_init.cc")
    arg_types = [
        in0_ty,
        in1_ty,
        wt_ty,
        out_ty,
        skip_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ]
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


def bn_conv2dk1_relu(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """Bottleneck 1x1 conv + ReLU kernel (int8 in, uint8 out).

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        ExternalFunction configured for the bn_conv2dk1_relu kernel.
    """
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.int8]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk1_relu.cc")
    arg_types = [in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32]
    return ExternalFunction(
        "conv2dk1_relu_i8_ui8",
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
    )


def bn_conv2dk3(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """Bottleneck 3x3 conv with stride-2 kernel (int8 in, uint8 out).

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        ExternalFunction configured for the bn_conv2dk3 kernel.
    """
    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[np.int8]]
    wt_ty = np.ndarray[(3 * 3 * input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk3.cc")
    arg_types = [
        line_ty,
        line_ty,
        line_ty,
        wt_ty,
        out_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ]
    return ExternalFunction(
        "conv2dk3_stride2_i8",
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
    )


def bn_conv2dk1_i8(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """Bottleneck 1x1 conv kernel (uint8 in, int8 out).

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        ExternalFunction configured for the bn_conv2dk1_i8 kernel.
    """
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.int8]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk1_i8.cc")
    arg_types = [in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32]
    return ExternalFunction(
        "conv2dk1_ui8_i8",
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
    )


def bn_conv2dk1_skip(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    skip_dtype=np.uint8,
) -> ExternalFunction:
    """Bottleneck 1x1 conv with skip connection (uint8 in).

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        skip_dtype: Skip connection data type (``np.uint8`` or ``np.int8``).

    Returns:
        ExternalFunction configured for the bn_conv2dk1_skip kernel.

    Raises:
        ValueError: When ``skip_dtype`` is not ``np.uint8`` or ``np.int8``.
    """
    if skip_dtype == np.uint8:
        func_name = "conv2dk1_skip_ui8_ui8_i8"
    elif skip_dtype == np.int8:
        func_name = "conv2dk1_skip_ui8_i8_i8"
    else:
        raise ValueError(
            f"bn_conv2dk1_skip(): skip_dtype must be np.uint8 or np.int8, "
            f"got {skip_dtype}"
        )

    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.int8]]
    skip_ty = np.ndarray[(input_width * output_channels,), np.dtype[skip_dtype]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk1_skip.cc")
    arg_types = [
        in_ty,
        wt_ty,
        out_ty,
        skip_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ]
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
    )


def bn_conv2dk3_dw(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    stride: int = 1,
) -> ExternalFunction:
    """Bottleneck depthwise 3x3 conv + ReLU kernel (uint8 in/out).

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        stride: Convolution stride (1 or 2).

    Returns:
        ExternalFunction configured for the bn_conv2dk3_dw kernel.

    Raises:
        ValueError: When ``stride`` is not 1 or 2.
    """
    if stride not in (1, 2):
        raise ValueError(f"bn_conv2dk3_dw(): stride must be 1 or 2, got {stride}")

    func_name = f"conv2dk3_dw_stride{stride}_relu_ui8_ui8"

    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(3 * 3 * input_channels,), np.dtype[np.int8]]
    out_size = (input_width // stride) * output_channels
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk3_dw.cc")

    if stride == 1:
        arg_types = [
            line_ty,
            line_ty,
            line_ty,
            wt_ty,
            out_ty,
            out_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ]
        return ExternalFunction(
            func_name,
            source_file=str(source),
            arg_types=arg_types,
            include_dirs=_include_dirs(),
        )
    else:
        arg_types = [
            line_ty,
            line_ty,
            line_ty,
            wt_ty,
            out_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ]
        return ExternalFunction(
            func_name,
            source_file=str(source),
            arg_types=arg_types,
            include_dirs=_include_dirs(),
        )
