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
    _default_source_path,
    _make_extern,
)


def _i32s(n: int) -> list:
    """Return a list of *n* ``np.int32`` types — for trailing scalar conv args."""
    return [np.int32] * n


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
    return _make_extern(
        func_name,
        _default_source_path("conv2dk1.cc"),
        [in_ty, wt_ty, out_ty, *_i32s(4)],
        compile_flags=flags,
    )


def conv2dk3(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
    weight_output_channels: int | None = None,
) -> ExternalFunction:
    """3x3 convolution kernel.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels produced by this call.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).
        weight_output_channels: Total number of output channels stored in the
            weights buffer. Defaults to ``output_channels``. Set higher than
            ``output_channels`` when the weights buffer is shared across
            multiple workers that each produce a slice of the output (the
            ``channel_offset`` runtime arg selects a worker's slice).

    Returns:
        ExternalFunction configured for the conv2dk3 kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk3", act_dtype, factory_name="conv2dk3"
    )
    if weight_output_channels is None:
        weight_output_channels = output_channels
    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[act_dtype]]
    wt_ty = np.ndarray[
        (3 * 3 * input_channels * weight_output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]
    return _make_extern(
        func_name,
        _default_source_path("conv2dk3.cc"),
        [line_ty, line_ty, line_ty, wt_ty, out_ty, *_i32s(8)],
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
    return _make_extern(
        func_name,
        _default_source_path("conv2dk1_skip.cc", subdir="aie2"),
        [in0_ty, in1_ty, wt_ty, out_ty, skip_ty, *_i32s(5)],
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
    return _make_extern(
        "conv2dk1_i8",
        _default_source_path("conv2dk1_i8.cc"),
        [in_ty, wt_ty, out_ty, *_i32s(4)],
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
    return _make_extern(
        "conv2dk14_i8",
        _default_source_path("conv2dk14.cc", subdir="aie2p"),
        [in_ty, wt_ty, out_ty, *_i32s(5)],
    )


def conv2dk1_skip_init(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
    skip_input_channels: int | None = None,
) -> ExternalFunction:
    """1x1 convolution kernel with skip-init connection.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).
        skip_input_channels: Number of input channels for the skip-projection
            1x1 conv whose weights are concatenated after the main conv
            weights in the same buffer. Defaults to ``input_channels``.

    Returns:
        ExternalFunction configured for the conv2dk1_skip_init kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1_skip_init", act_dtype, factory_name="conv2dk1_skip_init"
    )
    if skip_input_channels is None:
        skip_input_channels = input_channels
    half_ch = input_channels // 2
    total_in_ch = input_channels + skip_input_channels
    in0_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    in1_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(total_in_ch * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]
    skip_ty = np.ndarray[(input_width * skip_input_channels,), np.dtype[act_dtype]]
    return _make_extern(
        func_name,
        _default_source_path("conv2dk1_skip_init.cc", subdir="aie2"),
        [in0_ty, in1_ty, wt_ty, out_ty, skip_ty, *_i32s(7)],
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
    return _make_extern(
        "conv2dk1_relu_i8_ui8",
        _default_source_path("bottleneck/bn_conv2dk1_relu.cc", subdir="aie2"),
        [in_ty, wt_ty, out_ty, *_i32s(4)],
        compile_flags=["-DREGULAR", "-DINT8_ACT"],
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
    # Output is half-resolution because the kernel is stride-2.
    out_ty = np.ndarray[((input_width // 2) * output_channels,), np.dtype[np.uint8]]
    return _make_extern(
        "conv2dk3_stride2_i8",
        _default_source_path("bottleneck/bn_conv2dk3.cc", subdir="aie2"),
        [line_ty, line_ty, line_ty, wt_ty, out_ty, *_i32s(8)],
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
    return _make_extern(
        "conv2dk1_ui8_i8",
        _default_source_path("bottleneck/bn_conv2dk1_i8.cc", subdir="aie2"),
        [in_ty, wt_ty, out_ty, *_i32s(4)],
        compile_flags=["-DREGULAR", "-DSCALAR"],
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
        flags = ["-DREGULAR", "-DSCALAR", "-DUNSIGNED_SKIP"]
    elif skip_dtype == np.int8:
        func_name = "conv2dk1_skip_ui8_i8_i8"
        flags = ["-DREGULAR", "-DSCALAR"]
    else:
        raise ValueError(
            f"bn_conv2dk1_skip(): skip_dtype must be np.uint8 or np.int8, "
            f"got {skip_dtype}"
        )

    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(input_channels * output_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.int8]]
    skip_ty = np.ndarray[(input_width * output_channels,), np.dtype[skip_dtype]]
    return _make_extern(
        func_name,
        _default_source_path("bottleneck/bn_conv2dk1_skip.cc", subdir="aie2"),
        [in_ty, wt_ty, out_ty, skip_ty, *_i32s(5)],
        compile_flags=flags,
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

    return _make_extern(
        func_name,
        _default_source_path("bottleneck/bn_conv2dk3_dw.cc", subdir="aie2"),
        [line_ty, line_ty, line_ty, wt_ty, out_ty, *_i32s(8)],
        compile_flags=["-DREGULAR", "-DSCALAR", f"-DSTRIDE{stride}"],
    )


def bn_conv2dk1_relu_xy_pool_padded(
    input_width: int = 7,
    input_channels: int = 80,
    output_channels: int = 1280,
    weight_chunk_count: int | None = None,
) -> ExternalFunction:
    """Fused 1x1 conv + ReLU + xy-pool with channel padding (int8 in, uint16 out).

    A post-stage kernel that fuses a pointwise (1x1) convolution, ReLU
    activation, and global xy avg-pool into a single pass, with output
    channels padded to a DMA-friendly multiple.  Sized for MobileNet V3's
    post-bottleneck stage where the final 1x1 expand-conv collapses the
    7x7 feature map into a 1x1 vector.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Logical output channels (e.g. 1280).  Sets both
            the output buffer length AND, when ``weight_chunk_count`` is
            None, the weight buffer length (``input_channels * output_channels``).
        weight_chunk_count: Override the weight buffer's element count when
            the design streams weights in chunks (cascade/output-split).
            ``None`` means use the full ``input_channels * output_channels``
            tile.

    Returns:
        ExternalFunction configured for the fused conv+relu+xy_pool kernel.
    """
    wts_count = (
        weight_chunk_count
        if weight_chunk_count is not None
        else input_channels * output_channels
    )
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.int8]]
    wt_ty = np.ndarray[(wts_count,), np.dtype[np.int8]]
    out_ty = np.ndarray[(output_channels,), np.dtype[np.uint16]]
    return _make_extern(
        "conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8",
        _default_source_path("bottleneck/bn_conv2dk1_relu.cc", subdir="aie2"),
        [in_ty, wt_ty, out_ty, *_i32s(8)],
        compile_flags=["-DSCALAR", "-DCONV_XYPOOL_FUSED_LARGE_PADDED", "-DINT8_ACT"],
    )


def bn_fc_relu_ui16_pad(
    input_channels: int = 1280,
    output_channels: int = 16,
    weight_chunk_count: int | None = None,
) -> ExternalFunction:
    """Fully-connected layer (1x1 conv on (1,1,C)) + ReLU, uint16 in/out, with padding.

    A post-stage FC kernel used by MobileNet V3's classifier head.  Input is
    a (1,1,input_channels) feature vector held as uint16; output is
    ``output_channels`` uint16 logits.  Weights stored in a padded layout
    (the ``input_channels_pad`` runtime arg selects the actual stride).

    Args:
        input_channels: Number of input channels (e.g. 1280).
        output_channels: Number of output channels per call (slice width,
            since the full FC is split across multiple tiles).
        weight_chunk_count: Override the weight buffer's element count when
            the design streams weights in chunks (cascade/ping-pong).
            ``None`` means use the full ``input_channels * output_channels``
            tile.

    Returns:
        ExternalFunction configured for the post-L2 FC kernel.
    """
    wts_count = (
        weight_chunk_count
        if weight_chunk_count is not None
        else input_channels * output_channels
    )
    in_ty = np.ndarray[(input_channels,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(wts_count,), np.dtype[np.int8]]
    out_ty = np.ndarray[(output_channels,), np.dtype[np.uint16]]
    return _make_extern(
        "post_L2_conv2dk1_relu_i16_ui16_pad",
        _default_source_path("bottleneck/bn_conv2dk1_relu.cc", subdir="aie2"),
        [in_ty, wt_ty, out_ty, *_i32s(5)],
        compile_flags=["-DSCALAR", "-DPOSTL2_PAD", "-DUINT16_ACT"],
    )
