# kernels/conv.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Convolution kernel factories: conv2dk1/3/14, bottleneck (bn_*) variants.

The conv2dk1, conv2dk1_i8, conv2dk1_skip, conv2dk3 and conv2dk14 factories
specialize their dimensions at compile time. Their runtime dimension arguments
remain in the ABI and must match the factory dimensions; scales, region checks
and channel offsets remain runtime values.
"""

from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    _conv_act_dtype_info,
    _default_source_path,
    _detect_arch,
    _make_extern,
    dtypes,
)
from .core import conv_even


def _i32s(n: int) -> list:
    """Return a list of *n* ``np.int32`` types — for trailing scalar conv args."""
    return [np.int32] * n


def _conv_dimensions(input_width, input_channels, output_channels):
    """Specialize loop bounds while retaining the runtime scalar ABI."""
    return [
        f"-DCONV_INPUT_WIDTH={input_width}",
        f"-DCONV_INPUT_CHANNELS={input_channels}",
        f"-DCONV_OUTPUT_CHANNELS={output_channels}",
    ]


def _requant(acc, scale: int, lo: int = 0, hi: int = 255, dtype: type = np.uint8):
    """``(acc + 2**(scale-1)) >> scale`` saturated to ``[lo, hi]``, as the kernels do."""
    scale = int(scale)
    out = (acc + (1 << (scale - 1))) >> scale if scale > 0 else acc
    return np.clip(out, lo, hi).astype(dtype)


def _conv1x1_acc(x, weights, W: int, IC: int, OC: int):
    """int64 ``[OC/8][W][8]`` sums of a 1x1 conv over ``[IC/8][W][8]`` activations.

    Returns the sums and the leading (``calls``) shape of ``x``.
    """
    x = np.asarray(x)
    lead = x.shape[:-1]
    xi = x.reshape(*lead, IC // 8, W, 8).astype(np.int64)
    w = (
        np.asarray(weights, dtype=np.int8)
        .reshape(OC // 8, IC // 8, 8, 8)
        .astype(np.int64)
    )
    return np.einsum("...iwc,oicp->...owp", xi, w), lead


def conv2dk1_ref(x, weights, input_width, input_channels, output_channels, scale):
    """Numpy reference for [`conv2dk1`][iron.kernels.conv.conv2dk1]: 1x1 conv, requantized.

    Layouts are the kernel's: activations ``[C/8][W][8]`` (``x`` is one
    line of ``input_width * input_channels`` values, or ``(calls, ...)`` of
    them), weights ``[OC/8][IC/8][ic8][oc8]``, output ``[OC/8][W][8]`` as
    ``uint8``. ``out = sat_u8((sum_ic x * w + 2**(scale-1)) >> scale)``,
    i.e. a fused ReLU. Exact for the scalar path of ``conv2dk1.cc``.
    """
    W, IC, OC = int(input_width), int(input_channels), int(output_channels)
    acc, lead = _conv1x1_acc(x, weights, W, IC, OC)
    return _requant(acc, scale).reshape(*lead, W * OC)


def conv2dk1_i8_ref(x, weights, input_width, input_channels, output_channels, scale):
    """Numpy reference for [`conv2dk1_i8`][iron.kernels.conv.conv2dk1_i8]: 1x1 conv to ``int8``.

    The layouts of [`conv2dk1_ref`][iron.kernels.conv.conv2dk1_ref] with an
    ``int8`` output and no ReLU:
    ``out = sat_i8((sum_ic x * w + 2**(scale-1)) >> scale)``. Exact for the
    scalar path of ``conv2dk1_i8.cc``.
    """
    W, IC, OC = int(input_width), int(input_channels), int(output_channels)
    acc, lead = _conv1x1_acc(x, weights, W, IC, OC)
    return _requant(acc, scale, -128, 127, np.int8).reshape(*lead, W * OC)


def conv2dk1_skip_ref(
    x0,
    x1,
    weights,
    skip,
    input_width,
    input_channels,
    output_channels,
    scale,
    skip_scale,
):
    """Numpy reference for [`conv2dk1_skip`][iron.kernels.conv.conv2dk1_skip]: 1x1 conv plus residual.

    ``x0`` and ``x1`` each hold half the input channels (``[IC/16][W][8]``
    lines, ``x1`` the upper half); weights are ``[OC/8][IC/8][ic8][oc8]``
    over all of them and ``skip`` is an ``[OC/8][W][8]`` line. The conv sum
    is requantized and saturated to ``int8`` first, then the residual is
    added and the total requantized to ``uint8``:

    ```text
    conv = sat_i8((sum_ic x * w + 2**(scale-1)) >> scale)
    out  = sat_u8((conv + skip + 2**(skip_scale-1)) >> skip_scale)
    ```

    Exact for the scalar path of ``conv2dk1_skip.cc``. A shift of 0 is no
    shift (the scalar path's ``1 << -1`` is not defined for it; the vector
    path handles 0).
    """
    W, IC, OC = int(input_width), int(input_channels), int(output_channels)
    x0, x1 = np.asarray(x0), np.asarray(x1)
    lead = x0.shape[:-1]
    halves = (x0.reshape(*lead, IC // 16, W, 8), x1.reshape(*lead, IC // 16, W, 8))
    x = np.concatenate(halves, axis=-3).reshape(*lead, W * IC)
    acc, _ = _conv1x1_acc(x, weights, W, IC, OC)
    conv = _requant(acc, scale, -128, 127, np.int64)
    total = conv + np.asarray(skip).reshape(*lead, OC // 8, W, 8).astype(np.int64)
    return _requant(total, skip_scale).reshape(*lead, W * OC)


def conv2dk3_ref(
    line0,
    line1,
    line2,
    weights,
    input_width,
    input_channels,
    output_channels,
    kernel_width,
    kernel_height,
    check,
    scale,
    channel_offset,
):
    """Numpy reference for [`conv2dk3`][iron.kernels.conv.conv2dk3]: 3x3 conv over three lines.

    Produces the output line for ``line1``. Activations are ``[C/8][W][8]``
    lines; weights ``[WOC/8][IC/8][3 rows][kernel_width][ic8][oc8]`` where
    ``WOC`` may exceed ``output_channels`` (``channel_offset`` selects this
    call's slice, in units of 8 channels). The spatial border is zero
    padded; ``check`` is 0 (top: ``line0`` ignored), 1 (middle) or 2
    (bottom: ``line2`` ignored), matching the kernel's ``region`` enum.
    ``out = sat_u8((sum + 2**(scale-1)) >> scale)``. Exact for the scalar
    path of ``conv2dk3.cc``; ``kernel_height`` is accepted for the
    signature and is 3.
    """
    acc, lead, W, OC = _conv3x3_acc(
        (line0, line1, line2),
        weights,
        input_width,
        input_channels,
        output_channels,
        kernel_width,
        check,
        channel_offset,
    )
    del kernel_height
    return _requant(acc, scale).reshape(*lead, W * OC)


def _conv3x3_acc(
    lines,
    weights,
    input_width,
    input_channels,
    output_channels,
    kernel_width,
    check,
    channel_offset,
    stride: int = 1,
):
    """int64 ``[OC/8][W'][8]`` sums of a 3-line 3xKW conv over ``[IC/8][W][8]`` lines.

    Weights ``[WOC/8][IC/8][3][KW][ic8][oc8]``, zero-padded horizontally,
    ``check`` dropping the top or bottom line, ``channel_offset`` selecting
    the output-channel slice; ``stride`` 2 halves the output width.
    Returns ``(acc, lead, W', OC)``.
    """
    W, IC, OC = int(input_width), int(input_channels), int(output_channels)
    KW, check, off = int(kernel_width), int(check), int(channel_offset) // 8
    lines = [np.asarray(line) for line in lines]
    lead = lines[0].shape[:-1]
    w = np.asarray(weights, dtype=np.int8)
    woc = w.size // (IC * 3 * KW * 8)
    w = w.reshape(woc, IC // 8, 3, KW, 8, 8).astype(np.int64)[off : off + OC // 8]
    Wo = W // stride
    acc = np.zeros(lead + (OC // 8, Wo, 8), dtype=np.int64)
    rows = [
        r
        for r in range(3)
        if not (check == 0 and r == 0) and not (check == 2 and r == 2)
    ]
    for r in rows:
        v = lines[r].reshape(*lead, IC // 8, W, 8).astype(np.int64)
        padded = np.pad(v, [(0, 0)] * len(lead) + [(0, 0), (1, 1), (0, 0)])
        for ki in range(KW):
            # output x reads input pixel stride*x + ki - 1
            shifted = padded[..., :, ki : ki + stride * Wo : stride, :]
            acc += np.einsum("...iwc,oicp->...owp", shifted, w[:, :, r, ki])
    return acc, lead, Wo, OC


def conv2dk1_skip_init_ref(
    x0,
    x1,
    weights,
    skip,
    input_width,
    input_channels,
    output_channels,
    skip_input_channels,
    scale,
    skip_scale,
    scale_skip_conv,
):
    """Numpy reference for [`conv2dk1_skip_init`][iron.kernels.conv.conv2dk1_skip_init]: 1x1 conv plus a projected residual.

    Like [`conv2dk1_skip_ref`][iron.kernels.conv.conv2dk1_skip_ref], but the
    residual is itself a 1x1 conv of ``skip`` (``[ICs/8][W][8]``) with the
    weights stored after the main ones (``[OC/8][ICs/8][ic8][oc8]`` at
    offset ``OC * IC``):

    ```text
    conv = sat_i8((sum_ic x * w + 2**(scale-1)) >> scale)
    proj = sat_i8((sum_ics skip * ws + 2**(scale_skip_conv-1)) >> scale_skip_conv)
    out  = sat_u8((conv + proj + 2**(skip_scale-1)) >> skip_scale)
    ```

    Exact for the scalar path of ``conv2dk1_skip_init.cc``; a shift of 0
    is no shift.
    """
    W, IC, OC = int(input_width), int(input_channels), int(output_channels)
    ICs = int(skip_input_channels)
    x0, x1 = np.asarray(x0), np.asarray(x1)
    lead = x0.shape[:-1]
    halves = (x0.reshape(*lead, IC // 16, W, 8), x1.reshape(*lead, IC // 16, W, 8))
    x = np.concatenate(halves, axis=-3).reshape(*lead, W * IC)
    w = np.asarray(weights, dtype=np.int8)
    acc, _ = _conv1x1_acc(x, w[: OC * IC], W, IC, OC)
    conv = _requant(acc, scale, -128, 127, np.int64)
    acc_s, _ = _conv1x1_acc(skip, w[OC * IC : OC * IC + OC * ICs], W, ICs, OC)
    proj = _requant(acc_s, scale_skip_conv, -128, 127, np.int64)
    return _requant(conv + proj, skip_scale).reshape(*lead, W * OC)


def conv2dk14_ref(
    x, weights, input_width, input_channels, output_channels, kernel_width, scale
):
    """Numpy reference for [`conv2dk14`][iron.kernels.conv.conv2dk14]: a KxK patch conv (stride K) to ``int8``.

    One call covers ``T = input_width / kernel_width`` patches of ``K*K``
    RGBA pixels. Layouts are the kernel's: input ``[T/8][P/2][t8][p2][4]``
    ``uint8`` (``P = K*K`` pixels of 4 channels), weights
    ``[OC/8][P/2][p2][4][oc8]`` ``int8``, output ``[OC/8][T][oc8]``
    ``int8``: ``out = sat_i8((sum_{p,c} x * w + 2**(scale-1)) >> scale)``.
    ``input_channels`` is accepted for the signature (the pixel is RGBA).
    Exact for the scalar path of ``conv2dk14.cc``.
    """
    del input_channels
    W, OC, K = int(input_width), int(output_channels), int(kernel_width)
    T, P = W // K, K * K
    x = np.asarray(x)
    lead = x.shape[:-1]
    xi = x.reshape(*lead, T // 8, P // 2, 8, 2, 4).astype(np.int64)
    xi = np.moveaxis(xi, -3, -4).reshape(*lead, T, P, 4)  # [T][P][c]
    w = np.asarray(weights, dtype=np.int8).reshape(OC // 8, P, 4, 8).astype(np.int64)
    acc = np.einsum("...tpc,opcq->...otq", xi, w)  # [OC/8][T][oc8]
    return _requant(acc, scale, -128, 127, np.int8).reshape(*lead, OC * T)


# dwconv1d_channels_first.cc reads 16 elements past the last tap (aligned vector
# loads), so a padded input row carries this much slack after the halo.
DWCONV1D_TAIL = 16


def dwconv1d_channels_first(
    seq_len: int = 1024, kernel_size: int = 9, bias: bool = True
) -> ExternalFunction:
    """Depthwise 1-D cross-correlation on one bf16 channel (aie2p only).

    ``out[t] = bias + sum_p w[p] * x_pad[t + p]`` for ``t < seq_len``: a
    'same' convolution when ``x_pad`` is the channel zero-padded by
    ``(kernel_size - 1) // 2`` on each side plus ``DWCONV1D_TAIL`` don't-care
    elements (programming_examples/ml/dwconv1d builds it that way). The
    weight row holds ``kernel_size`` taps followed by the bias, whether or
    not ``bias`` is enabled.

    One channel per call with time contiguous, vectorized along time with
    scalar taps. See
    [`dwconv1d_channels_last`][iron.kernels.conv.dwconv1d_channels_last] for
    the transposed layout, and the "Choosing a depthwise conv1d" section of
    ``aie_kernels/README.md`` for which to reach for.

    Args:
        seq_len: Outputs per call (multiple of 16).
        kernel_size: Taps, 1 to 17.
        bias: Add the trailing weight as a bias.
    """
    if _detect_arch() != "aie2p":
        raise NotImplementedError(
            "dwconv1d_channels_first: aie_kernels/aie2p/dwconv1d_channels_first.cc "
            "has no aie2 port; select an NPU2 device"
        )
    if not 1 <= kernel_size <= 17:
        raise ValueError(
            f"dwconv1d_channels_first: kernel_size must be 1..17, got {kernel_size}"
        )
    if seq_len <= 0 or seq_len % 16:
        raise ValueError(
            "dwconv1d_channels_first: seq_len must be a positive multiple of 16, "
            f"got {seq_len}"
        )
    in_ty = np.ndarray[(seq_len + DWCONV1D_TAIL,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(kernel_size + 1,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(seq_len,), np.dtype[bfloat16]]
    return _make_extern(
        "dwconv1d_channels_first_bf16",
        _default_source_path("dwconv1d_channels_first.cc", subdir="aie2p"),
        [in_ty, w_ty, out_ty, np.int32],
        compile_flags=[
            f"-DDWCONV1D_CF_K={kernel_size}",
            f"-DDWCONV1D_CF_BIAS={int(bias)}",
        ],
        contract=KernelContract(
            roles=(In, In, Out, Param),
            reference=lambda x, w, n: dwconv1d_channels_first_ref(
                x, w, n, kernel_size=kernel_size, bias=bias
            ),
            acc_dtype=np.float32,
            reduction=kernel_size,
            tolerance=Tolerance.relative(
                0.128, 0.05, note="programming_examples/ml/dwconv1d: atol 0.05"
            ),
            ops_per_call=2 * kernel_size * seq_len,
        ),
    )


def dwconv1d_channels_first_ref(x_pad, w, seq_len, *, kernel_size: int, bias: bool):
    """Numpy reference for [`dwconv1d_channels_first`][iron.kernels.conv.dwconv1d_channels_first] on the padded row(s).

    ``x_pad`` is ``(..., seq_len + DWCONV1D_TAIL)``; ``w`` is ``(..., kernel_size + 1)``.
    """
    n = int(seq_len)
    x32 = np.asarray(x_pad).astype(np.float32)
    w32 = np.asarray(w).astype(np.float32)
    out = np.zeros(x32.shape[:-1] + (n,), dtype=np.float32)
    for p in range(kernel_size):
        out += w32[..., p : p + 1] * x32[..., p : p + n]
    if bias:
        out += w32[..., kernel_size : kernel_size + 1]
    return out.astype(np.asarray(x_pad).dtype)


def dwconv1d(
    seq_len: int = 1024, kernel_size: int = 9, bias: bool = True
) -> ExternalFunction:
    """Compatibility alias for [`dwconv1d_channels_first`][iron.kernels.conv.dwconv1d_channels_first]."""
    return dwconv1d_channels_first(seq_len, kernel_size, bias)


def dwconv1d_ref(x_pad, w, seq_len, *, kernel_size: int, bias: bool):
    """Compatibility alias for [`dwconv1d_channels_first_ref`][iron.kernels.conv.dwconv1d_channels_first_ref]."""
    return dwconv1d_channels_first_ref(
        x_pad, w, seq_len, kernel_size=kernel_size, bias=bias
    )


# The clamp bounds dwconv1d_channels_last is judged against. Wide enough that a
# bf16 5-tap product only reaches it on the 'large' data case, so the clamped
# and unclamped paths are both exercised.
_CLAMP_LIMIT = 6.0


def dwconv1d_channels_last(channels: int = 256, clamp: bool = True) -> ExternalFunction:
    """Depthwise 1-D conv over a channels-last layout, 5 taps (aie2p only).

    ``y[c] = clamp(sum_{t=0..4} w_t[c] * x_t[c], lo, hi)`` for ``c < channels``:
    one output timestep across every channel, with per-channel taps. The five
    taps arrive as five separate base pointers, oldest first, so a depth-5
    ObjectFifo is itself the sliding window.

    Counterpart to
    [`dwconv1d_channels_first`][iron.kernels.conv.dwconv1d_channels_first];
    layout picks the vectorization axis, so neither subsumes the other. See
    the "Choosing a depthwise conv1d" section of ``aie_kernels/README.md``.

    The five weight planes are five independent arguments, like the taps, so
    where they live is the design's business: they need not be one buffer, or
    evenly spaced.

    ``lo``/``hi`` are runtime buffers the design writes, bound here to
    ``+/-_CLAMP_LIMIT`` so the kernel is judged against a reference clamping to
    the same pair.

    Args:
        channels: Channels per call (multiple of 32).
        clamp: Clamp the result to the runtime ``lo``/``hi`` buffers.
    """
    if _detect_arch() != "aie2p":
        raise NotImplementedError(
            "dwconv1d_channels_last: aie_kernels/aie2p/dwconv1d_channels_last.cc "
            "has no aie2 port; select an NPU2 device"
        )
    if channels <= 0 or channels % 32:
        raise ValueError(
            "dwconv1d_channels_last: channels must be a positive multiple of the "
            f"32-lane store, got {channels}"
        )
    _TAPS = 5
    plane_ty = np.ndarray[(channels,), np.dtype[bfloat16]]
    lim_ty = np.ndarray[(1,), np.dtype[np.float32]]
    return _make_extern(
        "dwconv1d_channels_last_k5_bf16",
        _default_source_path("dwconv1d_channels_last.cc", subdir="aie2p"),
        [*([plane_ty] * 2 * _TAPS), plane_ty, lim_ty, lim_ty],
        compile_flags=[
            f"-DDWCONV1D_CL_C={channels}",
            f"-DDWCONV1D_CL_CLAMP={int(clamp)}",
        ],
        contract=KernelContract(
            stack_bytes=1280,  # aiecc measured_stack_size
            setup=conv_even,
            # lo/hi are buffers the design writes, so they are Param like
            # mha's idx gate: bound here rather than sampled, which also keeps
            # lo <= hi (aie::clamp does not define the inverted pair).
            roles=(*((In,) * 2 * _TAPS), Out, Param, Param),
            parameter_bindings=(
                (11, np.array([-_CLAMP_LIMIT], np.float32)),
                (12, np.array([_CLAMP_LIMIT], np.float32)),
            ),
            reference=partial(
                dwconv1d_channels_last_ref,
                lo=-_CLAMP_LIMIT,
                hi=_CLAMP_LIMIT,
                clamp=clamp,
            ),
            acc_dtype=np.float32,
            reduction=_TAPS,
            # Derived: the five products are exact in f32 (bf16 carries 8
            # mantissa bits, 8 + 8 < 24) and the sum rounds at most 5 * 2**-24
            # before one bf16 narrowing on store, which is 2**-9 relative. The
            # narrowing dominates by four orders of magnitude, so the bound is
            # one bf16 ulp with no room to spare for a real error.
            tolerance=Tolerance.relative(
                2**-8,
                0.0,
                note="one bf16 ulp from the single narrowing store; the f32 "
                "5-term sum contributes 5 * 2**-24. Verified on npu2",
            ),
            ops_per_call=2 * _TAPS * channels,
        ),
    )


def dwconv1d_channels_last_ref(
    w_0, w_1, w_2, w_3, w_4, x_0, x_1, x_2, x_3, x_4, *, lo, hi, clamp: bool
):
    """Numpy reference for [`dwconv1d_channels_last`][iron.kernels.conv.dwconv1d_channels_last]: one timestep over all channels.

    Each tap is an independent plane, so this is a plain ``sum_t w_t * x_t``.
    Accumulated in float32, which is what the kernel's ``accfloat`` is, then
    narrowed once on store; ``clamp`` applies after the narrowing, as the
    kernel's ``aie::clamp`` does on the already-bf16 vector.
    """
    ws = [np.asarray(v).astype(np.float32) for v in (w_0, w_1, w_2, w_3, w_4)]
    xs = [np.asarray(v).astype(np.float32) for v in (x_0, x_1, x_2, x_3, x_4)]
    acc = np.zeros(xs[0].shape, dtype=np.float32)
    for wt, xt in zip(ws, xs):
        acc += wt * xt
    out = acc.astype(np.asarray(x_0).dtype)
    if clamp:
        out = np.clip(out, np.asarray(lo, out.dtype), np.asarray(hi, out.dtype))
    return out


@dtypes(({"act_dtype": np.int8}, {"act_dtype": np.uint8}))
def conv2dk1(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype: type = np.int8,
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
        compile_flags=flags
        + _conv_dimensions(input_width, input_channels, output_channels),
        contract=KernelContract(
            stack_bytes=2752,  # aiecc measured_stack_size (Peano 22)
            roles=(In, Param, Out, Param, Param, Param, Param),
            reference=conv2dk1_ref,
            acc_dtype=np.int32,
            reduction=input_channels,
            tolerance=Tolerance.exact(
                note="measured bit-exact against the reference over every data case"
            ),
            ops_per_call=2 * input_width * input_channels * output_channels,
        ),
    )


@dtypes(({"act_dtype": np.int8}, {"act_dtype": np.uint8}))
def conv2dk3(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype: type = np.int8,
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
        compile_flags=flags
        + _conv_dimensions(input_width, input_channels, output_channels)
        + ["-DCONV_KERNEL_WIDTH=3", "-DCONV_KERNEL_HEIGHT=3"],
        contract=KernelContract(
            stack_bytes=4736,  # aiecc measured_stack_size (Peano 22)
            roles=(In, In, In, Param, Out, *((Param,) * 8)),
            reference=conv2dk3_ref,
            acc_dtype=np.int32,
            reduction=9 * input_channels,
            tolerance=Tolerance.exact(
                note="measured bit-exact against the reference over every data case"
            ),
            ops_per_call=2 * 9 * input_width * input_channels * output_channels,
        ),
    )


@dtypes(({"act_dtype": np.int8}, {"act_dtype": np.uint8}))
def conv2dk1_skip(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype: type = np.int8,
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

    Note:
        The activations are ``uint8`` in two half-channel tensors whatever
        ``act_dtype``, which types the residual (``skip``) only. The generic
        harness streams the three tensors through one packed fifo, which
        needs them to share a type: ``input_channels == 2 * output_channels``
        with ``act_dtype=np.uint8``. The ``int8`` residual build has the
        same contract but needs a design of its own to run.
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
        compile_flags=flags
        + _conv_dimensions(input_width, input_channels, output_channels),
        contract=KernelContract(
            stack_bytes=2752,  # aiecc measured_stack_size (Peano 22, uint8)
            roles=(In, In, Param, Out, In, *((Param,) * 5)),
            reference=conv2dk1_skip_ref,
            acc_dtype=np.int32,
            reduction=input_channels,
            tolerance=Tolerance.exact(
                note="measured bit-exact against the reference over every data case"
            ),
            ops_per_call=2 * input_width * input_channels * output_channels
            + input_width * output_channels,
        ),
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
        compile_flags=["-DINT8_ACT"]
        + _conv_dimensions(input_width, input_channels, output_channels),
        contract=KernelContract(
            stack_bytes=1504,  # aiecc measured_stack_size
            roles=(In, Param, Out, Param, Param, Param, Param),
            reference=conv2dk1_i8_ref,
            acc_dtype=np.int32,
            reduction=input_channels,
            # The only conv kernel that is not bit-exact: its vector path
            # ends in a symmetric_inf srs the scalar reference does not model,
            # measured at 2 of 98304 values, each one LSB out.
            tolerance=Tolerance.lsb(
                1, note="vector path srs rounding; measured within one LSB"
            ),
            ops_per_call=2 * input_width * input_channels * output_channels,
        ),
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
    in_ty = np.ndarray[(tiles * pixels * _RGBA,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(output_channels * pixels * _RGBA,), np.dtype[np.int8]]
    # One int8 per (output channel, tile): conv2dk14.cc writes
    # output[oc * tiles + tile] and nothing beyond it.
    out_ty = np.ndarray[(output_channels * tiles,), np.dtype[np.int8]]
    return _make_extern(
        "conv2dk14_i8",
        _default_source_path("conv2dk14.cc", subdir="aie2p"),
        [in_ty, wt_ty, out_ty, *_i32s(5)],
        compile_flags=_conv_dimensions(input_width, input_channels, output_channels)
        + [f"-DCONV_KERNEL_WIDTH={kernel_width}"],
        contract=KernelContract(
            roles=(In, Param, Out, *((Param,) * 5)),
            reference=conv2dk14_ref,
            acc_dtype=np.int32,
            reduction=pixels * _RGBA,
            tolerance=Tolerance.exact(
                note="measured bit-exact against the reference over every data case"
            ),
            ops_per_call=2 * tiles * pixels * _RGBA * output_channels,
        ),
    )


@dtypes(({"act_dtype": np.int8}, {"act_dtype": np.uint8}))
def conv2dk1_skip_init(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype: type = np.int8,
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
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``,
            when ``input_width`` is not a positive multiple of 32, or when a
            channel count is not a positive multiple of what the source
            steps by (16 input channels, 8 output and skip channels).
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1_skip_init", act_dtype, factory_name="conv2dk1_skip_init"
    )
    if input_width <= 0 or input_width % 32:
        # The kernel computes whole 32-wide blocks and its tail path was never
        # implemented, so a width that is not a multiple of 32 silently leaves
        # its last columns unwritten. Refuse it rather than return part of an
        # answer.
        raise ValueError(
            f"conv2dk1_skip_init: input_width must be a positive multiple of 32, "
            f"got {input_width}"
        )
    if skip_input_channels is None:
        skip_input_channels = input_channels
    # Both paths in the source count channels in whole steps: the two input
    # halves in 16s (an 8x8 weight block per half), the output and the skip
    # projection in 8s. A count that is not a whole number of steps is
    # silently truncated by the integer division in the loop bounds.
    for label, count, step in (
        ("input_channels", input_channels, 16),
        ("output_channels", output_channels, 8),
        ("skip_input_channels", skip_input_channels, 8),
    ):
        if count <= 0 or count % step:
            raise ValueError(
                f"conv2dk1_skip_init: {label} must be a positive multiple of "
                f"{step}, got {count}"
            )
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
        contract=KernelContract(
            stack_bytes=0x2000,  # >=2144 measured; __modsi3 has no .stack_sizes
            roles=(In, In, Param, Out, In, *((Param,) * 7)),
            reference=conv2dk1_skip_init_ref,
            acc_dtype=np.int32,
            reduction=max(input_channels, skip_input_channels),
            # Measured bit-exact over every data case at three seeds. The uint8
            # entry point was an empty function and the one-LSB slack these
            # conv kernels used to share was not what hid it, but an exact
            # contract states what this kernel actually owes.
            tolerance=Tolerance.exact(
                note="both paths match the reference bit-for-bit"
            ),
            ops_per_call=2
            * input_width
            * output_channels
            * (input_channels + skip_input_channels),
        ),
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


@dtypes(({"skip_dtype": np.uint8}, {"skip_dtype": np.int8}))
def bn_conv2dk1_skip(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    skip_dtype: type = np.uint8,
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


def _validate_bn_block_index(block_index: int, factory_name: str) -> None:
    if block_index not in (13, 14):
        raise ValueError(
            f"{factory_name}(): block_index must be 13 or 14 (the only "
            f"per-block symbols defined in the .cc), got {block_index}."
        )


def bn_conv2dk1_partial_put_i8(
    input_width: int = 7,
    input_channels: int = 80,
    weight_count: int = 4800,
    *,
    block_index: int = 13,
) -> ExternalFunction:
    """Cascade-PUT half of a width-split 1x1 conv on int8 activations.

    The PUT tile of a two-tile cascade-split pointwise conv: consumes a
    width slice of the activation, multiplies against its weight half,
    and emits the partial sum onto the cascade stream (no separate
    output buffer — cascade-only).  Sister of
    [`bn_conv2dk1_partial_get_relu_i8`][iron.kernels.conv.bn_conv2dk1_partial_get_relu_i8].

    Currently defined in the .cc only for MobileNet V3's bn13 / bn14
    (one wrapper symbol per block); ``block_index`` selects which.
    Generalising this to arbitrary block names would require adding a
    non-prefixed wrapper to ``bn_conv2dk1_i8.cc``.

    Args:
        input_width: Spatial width of the input slice.
        input_channels: Number of input channels.
        weight_count: Per-call weight chunk size in elements (the design
            streams weights in chunks; full weight tensor is shared
            across multiple kernel invocations).
        block_index: ``13`` or ``14``; selects the per-block C++ wrapper.

    Returns:
        ExternalFunction configured for the PUT tile.

    Raises:
        ValueError: When ``block_index`` is not 13 or 14.
    """
    _validate_bn_block_index(block_index, "bn_conv2dk1_partial_put_i8")
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.int8]]
    wt_ty = np.ndarray[(weight_count,), np.dtype[np.int8]]
    return _make_extern(
        f"bn{block_index}_1_conv2dk1_i8_ui8_partial_width_put_new",
        _default_source_path("bottleneck/bn_conv2dk1_i8.cc", subdir="aie2"),
        [in_ty, wt_ty, *_i32s(7)],
        compile_flags=[f"-DBN{block_index}_1_PARTIAL_PUT_I8_CAS_WIDTH_NEW"],
    )


def bn_conv2dk1_partial_get_relu_i8(
    input_width: int = 7,
    input_channels: int = 80,
    output_channels: int = 480,
    weight_count: int = 4800,
    *,
    block_index: int = 13,
) -> ExternalFunction:
    """Cascade-GET half of a width-split 1x1 conv + ReLU on int8 activations.

    The GET tile of a two-tile cascade-split pointwise conv: consumes
    the cascade partial sum from its sister PUT tile, finishes the dot
    product against its weight half, applies ReLU, and writes the full
    output buffer.  Sister of [`bn_conv2dk1_partial_put_i8`][iron.kernels.conv.bn_conv2dk1_partial_put_i8].

    Currently defined in the .cc only for MobileNet V3's bn13 / bn14
    (one wrapper symbol per block); ``block_index`` selects which.

    Args:
        input_width: Spatial width of the input slice.
        input_channels: Number of input channels.
        output_channels: Number of output channels (full L1 output width).
        weight_count: Per-call weight chunk size in elements.
        block_index: ``13`` or ``14``; selects the per-block C++ wrapper.

    Returns:
        ExternalFunction configured for the GET tile.

    Raises:
        ValueError: When ``block_index`` is not 13 or 14.
    """
    _validate_bn_block_index(block_index, "bn_conv2dk1_partial_get_relu_i8")
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.int8]]
    wt_ty = np.ndarray[(weight_count,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.uint8]]
    return _make_extern(
        f"bn{block_index}_1_conv2dk1_i8_ui8_partial_width_get_new",
        _default_source_path("bottleneck/bn_conv2dk1_relu.cc", subdir="aie2"),
        [in_ty, wt_ty, out_ty, *_i32s(9)],
        compile_flags=[f"-DBN{block_index}_1_PARTIAL_GET_I8_CAS_WIDTH_NEW"],
    )


def bn_conv2dk3_dw_out_split(
    input_width: int = 7,
    input_channels: int = 480,
    output_split_channels: int = 240,
    *,
    block_index: int = 13,
) -> ExternalFunction:
    """Depthwise 3x3 stride-1 conv with split output stream (uint8 in/out).

    A variant of [`bn_conv2dk3_dw`][iron.kernels.conv.bn_conv2dk3_dw] (stride=1) that writes its output
    to TWO separate buffers — the channel dimension is split in half so
    downstream cascade-PUT tiles can each consume one slice.  Used by
    MobileNet V3's bn13 / bn14 depthwise stage to feed the L3 cascade.

    Currently defined in the .cc only via per-block extern wrappers
    (BN13 or BN14 macro picks the symbol prefix); ``block_index`` selects
    which.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels (== output channels —
            depthwise).
        output_split_channels: Channels per output slice (half of
            ``input_channels`` for the typical 2-way split).
        block_index: ``13`` or ``14``; selects the per-block C++ wrapper.

    Returns:
        ExternalFunction configured for the split-output DW kernel.

    Raises:
        ValueError: When ``block_index`` is not 13 or 14.
    """
    _validate_bn_block_index(block_index, "bn_conv2dk3_dw_out_split")
    if (
        input_width < 2
        or output_split_channels <= 0
        or input_channels != 2 * output_split_channels
        or output_split_channels % 8
    ):
        raise ValueError(
            "bn_conv2dk3_dw_out_split requires width >= 2 and equal channel "
            "halves divisible by 8"
        )
    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(3 * 3 * input_channels,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_split_channels,), np.dtype[np.uint8]]

    return _make_extern(
        f"bn{block_index}_conv2dk3_ui8_out_split",
        _default_source_path("bottleneck/bn_conv2dk3_dw.cc", subdir="aie2"),
        [line_ty, line_ty, line_ty, wt_ty, out_ty, out_ty, *_i32s(8)],
        compile_flags=["-DSCALAR", f"-DBN{block_index}", "-DSTRIDE1_OUT_SPLIT"],
    )


def bn_conv2dk1_input_split_partial_put_ui8(
    input_width: int = 7,
    input_channels: int = 240,
    weight_count: int = 9600,
    *,
    block_index: int = 13,
) -> ExternalFunction:
    """Input-split cascade-PUT half of a 1x1 conv on uint8 activations.

    Like [`bn_conv2dk1_partial_put_i8`][iron.kernels.conv.bn_conv2dk1_partial_put_i8] but consumes a CHANNEL slice
    (input-split) of a uint8 activation instead of a width slice of int8.
    Used by MobileNet V3's bn13 / bn14 L3 stage.

    Args:
        input_width: Spatial width of the input slice.
        input_channels: Number of input channels (one half of the
            full input after split).
        weight_count: Per-call weight chunk size in elements.
        block_index: ``13`` or ``14``; selects the per-block C++ wrapper.

    Returns:
        ExternalFunction configured for the input-split PUT tile.

    Raises:
        ValueError: When ``block_index`` is not 13 or 14.
    """
    _validate_bn_block_index(block_index, "bn_conv2dk1_input_split_partial_put_ui8")
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(weight_count,), np.dtype[np.int8]]
    return _make_extern(
        f"bn{block_index}_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
        _default_source_path("bottleneck/bn_conv2dk1_i8.cc", subdir="aie2"),
        [in_ty, wt_ty, *_i32s(7)],
        compile_flags=[
            f"-DBN{block_index}_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH_NEW"
        ],
    )


def bn_conv2dk1_input_split_partial_skip_get(
    input_width: int = 7,
    input_channels: int = 240,
    output_channels: int = 80,
    weight_count: int = 9600,
    *,
    block_index: int = 13,
) -> ExternalFunction:
    """Input-split cascade-GET half of a 1x1 conv + skip-add (uint8 in, int8 out).

    The GET tile completes the cascade-split 1x1 + ReLU + residual add
    pattern: consumes the partial sum from its sister PUT tile, finishes
    the dot product, adds a skip row of int8 activations, and writes int8
    output.  Sister of [`bn_conv2dk1_input_split_partial_put_ui8`][iron.kernels.conv.bn_conv2dk1_input_split_partial_put_ui8].

    Args:
        input_width: Spatial width of the input slice.
        input_channels: Number of input channels (one half after split).
        output_channels: Final output channels.
        weight_count: Per-call weight chunk size in elements.
        block_index: ``13`` or ``14``; selects the per-block C++ wrapper.

    Returns:
        ExternalFunction configured for the input-split skip-GET tile.

    Raises:
        ValueError: When ``block_index`` is not 13 or 14.
    """
    _validate_bn_block_index(block_index, "bn_conv2dk1_input_split_partial_skip_get")
    in_ty = np.ndarray[(input_width * input_channels,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[(weight_count,), np.dtype[np.int8]]
    out_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.int8]]
    skip_ty = np.ndarray[(input_width * output_channels,), np.dtype[np.int8]]
    return _make_extern(
        f"bn_{block_index}_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
        _default_source_path("bottleneck/bn_conv2dk1_skip.cc", subdir="aie2"),
        [in_ty, wt_ty, out_ty, skip_ty, *_i32s(10)],
        compile_flags=[
            f"-DBN{block_index}_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH_NEW"
        ],
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
