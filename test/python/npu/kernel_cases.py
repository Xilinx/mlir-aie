# kernel_cases.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The kernel cases the device tests check and the nightly benchmark times.

One table, three readers: ``test_kernels_e2e.py`` runs the ``smoke`` cases on
every pull request and every case x edge-data case x seed under the
``extensive`` marker; ``test_kernels_bench.py`` times the ``perf`` cases. What a kernel
computes, and how close the device must come, is the factory's
``KernelContract``; a case only says which tile to build and how many
independent calls to make.

Tile sizes are chosen so two sets of tiles (ping-pong) plus the stack fit a
core's 64 KB: a 64x32x64 matmul tile set is 8 KB + 8 KB + 16 KB of C. The
harness drops to depth 1 when a set does not fit, which still checks the
kernel but is not the buffering anyone benchmarks. ``devices=("npu2",)``
marks kernels whose source exists only for AIE2P.
"""

import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from cases import Case
from ml_dtypes import bfloat16

_bf16 = dict(dtype=bfloat16)
_mm = dict(dim_m=64, dim_k=32, dim_n=64)
_mm_bf16 = dict(**_mm, input_dtype=bfloat16, output_dtype=np.float32)
_mm_bfp = dict(dim_m=64, dim_k=64, dim_n=64)  # the block_datatypes examples' tile

# The exact-copy and one-op bf16 kernels propagate NaN/inf and preserve
# subnormals, and their references do the same. That is a claim about
# behavior, so it is made where it is exercised rather than declared on the
# contract and never checked.
IEEE_FLOAT = (
    "random",
    "zeros",
    "ones",
    "large",
    "alternating",
    "nan_inf",
    "subnormal",
)

CASES: list[Case] = [
    Case("zero", dict(tile_size=64), calls=4, smoke=True, perf=False),
    Case("zero", dict(tile_size=64, dtype=bfloat16), calls=4, smoke=True, perf=False),
    Case(
        "zero",
        dict(tile_size=64, dtype=v8bfp16ebs8),
        calls=4,
        smoke=True,
        perf=False,
        devices=("npu2",),
    ),
    Case(
        "zero",
        dict(tile_size=68, dtype=np.uint8),
        calls=3,
        tag="vector-tail",
        perf=False,
    ),
    Case(
        "zero",
        dict(tile_size=34, dtype=np.int16, vectorized=False),
        calls=3,
        perf=False,
    ),
    Case(
        "zero",
        dict(tile_size=12, dtype=v8bfp16ebs8),
        calls=3,
        tag="vector-tail",
        perf=False,
        devices=("npu2",),
    ),
    # eltwise
    Case("passthrough", dict(tile_size=2048), calls=16),
    Case("passthrough", dict(tile_size=2048), calls=256),
    Case("passthrough", dict(dtype=np.int16), calls=16, smoke=True),
    Case("passthrough", dict(dtype=np.uint8), calls=16, smoke=True),
    # Four iterations hung with the old runtime-bound minimum-trip promise.
    Case(
        "passthrough",
        dict(tile_size=64),
        calls=4,
        tag="edge-tiny",
        smoke=True,
        perf=False,
    ),
    Case(
        "passthrough",
        dict(tile_size=128, dtype=np.int16),
        calls=4,
        tag="edge-tiny",
        perf=False,
    ),
    Case(
        "passthrough",
        dict(tile_size=256, dtype=np.uint8),
        calls=4,
        tag="edge-tiny",
        perf=False,
    ),
    Case("passthrough", dict(tile_size=16), calls=4, tag="edge-one-vector", perf=False),
    Case("scale", dict(dtype=np.int16), calls=16, smoke=True),
    Case("scale", dict(dtype=np.int16), calls=256),
    Case("scale", dict(dtype=np.int32), calls=16, smoke=True),
    Case("scale", dict(tile_size=64), calls=4, tag="edge-tiny", perf=False),
    Case(
        "scale",
        dict(tile_size=32, dtype=np.int16),
        calls=4,
        tag="edge-one-vector",
        perf=False,
    ),
    # No int16 overflow case: scale.cc stores acc32 with to_vector(0) and no
    # set_sat, so whether a product beyond int16 wraps or saturates is a core
    # setting the source leaves open (overflow="undefined"); the judge refuses
    # to grade such a reference until the kernel declares it.
    Case(
        "scale",
        dict(dtype=np.int32),
        calls=16,
        params=(-7,),
        tag="edge-negfactor",
        perf=False,
    ),
    Case("add", calls=16, smoke=True),
    Case("add", calls=256, data_cases=IEEE_FLOAT),
    Case("mul", calls=16, smoke=True),
    Case("mul", calls=256, data_cases=IEEE_FLOAT),
    Case("relu", calls=16, smoke=True),
    Case("relu", calls=256),
    # reduce
    Case("reduce_add", calls=16, smoke=True),
    Case("reduce_add", calls=256),
    Case("reduce_add", calls=1, tag="edge-single", perf=False),
    Case(
        "reduce_add",
        dict(tile_size=64),
        calls=4,
        tag="edge-tiny",
        smoke=True,
        perf=False,
    ),
    Case("reduce_add", dict(tile_size=16), calls=4, tag="edge-one-vector", perf=False),
    Case("reduce_min", calls=16, smoke=True),
    Case("reduce_min", calls=256),
    Case("reduce_min", dict(tile_size=16), calls=4, tag="edge-one-vector", perf=False),
    Case("reduce_max", calls=16, smoke=True),
    Case("reduce_max", calls=256),
    Case("reduce_max", _bf16, calls=16, smoke=True),
    Case("reduce_max", _bf16, calls=256),
    Case("reduce_max", dict(tile_size=16), calls=4, tag="edge-one-vector", perf=False),
    Case(
        "reduce_max",
        dict(tile_size=32, dtype=bfloat16),
        calls=4,
        tag="edge-one-vector",
        perf=False,
    ),
    # activation
    Case("gelu", calls=16, smoke=True),
    Case("gelu", calls=256),
    Case("silu", calls=16, smoke=True),
    Case(
        "silu", dict(use_lut=True), calls=16, tag="lut", devices=("npu2",), smoke=True
    ),
    Case("silu", calls=256),
    Case("bf16_exp", calls=16, smoke=True),
    Case("bf16_exp", calls=256),
    Case("tanh", calls=16, smoke=True),
    Case("tanh", calls=256),
    # The LUT tanh, which on aie2p is the alternative to the vtanh instruction
    # and 7.5x closer to the true function. It is judged against an exact model
    # of the table rather than against tanh itself, so this case is the tight
    # one: one bf16 ulp, where the vtanh build needs 5% relative.
    Case(
        "tanh",
        dict(use_lut=True),
        calls=16,
        tag="lut",
        devices=("npu2",),
        smoke=True,
    ),
    Case("sigmoid", calls=16, smoke=True),
    Case(
        "sigmoid",
        dict(use_lut=True),
        calls=16,
        tag="lut",
        devices=("npu2",),
        smoke=True,
    ),
    Case("sigmoid", calls=256),
    Case("softmax", calls=16, smoke=True),
    Case("softmax", calls=256),
    Case("leaky_relu", calls=16, scalars=(0.5,), smoke=True),
    Case("leaky_relu", calls=256, scalars=(0.5,)),
    Case("exp2f_vec", calls=16, devices=("npu2",), smoke=True),
    Case("exp2f_vec", calls=256, devices=("npu2",)),
    # Sized kernels retaining their runtime-count ABI.
    Case("add_sized", calls=16, smoke=True, perf=False),
    Case("mul_sized", calls=16, smoke=True, perf=False),
    Case("relu_sized", calls=16, smoke=True, perf=False),
    Case("silu_sized", calls=16, smoke=True, perf=False),
    Case("gelu_sized", calls=16, smoke=True, perf=False),
    *[
        Case(
            name,
            dict(tile_size=32),
            calls=4,
            tag="edge-tiny",
            smoke=True,
            perf=False,
        )
        for name in ("add_sized", "mul_sized", "relu_sized", "silu_sized", "gelu_sized")
    ],
    # datamovement
    Case("axpy", calls=16, scalars=(2.5,), smoke=True),
    Case("axpy", calls=256, scalars=(2.5,), data_cases=IEEE_FLOAT),
    Case("convert_copy", calls=16, devices=("npu2",), smoke=True),
    Case("convert_copy", calls=256, devices=("npu2",)),
    Case("expand", calls=16, smoke=True),
    Case("expand", calls=256),
    Case("transpose", dict(subtile=4), calls=16, smoke=True),
    Case("transpose", dict(subtile=8), calls=16, smoke=True),
    Case("transpose", dict(subtile=4, dtype=np.uint8), calls=16, smoke=True),
    Case("transpose", dict(subtile=8, dtype=np.uint32), calls=16),
    # linalg
    Case("mm", _mm_bf16, calls=16),
    Case("mm", _mm_bf16, calls=256),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32),
        calls=16,
    ),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int8, output_dtype=np.int32),
        calls=16,
    ),
    Case("mm", _mm_bf16, calls=1, tag="edge-single-tile", perf=False),
    # Bounded fused composition: two A bands and multiple K/drain chunks.
    Case("fused_mm", calls=4, smoke=True),
    *[
        Case(
            "fused_mm",
            dict(dim_k=48, epilogue=mode),
            calls=4,
            perf=False,
            # Keep tanh's AIE2 LUT input in its representable domain.
            data_cases=("random", "zeros", "ones", "alternating"),
        )
        for mode in ("gelu", "silu", "sigmoid")
    ],
    Case(
        "fused_mm",
        dict(dim_k=48, clamp=(-0.125, 0.75)),
        calls=4,
        perf=False,
    ),
    Case(
        "fused_mm",
        dict(dim_k=48, epilogue="silu", clamp=(-0.125, 0.75)),
        calls=4,
        perf=False,
        data_cases=("random", "zeros", "ones", "alternating"),
    ),
    # block floating point (aie2p): bfp16ebs8 A, B and C, and the mixed
    # kernel with bf16 A and C; host encode/shuffle via aie.utils.bfp.
    Case("mm_bfp", _mm_bfp, calls=16, devices=("npu2",)),
    Case("mm_bfp_shuffle", calls=4, devices=("npu2",), smoke=True),
    Case("q4nx_dequant", calls=4, devices=("npu2",), smoke=True),
    Case(
        "q4nx_dequant",
        dict(m_tile=16, k_tile=32, group=8, ct_k=16),
        calls=2,
        devices=("npu2",),
        perf=False,
    ),
    # Quantization groups need not divide the GEMM K slice.
    Case(
        "q4nx_dequant",
        dict(m_tile=48, k_tile=48, group=24, ct_k=16),
        calls=2,
        devices=("npu2",),
        perf=False,
    ),
    Case(
        "mm_bfp",
        dict(**_mm_bfp, mixed=True),
        calls=16,
        devices=("npu2",),
    ),
    Case(
        "mm",
        dict(
            dim_m=32, dim_k=32, dim_n=32, input_dtype=bfloat16, output_dtype=np.float32
        ),
        calls=16,
        tag="edge-small-tile",
        perf=False,
    ),
    Case("mv", dict(dim_m=32, dim_k=32), calls=16),
    # The i16 matvec takes two 16-row blocks per pass over the columns, so an
    # odd number of blocks leaves the last one to the tail.
    Case(
        "mv",
        dict(dim_m=48, dim_k=32),
        calls=4,
        tag="edge-rows-not-multiple-of-32",
        smoke=True,
        perf=False,
    ),
    Case(
        "mv",
        dict(
            dim_m=6, dim_k=128, input_dtype=bfloat16, output_dtype=bfloat16, vec_size=64
        ),
        calls=4,
        tag="edge-rows-not-multiple-of-4",
        smoke=True,
        perf=False,
    ),
    Case(
        "mv",
        dict(dim_m=32, dim_k=256, input_dtype=bfloat16, output_dtype=bfloat16),
        calls=4,
        smoke=True,
        perf=False,
    ),
    *[
        Case(
            "mv",
            dict(
                dim_m=4,
                dim_k=dim_k,
                input_dtype=bfloat16,
                output_dtype=bfloat16,
                vec_size=64,
            ),
            calls=4,
            tag=tag,
            smoke=True,
            perf=False,
        )
        for dim_k, tag in ((64, "edge-one-chunk"), (128, "edge-two-chunks"))
    ],
    # reduce companion, gated activation
    Case("compute_max", calls=16, smoke=True),
    Case("compute_max", _bf16, calls=16, smoke=True),
    Case("swiglu", calls=16, smoke=True),
    Case(
        "swiglu", dict(use_lut=True), calls=16, tag="lut", devices=("npu2",), smoke=True
    ),
    Case("swiglu", calls=256),
    # vision: uint8 lines of 1920 pixels
    Case("gray2rgba", calls=16, smoke=True),
    Case("rgba2gray", calls=16, smoke=True),
    Case("threshold", calls=16, scalars=(100, 255, 0), smoke=True),
    Case("threshold", calls=16, scalars=(100, 255, 2), tag="trunc", perf=False),
    Case("threshold", calls=16, scalars=(100, 255, 4), tag="tozero-inv", perf=False),
    Case(
        "threshold", dict(dtype=np.int16), calls=16, scalars=(100, 255, 1), perf=False
    ),
    Case("bitwise_or", calls=16, smoke=True),
    Case("bitwise_and", calls=16, smoke=True),
    Case("bitwise_or", dict(line_width=64), calls=4, tag="edge-one-vector", perf=False),
    Case(
        "bitwise_and", dict(line_width=64), calls=4, tag="edge-one-vector", perf=False
    ),
    *[
        Case(
            "threshold",
            dict(line_width=64),
            calls=4,
            scalars=(100, 255, mode),
            tag=f"edge-one-vector-mode-{mode}",
            perf=False,
        )
        for mode in range(5)
    ],
    # alpha = beta = 0.5 in Q2.14; gamma = 0, where the kernel's two paths agree.
    Case("add_weighted", calls=16, scalars=(8192, 8192, 0), smoke=True),
    Case(
        "add_weighted",
        dict(line_width=32),
        calls=4,
        scalars=(8192, 8192, 0),
        tag="edge-one-vector",
        perf=False,
    ),
    Case("filter2d", calls=16, smoke=True),
    Case("rgba2hue", calls=16, smoke=True),
    # conv: full-range int8 data (the kernels saturate, so `input_limit` only
    # keeps the int32 accumulator safe); the shift puts random sums around
    # uint8's range (64 channels x 127^2 ~ 2**20 >> 12 for k1; 9x that >> 15
    # for k3) so saturation is exercised without being the whole picture.
    Case("conv2dk1", calls=8, scalars=(32, 64, 64, 12), smoke=True),
    Case("conv2dk1_i8", calls=8, scalars=(32, 64, 64, 12), smoke=True),
    # conv2dk1_skip streams three tensors; the harness packs them into one
    # fifo only when they share a type, i.e. input_channels == 2 *
    # output_channels with a uint8 residual (the int8 residual build shares
    # the contract and is covered by the host reference tests).
    Case(
        "conv2dk1_skip",
        dict(input_channels=128, output_channels=64, act_dtype=np.uint8),
        calls=8,
        scalars=(32, 128, 64, 12, 1),
        smoke=True,
    ),
    # conv2dk1_skip_init: the residual is a 1x1 conv of its own; packable with
    # a uint8 residual of half the input channels (one type across the fifo).
    Case(
        "conv2dk1_skip_init",
        dict(input_channels=64, skip_input_channels=32, act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 32, 12, 1, 11),
        smoke=True,
    ),
    # The int8 entry point of the same source. Its residual is int8 beside
    # uint8 activations, so the three 'in' tensors group into two fifos, one
    # per type, which is exactly the two input channels a core tile has. Only
    # the uint8 half used to be covered, which is how that entry point stayed
    # an empty function.
    Case(
        "conv2dk1_skip_init",
        dict(input_channels=64, skip_input_channels=32, act_dtype=np.int8),
        calls=8,
        scalars=(32, 64, 64, 32, 12, 1, 11),
        tag="int8_skip",
        smoke=True,
    ),
    # conv2dk14 (aie2p): 16 patches of 14x14 RGBA pixels per call, 784 taps.
    Case(
        "conv2dk14",
        calls=4,
        scalars=(224, 4, 16, 14, 17),
        devices=("npu2",),
        smoke=True,
    ),
    Case(
        "conv2dk1",
        dict(act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 12),
    ),
    Case("conv2dk3", calls=8, scalars=(32, 64, 64, 3, 3, 1, 15, 0), smoke=True),
    Case(
        "conv2dk3",
        dict(act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 15, 0),
        smoke=True,
    ),
    Case(
        "conv2dk3",
        dict(act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 3, 3, 0, 15, 0),
        tag="top-row",
        perf=False,
    ),
    # eltwise mul/add selected per call (programming_examples/ml/scale_shift)
    Case("mul_add", calls=16, scalars=(1,), smoke=True),
    Case("mul_add", calls=16, scalars=(0,), tag="add", smoke=True),
    # transformer blocks (aie2p): one row per call
    Case("rms_norm", dict(cols=1024), calls=16, devices=("npu2",), smoke=True),
    Case(
        "rms_norm_eps",
        dict(cols=1024),
        calls=16,
        scalars=(1e-5,),
        devices=("npu2",),
        smoke=True,
        perf=False,
    ),
    Case("layer_norm", dict(cols=1024), calls=16, devices=("npu2",), smoke=True),
    Case(
        "layer_norm_f32",
        dict(cols=1024),
        calls=16,
        devices=("npu2",),
        smoke=True,
    ),
    Case(
        "layer_norm_affine_cast",
        dict(cols=1024),
        calls=16,
        devices=("npu2",),
        smoke=True,
    ),
    Case("rope", dict(cols=1024), calls=16, devices=("npu2",), smoke=True),
    Case(
        "rope",
        dict(cols=1024, two_halves=True),
        calls=16,
        devices=("npu2",),
        smoke=True,
        perf=False,
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(0,),
        tag="identity",
        devices=("npu2",),
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(1,),
        tag="silu",
        devices=("npu2",),
        smoke=True,
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(2,),
        tag="gelu",
        devices=("npu2",),
        smoke=True,
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(3,),
        tag="relu",
        devices=("npu2",),
    ),
    # depthwise 1-D conv (aie2p): 1024 outputs per call from a padded row
    Case(
        "dwconv1d_channels_first",
        dict(seq_len=1024, kernel_size=9),
        calls=16,
        scalars=(1024,),
        devices=("npu2",),
        smoke=True,
    ),
    # the transposed layout: one timestep across 256 channels, 5 per-channel taps
    Case(
        "dwconv1d_channels_last",
        dict(channels=256),
        calls=16,
        devices=("npu2",),
        smoke=True,
    ),
]


# Smaller shapes for the per-PR smoke test; the nightly times the shapes above.
CASES += [
    Case("passthrough", calls=4, smoke=True, perf=False),
    Case("mm", _mm_bf16, calls=4, smoke=True, perf=False),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32),
        calls=4,
        smoke=True,
        perf=False,
    ),
    Case(
        "mm",
        dict(**_mm_bf16, b_col_maj=True),
        calls=4,
        smoke=True,
        perf=False,
    ),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32, c_col_maj=True),
        calls=4,
        smoke=True,
        perf=False,
    ),
    Case(
        "mm",
        dict(**_mm_bf16, c_col_maj=True),
        calls=4,
        smoke=True,
        perf=False,
    ),
    Case(
        "mm",
        dict(
            **_mm,
            input_dtype=np.int16,
            output_dtype=np.int32,
            b_col_maj=True,
            c_col_maj=True,
        ),
        calls=3,
        smoke=True,
        perf=False,
    ),
    Case("mv", dict(dim_m=32, dim_k=32), calls=4, smoke=True, perf=False),
    # The attention toolkit's QK^T product: mm.cc's bf16 tile matmul.
    Case("mha", calls=4, devices=("npu2",), smoke=True, perf=False),
    # The prefill toolkit's S*V accumulate, one case per geometry. Each
    # -DPREFILL_HEAD_DIM build is its own object with its own blocked V order;
    # the 512 one has a degenerate k-block term and so cannot tell a wrong V
    # order from a right one, which is why both are smoke cases.
    Case("prefill_fv", dict(head_dim=512), calls=4, devices=("npu2",), smoke=True),
    Case("prefill_fv", dict(head_dim=256), calls=4, devices=("npu2",), smoke=True),
    Case(
        "mm_bfp",
        _mm_bfp,
        calls=4,
        devices=("npu2",),
        smoke=True,
        perf=False,
    ),
    Case(
        "mm_bfp",
        dict(**_mm_bfp, mixed=True),
        calls=4,
        devices=("npu2",),
        smoke=True,
        perf=False,
    ),
]
