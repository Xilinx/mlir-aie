# kernel_cases.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The kernel cases the device tests check and the nightly benchmark times.

One table, three readers: ``test_kernels_e2e.py`` runs the ``smoke`` cases on
every pull request and every case x edge-data case x seed under the
``extensive`` marker; ``python -m aie.utils.kernel_harness --cases
test/python/npu/kernel_cases.py`` times the ``perf`` cases. What a kernel
computes, and how close the device must come, is the factory's
``KernelContract``; a case only says which shape to run.

Tile sizes are chosen so two sets of tiles (ping-pong) plus the stack fit a
core's 64 KB: a 64x32x64 matmul tile set is 8 KB + 8 KB + 16 KB of C. The
harness drops to depth 1 when a set does not fit, which still checks the
kernel but is not the buffering anyone benchmarks. ``devices=("npu2",)``
marks kernels whose source exists only for AIE2P.
"""

import numpy as np
from aie.utils.kernel_harness.cases import Case
from ml_dtypes import bfloat16

_bf16 = dict(dtype=bfloat16)
_mm = dict(dim_m=64, dim_k=32, dim_n=64)
_mm_bf16 = dict(**_mm, input_dtype=bfloat16, output_dtype=np.float32)
_mm_bfp = dict(dim_m=64, dim_k=64, dim_n=64)  # the block_datatypes examples' tile

CASES: list[Case] = [
    # eltwise
    Case("passthrough", dict(tile_size=2048), calls=16),
    Case("passthrough", dict(tile_size=2048), calls=256),
    Case("passthrough", dict(dtype=np.int16), calls=16, smoke=True),
    Case("passthrough", dict(dtype=np.uint8), calls=16, smoke=True),
    Case("passthrough", dict(tile_size=64), calls=4, tag="edge-tiny", perf=False),
    Case("scale", dict(dtype=np.int16), calls=16, smoke=True),
    Case("scale", dict(dtype=np.int16), calls=256),
    Case("scale", dict(dtype=np.int32), calls=16, smoke=True),
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
    Case("add", calls=256),
    Case("mul", calls=16, smoke=True),
    Case("mul", calls=256),
    Case("relu", calls=16, smoke=True),
    Case("relu", calls=256),
    # reduce
    Case("reduce_add", calls=16, smoke=True),
    Case("reduce_add", calls=256),
    Case("reduce_add", calls=1, tag="edge-single", perf=False),
    Case("reduce_min", calls=16, smoke=True),
    Case("reduce_min", calls=256),
    Case("reduce_max", calls=16, smoke=True),
    Case("reduce_max", calls=256),
    Case("reduce_max", _bf16, calls=16, smoke=True),
    Case("reduce_max", _bf16, calls=256),
    # activation
    Case("gelu", calls=16, smoke=True),
    Case("gelu", calls=256),
    Case("silu", calls=16, smoke=True),
    Case("silu", calls=256),
    Case("bf16_exp", calls=16, smoke=True),
    Case("bf16_exp", calls=256),
    Case("tanh", calls=16, smoke=True),
    Case("tanh", calls=256),
    Case("sigmoid", calls=16, smoke=True),
    Case("sigmoid", calls=256),
    Case("softmax", calls=16, smoke=True),
    Case("softmax", calls=256),
    Case("leaky_relu", calls=16, scalars=(0.5,), smoke=True),
    Case("leaky_relu", calls=256, scalars=(0.5,)),
    Case("exp2f_vec", calls=16, devices=("npu2",), smoke=True),
    Case("exp2f_vec", calls=256, devices=("npu2",)),
    # datamovement
    Case("axpy", calls=16, scalars=(2.5,), smoke=True),
    Case("axpy", calls=256, scalars=(2.5,)),
    Case("convert_copy", calls=16, devices=("npu2",), smoke=True),
    Case("convert_copy", calls=256, devices=("npu2",)),
    Case("expand", calls=16, smoke=True),
    Case("expand", calls=256),
    Case("transpose", dict(subtile=4), calls=16, smoke=True),
    Case("transpose", dict(subtile=8), calls=16, smoke=True),
    Case("transpose", dict(subtile=4, dtype=np.uint8), calls=16, smoke=True),
    Case("transpose", dict(subtile=8, dtype=np.uint32), calls=16),
    # linalg
    Case("mm", _mm_bf16, shape=(256, 256, 256)),
    Case("mm", _mm_bf16, shape=(512, 512, 512)),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32),
        shape=(256, 256, 256),
    ),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int8, output_dtype=np.int32),
        shape=(256, 256, 256),
    ),
    Case("mm", _mm_bf16, shape=(64, 64, 64), tag="edge-single-tile", perf=False),
    # block floating point (aie2p): bfp16ebs8 A, B and C, and the mixed
    # kernel with bf16 A and C; host encode/shuffle via aie.utils.bfp.
    Case("mm_bfp", _mm_bfp, shape=(256, 256, 256), devices=("npu2",)),
    Case(
        "mm_bfp",
        dict(**_mm_bfp, mixed=True),
        shape=(256, 256, 256),
        devices=("npu2",),
    ),
    Case(
        "mm",
        dict(
            dim_m=32, dim_k=32, dim_n=32, input_dtype=bfloat16, output_dtype=np.float32
        ),
        shape=(256, 256, 256),
        tag="edge-small-tile",
        perf=False,
    ),
    Case("mv", dict(dim_m=32, dim_k=32), shape=(256, 256)),
    # reduce companion, gated activation
    Case("compute_max", calls=16, smoke=True),
    Case("compute_max", _bf16, calls=16, smoke=True),
    Case("swiglu", calls=16, smoke=True),
    Case("swiglu", calls=256),
    # vision: uint8 lines of 1920 pixels
    Case("gray2rgba", calls=16, smoke=True),
    Case("rgba2gray", calls=16, smoke=True),
    Case("threshold", calls=16, scalars=(100, 255, 0), smoke=True),
    Case("threshold", calls=16, scalars=(100, 255, 2), tag="trunc", perf=False),
    Case("threshold", calls=16, scalars=(100, 255, 4), tag="tozero-inv", perf=False),
    Case("bitwise_or", calls=16, smoke=True),
    Case("bitwise_and", calls=16, smoke=True),
    # alpha = beta = 0.5 in Q2.14; gamma = 0, where the kernel's two paths agree.
    Case("add_weighted", calls=16, scalars=(8192, 8192, 0), smoke=True),
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
    # The int8 entry point of the same source has no case, and cannot: its
    # three 'in' tensors exceed the two input DMA channels a core tile has, so
    # they must share one packed fifo, and an int8 residual beside uint8
    # activations gives that fifo two types (see _fifo_plan). Driving it needs
    # a design with more than one Worker, not another Case. Worth knowing that
    # this is why only half of conv2dk1_skip_init is covered here -- the uint8
    # entry point was an empty function and nothing caught it.
    # conv2dk14 (aie2p): 16 patches of 14x14 RGBA pixels per call, 784 taps.
    Case(
        "conv2dk14",
        calls=4,
        scalars=(224, 4, 16, 14, 17),
        devices=("npu2",),
        smoke=True,
    ),
    # bottleneck (bn_*) single-core kernels: scalar sources, round-half-even.
    Case("bn_conv2dk1_relu", calls=8, scalars=(32, 64, 64, 12), smoke=True),
    Case("bn_conv2dk1_i8", calls=8, scalars=(32, 64, 64, 13), smoke=True),
    Case("bn_conv2dk1_skip", calls=8, scalars=(32, 64, 64, 13, 1), smoke=True),
    Case(
        "bn_conv2dk1_skip",
        dict(skip_dtype=np.int8),
        calls=8,
        scalars=(32, 64, 64, 13, 1),
    ),
    Case(
        "bn_conv2dk3_dw",
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 11, 0),
        smoke=True,
    ),
    Case(
        "bn_conv2dk3_dw",
        dict(stride=2),
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 11, 0),
        smoke=True,
    ),
    Case(
        "bn_conv2dk3",
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 15, 0),
        smoke=True,
    ),
    # MobileNet's classifier FC: one (1, 1, 1280) uint16 vector in, 16 uint16
    # logits per call; weights [16/8][1280/8][8][8] unpadded (pad == IC).
    Case(
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=16),
        calls=8,
        scalars=(1, 1280, 1280, 16, 13),
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
        "dwconv1d",
        dict(seq_len=1024, kernel_size=9),
        calls=16,
        scalars=(1024,),
        devices=("npu2",),
        smoke=True,
    ),
]


# Smaller shapes for the per-PR smoke test; the nightly times the shapes above.
CASES += [
    Case("passthrough", calls=4, smoke=True, perf=False),
    Case("mm", _mm_bf16, shape=(128, 128, 128), smoke=True, perf=False),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32),
        shape=(128, 128, 128),
        smoke=True,
        perf=False,
    ),
    Case(
        "mm",
        dict(**_mm_bf16, b_col_maj=True),
        shape=(128, 128, 128),
        smoke=True,
        perf=False,
    ),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32, c_col_maj=True),
        shape=(128, 128, 128),
        smoke=True,
        perf=False,
    ),
    Case("mv", dict(dim_m=32, dim_k=32), shape=(128, 128), smoke=True, perf=False),
    Case(
        "mm_bfp",
        _mm_bfp,
        shape=(128, 128, 128),
        devices=("npu2",),
        smoke=True,
        perf=False,
    ),
    Case(
        "mm_bfp",
        dict(**_mm_bfp, mixed=True),
        shape=(128, 128, 128),
        devices=("npu2",),
        smoke=True,
        perf=False,
    ),
]
