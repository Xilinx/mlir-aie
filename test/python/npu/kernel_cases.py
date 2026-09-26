# kernel_cases.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The kernel cases the device tests check and the nightly performance checks time.

One table, three readers: ``test_kernels_e2e.py`` runs the ``smoke`` cases on
every pull request and every case x edge-data case x seed under the
``extensive`` marker; ``test_kernels_perf.py`` times the ``perf`` cases. What a kernel
computes, and how close the device must come, is the factory's
``KernelContract``; a case only says which tile to build and how many
independent calls to make.

Tile sizes are chosen so two sets of tiles (ping-pong) plus the stack fit a
core's 64 KB: a 64x32x64 matmul tile set is 8 KB + 8 KB + 16 KB of C. The
harness drops to depth 1 when a set does not fit, which still checks the
kernel but is not the buffering anyone times. ``devices=("npu2",)``
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
# The tile amd/IRON's mm operator builds: square, and B stored transposed.
_mm_bf16_col_maj = dict(
    dim_m=64,
    dim_k=64,
    dim_n=64,
    input_dtype=bfloat16,
    output_dtype=np.float32,
    b_col_maj=True,
)

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


def check(factory: str, kwargs: dict | None = None, calls: int = 4, **opts) -> Case:
    """A case the device tests check and the performance run does not time."""
    return Case(factory, kwargs or {}, calls=calls, perf=False, **opts)


CASES: list[Case] = [
    check("zero", dict(tile_size=64), smoke=True),
    check("zero", dict(tile_size=64, dtype=bfloat16), smoke=True),
    check("zero", dict(tile_size=64, dtype=v8bfp16ebs8), smoke=True, devices=("npu2",)),
    check("zero", dict(tile_size=68, dtype=np.uint8), calls=3, tag="vector-tail"),
    check("zero", dict(tile_size=34, dtype=np.int16, vectorized=False), calls=3),
    check(
        "zero",
        dict(tile_size=12, dtype=v8bfp16ebs8),
        calls=3,
        tag="vector-tail",
        devices=("npu2",),
    ),
    # eltwise
    Case("passthrough", dict(tile_size=2048), calls=16),
    Case("passthrough", dict(tile_size=2048), calls=256),
    Case("passthrough", dict(dtype=np.int16), calls=16, smoke=True),
    Case("passthrough", dict(dtype=np.uint8), calls=16, smoke=True),
    # The tile amd/IRON's passthrough operator compiles (PASSTHROUGH_ELEMS=1024
    # with BIT_WIDTH=16); the cases above all build a different loop bound.
    Case("passthrough", dict(tile_size=1024, dtype=np.int16), calls=16),
    # Four iterations hung with the old runtime-bound minimum-trip promise.
    check("passthrough", dict(tile_size=64), tag="edge-tiny", smoke=True),
    check("passthrough", dict(tile_size=128, dtype=np.int16), tag="edge-tiny"),
    check("passthrough", dict(tile_size=256, dtype=np.uint8), tag="edge-tiny"),
    check("passthrough", dict(tile_size=16), tag="edge-one-vector"),
    Case("scale", dict(dtype=np.int16), calls=16, smoke=True),
    Case("scale", dict(dtype=np.int16), calls=256),
    Case("scale", dict(dtype=np.int32), calls=16, smoke=True),
    check("scale", dict(tile_size=64), tag="edge-tiny"),
    check("scale", dict(tile_size=32, dtype=np.int16), tag="edge-one-vector"),
    check("scale", dict(tile_size=16), tag="edge-one-vector"),
    check("scale", dict(tile_size=48), tag="edge-tail"),
    # No int16 overflow case: scale.cc stores acc32 with to_vector(0) and no
    # set_sat, so whether a product beyond int16 wraps or saturates is a core
    # setting the source leaves open (overflow="undefined"); the judge refuses
    # to grade such a reference until the kernel declares it.
    check("scale", dict(dtype=np.int32), calls=16, params=(-7,), tag="edge-negfactor"),
    Case("add", calls=16, smoke=True),
    Case("add", calls=256, data_cases=IEEE_FLOAT),
    Case("mul", calls=16, smoke=True),
    Case("mul", calls=256, data_cases=IEEE_FLOAT),
    Case("relu", calls=16, smoke=True),
    Case("relu", calls=256),
    # reduce
    Case("reduce_add", calls=16, smoke=True),
    Case("reduce_add", calls=256),
    check("reduce_add", calls=1, tag="edge-single"),
    check("reduce_add", dict(tile_size=64), tag="edge-tiny", smoke=True),
    check("reduce_add", dict(tile_size=16), tag="edge-one-vector"),
    Case("reduce_add", _bf16, calls=16, smoke=True),
    Case("reduce_add", _bf16, calls=256),
    check("reduce_add", dict(tile_size=32, dtype=bfloat16), tag="edge-one-vector"),
    Case("reduce_min", calls=16, smoke=True),
    Case("reduce_min", calls=256),
    check("reduce_min", dict(tile_size=16), tag="edge-one-vector"),
    Case("reduce_min", _bf16, calls=16, smoke=True),
    Case("reduce_min", _bf16, calls=256),
    check("reduce_min", dict(tile_size=32, dtype=bfloat16), tag="edge-one-vector"),
    Case("reduce_max", calls=16, smoke=True),
    Case("reduce_max", calls=256),
    Case("reduce_max", _bf16, calls=16, smoke=True),
    Case("reduce_max", _bf16, calls=256),
    check("reduce_max", dict(tile_size=16), tag="edge-one-vector"),
    check("reduce_max", dict(tile_size=32, dtype=bfloat16), tag="edge-one-vector"),
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
    # AIE2 tanh takes 64 elements per trip, so 1056 leaves a 32-element tail.
    check("tanh", dict(tile_size=1056), tag="tail", smoke=True),
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
    # As tanh above.
    check("sigmoid", dict(tile_size=1056), tag="tail", smoke=True),
    Case("softmax", calls=16, smoke=True),
    Case("softmax", calls=256),
    # The AIE2 exp loop is rotated by one and pipelined only from 144 elements
    # up: 32 runs a single trip on the short path, 160 is the smallest tile on
    # the pipelined path.
    check("softmax", dict(tile_size=32), tag="short-trip", smoke=True),
    check("softmax", dict(tile_size=160), tag="min-pipelined", smoke=True),
    Case("leaky_relu", calls=16, scalars=(0.5,), smoke=True),
    Case("leaky_relu", calls=256, scalars=(0.5,)),
    # 160 is a multiple of the kernel's 32-element step but not of the 128 its
    # unrolled loop consumes per pass, so the remainder pass runs.
    check(
        "leaky_relu", dict(tile_size=160), scalars=(0.5,), tag="unroll-tail", smoke=True
    ),
    Case("exp2f_vec", calls=16, smoke=True),
    Case("exp2f_vec", calls=256),
    # 48 is not a multiple of the 32 elements one block handles, so the
    # 16-element tail runs.
    check("exp2f_vec", dict(tile_size=48), tag="vector-tail", smoke=True),
    # Sized kernels retaining their runtime-count ABI.
    check("add_sized", calls=16, smoke=True),
    check("mul_sized", calls=16, smoke=True),
    check("relu_sized", calls=16, smoke=True),
    check("silu_sized", calls=16, smoke=True),
    # Same remainder pass as the leaky_relu case above on aie2p: 160 steps the
    # 32-lane loop five times, where its unrolled body consumes four. On aie2,
    # 160 is two 64-element trips and a 32-element remainder.
    check("silu_sized", dict(tile_size=160), tag="unroll-tail", smoke=True),
    # AIE2 silu runs the same 64-element trips as gelu below.
    check("silu_sized", dict(tile_size=96), tag="short-trip", smoke=True),
    check("gelu_sized", calls=16, smoke=True),
    # On aie2p gelu's 32-lane loop is unrolled four ways too, so it has the
    # same remainder pass and the same need for a size that is not a multiple
    # of it. On aie2, as for silu, 160 leaves a remainder after two trips.
    check("gelu_sized", dict(tile_size=160), tag="unroll-tail", smoke=True),
    # AIE2 gelu takes 64 elements per trip and loads the next trip's input
    # ahead: 96 runs a single trip, reloading its own input, then the remainder.
    check("gelu_sized", dict(tile_size=96), tag="short-trip", smoke=True),
    *[
        check(name, dict(tile_size=32), tag="edge-tiny", smoke=True)
        for name in ("add_sized", "mul_sized", "relu_sized", "silu_sized", "gelu_sized")
    ],
    # datamovement
    Case("axpy", calls=16, scalars=(2.5,), smoke=True),
    Case("axpy", calls=256, scalars=(2.5,), data_cases=IEEE_FLOAT),
    Case("convert_copy", calls=16, smoke=True),
    Case("convert_copy", calls=256),
    # 272 is a multiple of the kernel's 16-element step but not of the 128 its
    # unrolled loop consumes per pass, so the remainder pass runs.
    check("convert_copy", dict(tile_size=272), tag="unroll-tail", smoke=True),
    # On AIE2 a row this short takes the loop that is not software-pipelined.
    *[
        check(name, dict(tile_size=64), tag="short-row", **kw)
        for name, kw in (("axpy", dict(scalars=(2.5,))), ("convert_copy", {}))
    ],
    Case("expand", calls=16, smoke=True),
    Case("expand", calls=256),
    # AIE2 builds a group's scale once when a group spans blocks; 96 leaves one
    # block of each group unpaired.
    check("expand", dict(tile_size=1024, group_size=64), calls=16, tag="group-64"),
    check("expand", dict(tile_size=768, group_size=96), calls=16, tag="group-96"),
    Case("transpose", dict(subtile=4), calls=16, smoke=True),
    Case("transpose", dict(subtile=8), calls=16, smoke=True),
    Case("transpose", dict(subtile=4, dtype=np.uint8), calls=16, smoke=True),
    Case("transpose", dict(subtile=8, dtype=np.uint32), calls=16),
    # linalg
    Case("mm", _mm_bf16, calls=16),
    Case("mm", _mm_bf16, calls=256),
    # amd/IRON's mm operator compiles the square tile with -DB_COL_MAJ, which
    # is a separate load path in mm.cc; every case above leaves B row-major.
    Case("mm", _mm_bf16_col_maj, calls=16),
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
    # Peano miscompiles the fully unrolled int8 -> int32 K loop from K = 416,
    # so mm_aie2p.h rolls K up there; the 16x16 tile still unrolls.
    *[
        check(
            "mm",
            dict(
                dim_m=m,
                dim_k=k,
                dim_n=n,
                **layout,
                input_dtype=np.int8,
                output_dtype=np.int32,
            ),
            devices=("npu2",),
        )
        for m, k, n, layout in (
            (16, 512, 32, {}),
            (16, 512, 32, dict(b_col_maj=True)),
            (16, 512, 32, dict(c_col_maj=True)),
            (32, 512, 16, dict(b_col_maj=True)),
            (48, 416, 16, dict(c_col_maj=True)),
            (16, 1008, 16, {}),
        )
    ],
    check("mm", _mm_bf16, calls=1, tag="edge-single-tile"),
    # Bounded fused composition: two A bands and multiple K/drain chunks.
    Case("fused_mm", calls=4, smoke=True),
    *[
        check(
            "fused_mm",
            dict(dim_k=48, epilogue=mode),
            # Keep tanh's AIE2 LUT input in its representable domain.
            data_cases=("random", "zeros", "ones", "alternating"),
        )
        for mode in ("gelu", "silu", "sigmoid")
    ],
    check("fused_mm", dict(dim_k=48, clamp=(-0.125, 0.75))),
    check(
        "fused_mm",
        dict(dim_k=48, epilogue="silu", clamp=(-0.125, 0.75)),
        data_cases=("random", "zeros", "ones", "alternating"),
    ),
    # Prepacked bfp16ebs8 B. The first is amd/IRON's aie2p flm GEMM tile
    # (M64, MA32, N64, CT_K128, 8x8x8, OUT_CHUNK 512), so its k step and
    # drain are IRON's; K is one CT_K chunk rather than 512 to fit L1. The
    # last walks two k chunks, which moves B by its byte stride.
    Case(
        "fused_mm",
        dict(
            dim_m=64,
            band_m=32,
            dim_k=128,
            dim_n=64,
            chunk_k=128,
            out_chunk=512,
            bfp16_b=True,
        ),
        calls=4,
        devices=("npu2",),
    ),
    check(
        "fused_mm",
        dict(
            dim_m=64,
            band_m=32,
            dim_k=128,
            dim_n=64,
            chunk_k=128,
            out_chunk=512,
            epilogue="silu",
            bfp16_b=True,
        ),
        devices=("npu2",),
        data_cases=("random", "zeros", "ones", "alternating"),
    ),
    # IRON's other aie2p flm tile, picked when K is a single 512 slice
    # (N128, CT_K32, MA64). Its 32 KiB accumulator leaves room for only
    # one CT_K chunk of K.
    Case(
        "fused_mm",
        dict(
            dim_m=64,
            band_m=64,
            dim_k=32,
            dim_n=128,
            chunk_k=32,
            out_chunk=512,
            bfp16_b=True,
        ),
        calls=4,
        devices=("npu2",),
    ),
    Case(
        "fused_mm",
        dict(
            dim_m=16,
            band_m=16,
            dim_k=64,
            dim_n=16,
            chunk_k=32,
            out_chunk=64,
            bfp16_b=True,
        ),
        calls=4,
        devices=("npu2",),
        smoke=True,
    ),
    # block floating point (aie2p): bfp16ebs8 A, B and C, and the mixed
    # kernel with bf16 A and C; host encode/shuffle via aie.utils.bfp.
    Case("mm_bfp", _mm_bfp, calls=16, devices=("npu2",)),
    Case("mm_bfp_shuffle", calls=4, devices=("npu2",), smoke=True),
    Case("mm_bfp_shuffle", dict(unshuffle=True), calls=4, devices=("npu2",)),
    # A non-square tile: rows and columns walk different strides.
    *[
        Case(
            "mm_bfp_shuffle",
            dict(dim_m=32, unshuffle=unshuffle),
            calls=4,
            devices=("npu2",),
        )
        for unshuffle in (False, True)
    ],
    Case("q4nx_dequant", calls=4, devices=("npu2",), smoke=True),
    check(
        "q4nx_dequant",
        dict(m_tile=16, k_tile=32, group=8, ct_k=16),
        calls=2,
        devices=("npu2",),
    ),
    # Quantization groups need not divide the GEMM K slice.
    check(
        "q4nx_dequant",
        dict(m_tile=48, k_tile=48, group=24, ct_k=16),
        calls=2,
        devices=("npu2",),
    ),
    Case(
        "mm_bfp",
        dict(**_mm_bfp, mixed=True),
        calls=16,
        devices=("npu2",),
    ),
    check(
        "mm",
        dict(
            dim_m=32, dim_k=32, dim_n=32, input_dtype=bfloat16, output_dtype=np.float32
        ),
        calls=16,
        tag="edge-small-tile",
    ),
    Case("mv", dict(dim_m=32, dim_k=32), calls=16),
    # The i16 case above builds mv_i16.cc; amd/IRON's mv operator builds the
    # bf16 kernel out of mv_bf16.cc instead, which no timed case reached.
    Case(
        "mv",
        dict(dim_m=32, dim_k=256, input_dtype=bfloat16, output_dtype=bfloat16),
        calls=16,
    ),
    # The i16 matvec takes two 16-row blocks per pass over the columns, so an
    # odd number of blocks leaves the last one to the tail.
    check(
        "mv", dict(dim_m=48, dim_k=32), tag="edge-rows-not-multiple-of-32", smoke=True
    ),
    check(
        "mv",
        dict(
            dim_m=6, dim_k=128, input_dtype=bfloat16, output_dtype=bfloat16, vec_size=64
        ),
        tag="edge-rows-not-multiple-of-4",
        smoke=True,
    ),
    check(
        "mv",
        dict(dim_m=32, dim_k=256, input_dtype=bfloat16, output_dtype=bfloat16),
        smoke=True,
    ),
    *[
        check(
            "mv",
            dict(
                dim_m=4,
                dim_k=dim_k,
                input_dtype=bfloat16,
                output_dtype=bfloat16,
                vec_size=64,
            ),
            tag=tag,
            smoke=True,
        )
        for dim_k, tag in ((64, "edge-one-chunk"), (128, "edge-two-chunks"))
    ],
    # Past four chunks the bf16 kernel keeps its mac loop and carries only the
    # folded sums from one group of four rows to the next.
    check(
        "mv",
        dict(
            dim_m=10,
            dim_k=512,
            input_dtype=bfloat16,
            output_dtype=bfloat16,
            vec_size=64,
        ),
        tag="edge-mac-loop",
        smoke=True,
    ),
    # The per-call shapes amd/IRON's llama 3.2 1B decode hands the bf16 GEMV:
    # four rows per call over the hidden (2048) and head (64) dims, at the
    # widest vec_size that leaves at least two chunks. The 2048 case is smoke:
    # no other smoke case runs the mac loop over this many chunks.
    *[
        Case(
            "mv",
            dict(
                dim_m=4,
                dim_k=dim_k,
                input_dtype=bfloat16,
                output_dtype=bfloat16,
                vec_size=vec_size,
            ),
            calls=16,
            smoke=smoke,
        )
        for dim_k, vec_size, smoke in ((64, 32, False), (2048, 64, True))
    ],
    # Its ffn down projection runs one row of 8192 per call instead, and 256
    # calls fill one 256-row C tile through row_offset while b stays put
    # (tile_size_input=1, tile_size_output=256 in IRON's GEMV core).
    Case(
        "mv",
        dict(
            dim_m=1,
            dim_k=8192,
            input_dtype=bfloat16,
            output_dtype=bfloat16,
            vec_size=64,
            output_rows=256,
        ),
        calls=256,
        tag="llama-decode-ffn-down",
    ),
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
    # AIE2 steps 32 pixels at a time, pipelined from 4 steps, then 16 at a
    # time: 4 steps and a tail, and 3 steps and a tail.
    check("gray2rgba", dict(line_width=144), tag="tail"),
    check("gray2rgba", dict(line_width=112), tag="short-row"),
    # AIE2P steps 64 pixels at a time: 144 and 112 are 2 and 1 steps and a
    # tail, and 48 is the tail alone.
    check("gray2rgba", dict(line_width=48), tag="tail-only"),
    Case("rgba2gray", calls=16, smoke=True),
    # Five vectors, one under the count the AIE2 pipelined loop requires.
    check("rgba2gray", dict(line_width=160), tag="short-row"),
    # AIE2P steps 64 pixels at a time, pipelined from 4 steps, then 32: 4
    # steps and a tail, and the tail alone.
    check("rgba2gray", dict(line_width=288), tag="tail"),
    check("rgba2gray", dict(line_width=32), tag="one-vector"),
    Case("threshold", calls=16, scalars=(100, 255, 0), smoke=True),
    check("threshold", calls=16, scalars=(100, 255, 2), tag="trunc"),
    check("threshold", calls=16, scalars=(100, 255, 4), tag="tozero-inv"),
    check("threshold", dict(dtype=np.int16), calls=16, scalars=(100, 255, 1)),
    Case("bitwise_or", calls=16, smoke=True),
    Case("bitwise_and", calls=16, smoke=True),
    check("bitwise_or", dict(line_width=64), tag="edge-one-vector"),
    check("bitwise_and", dict(line_width=64), tag="edge-one-vector"),
    *[
        check(
            "threshold",
            dict(line_width=64),
            scalars=(100, 255, mode),
            tag=f"edge-one-vector-mode-{mode}",
        )
        for mode in range(5)
    ],
    # alpha = beta = 0.5 in Q2.14.
    Case("add_weighted", calls=16, scalars=(8192, 8192, 0), smoke=True),
    # The vector path once dropped gamma, so a gamma case runs on every PR.
    check("add_weighted", calls=16, scalars=(8192, 8192, 40), tag="gamma", smoke=True),
    check(
        "add_weighted",
        dict(dtype=np.int16),
        calls=16,
        scalars=(8192, 8192, -300),
        tag="gamma",
    ),
    check(
        "add_weighted",
        dict(line_width=32),
        scalars=(8192, 8192, 0),
        tag="edge-one-vector",
    ),
    Case("filter2d", calls=16, smoke=True),
    # Three middle vectors, one under the count the AIE2 pipelined loop needs.
    check("filter2d", dict(line_width=160), tag="short-row"),
    # AIE2P steps 64 pixels at a time, pipelined from 4 steps, then a last
    # 32 pixels when the row has them: one block with a tail, two blocks, and
    # a pipelined row with a tail.
    check("filter2d", dict(line_width=96), tag="one-block"),
    check("filter2d", dict(line_width=128), tag="two-blocks"),
    check("filter2d", dict(line_width=352), tag="odd-pipelined"),
    Case("rgba2hue", calls=16, smoke=True),
    # Under four vectors, so AIE2 takes the loop that is not software-pipelined.
    check("rgba2hue", dict(line_width=96), tag="short-row"),
    # AIE2P steps 64 pixels at a time, pipelined from 4 steps, then a last
    # 32 pixels when the row has them: a pipelined row with a tail, and a row
    # that is only the tail.
    check("rgba2hue", dict(line_width=288), tag="tail"),
    check("rgba2hue", dict(line_width=32), tag="one-vector"),
    # conv: full-range int8 data (the kernels saturate, so `input_limit` only
    # keeps the int32 accumulator safe); the shift puts random sums around
    # uint8's range (64 channels x 127^2 ~ 2**20 >> 12 for k1; 9x that >> 15
    # for k3) so saturation is exercised without being the whole picture.
    Case("conv2dk1", calls=8, scalars=(32, 64, 64, 12), smoke=True),
    Case("conv2dk1_i8", calls=8, scalars=(32, 64, 64, 12), smoke=True),
    # Two 32-pixel blocks per row; at width 32 the block loop runs once.
    check(
        "conv2dk1",
        dict(input_width=64),
        calls=8,
        scalars=(64, 64, 64, 12),
        tag="two-blocks",
    ),
    check(
        "conv2dk1_i8",
        dict(input_width=64),
        calls=8,
        scalars=(64, 64, 64, 12),
        tag="two-blocks",
    ),
    # conv2dk1_skip streams three tensors; the harness packs them into one
    # fifo when they share a type, i.e. input_channels == 2 * output_channels
    # with a uint8 residual. An int8 residual goes in a fifo of its own.
    Case(
        "conv2dk1_skip",
        dict(input_channels=128, output_channels=64, act_dtype=np.uint8),
        calls=8,
        scalars=(32, 128, 64, 12, 1),
        smoke=True,
    ),
    Case(
        "conv2dk1_skip",
        dict(input_channels=128, output_channels=64, act_dtype=np.int8),
        calls=8,
        scalars=(32, 128, 64, 12, 1),
        tag="int8_skip",
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
    # One ic/16 step on each input half and one ic/8 skip step: loops of a
    # single trip, which the AIE2 kernel runs on a path of their own.
    check(
        "conv2dk1_skip_init",
        dict(input_channels=16, skip_input_channels=8, act_dtype=np.uint8),
        calls=8,
        scalars=(32, 16, 64, 8, 10, 1, 9),
        tag="one-step",
    ),
    # conv2dk14: 16 patches of 14x14 RGBA pixels per call, 784 taps.
    Case(
        "conv2dk14",
        calls=4,
        scalars=(224, 4, 16, 14, 17),
        smoke=True,
    ),
    # An odd number of 8-channel groups, and two 16-patch groups per call.
    check(
        "conv2dk14",
        dict(output_channels=24),
        scalars=(224, 4, 24, 14, 17),
        tag="three-groups",
    ),
    check(
        "conv2dk14",
        dict(input_width=448, output_channels=8),
        scalars=(448, 4, 8, 14, 17),
        tag="two-tile-groups",
    ),
    check(
        "conv2dk14",
        dict(input_width=256, kernel_width=16),
        scalars=(256, 4, 16, 16, 17),
        tag="kernel-width-16",
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
    check(
        "conv2dk3",
        dict(act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 3, 3, 0, 15, 0),
        tag="top-row",
    ),
    check("conv2dk3", calls=8, scalars=(32, 64, 64, 3, 3, 0, 15, 0), tag="top-row"),
    *[
        check(
            "conv2dk3",
            dict(act_dtype=dt),
            calls=8,
            scalars=(32, 64, 64, 3, 3, 2, 15, 0),
            tag="bottom-row",
        )
        for dt in (np.int8, np.uint8)
    ],
    # The ResNet/bottleneck split: two workers share the weights buffer and
    # each computes half the output channels, selected by channel_offset.
    *[
        check(
            "conv2dk3",
            dict(act_dtype=dt, output_channels=32, weight_output_channels=64),
            calls=8,
            scalars=(32, 64, 32, 3, 3, 1, 15, 32),
            tag="channel-offset",
        )
        for dt in (np.int8, np.uint8)
    ],
    # MobileNet bottleneck kernels at MobileNet V3 layer shapes; shifts put
    # the random sums around the output range, as for conv above.
    Case(
        "bn_conv2dk1_relu",
        dict(input_width=28, input_channels=40, output_channels=120),
        calls=8,
        scalars=(28, 40, 120, 8),
        smoke=True,
    ),
    Case(
        "bn_conv2dk1_relu",
        dict(input_width=112, input_channels=16, output_channels=64),
        calls=8,
        scalars=(112, 16, 64, 7),
    ),
    Case(
        "bn_conv2dk1_relu",
        dict(input_width=14, input_channels=80, output_channels=184),
        calls=8,
        scalars=(14, 80, 184, 8),
    ),
    Case(
        "bn_conv2dk1_relu",
        dict(input_width=7, input_channels=80, output_channels=120),
        calls=8,
        scalars=(7, 80, 120, 8),
    ),
    Case(
        "bn_conv2dk1_i8",
        dict(input_width=28, input_channels=120, output_channels=40),
        calls=8,
        scalars=(28, 120, 40, 10),
        smoke=True,
    ),
    Case(
        "bn_conv2dk1_i8",
        dict(input_width=56, input_channels=64, output_channels=24),
        calls=8,
        scalars=(56, 64, 24, 10),
    ),
    Case(
        "bn_conv2dk1_i8",
        dict(input_width=14, input_channels=240, output_channels=80),
        calls=8,
        scalars=(14, 240, 80, 11),
    ),
    Case(
        "bn_conv2dk1_i8",
        dict(input_width=7, input_channels=336, output_channels=80),
        calls=8,
        scalars=(7, 336, 80, 11),
    ),
    Case(
        "bn_conv2dk1_skip",
        dict(input_width=28, input_channels=120, output_channels=40),
        calls=8,
        scalars=(28, 120, 40, 10, 1),
        smoke=True,
    ),
    Case(
        "bn_conv2dk1_skip",
        dict(
            input_width=28,
            input_channels=120,
            output_channels=40,
            skip_dtype=np.int8,
        ),
        calls=8,
        scalars=(28, 120, 40, 10, 1),
    ),
    Case(
        "bn_conv2dk1_skip",
        dict(input_width=56, input_channels=72, output_channels=24),
        calls=8,
        scalars=(56, 72, 24, 10, 1),
    ),
    Case(
        "bn_conv2dk1_skip",
        dict(input_width=7, input_channels=240, output_channels=40),
        calls=8,
        scalars=(7, 240, 40, 11, 1),
    ),
    Case(
        "bn_conv2dk1_skip",
        dict(
            input_width=14,
            input_channels=184,
            output_channels=80,
            skip_dtype=np.int8,
        ),
        calls=8,
        scalars=(14, 184, 80, 11, 1),
    ),
    Case(
        "bn_conv2dk3",
        dict(input_width=224, input_channels=8, output_channels=16),
        calls=8,
        scalars=(224, 8, 16, 3, 3, 1, 8, 0),
        smoke=True,
    ),
    check(
        "bn_conv2dk3",
        dict(input_width=224, input_channels=8, output_channels=16),
        calls=8,
        scalars=(224, 8, 16, 3, 3, 0, 8, 0),
        tag="top-row",
    ),
    check(
        "bn_conv2dk3",
        dict(input_width=56, input_channels=16, output_channels=24),
        calls=8,
        scalars=(56, 16, 24, 3, 3, 2, 9, 0),
        tag="bottom-row",
    ),
    check(
        "bn_conv2dk3",
        dict(input_width=112, input_channels=32, output_channels=16),
        calls=8,
        scalars=(112, 32, 16, 3, 3, 1, 10, 0),
    ),
    check(
        "bn_conv2dk3",
        dict(input_width=56, input_channels=8, output_channels=24),
        calls=8,
        scalars=(56, 8, 24, 3, 3, 2, 8, 0),
        tag="bottom-row",
    ),
    check(
        "bn_conv2dk3",
        dict(input_width=40, input_channels=24, output_channels=8),
        calls=8,
        scalars=(40, 24, 8, 3, 3, 0, 9, 0),
        tag="top-row",
    ),
    *[
        check(
            "bn_conv2dk3",
            dict(input_width=w, input_channels=8, output_channels=16),
            calls=8,
            scalars=(w, 8, 16, 3, 3, 1, 8, 0),
        )
        for w in (8, 24, 88)
    ],
    # The second of two workers sharing the weights buffer.
    *[
        check(
            "bn_conv2dk3",
            dict(
                input_width=w,
                input_channels=ic,
                output_channels=8,
                weight_output_channels=16,
            ),
            calls=8,
            scalars=(w, ic, 8, 3, 3, 1, 9, 8),
            tag="channel-offset",
        )
        for w, ic in ((64, 16), (32, 8))
    ],
    Case(
        "bn_conv2dk3_dw",
        dict(input_width=28, input_channels=120, output_channels=120),
        calls=8,
        scalars=(28, 120, 120, 3, 3, 1, 7, 0),
        smoke=True,
    ),
    Case(
        "bn_conv2dk3_dw",
        dict(input_width=56, input_channels=72, output_channels=72, stride=2),
        calls=8,
        scalars=(56, 72, 72, 3, 3, 1, 7, 0),
    ),
    check(
        "bn_conv2dk3_dw",
        dict(input_width=56, input_channels=72, output_channels=72, stride=2),
        calls=8,
        scalars=(56, 72, 72, 3, 3, 0, 7, 0),
        tag="top-row",
    ),
    Case(
        "bn_conv2dk3_dw",
        dict(input_width=14, input_channels=336, output_channels=336, stride=2),
        calls=8,
        scalars=(14, 336, 336, 3, 3, 1, 7, 0),
    ),
    Case(
        "bn_conv2dk3_dw",
        dict(input_width=14, input_channels=184, output_channels=184),
        calls=8,
        scalars=(14, 184, 184, 3, 3, 1, 7, 0),
    ),
    check(
        "bn_conv2dk3_dw",
        dict(input_width=28, input_channels=120, output_channels=120),
        calls=8,
        scalars=(28, 120, 120, 3, 3, 2, 7, 0),
        tag="bottom-row",
    ),
    Case(
        "bn_conv2dk3_dw_out_split",
        dict(input_width=7, input_channels=480, output_split_channels=240),
        calls=8,
        scalars=(7, 480, 480, 3, 3, 1, 7, 0),
        smoke=True,
    ),
    check(
        "bn_conv2dk3_dw_out_split",
        dict(input_width=7, input_channels=480, output_split_channels=240),
        calls=8,
        scalars=(7, 480, 480, 3, 3, 0, 7, 0),
        tag="top-row",
    ),
    # Row widths of one to four 8-pixel chunks, unaligned ones, and fewer
    # channel blocks than AIE2P's chunked path takes. The 6 to 8 pixel rows
    # take AIE2P's whole-granule stores.
    *[
        check(
            "bn_conv2dk3_dw",
            dict(input_width=w, input_channels=c, output_channels=c),
            calls=8,
            scalars=(w, c, c, 3, 3, row, 7, 0),
            tag=tag,
        )
        for w, c, row, tag in (
            (5, 40, 0, "top-row"),
            (8, 32, 1, "middle-row"),
            (20, 48, 2, "bottom-row"),
            (30, 32, 1, "middle-row"),
            (32, 32, 0, "top-row"),
            (12, 24, 1, "middle-row"),
            (6, 40, 2, "bottom-row"),
            (7, 56, 1, "middle-row"),
            (8, 64, 0, "top-row"),
        )
    ],
    check(
        "bn_conv2dk3_dw_out_split",
        dict(input_width=6, input_channels=80, output_split_channels=40),
        calls=8,
        scalars=(6, 80, 80, 3, 3, 2, 7, 0),
        tag="bottom-row",
    ),
    # One of MobileNet's eight 120-channel weight slices, on the last row so
    # the average is taken.
    Case(
        "bn_conv2dk1_relu_xy_pool_padded",
        dict(input_width=7, input_channels=80, output_channels=120),
        calls=8,
        scalars=(7, 80, 120, 120, 8, 6, 1, 0),
        smoke=True,
    ),
    check(
        "bn_conv2dk1_relu_xy_pool_padded",
        dict(input_width=7, input_channels=80, output_channels=120),
        calls=8,
        scalars=(7, 80, 120, 120, 8, 0, 1, 0),
        tag="first-row",
    ),
    check(
        "bn_conv2dk1_relu_xy_pool_padded",
        dict(input_width=7, input_channels=80, output_channels=120),
        calls=8,
        scalars=(7, 80, 120, 120, 8, 3, 1, 0),
        tag="mid-row",
    ),
    check(
        "bn_conv2dk1_relu_xy_pool_padded",
        dict(
            input_width=7,
            input_channels=80,
            output_channels=128,
            weight_chunk_count=80 * 64,
        ),
        calls=8,
        scalars=(7, 80, 128, 128, 8, 6, 2, 1),
        tag="split",
    ),
    Case(
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=8),
        calls=8,
        scalars=(1, 1280, 1280, 8, 12),
        smoke=True,
    ),
    # MobileNet's FC1: 960 of the 1280 weight rows used.
    Case(
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=8),
        calls=8,
        scalars=(1, 960, 1280, 8, 11),
        tag="fc1",
    ),
    check(
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=16),
        calls=8,
        scalars=(1, 960, 1280, 16, 12),
        tag="two-blocks",
    ),
    check(
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=8),
        calls=8,
        scalars=(1, 1264, 1280, 8, 12),
        tag="ic-tail",
    ),
    check(
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=8),
        calls=8,
        scalars=(1, 48, 1280, 8, 8),
        tag="short",
    ),
    # eltwise mul/add selected per call (programming_examples/ml/scale_shift)
    Case("mul_add", calls=16, scalars=(1,), smoke=True),
    Case("mul_add", calls=16, scalars=(0,), tag="add", smoke=True),
    # transformer blocks: one row per call
    Case("rms_norm", dict(cols=1024), calls=16, smoke=True),
    check("rms_norm_eps", dict(cols=1024), calls=16, scalars=(1e-5,), smoke=True),
    # 200 is too short for the pipelined loops and 1000 has an odd chunk
    # count; both end in a scalar tail.
    check("rms_norm", dict(cols=200), calls=16, tag="row-tail"),
    check("rms_norm", dict(cols=1000), calls=16, tag="row-tail"),
    Case("layer_norm", dict(cols=1024), calls=16, smoke=True),
    # Rows of an odd number of 16-lane halves: 208 is too short for the
    # pipelined loops and 1008 leaves an odd chunk (aie2p ends in a half vector).
    check("layer_norm", dict(cols=208), calls=16, tag="row-tail"),
    check("layer_norm", dict(cols=1008), calls=16, tag="row-tail"),
    Case(
        "layer_norm_f32",
        dict(cols=1024),
        calls=16,
        smoke=True,
    ),
    Case(
        "layer_norm_affine_cast",
        dict(cols=1024),
        calls=16,
        smoke=True,
    ),
    # On aie2, 112 is too short for the pipelined loops and 1008 leaves an odd
    # chunk and three past a multiple of four.
    *[
        check(name, dict(cols=cols), calls=16, tag="row-tail")
        for name in ("layer_norm_f32", "layer_norm_affine_cast")
        for cols in (112, 1008)
    ],
    Case("rope", dict(cols=1024), calls=16, smoke=True),
    # 1008 is a multiple of 16 but not of the 64 the interleaved row walks in,
    # and 96 halves into 48, which is a multiple of 16 but not of the 32 the
    # two-halves row walks in. Both rows are therefore the shortest ones that
    # reach each kernel's close-out step, which 1024 never does.
    check("rope", dict(cols=1008), calls=16, tag="row-tail", smoke=True),
    check("rope", dict(cols=1024, two_halves=True), calls=16, smoke=True),
    check("rope", dict(cols=96, two_halves=True), calls=16, tag="row-tail", smoke=True),
    # On aie2, 112 is too short for the pipelined interleaved loop.
    check("rope", dict(cols=112), calls=16, tag="row-tail"),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(0,),
        tag="identity",
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(1,),
        tag="silu",
        smoke=True,
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(2,),
        tag="gelu",
        smoke=True,
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(3,),
        tag="relu",
    ),
    # On aie2 the main loops need 4 trips, and the SiLU and GELU loops run one
    # fewer than the row has vectors: 16 runs none of those, 48 only the
    # fallback loops, and 80 the main loops at their minimum count.
    *[
        check(
            "mm_activation_epilogue",
            dict(tile_size=size),
            scalars=(mode,),
            tag=f"{act}-{tag}",
        )
        for size, tag in ((16, "one-vector"), (48, "short-row"), (80, "min-trips"))
        for mode, act in enumerate(("identity", "silu", "gelu", "relu"))
    ],
    # On aie2, identity and ReLU select on the float's bit pattern.
    *[
        check(
            "mm_activation_epilogue",
            calls=16,
            scalars=(mode,),
            tag=f"{act}-ieee",
            data_cases=IEEE_FLOAT,
            devices=("npu1",),
        )
        for mode, act in ((0, "identity"), (3, "relu"))
    ],
    # depthwise 1-D conv: 1024 outputs per call from a padded row
    Case(
        "dwconv1d",
        dict(seq_len=1024, kernel_size=9),
        calls=16,
        scalars=(1024,),
    ),
    Case(
        "dwconv1d_channels_first",
        dict(seq_len=1024, kernel_size=9),
        calls=16,
        scalars=(1024,),
        smoke=True,
    ),
    # one tap is the degenerate split: the kernel halves its taps across two
    # sliding_mul chains and kernel_size=1 is the only shape with nothing in
    # the second chain.
    check(
        "dwconv1d_channels_first",
        dict(seq_len=1024, kernel_size=1),
        calls=16,
        scalars=(1024,),
    ),
    # 1008 is 63 blocks of 16, so the kernel's two-blocks-per-pass loop has to
    # run its tail pass; 1024 is 64 blocks and never does.
    check(
        "dwconv1d_channels_first",
        dict(seq_len=1008, kernel_size=9),
        calls=16,
        scalars=(1008,),
        tag="odd-block-count",
    ),
    # three blocks are under the four AIE2's pipelined loop requires; 17 taps
    # is the widest window.
    check(
        "dwconv1d_channels_first",
        dict(seq_len=48, kernel_size=17),
        calls=16,
        scalars=(48,),
        tag="short-row",
    ),
    # the transposed layout: one timestep across 256 channels, 5 per-channel taps
    Case(
        "dwconv1d_channels_last",
        dict(channels=256),
        calls=16,
        smoke=True,
    ),
    check(
        "dwconv1d_channels_last",
        dict(channels=256, clamp=False),
        calls=16,
        tag="unclamped",
    ),
    # a channel count AIE2P's 64-lane groups do not divide
    check(
        "dwconv1d_channels_last",
        dict(channels=96),
        calls=16,
        tag="odd-group",
    ),
    # channel counts that walk the rolled chunk loop and then a tail
    check(
        "dwconv1d_channels_last",
        dict(channels=320),
        calls=16,
        tag="chunk-tail",
    ),
    check(
        "dwconv1d_channels_last",
        dict(channels=480),
        calls=16,
        tag="odd-chunk-tail",
    ),
    # overflowed its declared stack on three builds when every channel unrolled
    check(
        "dwconv1d_channels_last",
        dict(channels=512),
        calls=16,
        tag="two-chunks",
    ),
    # amd/IRON model shapes: Llama 3.2 1B. Each is one core's per-call tile as
    # IRON instantiates the model at a 2048-token context. The decode GEMVs over
    # 2048 and 64 columns are the mv cases above. The one-row ffn down GEMV is
    # not here: its 2-byte output is below the 4-byte DMA transfer minimum.
    # prefill q/k/v/o, ffn and attention-score GEMMs: bf16 in and out, 8x8x8
    # mmul emulated with bfp16. No "large" data: bfp16's shared block
    # exponent leaves an error proportional to the operands, which at 1e4
    # scale dwarfs the matmul tolerance's 0.5 atol wherever a sum cancels.
    Case(
        "mm",
        dict(
            dim_m=64,
            dim_k=64,
            dim_n=64,
            input_dtype=bfloat16,
            output_dtype=bfloat16,
            emulate_bf16_mmul_with_bfp16=True,
        ),
        calls=16,
        tag="llama-prefill",
        devices=("npu2",),
        data_cases=("random", "zeros", "ones", "alternating"),
    ),
    # prefill LM head GEMM, B column-major
    Case(
        "mm",
        dict(
            dim_m=64,
            dim_k=64,
            dim_n=64,
            input_dtype=bfloat16,
            output_dtype=bfloat16,
            b_col_maj=True,
            emulate_bf16_mmul_with_bfp16=True,
        ),
        calls=16,
        tag="llama-prefill-lm-head",
        devices=("npu2",),
        data_cases=("random", "zeros", "ones", "alternating"),
    ),
    # ffn gate activation
    Case("silu_sized", dict(tile_size=4096), calls=16, tag="llama-prefill"),
    Case("silu_sized", dict(tile_size=1024), calls=16, tag="llama-decode"),
    # ffn gate product and attention-score scaling
    Case("mul_sized", dict(tile_size=4096), calls=16, tag="llama-prefill-ffn"),
    Case("mul_sized", dict(tile_size=2048), calls=16, tag="llama-prefill-attn-scale"),
    Case("mul_sized", dict(tile_size=1024), calls=16, tag="llama-decode-ffn"),
    Case("mul_sized", dict(tile_size=256), calls=16, tag="llama-decode-attn-scale"),
    # residual adds
    Case("add_sized", dict(tile_size=2048), calls=16, tag="llama-prefill"),
    Case("add_sized", dict(tile_size=256), calls=16, tag="llama-decode"),
    # attention weights, one prompt-length row of scores per call
    Case("softmax", dict(tile_size=2048), calls=16, tag="llama-prefill"),
    # q/k rotary embedding, one 64-wide head row per call
    Case("rope", dict(cols=64, two_halves=True), calls=16, tag="llama"),
    # decode KV-cache transpose, 256x32 in 8x8 subtiles
    Case(
        "transpose",
        dict(dim_m=256, dim_n=32, subtile=8),
        calls=16,
        tag="llama-decode",
    ),
]


# Smaller shapes for the per-PR smoke test; the nightly times the shapes above.
CASES += [
    check("passthrough", smoke=True),
    check("mm", _mm_bf16, smoke=True),
    check("mm", dict(**_mm, input_dtype=np.int16, output_dtype=np.int32), smoke=True),
    check("mm", dict(**_mm_bf16, b_col_maj=True), smoke=True),
    check(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32, c_col_maj=True),
        smoke=True,
    ),
    check(
        "mm",
        dict(**_mm, input_dtype=np.int8, output_dtype=np.int32, c_col_maj=True),
        smoke=True,
    ),
    check("mm", dict(**_mm_bf16, c_col_maj=True), smoke=True),
    check(
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
    ),
    check("mv", dict(dim_m=32, dim_k=32), smoke=True),
    # The attention toolkit's QK^T product: mm.cc's bf16 tile matmul.
    check("mha", smoke=True),
    # ...and its P*V product, mha.cc's own 8x8x8 expansion. Same tile, but a
    # different micro-tile and so a different blocked operand order, which is
    # the part a shared case could not check.
    check("mha", dict(pv=True), smoke=True),
    # The toolkit's online softmax over one key block: params is (key block,
    # query block), scalars the two sequence lengths. The padded diagonal
    # block takes every branch the full one skips (the causal mask, masked
    # tail keys, padded query rows), so the two smoke cases cover the kernel;
    # the plain diagonal block is its own cycle series.
    Case(
        "mha_softmax",
        calls=4,
        params=((0, 1),),
        scalars=(128, 64),
        smoke=True,
    ),
    Case(
        "mha_softmax",
        calls=4,
        params=((0, 0),),
        scalars=(64, 64),
        tag="diagonal",
    ),
    check(
        "mha_softmax",
        params=((0, 0),),
        scalars=(37, 37),
        tag="diagonal-padded",
        smoke=True,
    ),
    # The prefill toolkit's S*V accumulate, one case per geometry. Each
    # -DPREFILL_HEAD_DIM build is its own object with its own blocked V order;
    # the 512 one has a degenerate k-block term and so cannot tell a wrong V
    # order from a right one, which is why both are smoke cases.
    Case("prefill_fv", dict(head_dim=512), calls=4, smoke=True),
    Case("prefill_fv", dict(head_dim=256), calls=4, smoke=True),
    check("mm_bfp", _mm_bfp, devices=("npu2",), smoke=True),
    check("mm_bfp", dict(**_mm_bfp, mixed=True), devices=("npu2",), smoke=True),
    # The mixed kernel walks its 2x2 output tiles with one loop that wraps at
    # the end of each tile row; with fewer tile columns than rows, a wrap that
    # used the wrong extent would land a tile on the wrong row.
    check(
        "mm_bfp",
        dict(dim_m=64, dim_k=32, dim_n=32, mixed=True),
        devices=("npu2",),
        tag="non-square",
        smoke=True,
    ),
    # The bfp16 kernel pops two k blocks per row at a time only when the block
    # count is even; K=24 takes its odd path. N wider than M checks the row
    # wrap from the other side of the mixed case above.
    check(
        "mm_bfp",
        dict(dim_m=32, dim_k=24, dim_n=48),
        devices=("npu2",),
        tag="odd-k",
        smoke=True,
    ),
]
