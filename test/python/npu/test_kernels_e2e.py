# test_kernels_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

"""End-to-end tests for the IRON kernel library factories.

``test/python/test_kernels_specs.py`` pins each factory's declared surface
and ``test/python/test_kernel_contracts.py`` proves the contract matches it
and lowers to MLIR, all on the host. This file is the tier that needs a
device: one run per kernel through ``aie.utils.kernel_harness``, judged under
the tolerance the kernel declares. It catches the mismatches types cannot --
wrong exported symbol, wrong compile flag, DMA-alignment bugs, a reference
that disagrees with the C++.

Adding a kernel to this file is one line in ``CASES``. The design, inputs,
reference and tolerance all come from the factory's ``KernelContract``.

Full sweeps (every case x data case x seed) are the nightly benchmark's
correctness gate (``benchmarks/kernels/test_correctness.py``); this file
runs on pull requests and stays one representative case per kernel.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.kernels._common import _detect_arch
from aie.utils import kernel_harness as kh
from ml_dtypes import bfloat16

# (factory name, factory kwargs, harness options). ``aie2p`` marks kernels
# whose source exists only for that arch.
CASES = {
    "passthrough/int32": ("passthrough", {}, dict(calls=4)),
    "passthrough/int16": ("passthrough", dict(dtype=np.int16), dict(calls=4)),
    "passthrough/uint8": ("passthrough", dict(dtype=np.uint8), dict(calls=4)),
    "scale/int16": ("scale", dict(dtype=np.int16), dict(calls=4)),
    "scale/int32": ("scale", dict(dtype=np.int32), dict(calls=4)),
    "add": ("add", {}, dict(calls=4)),
    "mul": ("mul", {}, dict(calls=4)),
    "relu": ("relu", {}, dict(calls=4)),
    "reduce_add": ("reduce_add", {}, dict(calls=4)),
    "reduce_min": ("reduce_min", {}, dict(calls=4)),
    "reduce_max/int32": ("reduce_max", {}, dict(calls=4)),
    # Pins the DMA-alignment fix: the bf16 output tile is 2 elements wide.
    "reduce_max/bf16": ("reduce_max", dict(dtype=bfloat16), dict(calls=4)),
    "gelu": ("gelu", {}, dict(calls=2)),
    "silu": ("silu", {}, dict(calls=2)),
    "bf16_exp": ("bf16_exp", {}, dict(calls=2)),
    "tanh": ("tanh", {}, dict(calls=2)),
    "sigmoid": ("sigmoid", {}, dict(calls=2)),
    "softmax": ("softmax", {}, dict(calls=2)),
    # alpha = 0.5 is exact in bf16, so no rounding masks a transport bug.
    "leaky_relu": ("leaky_relu", {}, dict(calls=4, scalars=(0.5,))),
    "exp2f_vec": ("exp2f_vec", {}, dict(calls=2, aie2p=True)),
    # a is a fractional f32 scalar: the call builder types it by the kernel's arg.
    "axpy": ("axpy", {}, dict(calls=4, scalars=(2.5,))),
    "convert_copy": ("convert_copy", {}, dict(calls=2, aie2p=True)),
    "expand": ("expand", {}, dict(calls=2)),
    "transpose/4": ("transpose", dict(subtile=4), dict(calls=2)),
    "transpose/8": ("transpose", dict(subtile=8), dict(calls=2)),
    "transpose/uint8": ("transpose", dict(subtile=4, dtype=np.uint8), dict(calls=2)),
    "mm/bf16_f32": (
        "mm",
        dict(
            dim_m=64, dim_k=32, dim_n=64, input_dtype=bfloat16, output_dtype=np.float32
        ),
        dict(shape=(128, 128, 128)),
    ),
    "mm/i16_i32": (
        "mm",
        dict(dim_m=64, dim_k=32, dim_n=64, input_dtype=np.int16, output_dtype=np.int32),
        dict(shape=(128, 128, 128)),
    ),
    "mv": ("mv", dict(dim_m=32, dim_k=32), dict(shape=(128, 128))),
    "mm_bfp": (
        "mm_bfp",
        dict(dim_m=64, dim_k=64, dim_n=64),
        dict(shape=(128, 128, 128), aie2p=True),
    ),
    "mm_bfp/mixed": (
        "mm_bfp",
        dict(dim_m=64, dim_k=64, dim_n=64, mixed=True),
        dict(shape=(128, 128, 128), aie2p=True),
    ),
    "mm/bf16_f32/b_col_maj": (
        "mm",
        dict(
            dim_m=64,
            dim_k=32,
            dim_n=64,
            input_dtype=bfloat16,
            output_dtype=np.float32,
            b_col_maj=True,
        ),
        dict(shape=(128, 128, 128)),
    ),
    "mm/i16_i32/c_col_maj": (
        "mm",
        dict(
            dim_m=64,
            dim_k=32,
            dim_n=64,
            input_dtype=np.int16,
            output_dtype=np.int32,
            c_col_maj=True,
        ),
        dict(shape=(128, 128, 128)),
    ),
    "compute_max/int32": ("compute_max", {}, dict(calls=4)),
    "compute_max/bf16": ("compute_max", dict(dtype=bfloat16), dict(calls=4)),
    "swiglu": ("swiglu", {}, dict(calls=2)),
    "gray2rgba": ("gray2rgba", {}, dict(calls=2)),
    "rgba2gray": ("rgba2gray", {}, dict(calls=2)),
    "threshold": ("threshold", {}, dict(calls=2, scalars=(100, 255, 0))),
    "bitwise_or": ("bitwise_or", {}, dict(calls=2)),
    "bitwise_and": ("bitwise_and", {}, dict(calls=2)),
    "add_weighted": ("add_weighted", {}, dict(calls=2, scalars=(8192, 8192, 0))),
    "filter2d": ("filter2d", {}, dict(calls=2)),
    "rgba2hue": ("rgba2hue", {}, dict(calls=2)),
    "conv2dk1": ("conv2dk1", {}, dict(calls=2, scalars=(32, 64, 64, 12))),
    "conv2dk3": (
        "conv2dk3",
        dict(act_dtype=np.uint8),
        dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 15, 0)),
    ),
    "conv2dk3/int8": (
        "conv2dk3",
        {},
        dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 15, 0)),
    ),
    "conv2dk1_i8": ("conv2dk1_i8", {}, dict(calls=2, scalars=(32, 64, 64, 12))),
    "conv2dk1_skip": (
        "conv2dk1_skip",
        dict(input_channels=128, output_channels=64, act_dtype=np.uint8),
        dict(calls=2, scalars=(32, 128, 64, 12, 1)),
    ),
    "conv2dk1_skip_init": (
        "conv2dk1_skip_init",
        dict(input_channels=64, skip_input_channels=32, act_dtype=np.uint8),
        dict(calls=2, scalars=(32, 64, 64, 32, 12, 1, 11)),
    ),
    "conv2dk14": (
        "conv2dk14",
        {},
        dict(calls=2, scalars=(224, 4, 16, 14, 17), aie2p=True),
    ),
    "bn_conv2dk1_relu": (
        "bn_conv2dk1_relu",
        {},
        dict(calls=2, scalars=(32, 64, 64, 12)),
    ),
    "bn_conv2dk1_i8": ("bn_conv2dk1_i8", {}, dict(calls=2, scalars=(32, 64, 64, 13))),
    "bn_conv2dk1_skip": (
        "bn_conv2dk1_skip",
        {},
        dict(calls=2, scalars=(32, 64, 64, 13, 1)),
    ),
    "bn_conv2dk3_dw": (
        "bn_conv2dk3_dw",
        {},
        dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 11, 0)),
    ),
    "bn_conv2dk3_dw/stride2": (
        "bn_conv2dk3_dw",
        dict(stride=2),
        dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 11, 0)),
    ),
    "bn_conv2dk3": (
        "bn_conv2dk3",
        {},
        dict(calls=2, scalars=(32, 64, 64, 3, 3, 1, 15, 0)),
    ),
    "mul_add/mul": ("mul_add", {}, dict(calls=2, scalars=(1,))),
    "mul_add/add": ("mul_add", {}, dict(calls=2, scalars=(0,))),
    "rms_norm": ("rms_norm", dict(cols=1024), dict(calls=2, aie2p=True)),
    "layer_norm": ("layer_norm", dict(cols=1024), dict(calls=2, aie2p=True)),
    "layer_norm_f32": ("layer_norm_f32", dict(cols=1024), dict(calls=2, aie2p=True)),
    "layer_norm_affine_cast": (
        "layer_norm_affine_cast",
        dict(cols=1024),
        dict(calls=2, aie2p=True),
    ),
    "rope": ("rope", dict(cols=1024), dict(calls=2, aie2p=True)),
    "mm_activation_epilogue/silu": (
        "mm_activation_epilogue",
        {},
        dict(calls=2, scalars=(1,), aie2p=True),
    ),
    "mm_activation_epilogue/gelu": (
        "mm_activation_epilogue",
        {},
        dict(calls=2, scalars=(2,), aie2p=True),
    ),
    "dwconv1d": (
        "dwconv1d",
        dict(seq_len=1024, kernel_size=9),
        dict(calls=2, scalars=(1024,), aie2p=True),
    ),
    "bn_fc_relu_ui16_pad": (
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=16),
        dict(calls=2, scalars=(1, 1280, 1280, 16, 13)),
    ),
}


@pytest.mark.parametrize("case_id", list(CASES))
def test_kernel_e2e(case_id):
    name, fkw, opts = CASES[case_id]
    opts = dict(opts)
    if opts.pop("aie2p", False) and _detect_arch() != "aie2p":
        pytest.skip(f"{name} binds an aie2p-only source")
    verdict = kh.check(
        getattr(kernels, name), rng=np.random.default_rng(0), **opts, **fkw
    )
    assert verdict, f"{case_id}: {verdict.detail}"


# ---------------------------------------------------------------------------
# mha: flash-attention toolkit — compile regression only.
#
# mha.cc is not a single kernel but a set of composable symbols (matmul_PV,
# partial_softmax, rescale_O, init_scale_buffer, …) that ``#include`` sibling
# ``softmax.cc`` + ``mm.cc``.  A full attention dataflow needs a bespoke
# multi-core design; that's out of scope here.  What this test pins is that
# mha.cc must COMPILE against mlir-aie's softmax.cc, which relies on
# ``partial_softmax_bf16`` / ``partial_softmax_alias_bf16`` being defined there.
# If those regress, mha.cc stops compiling and this test fails.
# ``init_scale_buffer`` is the simplest symbol to instantiate the translation
# unit.
# ---------------------------------------------------------------------------

_MHA_TILE = 1024


@iron.jit
def _mha_compile_probe(
    a_in: In, b_out: Out, *, size: iron.CompileTime[int] = _MHA_TILE
):
    buf = np.ndarray[(_MHA_TILE,), np.dtype[bfloat16]]
    # kernels.mha compiles mha.cc once and binds its symbols; init_scale_buffer
    # is the simplest of them to instantiate the translation unit with.
    kern = kernels.mha(dim_m=64, dim_k=64, dim_n=64).init_scale_buffer
    of_in = ObjectFifo(buf, name="mhi")
    of_out = ObjectFifo(buf, name="mho")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(c, _MHA_TILE)  # init_scale_buffer writes its buffer arg
        for i in range_(_MHA_TILE):
            c[i] = a[i]
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])
    vec = np.ndarray[(size,), np.dtype[bfloat16]]

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [vec, vec, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_mha_compiles_against_mlir_aie_softmax():
    """mha.cc compiles + runs on aie2p (prereq: partial_softmax_* in softmax.cc).

    aie2p-only: mha.cc is an aie2p source and pulls in the aie2p mm/softmax
    micro-kernels.  Skip on any other arch rather than fail.
    """
    if _detect_arch() != "aie2p":
        pytest.skip("mha.cc is aie2p-only")
    a = iron.tensor(np.zeros(_MHA_TILE, dtype=bfloat16), dtype=bfloat16, device="npu")
    b = iron.zeros(_MHA_TILE, dtype=bfloat16, device="npu")
    _mha_compile_probe(a, b, size=_MHA_TILE)
    # Reaching here means aiecc compiled mha.cc (with its softmax.cc/mm.cc
    # includes) and the design ran — the partial_softmax_* symbols resolved.


# ---------------------------------------------------------------------------
# generic/mv.cc:  bf16 matrix-vector multiply (c = A @ b).
#
# No kernels.* factory exposes this yet (kernels.mv resolves aie2/mv.cc, an
# i16->i32 kernel), so the ExternalFunction is hand-built directly against the
# source — same approach as the mha probe above.  The entry is
# matvec_vectorized_bf16_bf16(m, row_offset,
# a, b, c); it needs -DDIM_K and assumes k >= 2*VEC_SIZE (VEC_SIZE=64 -> k>=128).
# Once it gets a factory and a contract, this collapses into CASES above.
# ---------------------------------------------------------------------------

_MV_M = 32
_MV_K = 128


@iron.jit
def _mv_design(a_in: In, b_in: In, c_out: Out):
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _include_dirs, _kernel_source

    # generic/ source is arch-independent; the subdir arg pins it to generic/mv.cc.
    src = _kernel_source("aie2", "generic", "mv.cc")
    a_ty = np.ndarray[(_MV_M * _MV_K,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(_MV_K,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(_MV_M,), np.dtype[bfloat16]]
    kern = ExternalFunction(
        "matvec_vectorized_bf16_bf16",
        source_file=str(src),
        arg_types=[np.int32, np.int32, a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DDIM_K={_MV_K}"],
    )
    of_a = ObjectFifo(a_ty, name="mva")
    of_b = ObjectFifo(b_ty, name="mvb")
    of_c = ObjectFifo(c_ty, name="mvc")

    def core(of_a, of_b, of_c, k):
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        c = of_c.acquire(1)
        k(_MV_M, 0, a, b, c)  # (m, row_offset=0, a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    w = Worker(core, fn_args=[of_a.cons(), of_b.cons(), of_c.prod(), kern])

    def seq(a, b, c, ah, bh, ch):
        ah.fill(a)
        bh.fill(b)
        ch.drain(c, wait=True)

    rt = Runtime(seq, [a_ty, b_ty, c_ty, of_a.prod(), of_b.prod(), of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_mv_bf16_e2e():
    from aie.utils.verify import Tolerance, compare

    rng = np.random.default_rng(7)
    mat = rng.uniform(-1, 1, size=(_MV_M, _MV_K)).astype(bfloat16)
    vec = rng.uniform(-1, 1, size=(_MV_K,)).astype(bfloat16)
    at = iron.tensor(mat.reshape(-1), dtype=bfloat16, device="npu")
    bt = iron.tensor(vec, dtype=bfloat16, device="npu")
    ct = iron.zeros(_MV_M, dtype=bfloat16, device="npu")

    _mv_design(at, bt, ct)

    expected = (mat.astype(np.float32) @ vec.astype(np.float32)).astype(bfloat16)
    verdict = compare(
        ct.numpy(),
        expected,
        Tolerance.relative(
            0.03, 0.05, max_mismatch_frac=0.02, note="fp32 accumulate, bf16 round"
        ),
    )
    assert verdict, verdict.detail
