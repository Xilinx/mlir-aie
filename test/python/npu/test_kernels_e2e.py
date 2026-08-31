# test_kernels_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

"""Starter end-to-end tests for the IRON kernel library factories.

``test/python/test_kernels.py`` is purely declarative: it pins each factory's
returned ``ExternalFunction`` (name, arg types, shapes, error paths) without
ever invoking ``aiecc``.  That catches API regressions but cannot catch
mismatches between the factory's metadata and the actual C++ kernel — e.g.
wrong exported function name, wrong compile flag, or DMA-alignment issues
like the bfloat16 ``reduce_max`` output that this test pins.

These tests compile + run each covered factory and verify output.
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_

# ---------------------------------------------------------------------------
# kernels.passthrough — already exercised by passthrough_kernel.py / 00_memcpy.
# This test pins the int32 path as the canonical "factory works end to end" check.
# ---------------------------------------------------------------------------


def test_passthrough_int32_e2e():
    """Compile + run kernels.passthrough(dtype=int32) and verify output == input."""
    LINE = 1024
    N = 4 * LINE  # 4 LINE-sized chunks

    @iron.jit
    def passthrough_design(x: In, y: Out, *, n: CompileTime[int]):
        line_ty = np.ndarray[(LINE,), np.dtype[np.int32]]
        vec_ty = np.ndarray[(n,), np.dtype[np.int32]]

        of_in = ObjectFifo(line_ty, name="in")
        of_out = ObjectFifo(line_ty, name="out")
        passthrough_fn = kernels.passthrough(tile_size=LINE, dtype=np.int32)

        def core(of_in, of_out, fn):
            for _ in range_(n // LINE):
                ein = of_in.acquire(1)
                eout = of_out.acquire(1)
                fn(ein, eout, LINE)
                of_in.release(1)
                of_out.release(1)

        worker = Worker(core, [of_in.cons(), of_out.prod(), passthrough_fn])

        def sequence(a, b, in_h, out_h):
            in_h.fill(a)
            out_h.drain(b, wait=True)

        rt = Runtime(
            sequence,
            [vec_ty, vec_ty, of_in.prod(), of_out.cons()],
        )

        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    x = iron.arange(N, dtype=np.int32, device="npu")
    y = iron.zeros(N, dtype=np.int32, device="npu")
    passthrough_design(x, y, n=N)
    np.testing.assert_array_equal(y.numpy(), x.numpy())


# ---------------------------------------------------------------------------
# kernels.reduce_max(dtype=bfloat16) — pins the alignment fix.
#
# The library's bf16 path produces an out_size of 2 elements (4 bytes) so the
# output tile satisfies the NPU's 4-byte shim-DMA alignment.  Before the fix,
# this code path was unreachable from the kernel library — callers had to
# build an ExternalFunction by hand (see the prior 02_vector_reduce_max).
# ---------------------------------------------------------------------------


def test_reduce_max_bfloat16_output_alignment_e2e():
    """kernels.reduce_max(dtype=bfloat16) must compile, run, and write the right value."""
    TILE = 1024

    @iron.jit
    def reduce_max_design(x: In, y: Out, *, n: CompileTime[int]):
        in_ty = np.ndarray[(n,), np.dtype[bfloat16]]
        tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]

        # Library-decided output shape; verifies the alignment fix is active.
        reduce_fn = kernels.reduce_max(tile_size=TILE, dtype=bfloat16)
        out_ty = reduce_fn.arg_types()[1]
        assert out_ty.__args__[0] == (
            2,
        ), "bfloat16 reduce_max output must be padded to 2 elements for DMA alignment"

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(out_ty, name="out")

        def core(of_in, of_out, fn):
            for _ in range_(n // TILE):
                ein = of_in.acquire(1)
                eout = of_out.acquire(1)
                fn(ein, eout, TILE)
                of_in.release(1)
                of_out.release(1)

        worker = Worker(core, [of_in.cons(), of_out.prod(), reduce_fn])

        def sequence(a, b, in_h, out_h):
            in_h.fill(a)
            out_h.drain(b, wait=True)

        rt = Runtime(
            sequence,
            [in_ty, out_ty, of_in.prod(), of_out.cons()],
        )

        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    x = iron.arange(TILE, dtype=bfloat16, device="npu")
    y = iron.zeros(2, dtype=bfloat16, device="npu")
    reduce_max_design(x, y, n=TILE)
    # Last call's per-tile reduction is the max of the final tile; for arange
    # input that's the largest element.
    expected = bfloat16(TILE - 1)
    assert y[0] == expected, f"reduce_max wrote {y[0]} but max of input was {expected}"


# ===========================================================================
# The following kernels need a bespoke hand-built design (their
# signatures don't fit the generic ``transform_parallel`` helper):
#
# * ``axpy``       — two inputs (x, y) + a runtime scalar ``a`` before the output.
# * ``transpose``  — a 2D blocked shuffle; verified against ``numpy`` block-wise T.
# * ``expand``     — int4 -> bf16 dequant with per-group scale factors packed in
#                    the input tile.
# * ``leaky_relu`` — one input + a trailing ``alpha`` (bf16) slope scalar.
#
# Correctness is checked against a numpy reference with a bf16-appropriate
# tolerance (the kernels accumulate in fp32 then round to bf16, so exact
# equality does not hold).
# ===========================================================================


def _bf16_close(actual, expected, rtol=0.03, atol=0.05):
    """Fraction of elements outside bf16 tolerance (kernels round fp32->bf16)."""
    a = actual.astype(np.float32)
    e = expected.astype(np.float32)
    return float(np.mean(np.abs(a - e) > (rtol * np.abs(e) + atol)))


# ---------------------------------------------------------------------------
# axpy:  z = a * x + y   (bf16 x/y/z, fp32 scalar a)
# ---------------------------------------------------------------------------

_AXPY_TILE = 1024


@iron.jit
def _axpy_design(
    x_in: In,
    y_in: In,
    z_out: Out,
    *,
    size: CompileTime[int] = 4096,
    a: CompileTime[float] = 2.0,
):
    kern = kernels.axpy(tile_size=_AXPY_TILE, vectorized=True)
    tile = np.ndarray[(_AXPY_TILE,), np.dtype[bfloat16]]
    of_x = ObjectFifo(tile, name="ax")
    of_y = ObjectFifo(tile, name="ay")
    of_z = ObjectFifo(tile, name="az")

    def core(of_x, of_y, of_z, k):
        for _ in range_(size // _AXPY_TILE):
            xv = of_x.acquire(1)
            yv = of_y.acquire(1)
            zv = of_z.acquire(1)
            k(xv, yv, a, zv, _AXPY_TILE)
            of_x.release(1)
            of_y.release(1)
            of_z.release(1)

    w = Worker(core, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), kern])
    vec = np.ndarray[(size,), np.dtype[bfloat16]]

    def seq(x, y, z, xh, yh, zh):
        xh.fill(x)
        yh.fill(y)
        zh.drain(z, wait=True)

    rt = Runtime(seq, [vec, vec, vec, of_x.prod(), of_y.prod(), of_z.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_axpy_e2e():
    # NOTE: ``a`` must be integer-valued here.  The kernel takes a C ``float``
    # scalar, but @iron.jit's CompileTime scalar-constant plumbing currently
    # emits an *integer* MLIR constant for the arg, truncating e.g. 2.5 -> 2.
    # The kernel math itself is correct (verified for a in {2.0, 3.0}); a
    # fractional slope needs the framework's float-scalar transport, which is a
    # separate framework gap.
    size, a = 4096, 3.0
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(size,)).astype(bfloat16)
    y = rng.uniform(-2, 2, size=(size,)).astype(bfloat16)
    xt = iron.tensor(x, dtype=bfloat16, device="npu")
    yt = iron.tensor(y, dtype=bfloat16, device="npu")
    zt = iron.zeros(size, dtype=bfloat16, device="npu")

    _axpy_design(xt, yt, zt, size=size, a=a)

    expected = (a * x.astype(np.float32) + y.astype(np.float32)).astype(bfloat16)
    frac = _bf16_close(zt.numpy(), expected)
    assert frac < 0.02, f"axpy mismatch fraction {frac:.4f}"


# ---------------------------------------------------------------------------
# transpose:  blockwise bf16 transpose of a DIM_n x DIM_m matrix
# ---------------------------------------------------------------------------

_T_M = 32  # inner (contiguous) dim
_T_N = 32  # outer dim


@iron.jit
def _transpose_design(
    a_in: In,
    b_out: Out,
    *,
    subtile: CompileTime[int] = 4,
):
    kern = kernels.transpose(dim_m=_T_M, dim_n=_T_N, subtile=subtile)
    tile = np.ndarray[(_T_M * _T_N,), np.dtype[bfloat16]]
    of_in = ObjectFifo(tile, name="tin")
    of_out = ObjectFifo(tile, name="tout")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(a, c)
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [tile, tile, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def _blockwise_transpose_ref(mat, subtile):
    """Reference: transpose each subtile x subtile block in place.

    The kernel transposes the sub-blocks of the matrix, not the whole matrix,
    so the reference does the same: for each block, swap block[i, j] <-> [j, i].
    """
    n, m = mat.shape
    out = mat.copy()
    for r in range(0, n, subtile):
        for c in range(0, m, subtile):
            out[r : r + subtile, c : c + subtile] = mat[
                r : r + subtile, c : c + subtile
            ].T
    return out


@pytest.mark.parametrize("subtile", [4, 8])
def test_transpose_e2e(subtile):
    rng = np.random.default_rng(1)
    mat = rng.uniform(-4, 4, size=(_T_N, _T_M)).astype(bfloat16)
    at = iron.tensor(mat.reshape(-1), dtype=bfloat16, device="npu")
    bt = iron.zeros(_T_M * _T_N, dtype=bfloat16, device="npu")

    _transpose_design(at, bt, subtile=subtile)

    got = bt.numpy().reshape(_T_N, _T_M)
    expected = _blockwise_transpose_ref(mat, subtile)
    frac = _bf16_close(got, expected, rtol=0.0, atol=0.0)  # transpose is lossless
    assert frac < 0.001, f"transpose(subtile={subtile}) mismatch fraction {frac:.4f}"


# ---------------------------------------------------------------------------
# expand:  int4 -> bf16 dequant with per-group bf16 scale factors
# ---------------------------------------------------------------------------

_EX_TILE = 1024
_EX_GROUP = 32


@iron.jit
def _expand_design(a_in: In, c_out: Out):
    kern = kernels.expand(tile_size=_EX_TILE, group_size=_EX_GROUP)
    # Input tile: _EX_TILE int4s packed as _EX_TILE//2 uint8, followed by
    # (_EX_TILE//_EX_GROUP) bf16 scale factors.  Model the whole payload as a
    # raw uint8 buffer sized to hold both regions.
    n_scales = _EX_TILE // _EX_GROUP
    in_bytes = _EX_TILE // 2 + n_scales * 2  # int4 payload + bf16 scales
    in_ty = np.ndarray[(in_bytes,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(_EX_TILE,), np.dtype[bfloat16]]
    of_in = ObjectFifo(in_ty, name="ein")
    of_out = ObjectFifo(out_ty, name="eout")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(a, c)
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])

    def seq(a, c, ih, oh):
        ih.fill(a)
        oh.drain(c, wait=True)

    rt = Runtime(seq, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_expand_e2e():
    n_scales = _EX_TILE // _EX_GROUP
    rng = np.random.default_rng(2)
    # int4 values 0..15, two packed per byte (low nibble = element 2i).
    nibbles = rng.integers(0, 16, size=(_EX_TILE,), dtype=np.uint8)
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)
    scales = rng.uniform(0.1, 1.0, size=(n_scales,)).astype(bfloat16)
    payload = np.concatenate([packed, scales.view(np.uint8)])
    at = iron.tensor(payload, dtype=np.uint8, device="npu")
    ct = iron.zeros(_EX_TILE, dtype=bfloat16, device="npu")

    _expand_design(at, ct)

    # Reference: each element = nibble * scale-of-its-group.
    grp = np.repeat(scales.astype(np.float32), _EX_GROUP)
    expected = (nibbles.astype(np.float32) * grp).astype(bfloat16)
    frac = _bf16_close(ct.numpy(), expected)
    assert frac < 0.02, f"expand mismatch fraction {frac:.4f}"


# ---------------------------------------------------------------------------
# leaky_relu:  f(x) = max(x, alpha * x)   (bf16, trailing bf16 alpha scalar)
# ---------------------------------------------------------------------------

_LRELU_TILE = 1024


@iron.jit
def _leaky_relu_design(
    a_in: In,
    b_out: Out,
    *,
    size: CompileTime[int] = 4096,
    alpha: CompileTime[float] = 0.1,
):
    kern = kernels.leaky_relu(tile_size=_LRELU_TILE)
    tile = np.ndarray[(_LRELU_TILE,), np.dtype[bfloat16]]
    of_in = ObjectFifo(tile, name="lin")
    of_out = ObjectFifo(tile, name="lout")

    def core(of_in, of_out, k):
        for _ in range_(size // _LRELU_TILE):
            a = of_in.acquire(1)
            c = of_out.acquire(1)
            # Pass alpha as a plain Python float: the call builder wraps int/float
            # literals into an MLIR constant of the kernel's declared arg type
            # (bf16 here).  A raw ml_dtypes.bfloat16 value is neither int nor
            # float and would reach the call unwrapped.
            k(a, c, _LRELU_TILE, alpha)
            of_in.release(1)
            of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])
    vec = np.ndarray[(size,), np.dtype[bfloat16]]

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [vec, vec, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_leaky_relu_e2e():
    # alpha=0.5 is exactly representable in bf16, so no rounding masks a
    # transport bug; the design casts it to the kernel's bf16 arg.
    size = 4096
    alpha = 0.5
    rng = np.random.default_rng(3)
    x = rng.uniform(-4, 4, size=(size,)).astype(bfloat16)
    xt = iron.tensor(x, dtype=bfloat16, device="npu")
    yt = iron.zeros(size, dtype=bfloat16, device="npu")

    _leaky_relu_design(xt, yt, size=size, alpha=alpha)

    expected = kernels.leaky_relu_ref(x, alpha=alpha)
    frac = _bf16_close(yt.numpy(), expected)
    assert frac < 0.02, f"leaky_relu mismatch fraction {frac:.4f}"


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
def _mha_compile_probe(a_in: In, b_out: Out, *, size: CompileTime[int] = _MHA_TILE):
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _include_dirs, _kernel_source

    src = _kernel_source("aie2p", "aie2p", "mha.cc")
    buf = np.ndarray[(_MHA_TILE,), np.dtype[bfloat16]]
    kern = ExternalFunction(
        "init_scale_buffer",
        source_file=str(src),
        arg_types=[buf, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=["-DDIM_M=64", "-DDIM_K=64", "-DDIM_N=64"],
    )
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
    from aie.iron.kernels._common import _detect_arch

    if _detect_arch() != "aie2p":
        pytest.skip("mha.cc is aie2p-only")
    a = iron.tensor(np.zeros(_MHA_TILE, dtype=bfloat16), dtype=bfloat16, device="npu")
    b = iron.zeros(_MHA_TILE, dtype=bfloat16, device="npu")
    _mha_compile_probe(a, b, size=_MHA_TILE)
    # Reaching here means aiecc compiled mha.cc (with its softmax.cc/mm.cc
    # includes) and the design ran — the partial_softmax_* symbols resolved.


# ---------------------------------------------------------------------------
# tanh / sigmoid:  LUT-approximated bf16 activations, signature (in, out, size).
# The kernels approximate via a lookup table, so verify against the numpy _ref
# with the wide bf16-LUT tolerance the refs document (rtol=0.128).
# ---------------------------------------------------------------------------

_ACT_TILE = 1024
_ACT_FACTORY = {"tanh": kernels.tanh, "sigmoid": kernels.sigmoid}


@iron.jit
def _activation_design(a_in: In, b_out: Out, *, which: CompileTime[str] = "tanh"):
    kern = _ACT_FACTORY[which](tile_size=_ACT_TILE)
    tile = np.ndarray[(_ACT_TILE,), np.dtype[bfloat16]]
    of_in = ObjectFifo(tile, name="actin")
    of_out = ObjectFifo(tile, name="actout")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(a, c, _ACT_TILE)
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [tile, tile, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


@pytest.mark.parametrize(
    "which, ref", [("tanh", kernels.tanh_ref), ("sigmoid", kernels.sigmoid_ref)]
)
def test_activation_lut_e2e(which, ref):
    rng = np.random.default_rng(4)
    x = rng.uniform(-4, 4, size=(_ACT_TILE,)).astype(bfloat16)
    xt = iron.tensor(x, dtype=bfloat16, device="npu")
    yt = iron.zeros(_ACT_TILE, dtype=bfloat16, device="npu")

    _activation_design(xt, yt, which=which)

    expected = ref(x)
    frac = _bf16_close(yt.numpy(), expected, rtol=0.128, atol=0.05)
    assert frac < 0.02, f"{which} mismatch fraction {frac:.4f}"


# ---------------------------------------------------------------------------
# convert_copy:  f32 -> bf16 narrowing cast (conv_even rounding), sig (in, out, size).
# ---------------------------------------------------------------------------

_CVT_TILE = 1024


@iron.jit
def _convert_copy_design(a_in: In, b_out: Out):
    kern = kernels.convert_copy(tile_size=_CVT_TILE)
    in_ty = np.ndarray[(_CVT_TILE,), np.dtype[np.float32]]
    out_ty = np.ndarray[(_CVT_TILE,), np.dtype[bfloat16]]
    of_in = ObjectFifo(in_ty, name="cvtin")
    of_out = ObjectFifo(out_ty, name="cvtout")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(a, c, _CVT_TILE)
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_convert_copy_e2e():
    from aie.iron.kernels._common import _detect_arch

    if _detect_arch() != "aie2p":
        pytest.skip("convert_copy binds aie2p/cast_f32_bf16.cc")
    rng = np.random.default_rng(5)
    x = rng.uniform(-8, 8, size=(_CVT_TILE,)).astype(np.float32)
    xt = iron.tensor(x, dtype=np.float32, device="npu")
    yt = iron.zeros(_CVT_TILE, dtype=bfloat16, device="npu")

    _convert_copy_design(xt, yt)

    # The kernel rounds f32->bf16 with conv_even (round-half-to-even), which is
    # the same rounding ml_dtypes' bf16 cast uses — so equality is bit-for-bit.
    expected = x.astype(bfloat16)
    np.testing.assert_array_equal(yt.numpy(), expected)


# ---------------------------------------------------------------------------
# aie2p add / mul:  the aie2p-specific eltwise .cc variants added in this PR.
# On aie2p the add()/mul() factories resolve aie_kernels/aie2p/{add,mul}.cc via
# _detect_arch(); this pins that the new 512-bit variants run correctly there.
# The factories hard-code a 1024-element tile with signature (a, b, c) — no size.
# ---------------------------------------------------------------------------

_ELT_TILE = 1024
_ELT_FACTORY = {"add": kernels.add, "mul": kernels.mul}


@iron.jit
def _eltwise_design(a_in: In, b_in: In, c_out: Out, *, which: CompileTime[str] = "add"):
    kern = _ELT_FACTORY[which](tile_size=_ELT_TILE)
    tile = np.ndarray[(_ELT_TILE,), np.dtype[bfloat16]]
    of_a = ObjectFifo(tile, name="ea")
    of_b = ObjectFifo(tile, name="eb")
    of_c = ObjectFifo(tile, name="ec")

    def core(of_a, of_b, of_c, k):
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        c = of_c.acquire(1)
        k(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    w = Worker(core, fn_args=[of_a.cons(), of_b.cons(), of_c.prod(), kern])

    def seq(a, b, c, ah, bh, ch):
        ah.fill(a)
        bh.fill(b)
        ch.drain(c, wait=True)

    rt = Runtime(seq, [tile, tile, tile, of_a.prod(), of_b.prod(), of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


@pytest.mark.parametrize("which, op", [("add", np.add), ("mul", np.multiply)])
def test_aie2p_eltwise_e2e(which, op):
    from aie.iron.kernels._common import _detect_arch

    if _detect_arch() != "aie2p":
        pytest.skip("exercises the aie2p-specific {add,mul}.cc variants")
    rng = np.random.default_rng(6)
    a = rng.uniform(-2, 2, size=(_ELT_TILE,)).astype(bfloat16)
    b = rng.uniform(-2, 2, size=(_ELT_TILE,)).astype(bfloat16)
    at = iron.tensor(a, dtype=bfloat16, device="npu")
    bt = iron.tensor(b, dtype=bfloat16, device="npu")
    ct = iron.zeros(_ELT_TILE, dtype=bfloat16, device="npu")

    _eltwise_design(at, bt, ct, which=which)

    expected = op(a.astype(np.float32), b.astype(np.float32)).astype(bfloat16)
    frac = _bf16_close(ct.numpy(), expected)
    assert frac < 0.02, f"aie2p {which} mismatch fraction {frac:.4f}"


# ---------------------------------------------------------------------------
# generic/mv.cc:  bf16 matrix-vector multiply (c = A @ b).
#
# No kernels.* factory exposes this yet (kernels.mv resolves aie2/mv.cc, an
# i16->i32 kernel), so the ExternalFunction is hand-built directly against the
# source — same approach as the mha probe above.  The entry is
# matvec_vectorized_bf16_bf16(m, row_offset,
# a, b, c); it needs -DDIM_K and assumes k >= 2*VEC_SIZE (VEC_SIZE=64 -> k>=128).
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
    rng = np.random.default_rng(7)
    mat = rng.uniform(-1, 1, size=(_MV_M, _MV_K)).astype(bfloat16)
    vec = rng.uniform(-1, 1, size=(_MV_K,)).astype(bfloat16)
    at = iron.tensor(mat.reshape(-1), dtype=bfloat16, device="npu")
    bt = iron.tensor(vec, dtype=bfloat16, device="npu")
    ct = iron.zeros(_MV_M, dtype=bfloat16, device="npu")

    _mv_design(at, bt, ct)

    expected = (mat.astype(np.float32) @ vec.astype(np.float32)).astype(bfloat16)
    frac = _bf16_close(ct.numpy(), expected)
    assert frac < 0.02, f"mv matvec mismatch fraction {frac:.4f}"
