# test_kernels_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest -m "not extensive" %s
# RUN: %run_on_npu2_xrt% %pytest -m "not extensive" %s
# RUN: %run_on_npu2_hrx% %pytest -m "not extensive" %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

"""Device tests for the IRON kernel library, driven by ``kernel_cases.CASES``.

``test/python/test_kernel_contracts.py`` proves every contract matches its
factory and lowers to MLIR on the host. This file is the tier that needs a
device: each case runs through ``aie.iron.algorithms.kernel_design`` and is judged
under the tolerance its kernel declares. It catches what types cannot -- a
wrong exported symbol, a wrong compile flag, a DMA-alignment bug, a
reference that disagrees with the C++.

Two tiers share one table (``kernel_cases.py``):

* ``test_kernel`` runs the ``smoke`` cases on random data: one representative
  shape per kernel, on every pull request.
* ``test_kernel_extensive`` (marker ``extensive``, deselected by the RUN
  lines above) runs every case under every edge-data case its contract
  admits, for ``--seeds`` random seeds. The nightly benchmark workflow runs
  it as the correctness gate before anything is timed.

Cases whose kernels exist only for one NPU generation carry
``supported_devices`` (see ``conftest.py``), so they skip elsewhere.
"""

from types import SimpleNamespace

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.controlflow import range_
from aie.iron.kernels._common import _detect_arch
from aie.utils.compile.jit import CompileTime
from aie.utils.verify import Tolerance, compare
from cases import inputs_for
from kernel_cases import CASES
from ml_dtypes import bfloat16


def _param(case):
    marks = [pytest.mark.supported_devices(*case.devices)] if case.devices else []
    return pytest.param(case, marks=marks, id=case.name)


def _run(design, fn, inputs, out_n, out_dt):
    """Upload, run and read back, with the output poisoned first.

    The result is a copy: ``Tensor.numpy()`` views the XRT buffer's mapped
    host memory, and ``out`` is the last reference to that buffer.
    """
    ins, out = kd.upload(inputs, out_n, out_dt, fn=fn, poison=True)
    outputs = out if isinstance(out, tuple) else (out,)
    design(*ins, *outputs)
    got = tuple(o.numpy().copy() for o in outputs)
    return got if len(got) > 1 else got[0]


def _run_case(case, data_case: str, seed: int):
    fn = case.fn()
    inputs = inputs_for(case, data_case, np.random.default_rng(1000 + seed))
    design = kd.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=fn.param_values(inputs),
        **case.kwargs,
    )
    ref = fn.expected(inputs, scalars=case.scalars)
    out_n = kd.output_size(fn, calls=case.calls, shape=case.shape)
    out_dt = fn.output_dtype()
    # The output is poisoned so a kernel that writes nothing cannot pass.
    got = _run(design, fn, inputs, out_n, out_dt)
    verdict = fn.judge(got, ref, calls=case.calls)
    assert verdict, f"{case.name} [{data_case}, seed {seed}]: {verdict.detail}"


@pytest.mark.parametrize("case", [_param(c) for c in CASES if c.smoke])
def test_kernel(case):
    _run_case(case, "random", 0)


def pytest_generate_tests(metafunc):
    if {"case", "data_case", "seed"} <= set(metafunc.fixturenames):
        seeds = metafunc.config.getoption("--seeds")
        params = [
            pytest.param(
                c,
                dc,
                seed,
                marks=[pytest.mark.supported_devices(*c.devices)] if c.devices else [],
                id=f"{c.name}/{dc}/s{seed}",
            )
            for c in CASES
            for dc in c.data_policy()
            for seed in range(seeds if dc == "random" else 1)
        ]
        metafunc.parametrize("case,data_case,seed", params)


@pytest.mark.extensive
def test_kernel_extensive(case, data_case, seed):
    _run_case(case, data_case, seed)


def test_case_names_are_unique():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names)), "two cases share a series name"


def test_run_reads_all_outputs(monkeypatch):
    arrays = (np.arange(4), np.arange(4) + 10)
    outputs = tuple(SimpleNamespace(numpy=lambda a=a: a) for a in arrays)
    inp = object()
    monkeypatch.setattr(kd, "upload", lambda *args, **kwargs: ([inp], outputs))
    calls = []
    got = _run(lambda *args: calls.append(args), None, [], (4, 4), (int, int))
    assert calls == [(inp, *outputs)]
    for actual, expected in zip(got, arrays):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)


# getExpBf16 reaches its tables through a Q8 fixed-point int16, so an input
# outside (-128, 128) used to wrap and fetch an unrelated entry. The two tests
# below pin the clamp that closes that (EXP_BF16_CLAMP in
# aie_runtime_lib/AIE2{,P}/lut_based_ops.h); the data-case sweep never reaches
# far enough past the wrap to catch a regression on its own.


def test_bf16_exp_saturates_outside_lut_domain():
    """Exp saturates rather than wrapping, over the whole real line."""
    fn = kernels.bf16_exp()
    xs = np.array(
        [88, 89, 128, 200, 4e4, np.inf, -88, -128, -200, -4e4, -np.inf],
        dtype=np.float32,
    )
    tile = np.zeros(1024, dtype=np.float32)
    tile[: len(xs)] = xs
    tile_bf16 = tile.astype(bfloat16)

    design = kd.design(kernels.bf16_exp, calls=1)
    got = _run(design, fn, [tile_bf16.reshape(1, 1024)], 1024, np.dtype(bfloat16))
    expected = fn.expected([tile_bf16.reshape(1, 1024)])
    verdict = fn.judge(got, expected, calls=1)
    assert verdict, verdict.detail
    # Padding must not dilute the mismatch budget for the boundary inputs.
    verdict = compare(
        got[: len(xs)], expected.ravel()[: len(xs)], fn.contract.tolerance
    )
    assert verdict, verdict.detail


@pytest.mark.supported_devices("npu2")
def test_bf16_exp_full_clamped_range():
    """Check every lane relatively, including the nonzero subnormal tail."""
    fn = kernels.bf16_exp()
    tile = np.linspace(-88, 88, 1024, dtype=np.float32)
    # Include bf16 neighbours of both ends, the normal/subnormal transition,
    # infinities, and inputs that wrapped the old AIE2 Q8 lookup.
    edges = [
        -np.inf,
        -4e4,
        -200,
        -128,
        -89,
        -88,
        -87.5,
        -87,
        -1,
        -0.5,
        0,
        0.5,
        1,
        87,
        87.5,
        88,
        89,
        128,
        200,
        4e4,
        np.inf,
    ]
    tile[: len(edges)] = edges
    tile_bf16 = tile.astype(bfloat16).reshape(1, 1024)
    design = kd.design(kernels.bf16_exp, calls=1)
    got = _run(design, fn, [tile_bf16], 1024, np.dtype(bfloat16))
    expected = fn.expected([tile_bf16])
    verdict = fn.judge(got, expected, calls=1)
    assert verdict, verdict.detail
    # No absolute floor or mismatch budget: zero at -88 and a single bad
    # exponent are failures, regardless of the other 1023 lanes.
    np.testing.assert_allclose(
        got.astype(np.float64),
        expected.ravel().astype(np.float64),
        rtol=0.01,
        atol=0,
    )


def test_softmax_wide_dynamic_range():
    """A tile whose x - max runs past the LUT domain still normalises.

    Softmax subtracts the per-tile max, so its exp input is <= 0 but not
    bounded below. Before the clamp, Q8 wrapped those deeply-negative values
    onto a large positive entry, which swamped the normalising sum and made
    every element of the tile wrong -- including the peak.
    """
    fn = kernels.softmax()
    tile = np.full(1024, -1000.0, dtype=np.float32)
    tile[0], tile[1] = 0.0, -5.0
    tile_bf16 = tile.astype(bfloat16)

    design = kd.design(kernels.softmax, calls=1)
    got = _run(design, fn, [tile_bf16.reshape(1, 1024)], 1024, np.dtype(bfloat16))
    verdict = fn.judge(got, fn.expected([tile_bf16.reshape(1, 1024)]), calls=1)
    assert verdict, verdict.detail
    # The peak must dominate: a wrapped index used to bury it at ~1e-14.
    assert got.astype(np.float32)[0] > 0.9


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

# init_scale_buffer's buffer argument is mha's per-row scale buffer, which is
# dim_m elements wide, so the probe's ObjectFifo has to be that wide too.
_MHA_DIM = 64
_MHA_TILE = _MHA_DIM


@iron.jit
def _mha_compile_probe(
    a_in: In, b_out: Out, *, size: iron.CompileTime[int] = _MHA_TILE
):
    buf = np.ndarray[(_MHA_TILE,), np.dtype[bfloat16]]
    # kernels.mha compiles mha.cc once and binds its symbols; init_scale_buffer
    # is the simplest of them to instantiate the translation unit with.
    kern = kernels.mha(
        dim_m=_MHA_DIM, dim_k=_MHA_DIM, dim_n=_MHA_DIM
    ).also.init_scale_buffer
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


def _bf16_from_bits(u):
    return (np.asarray(u, np.uint32) << 16).view(np.float32).astype(bfloat16)


def test_setup_reaches_the_core():
    """A contract's ``setup`` must change the core, not just declare an intent.

    A fresh core rounds ``floor``, so every contract judged against a numpy
    reference names ``conv_even`` as its ``setup``. Nothing else in the suite
    separates the two modes: ordinary data rounds the same way under both, so
    a ``setup`` that never ran would still pass. These inputs are exact ties
    -- their product falls precisely halfway between two bf16 values, with an
    odd lower neighbour -- which is the one case where floor and conv_even
    must disagree.
    """
    n = 1024
    rng = np.random.default_rng(0)
    xs, ys = [], []
    while len(xs) < n:
        a = _bf16_from_bits(rng.integers(0x3F00, 0x4000, 4096, dtype=np.uint32))
        b = _bf16_from_bits(rng.integers(0x3F00, 0x4000, 4096, dtype=np.uint32))
        p = (a.astype(np.float32) * b.astype(np.float32)).view(np.uint32)
        tie = ((p & 0xFFFF) == 0x8000) & (((p >> 16) & 1) == 1)
        xs.extend(a[tie].tolist())
        ys.extend(b[tie].tolist())
    x = np.asarray(xs[:n], bfloat16)
    y = np.asarray(ys[:n], bfloat16)

    bits = (x.astype(np.float32) * y.astype(np.float32)).view(np.uint32) >> 16
    floor_result = _bf16_from_bits(bits)
    conv_even_result = _bf16_from_bits(bits + 1)  # ties away from the odd neighbour

    fn = kernels.mul(tile_size=n)
    design = kd.design(kernels.mul, calls=1, tile_size=n)
    ins, out = kd.upload(
        [x.reshape(1, n), y.reshape(1, n)],
        kd.output_size(fn, calls=1),
        bfloat16,
        fn=fn,
        poison=True,
    )
    design(*ins, out)
    got = out.numpy().copy().reshape(-1)[:n].astype(bfloat16)

    assert np.array_equal(got, conv_even_result), (
        f"{int((got == floor_result).sum())} of {n} ties rounded floor: the "
        "contract's setup did not reach the core"
    )


# ---------------------------------------------------------------------------
# Runtime element-count variants (the *_sized factories added in this branch).
# Unlike the fixed-1024 add/mul/silu/gelu/relu, these read the element count at
# runtime, so the design passes it as a trailing arg and may use a non-1024
# tile.  A 1536-element tile (not a multiple of the fixed 1024) is what pins
# that the runtime-size path, including the scalar tail on add/mul, is exercised.
# ---------------------------------------------------------------------------

_SIZED_TILE = 1536

_SIZED_ELT_FACTORY = {"add": kernels.add_sized, "mul": kernels.mul_sized}


@iron.jit
def _sized_eltwise_design(
    a_in: In, b_in: In, c_out: Out, *, which: CompileTime[str] = "add"
):
    kern = _SIZED_ELT_FACTORY[which](tile_size=_SIZED_TILE)
    tile = np.ndarray[(_SIZED_TILE,), np.dtype[bfloat16]]
    of_a = ObjectFifo(tile, name="sea")
    of_b = ObjectFifo(tile, name="seb")
    of_c = ObjectFifo(tile, name="sec")

    def core(of_a, of_b, of_c, k):
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        c = of_c.acquire(1)
        k(a, b, c, _SIZED_TILE)
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
def test_sized_eltwise_e2e(which, op):
    rng = np.random.default_rng(11)
    a = rng.uniform(-2, 2, size=(_SIZED_TILE,)).astype(bfloat16)
    b = rng.uniform(-2, 2, size=(_SIZED_TILE,)).astype(bfloat16)
    at = iron.tensor(a, dtype=bfloat16, device="npu")
    bt = iron.tensor(b, dtype=bfloat16, device="npu")
    ct = iron.zeros(_SIZED_TILE, dtype=bfloat16, device="npu")

    _sized_eltwise_design(at, bt, ct, which=which)

    expected = op(a.astype(np.float32), b.astype(np.float32)).astype(bfloat16)
    verdict = compare(
        ct.numpy(), expected, _SIZED_ELT_FACTORY[which](_SIZED_TILE).contract.tolerance
    )
    assert verdict, verdict.detail


_SIZED_ACT_FACTORY = {
    "silu": (kernels.silu_sized, kernels.silu_ref),
    "gelu": (kernels.gelu_sized, kernels.gelu_ref),
    "relu": (kernels.relu_sized, kernels.relu_ref),
}


@iron.jit
def _sized_activation_design(a_in: In, b_out: Out, *, which: CompileTime[str] = "silu"):
    kern = _SIZED_ACT_FACTORY[which][0](tile_size=_SIZED_TILE)
    tile = np.ndarray[(_SIZED_TILE,), np.dtype[bfloat16]]
    of_in = ObjectFifo(tile, name="sactin")
    of_out = ObjectFifo(tile, name="sactout")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(a, c, _SIZED_TILE)
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [tile, tile, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


@pytest.mark.parametrize("which", ["silu", "gelu", "relu"])
def test_sized_activation_e2e(which):
    ref = _SIZED_ACT_FACTORY[which][1]
    rng = np.random.default_rng(12)
    x = rng.uniform(-4, 4, size=(_SIZED_TILE,)).astype(bfloat16)
    xt = iron.tensor(x, dtype=bfloat16, device="npu")
    yt = iron.zeros(_SIZED_TILE, dtype=bfloat16, device="npu")

    _sized_activation_design(xt, yt, which=which)

    expected = ref(x)
    verdict = compare(
        yt.numpy(),
        expected,
        _SIZED_ACT_FACTORY[which][0](_SIZED_TILE).contract.tolerance,
    )
    assert verdict, verdict.detail


# ---------------------------------------------------------------------------
# Norms: rms_norm / rms_norm_eps / layer_norm.  Each normalizes one bf16 row of
# `cols` elements (gamma=1, beta=0), reading cols at runtime.  A 2048-element
# row is a realistic embedding-dim width and a non-1024 tile.
# ---------------------------------------------------------------------------

_NORM_COLS = 2048


@iron.jit
def _rms_norm_design(a_in: In, b_out: Out, *, use_eps: CompileTime[bool] = False):
    tile = np.ndarray[(_NORM_COLS,), np.dtype[bfloat16]]
    kern = (kernels.rms_norm_eps if use_eps else kernels.rms_norm)(tile_size=_NORM_COLS)
    of_in = ObjectFifo(tile, name="rnin")
    of_out = ObjectFifo(tile, name="rnout")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        if use_eps:
            k(a, c, _NORM_COLS, 1e-5)
        else:
            k(a, c, _NORM_COLS)
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [tile, tile, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


@pytest.mark.parametrize("use_eps", [False, True])
def test_rms_norm_e2e(use_eps):
    rng = np.random.default_rng(13)
    x = rng.uniform(-2, 2, size=(_NORM_COLS,)).astype(bfloat16)
    xt = iron.tensor(x, dtype=bfloat16, device="npu")
    yt = iron.zeros(_NORM_COLS, dtype=bfloat16, device="npu")

    _rms_norm_design(xt, yt, use_eps=use_eps)

    expected = kernels.rms_norm_ref(x, eps=1e-5)
    verdict = compare(
        yt.numpy(), expected, kernels.rms_norm(_NORM_COLS).contract.tolerance
    )
    assert verdict, verdict.detail


@iron.jit
def _layer_norm_design(a_in: In, b_out: Out):
    tile = np.ndarray[(_NORM_COLS,), np.dtype[bfloat16]]
    kern = kernels.layer_norm(tile_size=_NORM_COLS)
    of_in = ObjectFifo(tile, name="lnin")
    of_out = ObjectFifo(tile, name="lnout")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(a, c, _NORM_COLS)
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [tile, tile, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_layer_norm_e2e():
    rng = np.random.default_rng(14)
    x = rng.uniform(-2, 2, size=(_NORM_COLS,)).astype(bfloat16)
    xt = iron.tensor(x, dtype=bfloat16, device="npu")
    yt = iron.zeros(_NORM_COLS, dtype=bfloat16, device="npu")

    _layer_norm_design(xt, yt)

    expected = kernels.layer_norm_ref(x, eps=1e-5)
    verdict = compare(
        yt.numpy(), expected, kernels.layer_norm(_NORM_COLS).contract.tolerance
    )
    assert verdict, verdict.detail


# ---------------------------------------------------------------------------
# RoPE (interleaved / Llama and two-halves / HuggingFace).  Applies a rotation
# per (cos, sin) pair from an interleaved LUT to one bf16 row of `dims`.
# ---------------------------------------------------------------------------

_ROPE_DIMS = 128


@iron.jit
def _rope_design(
    a_in: In,
    lut_in: In,
    b_out: Out,
    *,
    two_halves: CompileTime[bool] = False,
    dims: CompileTime[int] = _ROPE_DIMS,
):
    tile = np.ndarray[(dims,), np.dtype[bfloat16]]
    lut_ty = np.ndarray[(dims,), np.dtype[bfloat16]]
    kern = kernels.rope(tile_size=dims, two_halves=two_halves)
    of_in = ObjectFifo(tile, name="ropein")
    of_lut = ObjectFifo(lut_ty, name="ropelut")
    of_out = ObjectFifo(tile, name="ropeout")

    def core(of_in, of_lut, of_out, k):
        a = of_in.acquire(1)
        lut = of_lut.acquire(1)
        c = of_out.acquire(1)
        k(a, lut, c, dims)
        of_in.release(1)
        of_lut.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_lut.cons(), of_out.prod(), kern])

    def seq(a, lut, b, ih, lh, oh):
        ih.fill(a)
        lh.fill(lut)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [tile, lut_ty, tile, of_in.prod(), of_lut.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


@pytest.mark.parametrize("two_halves", [False, True])
@pytest.mark.parametrize("dims", [96, _ROPE_DIMS])
def test_rope_e2e(two_halves, dims):
    rng = np.random.default_rng(15)
    x = rng.uniform(-2, 2, size=(dims,)).astype(bfloat16)
    # dims/2 (cos, sin) pairs, interleaved.
    theta = rng.uniform(-np.pi, np.pi, size=(dims // 2,)).astype(np.float32)
    lut = np.empty((dims,), dtype=np.float32)
    lut[0::2] = np.cos(theta)
    lut[1::2] = np.sin(theta)
    lut = lut.astype(bfloat16)

    xt = iron.tensor(x, dtype=bfloat16, device="npu")
    lt = iron.tensor(lut, dtype=bfloat16, device="npu")
    yt = iron.zeros(dims, dtype=bfloat16, device="npu")

    _rope_design(xt, lt, yt, two_halves=two_halves, dims=dims)

    expected = kernels.rope_ref(x, lut, two_halves=two_halves)
    verdict = compare(
        yt.numpy(),
        expected,
        Tolerance.relative(0.05, 0.05, max_mismatch_frac=0.05),
    )
    assert verdict, verdict.detail
