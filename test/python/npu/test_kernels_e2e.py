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
device: each case runs through ``aie.utils.kernel_harness`` and is judged
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

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.kernels._common import _detect_arch
from aie.utils import kernel_harness as kh
from aie.utils.kernel_harness.cases import inputs_for
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
    ins, out = kh.upload(inputs, out_n, out_dt, fn=fn, poison=True)
    design(*ins, out)
    return out.numpy().copy()


def _run_case(case, data_case: str, seed: int):
    fn = case.fn()
    inputs = inputs_for(case, data_case, np.random.default_rng(1000 + seed))
    design = kh.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=fn.param_values(inputs),
        **case.kwargs,
    )
    ref = fn.expected(inputs, scalars=case.scalars)
    out_n = kh.output_size(fn, calls=case.calls, shape=case.shape)
    out_dt = fn.output_dtype(ref.dtype)
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


# getExpBf16 reaches its tables through a Q8 fixed-point int16, so an input
# outside (-128, 128) used to wrap and fetch an unrelated entry. The two tests
# below pin the clamp that closes that (EXP_BF16_CLAMP in
# aie_runtime_lib/AIE2{,P}/lut_based_ops.h); the data-case sweep never reaches
# far enough past the wrap to catch a regression on its own.


def test_bf16_exp_saturates_outside_lut_domain():
    """exp saturates rather than wrapping, over the whole real line."""
    fn = kernels.bf16_exp()
    xs = np.array(
        [88, 89, 128, 200, 4e4, np.inf, -88, -128, -200, -4e4, -np.inf],
        dtype=np.float32,
    )
    tile = np.zeros(1024, dtype=np.float32)
    tile[: len(xs)] = xs
    tile_bf16 = tile.astype(bfloat16)

    design = kh.design(kernels.bf16_exp, calls=1)
    got = _run(design, fn, [tile_bf16.reshape(1, 1024)], 1024, np.dtype(bfloat16))
    verdict = fn.judge(got, fn.expected([tile_bf16.reshape(1, 1024)]), calls=1)
    assert verdict, verdict.detail


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

    design = kh.design(kernels.softmax, calls=1)
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
    design = kh.design(kernels.mul, calls=1, tile_size=n)
    ins, out = kh.upload(
        [x.reshape(1, n), y.reshape(1, n)],
        kh.output_size(fn, calls=1),
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
