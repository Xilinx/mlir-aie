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
  admits, for ``--seeds`` random seeds.

Cases whose kernels exist only for one NPU generation carry
``supported_devices`` (see ``conftest.py``), so they skip elsewhere.
"""

from types import SimpleNamespace

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils.verify import compare
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
    out_n = kd.output_size(fn, calls=case.calls)
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
    # Include bf16 neighbors of both ends, the normal/subnormal transition,
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
    """A tile whose x - max runs past the LUT domain still normalizes.

    Softmax subtracts the per-tile max, so its exp input is <= 0 but not
    bounded below. Before the clamp, Q8 wrapped those deeply-negative values
    onto a large positive entry, which swamped the normalizing sum and made
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


def _bf16_from_bits(u):
    return (np.asarray(u, np.uint32) << 16).view(np.float32).astype(bfloat16)


def test_setup_reaches_the_core():
    """A contract's ``setup`` must change the core, not just declare an intent.

    A fresh core rounds ``floor``, so every contract judged against a numpy
    reference names ``conv_even`` as its ``setup``. Nothing else in the suite
    separates the two modes: ordinary data rounds the same way under both, so
    a ``setup`` that never ran would still pass. These inputs are exact ties
    -- their product falls precisely halfway between two bf16 values, with an
    odd lower neighbor -- which is the one case where floor and conv_even
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
    conv_even_result = _bf16_from_bits(bits + 1)  # ties away from the odd neighbor

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
# cascade_mm: a two-tile design. The PUT half streams A * B onto the cascade
# and the GET half adds its own product and the cascade term into C, so the
# generic one-Worker builder cannot run it; this test builds the pair by hand
# and judges it against the two products.
# ---------------------------------------------------------------------------

_CASCADE_DIM = 16


@iron.jit
def _cascade_design(a_put: In, b_put: In, a_get: In, b_get: In, c_out: Out):
    from aie.iron import CascadeFlow
    from aie.iron.device import Tile

    m = k = n = _CASCADE_DIM
    get = kernels.cascade_mm(dim_m=m, dim_k=k, dim_n=n)
    put = kernels.cascade_mm_put(dim_m=m, dim_k=k, dim_n=n)
    zero = get.contract.initializers[0][1](get)
    a_ty, b_ty, c_ty = get.arg_types()
    fifos = {
        name: ObjectFifo(ty, name=name)
        for name, ty in (("ap", a_ty), ("bp", b_ty), ("ag", a_ty), ("bg", b_ty))
    }
    of_c = ObjectFifo(c_ty, name="c")
    unused = np.zeros(m * n, dtype=np.dtype(kd.shape_dtype(c_ty)[1]))

    def put_core(of_a, of_b, scratch, k_put):
        a, b = of_a.acquire(1), of_b.acquire(1)
        k_put(a, b, scratch)
        of_a.release(1)
        of_b.release(1)

    def get_core(of_a, of_b, of_c, k_zero, k_get):
        a, b, c = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
        k_zero(c)
        k_get(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    from aie.iron.buffer import Buffer

    scratch = Buffer(c_ty, name="scratch", initial_value=unused)
    # The cascade runs north to south: the PUT tile sits above the GET tile.
    w_put = Worker(
        put_core,
        [fifos["ap"].cons(), fifos["bp"].cons(), scratch, put],
        tile=Tile(0, 3),
    )
    w_get = Worker(
        get_core,
        [fifos["ag"].cons(), fifos["bg"].cons(), of_c.prod(), zero, get],
        tile=Tile(0, 2),
    )
    CascadeFlow(w_put, w_get)

    def seq(ap, bp, ag, bg, c, h_ap, h_bp, h_ag, h_bg, h_c):
        for handle, host in ((h_ap, ap), (h_bp, bp), (h_ag, ag), (h_bg, bg)):
            handle.fill(host)
        h_c.drain(c, wait=True)

    rt = Runtime(
        seq,
        [a_ty, b_ty, a_ty, b_ty, c_ty]
        + [fifos[name].prod() for name in ("ap", "bp", "ag", "bg")]
        + [of_c.cons()],
    )
    return Program(
        iron.get_current_device(), rt, workers=[w_put, w_get]
    ).resolve_program()


def test_cascade_mm_pair():
    m = k = n = _CASCADE_DIM
    get = kernels.cascade_mm(dim_m=m, dim_k=k, dim_n=n)
    rng = np.random.default_rng(3)
    limit = get.input_limit(np.int16)
    a1, b1, a2, b2 = (
        rng.integers(-limit, limit, size=(m * k,)).astype(np.int16) for _ in range(4)
    )
    tensors = [iron.tensor(x, dtype=np.int16, device="npu") for x in (a1, b1, a2, b2)]
    c = iron.tensor(np.zeros(m * n, np.int16), dtype=np.int16, device="npu")
    _cascade_design(*tensors, c)
    expected = (
        a1.astype(np.int64).reshape(m, k) @ b1.astype(np.int64).reshape(k, n)
        + a2.astype(np.int64).reshape(m, k) @ b2.astype(np.int64).reshape(k, n)
    ).astype(np.int16)
    verdict = get.judge(c.numpy().copy(), expected.reshape(1, -1), calls=1)
    assert verdict, verdict.detail
