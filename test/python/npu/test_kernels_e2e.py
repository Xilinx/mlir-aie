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

A pass says the kernel is within tolerance, not how close it is.
``--report-error PATH`` records, for every run of either tier, pass or fail,
its error against the contract's reference run on float64 inputs
(``cases.error_report``): ulps, results not correctly rounded, max abs/rel.

Cases whose kernels exist only for one NPU generation carry
``supported_devices`` (see ``conftest.py``), so they skip elsewhere.
"""

import pathlib
from types import SimpleNamespace

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import (
    CompileTime,
    In,
    InOut,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.algorithms import kernel_design as kd
from aie.utils.verify import compare, poisoned
from cases import error_report, inputs_for
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


def _run_case(case, data_case: str, seed: int, report_error=None):
    fn = case.fn()
    inputs = inputs_for(case, data_case, np.random.default_rng(1000 + seed))
    design = kd.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=fn.param_values(inputs),
        guard=True,
        **case.kwargs,
    )
    ref = fn.expected(inputs, scalars=case.scalars)
    out_n = kd.output_size(fn, calls=case.calls, guard=True)
    out_dt = fn.output_dtype()
    # The output is poisoned so a kernel that writes nothing cannot pass.
    got, overrun = kd.strip_guard(
        fn, _run(design, fn, inputs, out_n, out_dt), calls=case.calls
    )
    assert not any(np.atleast_1d(overrun)), (
        f"{case.name} [{data_case}, seed {seed}]: changed {overrun} guard byte(s) past its output "
        f"({kd.GUARD_BYTES} after each tile)"
    )
    verdict = fn.judge(got, ref, calls=case.calls, inputs=inputs, scalars=case.scalars)
    if report_error:
        entries = error_report(fn, got, inputs, calls=case.calls, scalars=case.scalars)
        report_error(f"{case.name}/{data_case}/s{seed}", entries, bool(verdict))
    assert verdict, f"{case.name} [{data_case}, seed {seed}]: {verdict.detail}"


@pytest.mark.parametrize("case", [_param(c) for c in CASES if c.smoke])
def test_kernel(case, report_error):
    _run_case(case, "random", 0, report_error)


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
def test_kernel_extensive(case, data_case, seed, report_error):
    _run_case(case, data_case, seed, report_error)


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


# AIE2's getTanhBf16 multiplied its flat end segments' slope of 0 by x, so
# +-inf gave NaN, and every kernel built on it followed: gelu's tanh argument,
# x * (c + d * x^2), overflows to inf from |x| = 2.13e13. Silu multiplied -inf
# by its sigmoid of 0, swiglu an overflowed x * w1 by a silu of 0, and the
# matmul epilogue's silu split an f32 x past bf16's range into inf and -inf.
_HUGE = np.array([8, 9, 1e4, 2e13, 2.2e13, 1e20, 1e30, 3e38, np.inf], np.float32)
_HUGE = np.concatenate([_HUGE, -_HUGE])


def _huge_tile(dtype, values=_HUGE):
    tile = np.zeros(1024, dtype=np.float32)
    tile[: len(values)] = values
    return tile.astype(dtype).reshape(1, 1024)


def _relu(x):
    return np.where(x > 0, x, 0)


def _step(x):
    return np.where(x > 0, 1, 0)


def _assert_gelu_saturates(got, tile):
    edge = tile.ravel()[: len(_HUGE)].astype(bfloat16).astype(np.float32)
    np.testing.assert_array_equal(got[: len(_HUGE)].astype(np.float32), _relu(edge))


@pytest.mark.supported_devices("npu1")
@pytest.mark.parametrize(
    "factory,limit",
    [
        ("tanh", np.sign),
        ("sigmoid", _step),
        ("silu", _relu),
        ("silu_sized", _relu),
        ("gelu", _relu),
        ("gelu_sized", _relu),
    ],
)
def test_activation_saturates_for_huge_inputs(factory, limit):
    """Each activation takes its limit for large and infinite inputs."""
    fn = getattr(kernels, factory)()
    tile = _huge_tile(bfloat16)
    design = kd.design(getattr(kernels, factory), calls=1)
    got = _run(design, fn, [tile], 1024, np.dtype(bfloat16))
    edge = tile.ravel()[: len(_HUGE)].astype(np.float32)
    np.testing.assert_array_equal(got[: len(_HUGE)].astype(np.float32), limit(edge))


@pytest.mark.supported_devices("npu1")
def test_swiglu_zero_gate_hides_overflow():
    """An x * w1 overflowing to inf, times a silu of exactly 0, gives 0, not NaN."""
    fn = kernels.swiglu()
    x = _huge_tile(bfloat16, [1e20, 1e20, 1e20, 2, 1, 1])
    w1 = _huge_tile(bfloat16, [1e20, 1e20, -1e20, np.inf, 1, 1])
    w2 = _huge_tile(bfloat16, [-1e20, 1e20, 1e20, -8, np.inf, -np.inf])
    design = kd.design(kernels.swiglu, calls=1)
    got = _run(design, fn, [x, w1, w2], 1024, np.dtype(bfloat16))
    np.testing.assert_array_equal(
        got[:6].astype(np.float32), [0, np.inf, -np.inf, 0, np.inf, 0]
    )
    verdict = fn.judge(got, fn.expected([x, w1, w2]), calls=1)
    assert verdict, verdict.detail


@pytest.mark.supported_devices("npu1")
def test_epilogue_gelu_saturates_for_huge_inputs():
    """The matmul epilogue's gelu, which narrows its f32 input to bf16 first."""
    fn = kernels.mm_activation_epilogue()
    tile = _huge_tile(np.float32)
    design = kd.design(kernels.mm_activation_epilogue, calls=1, scalars=(2,))
    got = _run(design, fn, [tile], 1024, np.dtype(np.float32))
    _assert_gelu_saturates(got, tile)
    verdict = fn.judge(got, fn.expected([tile], scalars=(2,)), calls=1)
    assert verdict, verdict.detail


@pytest.mark.supported_devices("npu1")
def test_epilogue_silu_saturates_for_huge_inputs():
    """The epilogue's silu keeps f32 inputs past bf16's range finite."""
    fn = kernels.mm_activation_epilogue()
    finite = np.concatenate([_HUGE[np.isfinite(_HUGE)], [3.4e38, -3.4e38]])
    tile = _huge_tile(np.float32, finite)
    design = kd.design(kernels.mm_activation_epilogue, calls=1, scalars=(1,))
    got = _run(design, fn, [tile], 1024, np.dtype(np.float32))
    np.testing.assert_allclose(got[: len(finite)], _relu(finite), rtol=2**-16)
    verdict = fn.judge(got, fn.expected([tile], scalars=(1,)), calls=1)
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


def _skips_first_vector():
    tile = np.ndarray[(64,), np.dtype[np.int32]]
    return iron.ExternalFunction(
        "skips_first_vector",
        source_string="""extern "C" {
void skips_first_vector(int *in, int *out) {
  for (int i = 16; i < 64; i++)
    out[i] = in[i];
}
}""",
        arg_types=[tile, tile],
        contract=kernels.KernelContract(roles=(In, Out), reference=lambda x: x),
    )


def test_guard_poisons_the_core_tile():
    """An output element the kernel never writes reads back as poison.

    Poisoning the host buffer alone is not enough: the DMA overwrites it with
    the core's tile, which held zeros, so a kernel that skipped a vector
    passed every data case whose reference was zero there.
    """
    calls = 3
    fn = _skips_first_vector()
    x = np.arange(calls * 64, dtype=np.int32).reshape(calls, 64) + 1
    design = kd.design(_skips_first_vector, calls=calls, guard=True)
    out_n = kd.output_size(fn, calls=calls, guard=True)
    got, overrun = kd.strip_guard(
        fn, _run(design, fn, [x], out_n, np.int32), calls=calls
    )
    assert not any(np.atleast_1d(overrun))
    got = got.reshape(calls, 64)
    np.testing.assert_array_equal(got[:, 16:], x[:, 16:])
    skipped = got[:, :16]
    assert (skipped == poisoned(1, np.int32)[0]).all(), (
        f"the elements the kernel skipped read back as {np.unique(skipped)}, "
        "not the poison"
    )


def _adds_param(vector_loads=False):
    # The scalar loop reads w wherever it sits; aie::load_v assumes a
    # 64-byte aligned w and rounds a misaligned address down.
    name = f"adds_param_{'vector' if vector_loads else 'scalar'}"
    body = (
        "aie::store_v(out + i, aie::add(aie::load_v<16>(in + i), "
        "aie::load_v<16>(w + i)));"
        if vector_loads
        else "for (int j = i; j < i + 16; j++) out[j] = in[j] + w[j];"
    )
    tile = np.ndarray[(64,), np.dtype[np.int32]]
    return iron.ExternalFunction(
        name,
        source_string=f"""#include <aie_api/aie.hpp>
extern "C" void {name}(int *in, int *w, int *out) {{
  for (int i = 0; i < 64; i += 16) {{
    {body}
  }}
}}""",
        arg_types=[tile, tile, tile],
        contract=kernels.KernelContract(
            roles=(In, kernels.Param, Out), reference=lambda x, w: x + w
        ),
    )


def _run_adds_param(vector_loads, offset):
    calls = 2
    fn = _adds_param(vector_loads)
    x = np.arange(calls * 64, dtype=np.int32).reshape(calls, 64)
    w = np.arange(64, dtype=np.int32) * 1000 + 7
    design = kd.design(
        _adds_param,
        calls=calls,
        params=[w],
        arg_byte_offsets=((1, offset),),
        vector_loads=vector_loads,
    )
    got = _run(design, fn, [x, w], kd.output_size(fn, calls=calls), np.int32)
    return got.reshape(calls, 64), x + w


@pytest.mark.parametrize("offset", [4, 16])
def test_a_param_at_a_byte_offset_is_read_there(offset):
    got, ref = _run_adds_param(False, offset)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.supported_devices("npu2")
def test_a_vector_load_from_a_param_at_a_byte_offset_is_caught():
    """The misaligned-weights bug: 16 bytes past a 64-byte boundary, load_v reads the wrong words.

    mobilenet packs several layers' weights into one buffer, and a view 16
    bytes past alignment passed every aligned harness case.
    """
    got, ref = _run_adds_param(True, 0)
    np.testing.assert_array_equal(got, ref)
    got, ref = _run_adds_param(True, 16)
    assert (got != ref).any(), "a load_v from a misaligned param read it right"


# ---------------------------------------------------------------------------
# cascade_mm: a two-tile design. The PUT half streams A * B onto the cascade
# and the GET half adds its own product and the cascade term into C, so the
# generic one-Worker builder cannot run it; this test builds the chain by hand
# (PUT, then PUT_GET tiles, then GET) and judges it against the products.
# ---------------------------------------------------------------------------

_CASCADE_DTYPES = {
    "i16_i16": (np.int16, np.int16),
    "i16_i32": (np.int16, np.int32),
    "bf16_bf16": (bfloat16, bfloat16),
    "bf16_f32": (bfloat16, np.float32),
}
# Each tile runs its kernel twice on the same operands, so the GET half's
# second call must add into the C its first call wrote.
_CASCADE_CALLS = 2


@iron.jit
def _cascade_design(
    *tensors: InOut,
    combo: CompileTime[str],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    tiles: CompileTime[int],
):
    from aie.iron import CascadeFlow
    from aie.iron.buffer import Buffer
    from aie.iron.device import Tile

    in_dt, out_dt = _CASCADE_DTYPES[combo]
    kw = dict(dim_m=m, dim_k=k, dim_n=n, input_dtype=in_dt, output_dtype=out_dt)
    get = kernels.cascade_mm(**kw)
    put = kernels.cascade_mm_put(**kw)
    zero = get.contract.initializers[0][1](get)
    a_ty, b_ty, c_ty = get.arg_types()
    # Single-buffered, so 64x64 with a 32-bit C fits in a core tile's memory.
    of_a = [ObjectFifo(a_ty, name=f"a{i}", depth=1) for i in range(tiles)]
    of_b = [ObjectFifo(b_ty, name=f"b{i}", depth=1) for i in range(tiles)]
    of_c = ObjectFifo(c_ty, name="c", depth=1)
    unused = np.zeros(m * n, dtype=np.dtype(out_dt))

    def put_core(of_a, of_b, scratch, k_put):
        a, b = of_a.acquire(1), of_b.acquire(1)
        for _ in range(_CASCADE_CALLS):
            k_put(a, b, scratch)
        of_a.release(1)
        of_b.release(1)

    def get_core(of_a, of_b, of_c, k_zero, k_get):
        a, b, c = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
        k_zero(c)
        for _ in range(_CASCADE_CALLS):
            k_get(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    # The cascade runs north to south: tile i sits above tile i + 1.
    workers = []
    for i in range(tiles):
        ins = [of_a[i].cons(), of_b[i].cons()]
        if i == tiles - 1:
            fn, args = get_core, ins + [of_c.prod(), zero, get]
        else:
            scratch = Buffer(c_ty, name=f"scratch{i}", initial_value=unused)
            fn, args = put_core, ins + [scratch, put if i == 0 else get.put_get]
        workers.append(Worker(fn, args, tile=Tile(0, 1 + tiles - i)))
    for up, down in zip(workers, workers[1:]):
        CascadeFlow(up, down)

    def seq(*args):
        hosts, handles = args[: 2 * tiles + 1], args[2 * tiles + 1 :]
        for handle, host in zip(handles[:-1], hosts[:-1]):
            handle.fill(host)
        handles[-1].drain(hosts[-1], wait=True)

    producers = [f.prod() for pair in zip(of_a, of_b) for f in pair]
    rt = Runtime(seq, [a_ty, b_ty] * tiles + [c_ty] + producers + [of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _cascade_device_marks(combo):
    # The bf16 chains have only been run on npu2.
    if combo.startswith("bf16"):
        return [pytest.mark.supported_devices("npu2")]
    return []


def _cascade_cases():
    for combo in _CASCADE_DTYPES:
        for dim in (16, 32, 64):
            for tiles in (2, 3):
                smoke = (combo, dim, tiles) in (
                    ("i16_i16", 16, 2),
                    ("bf16_f32", 32, 3),
                )
                marks = [] if smoke else [pytest.mark.extensive]
                yield pytest.param(
                    combo,
                    (dim, dim, dim),
                    tiles,
                    False,
                    marks=marks + _cascade_device_marks(combo),
                    id=f"{combo}-{dim}-{tiles}",
                )
    extensive = [pytest.mark.extensive]
    for combo in _CASCADE_DTYPES:
        # 24 does not tile, so AIE2P falls back to the scalar kernel.
        yield pytest.param(
            combo,
            (24,) * 3,
            3,
            False,
            marks=extensive + _cascade_device_marks(combo),
            id=f"{combo}-24-3",
        )
    for combo in ("i16_i16", "i16_i32"):
        # K = 24 tiles, but in K steps of 8 rather than 16.
        yield pytest.param(
            combo, (16, 24, 16), 3, False, marks=extensive, id=f"{combo}-16x24x16-3"
        )
        yield pytest.param(
            combo, (64,) * 3, 3, True, marks=extensive, id=f"{combo}-64-3-full_range"
        )


@pytest.mark.parametrize("combo,shape,tiles,full_range", list(_cascade_cases()))
def test_cascade_mm_chain(combo, shape, tiles, full_range):
    in_dt, out_dt = _CASCADE_DTYPES[combo]
    m, k, n = shape
    get = kernels.cascade_mm(
        dim_m=m, dim_k=k, dim_n=n, input_dtype=in_dt, output_dtype=out_dt
    )
    rng = np.random.default_rng(3)
    sizes = [m * k, k * n] * tiles
    if in_dt is bfloat16:
        xs = [rng.standard_normal(s).astype(bfloat16) for s in sizes]
    else:
        # Full range overflows the output type: C wraps like the scalar kernel.
        limit = 32767 if full_range else get.input_limit(np.int16)
        xs = [rng.integers(-limit, limit, size=s).astype(np.int16) for s in sizes]
    tensors = [iron.tensor(x, dtype=in_dt, device="npu") for x in xs]
    c = iron.tensor(np.zeros(m * n, out_dt), dtype=out_dt, device="npu")
    _cascade_design(*tensors, c, combo=combo, m=m, k=k, n=n, tiles=tiles)
    got = c.numpy().copy()

    wide = np.float64 if in_dt is bfloat16 else np.int64
    a_s = [x.astype(wide).reshape(m, k) for x in xs[0::2]]
    b_s = [x.astype(wide).reshape(k, n) for x in xs[1::2]]
    exact = _CASCADE_CALLS * sum(a @ b for a, b in zip(a_s, b_s))
    if in_dt is not bfloat16:
        expected = exact.astype(out_dt).reshape(-1)
        bad = int((got != expected).sum())
        assert bad == 0, f"{bad} of {m * n} outputs differ"
        return
    # Every bf16 product is exact in fp32, so an fp32 sum of n terms, in any
    # order, is within n * 2**-24 * sum|a*b| of the exact result. A bf16 C is
    # rounded once a call, half a bf16 ulp each: 2**-8 of the final value
    # covers both roundings of two calls.
    terms = sum(abs(a) @ abs(b) for a, b in zip(a_s, b_s))
    n_terms = _CASCADE_CALLS * (tiles * k + 1)
    bound = n_terms * 2.0**-24 * _CASCADE_CALLS * terms
    if out_dt is bfloat16:
        bound = bound + 2.0**-8 * np.abs(exact)
    err = np.abs(got.astype(np.float64).reshape(m, n) - exact)
    assert (err <= bound).all(), (
        f"{int((err > bound).sum())} of {m * n} outputs outside the fp32 "
        f"summation bound; max |err| {err.max():.4g}"
    )


_LUT_PAIR_SRC = """#include <aie_api/aie.hpp>
#include <stdint.h>
__attribute__((section(".aie.bank1"), aligned(32))) int16 tbl_ab[512];
__attribute__((section(".aie.bank2"), aligned(32))) int16 tbl_cd[512];
extern "C" void lut_pair(uint8_t *out) {
  using lut_t = aie::lut<4, bfloat16, bfloat16>;
  lut_t l(256, (bfloat16 *)tbl_ab, (bfloat16 *)tbl_cd);
  aie::parallel_lookup<uint16, lut_t, aie::lut_oor_policy::truncate> lk(l, 0);
  aie::vector<int16, 16> idx = aie::load_v<16>((int16 *)out);
  *(v16bfloat16 *)out = lk.fetch(idx.cast_to<uint16>());
}
"""


def _lut_pair_object(tmp_path, aiecc_flags):
    """Compile a design whose kernel builds an aie::lut, returning its object."""
    from aie.iron import ExternalFunction
    from aie.utils.config import cxx_header_path

    src = tmp_path / f"lut_pair_{len(aiecc_flags)}.cc"
    src.write_text(_LUT_PAIR_SRC)

    @iron.jit(aiecc_flags=aiecc_flags)
    def design(out_tensor: Out):
        ty = np.ndarray[(64,), np.dtype[np.uint8]]
        of_out = ObjectFifo(ty, name="lpo")
        kern = ExternalFunction(
            "lut_pair",
            source_file=str(src),
            arg_types=[ty],
            include_dirs=[cxx_header_path()],
        )

        def core(of_out, k):
            e = of_out.acquire(1)
            k(e)
            of_out.release(1)

        # data_size bounds the unpinned region, which otherwise takes the
        # largest free run on the tile -- here, a bank the tables are pinned to.
        w = Worker(
            core,
            fn_args=[of_out.prod(), kern],
            while_true=False,
            stack_size=2048,
            data_size=4096,
        )

        def seq(c, cons):
            cons.drain(c, wait=True)

        rt = Runtime(seq, [ty, of_out.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    design.compile()
    # The cache root is resolved at import, so read back where the compile
    # actually put the kernel rather than trying to redirect it.
    kernel_dir = design.compilable._kernel_dir
    objects = list(pathlib.Path(kernel_dir).glob("lut_pair*.o"))
    assert objects, f"kernel object not found in {kernel_dir}"
    return objects[0]


def _carries_bitcode(obj, tmp_path):
    import os
    import subprocess

    from aie.utils import config

    ret = subprocess.run(
        [
            config.objcopy_path(),
            f"--dump-section=.llvmbc={tmp_path / 'kernel.bc'}",
            str(obj),
            os.devnull,
        ],
        capture_output=True,
    )
    return ret.returncode == 0


@pytest.mark.supported_devices("npu2")
def test_check_lut_banks_is_one_switch(tmp_path):
    """Asking aiecc for the LUT check is what makes the kernel keep its IR.

    The check reads bitcode that only the kernel compile can preserve, so the
    two are derived from one flag and cannot disagree. Preserving it is not
    free, which is why it stays off until asked for.
    """
    from aie.iron.kernels._common import _detect_arch

    if _detect_arch() != "aie2p":
        pytest.skip("the LUT gather this reads is aie2p")
    assert _carries_bitcode(_lut_pair_object(tmp_path, ["--check-lut-banks"]), tmp_path)
    assert not _carries_bitcode(_lut_pair_object(tmp_path, []), tmp_path)
