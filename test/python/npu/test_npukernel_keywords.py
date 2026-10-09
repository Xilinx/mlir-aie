# test_npukernel_keywords.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# RUN: %run_on_npu_hsa% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings || hsa_npu
"""How a kernel call routes its keywords: a declared dispatch scalar, the
runtime's ``retry`` option, and nothing else, which is rejected before
anything runs."""

import subprocess
import sys

import aie.iron as iron
import numpy as np
import pytest
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import DispatchTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup
from aie.iron import Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.npukernel import NPUKernel

TILE_SIZE = 256
MAX_TILES = 8
tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int32]]
max_ty = np.ndarray[(MAX_TILES * TILE_SIZE,), np.dtype[np.int32]]


def copy_tiles(count=None):
    """Copy the first `count` tiles of `a` into `b`: a dispatch-time scalar, or
    every tile when None."""
    of_in = ObjectFifo(tile_ty, name="of_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="of_out", depth=2)

    def core_fn(in_cons, out_prod):
        elem_in = in_cons.acquire(1)
        elem_out = out_prod.acquire(1)
        for i in range_(TILE_SIZE):
            elem_out[i] = elem_in[i]
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def seq(a_h, b_h, *rest):
        *scalar, in_prod, out_cons = rest
        tiles = TensorAccessPattern.full((MAX_TILES * TILE_SIZE,)).split(0, TILE_SIZE)
        for tile in range_(scalar[0] if scalar else MAX_TILES):
            tg = TaskGroup()
            out_cons.drain(b_h, tap=tiles[tile], wait=True, group=tg)
            in_prod.fill(a_h, tap=tiles[tile], group=tg)
            tg.finish()

    scalars = [] if count is None else [count]
    rt = Runtime(seq, [max_ty, max_ty, *scalars, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def copy_all(a: In, b: Out):
    return copy_tiles()


@iron.jit
def copy_n(a: In, b: Out, *, n_tiles: DispatchTime[np.int32] = 3):
    return copy_tiles(n_tiles)


@iron.jit
def copy_retry(a: In, b: Out, *, retry: DispatchTime[np.int32] = 3):
    return copy_tiles(retry)


def source(seed):
    rng = np.random.default_rng(seed)
    values = rng.integers(1, 2**16, size=(MAX_TILES * TILE_SIZE,), dtype=np.int32)
    return iron.tensor(values, dtype=np.int32, device="npu")


def output():
    return iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")


def assert_copied(a, b, count):
    expected = np.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32)
    expected[: count * TILE_SIZE] = a.numpy()[: count * TILE_SIZE]
    assert np.array_equal(b.numpy(), expected)


@pytest.fixture(scope="module")
def kernels():
    """Each design's NPUKernel, built and run once at its default."""
    built = {}
    for name, design, count in (
        ("static", copy_all, MAX_TILES),
        ("n_tiles", copy_n, 3),
        ("retry", copy_retry, 3),
    ):
        a, b = source(0), output()
        design(a, b)
        assert_copied(a, b, count)
        built[name] = next(iter(design._kernel_cache.values()))
    return built


def test_dispatch_signature_cannot_be_mutated(kernels):
    built = kernels["n_tiles"]
    names = ["n_tiles"]
    kernel = NPUKernel(
        built.xclbin_path,
        dispatch_params=names,
        dispatch_lib_path=built.dispatch_lib_path,
    )
    names.append("extra")
    kernel.dispatch_params.clear()
    three = kernel._generate_dispatch_insts({"n_tiles": 3})
    assert three.nbytes < kernel._generate_dispatch_insts({"n_tiles": 6}).nbytes
    kernel.dispatch_params.append("extra")
    with pytest.raises(HostRuntimeError, match="dispatch scalar mismatch"):
        kernel._generate_dispatch_insts({"n_tiles": 3, "extra": 4})
    assert kernel.dispatch_params == ["n_tiles"]


@pytest.mark.parametrize("name", ["static", "n_tiles"])
@pytest.mark.parametrize("unknown", ["n_tile", "rety", "dispatch_scalars"])
def test_unknown_keyword_rejected_before_runtime(kernels, name, unknown):
    scalars = {"n_tiles": 3} if name == "n_tiles" else {}
    a, b = source(1), output()
    with pytest.raises(TypeError, match=f"unexpected keyword.*'{unknown}'"):
        kernels[name](a, b, **scalars, **{unknown: 6})
    assert not b.numpy().any()


def test_unknown_keyword_does_not_initialize_runtime():
    script = (
        "import aie.utils as utils\n"
        "from aie.utils.npukernel import NPUKernel\n"
        "try:\n"
        "    NPUKernel(dispatch_params=['n_tiles'])(n_tiles=3, n_tile=6)\n"
        "except TypeError:\n"
        "    print('rejected', utils._DefaultNPURuntime is None)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "rejected True"


@pytest.mark.parametrize("name", ["static", "n_tiles", "retry"])
@pytest.mark.parametrize("retry", [None, False, True])
def test_valid_keywords_forwarded(kernels, name, retry):
    # A dispatch scalar named retry takes the keyword from the load option.
    scalars = {} if name == "static" else {name: 5}
    options = {} if retry is None or name == "retry" else {"retry": retry}
    a, b = source(2), output()
    kernels[name](a, b, **scalars, **options)
    assert_copied(a, b, MAX_TILES if name == "static" else 5)


@pytest.mark.parametrize("specialized", [False, True])
def test_jit_default_does_not_hide_unknown_keyword(kernels, specialized):
    design = copy_n.specialize(n_tiles=3) if specialized else copy_n
    a, b = source(3), output()
    with pytest.raises(TypeError, match="unexpected keyword.*'n_tile'"):
        design(a, b, n_tile=6)
    assert not b.numpy().any()


@pytest.mark.parametrize("kwargs,expected", [({}, 3), ({"n_tiles": 6}, 6)])
def test_jit_dispatch_default_and_override(kernels, kwargs, expected):
    a, b = source(4), output()
    copy_n(a, b, retry=False, **kwargs)
    assert_copied(a, b, expected)
