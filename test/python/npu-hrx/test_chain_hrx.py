# test_chain_hrx.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% env NPU_RUNTIME=hrx %pytest %s
# REQUIRES: hrx_python_bindings

"""HRX multi-dispatch / chain (runlist) backend test.

Exercises ``HRXHostRuntime.run_chain`` -- several dispatches recorded into one
HRX command buffer and submitted as a single batched ``ERT_CMD_CHAIN`` with an
execution + memory barrier between them, so a later run observes an earlier
run's device writes (producer -> consumer). This is an HRX-only capability (the
XRT Python runtime has no ``run_chain``), so it lives here as an explicit HRX
backend test rather than in the shared programming examples.

The design under test is a plain IRON ObjectFifo ``out = in + 1`` kernel built
through the normal ``@compileconfig`` path -- nothing in it is HRX-specific. Only
the batched dispatch (``run_chain`` via ``aie.utils.DefaultNPURuntime``, which is
the HRX runtime under ``NPU_RUNTIME=hrx``) is backend-specific, which is exactly
what this test is here to cover.
"""

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    compileconfig,
)
from aie.iron.controlflow import range_
from aie.utils.npukernel import NPUKernel

_TILE = 16
_SIZE = 1024


def _add_one_design(input_buf: In, output_buf: Out, N: CompileTime[int]):
    """Add 1 to every element of a length-N int32 vector."""
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE):
                elem_out[i] = elem_in[i] + 1
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, b):
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@compileconfig
def add_one(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _add_one_design(input_buf, output_buf, N=N)


def _hrx_runtime():
    """The default NPU runtime, which is the HRX runtime under NPU_RUNTIME=hrx.

    The RUN line forces ``NPU_RUNTIME=hrx`` (and the ``hrx_python_bindings``
    REQUIRES gate keeps this off non-HRX hosts), so ``DefaultNPURuntime`` here is
    the ``CachedHRXRuntime`` that provides the ``run_chain`` under test.
    """
    import aie.utils as u

    rt = u.DefaultNPURuntime
    assert rt is not None, "No default NPU runtime (is NPU_RUNTIME=hrx set?)"
    return rt


@pytest.fixture(scope="module")
def hrx_kernel():
    """Build the add-one design once and load it as an NPUKernel for the chain."""
    xclbin, insts = add_one.specialize(N=_SIZE).compile()
    return NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE")


def test_chain_producer_consumer(hrx_kernel):
    """run0: out0 = in + 1 ; run1: out1 = out0 + 1 -- in one batched submit.

    If chaining/ordering were broken, run1 would read out0 before run0 wrote it
    and out1 would be wrong, so out1 == in + 2 proves the in-chain dependency.
    """
    rt = _hrx_runtime()
    handle = rt.load(hrx_kernel)

    base = np.arange(1, _SIZE + 1, dtype=np.int32)
    in_a = iron.tensor(base, dtype=np.int32, device="npu")
    out0 = iron.zeros(_SIZE, dtype=np.int32, device="npu")
    out1 = iron.zeros(_SIZE, dtype=np.int32, device="npu")

    rt.run_chain([(handle, [in_a, out0]), (handle, [out0, out1])])

    out0.to("cpu")
    out1.to("cpu")
    np.testing.assert_array_equal(out0.numpy(), base + 1)
    np.testing.assert_array_equal(out1.numpy(), base + 2)


def test_deep_chain(hrx_kernel):
    """Depth-8 chain threaded through distinct buffers in one submit.

    Stresses the inter-dispatch barrier across many dispatches in a single
    ERT_CMD_CHAIN: stage k must equal in + (k + 1).
    """
    rt = _hrx_runtime()
    handle = rt.load(hrx_kernel)

    base = np.arange(1, _SIZE + 1, dtype=np.int32)
    in_a = iron.tensor(base, dtype=np.int32, device="npu")

    depth = 8
    stages = [iron.zeros(_SIZE, dtype=np.int32, device="npu") for _ in range(depth)]
    chain = [(handle, [in_a, stages[0]])]
    for k in range(1, depth):
        chain.append((handle, [stages[k - 1], stages[k]]))

    rt.run_chain(chain)

    for k, st in enumerate(stages):
        st.to("cpu")
        np.testing.assert_array_equal(st.numpy(), base + (k + 1))
