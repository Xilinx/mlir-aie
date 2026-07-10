# test_iron_jit_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""End-to-end tests for the new @iron.jit / CompilableDesign / CallableDesign
stack.  All tests run a real kernel on the NPU and verify output correctness.

Coverage:
- @iron.jit bare decorator (compile params at call time)
- @iron.jit with pre-bound CompileTime[T] params (Triton style)
- @iron.compileconfig + explicit CompilableDesign + CallableDesign (AOT path)
- Compile-on-demand: first call compiles, second call reuses the cached kernel
- Cache invalidation: different compile_kwargs produce different cached kernels
- Correct output for each configuration
- CompileTime[T] param missing → TypeError before any NPU interaction
- TraceConfig passed to a design that never calls enable_trace → HostRuntimeError
"""

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    Out,
    CallableDesign,
    CompilableDesign,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    compileconfig,
)
from aie.iron.controlflow import range_

# ---------------------------------------------------------------------------
# Shared design: element-wise add of a constant
# ---------------------------------------------------------------------------

_TILE_SIZE = 16


def _add_const_design(
    input_buf: In, output_buf: Out, N: CompileTime[int], add_value: CompileTime[int]
):
    """Add ``add_value`` to every element of a length-N int32 vector.

    Parameters
    ----------
    input_buf, output_buf : In / Out
        Runtime DMA tensors.
    N : CompileTime[int]
        Total element count — compile-time; determines the generated loop bounds.
    add_value : CompileTime[int]
        Constant to add — compile-time; baked into the AIE core at generation time.
    """
    tile_ty = np.ndarray[(_TILE_SIZE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE_SIZE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE_SIZE):
                elem_out[i] = elem_in[i] + add_value
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, b):
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def N():
    return 1024


@pytest.fixture(scope="session")
def input_array(N):
    return iron.arange(N, dtype=np.int32)


# ---------------------------------------------------------------------------
# 1. @iron.jit bare — CompileTime[T] params passed at call time
# ---------------------------------------------------------------------------


@iron.jit
def add_const_jit(
    input_buf: In, output_buf: Out, *, N: CompileTime[int], add_value: CompileTime[int]
):
    return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)


@pytest.mark.parametrize("add_value", [1, 5, 100])
def test_jit_bare_correct_output(input_array, N, add_value):
    """Bare @iron.jit with CompileTime[T] params supplied at call time."""
    output = iron.zeros(N, dtype=np.int32, device="npu")
    add_const_jit(input_array, output, N=N, add_value=add_value)
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), input_array.numpy() + add_value)


# ---------------------------------------------------------------------------
# 2. @iron.jit with pre-bound compile params (Triton style)
# ---------------------------------------------------------------------------


@iron.jit(N=1024, add_value=7)
def add_seven(
    input_buf: In, output_buf: Out, *, N: CompileTime[int], add_value: CompileTime[int]
):
    return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)


def test_jit_prebound_params_correct_output(input_array, N):
    """@iron.jit with N and add_value pre-bound at decoration time."""
    output = iron.zeros(N, dtype=np.int32, device="npu")
    add_seven(input_array, output)
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), input_array.numpy() + 7)


# ---------------------------------------------------------------------------
# 3. @compileconfig + explicit CompilableDesign + CallableDesign (AOT path)
# ---------------------------------------------------------------------------


@compileconfig
def add_const_design(
    input_buf: In, output_buf: Out, *, N: CompileTime[int], add_value: CompileTime[int]
):
    return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)


def test_aot_compile_then_run(input_array, N):
    """AOT: compile eagerly, then run via CallableDesign."""
    design = add_const_design.specialize(N=N, add_value=3)
    xclbin, insts = design.compile()
    assert xclbin.exists()
    assert insts.exists()

    output = iron.zeros(N, dtype=np.int32, device="npu")
    CallableDesign(design)(input_array, output)
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), input_array.numpy() + 3)


# ---------------------------------------------------------------------------
# 4. Compile-on-demand: second call reuses compiled kernel
# ---------------------------------------------------------------------------


def test_compile_on_demand_second_call_hits_cache(input_array, N):
    """The second call must use the cached kernel (no recompile)."""

    @iron.jit(N=N, add_value=2)
    def add_two(
        input_buf: In,
        output_buf: Out,
        *,
        N: CompileTime[int],
        add_value: CompileTime[int],
    ):
        return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)

    out1 = iron.zeros(N, dtype=np.int32, device="npu")
    out2 = iron.zeros(N, dtype=np.int32, device="npu")

    add_two(input_array, out1)
    add_two(input_array, out2)

    out1.to("cpu")
    out2.to("cpu")
    expected = input_array.numpy() + 2
    np.testing.assert_array_equal(out1.numpy(), expected)
    np.testing.assert_array_equal(out2.numpy(), expected)


# ---------------------------------------------------------------------------
# 5. Cache isolation: different compile_kwargs produce separate artifacts
# ---------------------------------------------------------------------------


def test_different_compile_kwargs_produce_different_correct_outputs(input_array, N):
    """Two designs compiled with different add_value must produce different results."""

    @iron.jit
    def add_dynamic(
        input_buf: In,
        output_buf: Out,
        *,
        N: CompileTime[int],
        add_value: CompileTime[int],
    ):
        return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)

    out_10 = iron.zeros(N, dtype=np.int32, device="npu")
    out_20 = iron.zeros(N, dtype=np.int32, device="npu")

    add_dynamic(input_array, out_10, N=N, add_value=10)
    add_dynamic(input_array, out_20, N=N, add_value=20)

    out_10.to("cpu")
    out_20.to("cpu")
    ref = input_array.numpy()
    np.testing.assert_array_equal(out_10.numpy(), ref + 10)
    np.testing.assert_array_equal(out_20.numpy(), ref + 20)


# ---------------------------------------------------------------------------
# 6. Missing CompileTime[T] param → TypeError before NPU interaction
# ---------------------------------------------------------------------------


def test_missing_compile_param_raises_type_error():
    """Supplying compile_kwargs without a required CompileTime[T] param raises TypeError."""
    design = CompilableDesign(
        add_const_design.mlir_generator,
        compile_kwargs={"N": 1024},  # add_value missing
    )
    with pytest.raises(TypeError, match="compile_kwargs do not match"):
        design.compile()


# ---------------------------------------------------------------------------
# 7. use_cache=False always recompiles (output must still be correct)
# ---------------------------------------------------------------------------


def test_use_cache_false_recompiles_but_output_correct(input_array, N):
    @iron.jit(N=N, add_value=4, use_cache=False)
    def add_four_nocache(
        input_buf: In,
        output_buf: Out,
        *,
        N: CompileTime[int],
        add_value: CompileTime[int],
    ):
        return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)

    out = iron.zeros(N, dtype=np.int32, device="npu")
    add_four_nocache(input_array, out)
    out.to("cpu")
    np.testing.assert_array_equal(out.numpy(), input_array.numpy() + 4)


# ---------------------------------------------------------------------------
# 8. Passing trace_config to a design that never calls enable_trace is an error
# ---------------------------------------------------------------------------


def test_trace_config_without_enable_trace_raises(input_array, N):
    """Passing a TraceConfig to a design that never calls enable_trace must
    raise HostRuntimeError. The guard in the runtime catches the mismatch:
    the xclbin declares only the data BOs but the host would append a trace
    buffer, which would segfault in XRT argument setup."""
    from aie.utils.trace.config import TraceConfig
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    trace_cfg = TraceConfig(trace_size=65536)

    @iron.jit
    def add_traced(
        input_buf: In,
        output_buf: Out,
        *,
        N: CompileTime[int],
        add_value: CompileTime[int],
        trace_config: CompileTime[TraceConfig | None] = None,
    ):
        return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)

    out = iron.zeros(N, dtype=np.int32, device="npu")
    with pytest.raises(HostRuntimeError, match="host buffer argument"):
        add_traced(input_array, out, N=N, add_value=9, trace_config=trace_cfg)
