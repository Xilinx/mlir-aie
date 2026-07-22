# test_jit_trace.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np
import os
import aie.iron as iron

from aie.utils import tensor
from aie.utils.trace import TraceConfig, parse_trace
from aie.iron import CompileTime, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_


# Define kernel function
def scale_scalar(of_in, of_out, factor, N):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(N):
        elem_out[i] = elem_in[i]
    of_in.release(1)
    of_out.release(1)


@iron.jit
def design(
    a_in: iron.In,
    c_out: iron.Out,
    *,
    trace_config: CompileTime[TraceConfig | None] = None
):
    N = 1024
    # Construct types for sequence
    a_type = np.ndarray[(1024,), np.dtype[np.int32]]
    c_type = np.ndarray[(1024,), np.dtype[np.int32]]

    # Define ObjectFifos
    of_in = ObjectFifo(a_type, depth=2)
    of_out = ObjectFifo(c_type, depth=2)

    # Define Worker
    worker = Worker(scale_scalar, fn_args=[of_in.cons(), of_out.prod(), 2, N])

    def sequence(a, c, in_h, out_h):
        # In runtime sequence:
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [a_type, c_type],
        fn_args=[of_in.prod(), of_out.cons()],
    )
    if trace_config:
        rt.enable_trace(trace_config.trace_size, workers=[worker])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@pytest.mark.parametrize("trace_size", [8192])
def test_jit_trace(trace_size):
    N = 1024
    ref = np.arange(N, dtype=np.int32)
    a = tensor(ref, dtype=np.int32)
    c = tensor(np.zeros(N, dtype=np.int32), dtype=np.int32)

    trace_config = TraceConfig(trace_size=trace_size)

    # Run JIT kernel with tracing
    design(a, c, trace_config=trace_config)

    # Sync output from device
    c.to("cpu")

    # Verify results
    assert np.array_equal(c.numpy(), ref)

    # Verify trace file exists
    assert os.path.exists(trace_config.trace_file)

    # Parse trace using physical MLIR (contains lowered npu_write32 ops)
    assert (
        trace_config.physical_mlir_path is not None
    ), "physical_mlir_path not set - trace parsing requires lowered MLIR"
    assert os.path.exists(trace_config.physical_mlir_path)

    with open(trace_config.physical_mlir_path, "r") as f:
        physical_mlir_str = f.read()

    trace_buffer = trace_config.read_trace()
    trace_events = parse_trace(trace_buffer, physical_mlir_str)

    assert len(trace_events) > 0
