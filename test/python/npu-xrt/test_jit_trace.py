# test_jit_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np
import os
import aie.iron as iron
from aie.utils.jit import jit
from aie.utils import tensor
from aie.utils.trace import TraceConfig, parse_trace
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


# Define kernel function
def scale_scalar(of_in, of_out, factor, N):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(N):
        elem_out[i] = elem_in[i]
    of_in.release(1)
    of_out.release(1)


@jit(is_placed=False)
def design(a_in, c_out, trace_config=None):
    N = 1024
    # Construct types for sequence
    a_type = np.ndarray[(1024,), np.dtype[np.int32]]
    c_type = np.ndarray[(1024,), np.dtype[np.int32]]

    # Define ObjectFifos
    of_in = ObjectFifo(a_type, depth=2)
    of_out = ObjectFifo(c_type, depth=2)

    # Define Worker
    worker = Worker(scale_scalar, fn_args=[of_in.cons(), of_out.prod(), 2, N])

    rt = Runtime()
    with rt.sequence(a_type, c_type) as (a, c):
        if trace_config:
            rt.enable_trace(trace_config.trace_size, workers=[worker])

        # In runtime sequence:
        rt.fill(of_in.prod(), a)
        rt.start(worker)
        rt.drain(of_out.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


@pytest.mark.parametrize("trace_size", [8192])
def test_jit_trace(trace_size):
    N = 1024
    ref = np.arange(N, dtype=np.int32)
    a = tensor(ref, dtype=np.int32)
    c = tensor(np.zeros(N, dtype=np.int32), dtype=np.int32)

    trace_config = TraceConfig(trace_size=trace_size, trace_after_last_tensor=False)

    # Run JIT kernel with tracing
    design(a, c, trace_config=trace_config)

    # Sync output from device
    c.to("cpu")

    # Verify results
    assert np.array_equal(c.numpy(), ref)

    # Verify trace file exists
    assert os.path.exists(trace_config.trace_file)

    # Parse trace
    # Get MLIR module from the wrapped function
    mlir_module = design.compilable.function(a, c, trace_config=trace_config)

    trace_buffer = trace_config.read_trace()
    trace_events = parse_trace(trace_buffer, str(mlir_module))

    assert len(trace_events) > 0
