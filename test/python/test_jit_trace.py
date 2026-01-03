# test_jit_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np
import aie.iron as iron
from aie.utils.jit import jit
from aie.utils import tensor
from aie.utils.trace import TraceConfig
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


# Define kernel function
def scale_scalar(of_in, of_out, factor, N):
    for i in range(N):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        elem_out[0] = factor * elem_in[0]
        of_in.release(1)
        of_out.release(1)


@jit(is_placed=False)
def design(a_in, c_out, trace_config=None):
    N = 1024
    # Construct types for sequence
    a_type = np.ndarray[a_in.shape, np.dtype(a_in.dtype)]
    c_type = np.ndarray[c_out.shape, np.dtype(c_out.dtype)]

    # Define ObjectFifos
    of_in = ObjectFifo(a_type, depth=2)
    of_out = ObjectFifo(c_type, depth=2)

    # Define Worker
    worker = Worker(scale_scalar, fn_args=[of_in.cons(), of_out.prod(), 2, N])

    rt = Runtime()
    with rt.sequence(a_type, c_type) as (a, c):
        if trace_config:
            rt.enable_trace(trace_config.trace_size)

        # In runtime sequence:
        rt.fill(of_in.prod(), a)
        rt.start(worker)
        rt.drain(of_out.cons(), c)

        if trace_config:
            rt.enable_trace(trace_config.trace_size, workers=[worker])

    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


@pytest.mark.parametrize("trace_size", [8192])
def test_jit_trace(trace_size):
    N = 1024
    a = tensor(np.arange(N, dtype=np.int32), dtype=np.int32)
    c = tensor(np.zeros(N, dtype=np.int32), dtype=np.int32)

    trace_config = TraceConfig(trace_size=trace_size, trace_after_last_tensor=True)

    # Run JIT kernel with tracing
    design(a, c, trace_config=trace_config)

    # Verify results
    expected = a * 2
    assert np.array_equal(c, expected)

    # Verify trace file exists?
    # trace_config.trace_file_name
    import os

    assert os.path.exists(trace_config.trace_file_name)
