# passthrough_kernel/passthrough_kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Passthrough kernel — IRON API design with ``@iron.jit`` compilation."""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.extras import types as T
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import print_cycles_summary
from aie.utils.verify import assert_pass


@iron.jit
def my_passthrough_kernel(
    in_tensor: In,
    out_tensor: Out,
    *,
    n: CompileTime[int],
    trace_config: CompileTime[TraceConfig | None] = None,
    dynamic_txn: CompileTime[bool] = False,
):
    in1_dtype = np.uint8
    n_lines = 4
    line_size = n // n_lines
    line_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]
    vector_type = np.ndarray[(n,), np.dtype[in1_dtype]]

    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    pass_through_line = kernels.passthrough(tile_size=line_size, dtype=in1_dtype)

    def core_fn(of_in, of_out, pass_through_line):
        # Worker wraps this body in `while True` by default (while_true=True).
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        pass_through_line(elem_in, elem_out, line_size)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), pass_through_line],
        trace=1 if trace_config else 0,
    )

    rt = Runtime()
    if dynamic_txn:
        # Runtime-parameterized transfer length: `buffer_length` is an SSA
        # value of the runtime sequence (not baked in at compile time), so a
        # single compiled design serves any size up to `n`.  Consumed by the
        # aiecc TXN-C++ flow (see passthrough_kernel_dynamic.py); this path
        # emits MLIR for that flow rather than executing on the NPU.
        with rt.sequence(vector_type, vector_type, T.i32) as (
            a_in,
            b_out,
            buffer_length,
        ):
            if trace_config:
                rt.enable_trace(trace_config.trace_size, workers=[worker])
            rt.start(worker)
            rt.fill(of_in.prod(), a_in, sizes=[1, 1, 1, buffer_length])
            rt.drain(of_out.cons(), b_out, sizes=[1, 1, 1, buffer_length], wait=True)
    else:
        with rt.sequence(vector_type, vector_type) as (a_in, b_out):
            if trace_config:
                rt.enable_trace(trace_config.trace_size, workers=[worker])
            rt.start(worker)
            rt.fill(of_in.prod(), a_in)
            rt.drain(of_out.cons(), b_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i1s",
        "--in1_size",
        type=int,
        default=4096,
        help=(
            "Input buffer size in bytes "
            "(equals element count here because the kernel is hardcoded to uint8)"
        ),
    )
    p.add_argument(
        "-t",
        "--trace_size",
        type=int,
        default=0,
        help="Trace buffer size in bytes (0 disables tracing)",
    )
    p.add_argument("-w", "--warmup", type=int, default=10)
    p.add_argument("-n", "--iters", type=int, default=20)
    opts = p.parse_args()

    in1_size = opts.in1_size
    if in1_size % 64 != 0 or in1_size < 512:
        sys.exit(f"in1_size ({in1_size}) must be a multiple of 64 and >= 512")

    in1_dtype = np.uint8
    n_elems = in1_size // np.dtype(in1_dtype).itemsize

    # iron.{arange,zeros_like} target the NPU; the actual NPU generation
    # (NPU1 vs NPU2) is auto-detected by DefaultNPURuntime at JIT time.
    in_tensor = iron.arange(n_elems, dtype=in1_dtype, device="npu")
    out_tensor = iron.zeros_like(in_tensor)

    trace_config = (
        TraceConfig(trace_size=opts.trace_size) if opts.trace_size > 0 else None
    )

    if trace_config is not None:
        # trace.txt is overwritten each call, so only one iteration is meaningful
        warmup, iters = 0, 1
    else:
        warmup, iters = opts.warmup, opts.iters

    bench = run_iters(
        my_passthrough_kernel,
        in_tensor,
        out_tensor,
        n=n_elems,
        trace_config=trace_config,
        warmup=warmup,
        iters=iters,
    )

    assert_pass(
        out_tensor.numpy(),
        in_tensor.numpy(),
        fail_msg="output does not match input",
        print_pass=False,
    )

    print()
    print_benchmark(bench)

    if trace_config is not None:
        if trace_config.physical_mlir_path is None:
            sys.exit(
                "trace requested but physical_mlir_path was not set by the JIT "
                "runtime — cannot parse trace events."
            )
        trace_json = "trace_passthrough_kernel.json"
        trace_config.trace_to_json(
            trace_config.physical_mlir_path, output_name=trace_json
        )
        print_cycles_summary(trace_json)

    print("PASS!")


if __name__ == "__main__":
    main()
