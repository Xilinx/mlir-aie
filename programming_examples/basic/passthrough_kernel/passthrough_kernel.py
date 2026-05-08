# passthrough_kernel/passthrough_kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Passthrough kernel — Iron API + ``@iron.jit`` + ``aie.iron.kernels``.

Same dataflow as the placed variant (one ObjectFifo in, one out, one compute
core calling ``passThroughLine``) but uses Iron's high-level ``Worker``/
``Runtime``/``Program`` builders so placement is automatic, ``@iron.jit`` so
kernel compilation and xclbin generation happen on the first call, and the
``aie.iron.kernels.passthrough`` factory so the C++ kernel wiring (source
path, include dirs, ``-DBIT_WIDTH=8``) does not need to be repeated here.

When ``--trace_size > 0``, a ``TraceConfig`` is passed at call time; the JIT
runtime writes the trace buffer to ``trace.txt``, this script then parses it
to ``trace_passthrough_kernel.json`` and prints the per-tile cycle summary —
matching the placed-flow ``make trace_py`` output.
"""

import argparse
import sys
import time

import numpy as np

import aie.iron as iron
from aie.iron import (
    Compile,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import get_cycles_summary


_TRACE_JSON = "trace_passthrough_kernel.json"


@iron.jit
def my_passthrough_kernel(
    in_tensor: In,
    out_tensor: Out,
    *,
    n: Compile[int],
    trace_config: Compile[TraceConfig | None] = None,
):
    """Passthrough generator specialised on element count.

    Mirrors the placed variant: input is streamed through a depth-2
    ObjectFifo as four sub-tensors of ``n // 4`` elements each so DMA
    transfers overlap with compute on the double-buffered FIFO.

    ``trace_config`` enables hardware tracing when supplied; the runtime
    writes the trace buffer to ``trace_config.trace_file`` after execution.
    """
    in1_dtype = np.uint8
    line_size = n // 4  # chop input in 4 sub-tensors (matches placed)
    line_type = np.ndarray[(line_size,), np.dtype[in1_dtype]]
    vector_type = np.ndarray[(n,), np.dtype[in1_dtype]]

    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    # Kernel-library factory wires source path, include dirs, and
    # -DBIT_WIDTH=8 automatically for the requested dtype.  The kernel
    # operates on one ``line_size``-element tile at a time.
    pass_through_line = kernels.passthrough(tile_size=line_size, dtype=in1_dtype)

    def core_fn(of_in, of_out, pass_through_line):
        for _ in range_(sys.maxsize):
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
    with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
        if trace_config:
            rt.enable_trace(trace_config.trace_size, workers=[worker])
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _print_trace_summary(json_path: str) -> None:
    """Mirror get_trace_summary.py's per-tile summary output."""
    cycles = get_cycles_summary(json_path)
    for entry in cycles:
        print(entry[0])
        runs = len(entry) - 1
        print(f"Total number of full kernel invocations is {runs}")
        if runs > 0:
            samples = entry[1:]
            print(
                "First/Min/Avg/Max cycles is "
                f"{entry[1]}/ {min(samples)}/ "
                f"{sum(samples) / runs}/ {max(samples)}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", default="npu", help="AIE device (npu|npu2)")
    p.add_argument(
        "-i1s",
        "--in1_size",
        type=int,
        default=4096,
        help="Input buffer size in bytes (uint8 elements)",
    )
    p.add_argument(
        "-os",
        "--out_size",
        type=int,
        default=4096,
        help="Output buffer size in bytes (must equal --in1_size)",
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
    out_size = opts.out_size
    trace_size = opts.trace_size

    if in1_size % 64 != 0 or in1_size < 512:
        sys.exit(f"in1_size ({in1_size}) must be a multiple of 64 and >= 512")
    assert out_size == in1_size, "out_size must equal in1_size"

    in1_dtype = np.uint8
    n_elems = in1_size // np.dtype(in1_dtype).itemsize

    # The actual NPU device class is auto-detected by DefaultNPURuntime;
    # opts.dev only selects the device-name string the runtime uses for
    # tensor allocation (matching the original CLI for compatibility).
    in_tensor = iron.tensor(
        np.arange(0, n_elems, dtype=in1_dtype), dtype=in1_dtype, device=opts.dev
    )
    out_tensor = iron.zeros([n_elems], dtype=in1_dtype, device=opts.dev)

    trace_config = TraceConfig(trace_size=trace_size) if trace_size > 0 else None

    # Trace runs do a single invocation: the runtime overwrites trace.txt
    # on every call so timing-loop iterations would only retain the last
    # sample anyway.
    if trace_config is not None:
        warmup, iters = 0, 1
    else:
        warmup, iters = opts.warmup, opts.iters

    npu_time_total = 0.0
    npu_time_min = float("inf")
    npu_time_max = 0.0
    for i in range(warmup + iters):
        start = time.perf_counter()
        my_passthrough_kernel(
            in_tensor, out_tensor, n=n_elems, trace_config=trace_config
        )
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        if i >= warmup:
            npu_time_total += elapsed_us
            npu_time_min = min(npu_time_min, elapsed_us)
            npu_time_max = max(npu_time_max, elapsed_us)

    expected = in_tensor.numpy()
    computed = out_tensor.numpy()
    if not np.array_equal(expected, computed):
        mismatches = int(np.sum(expected != computed))
        print(f"FAIL! {mismatches} mismatches out of {expected.size}")
        sys.exit(1)

    if iters > 0:
        avg_us = npu_time_total / iters
        print(f"\nAvg NPU time: {avg_us:.1f}us.")
        print(f"Min NPU time: {npu_time_min:.1f}us.")
        print(f"Max NPU time: {npu_time_max:.1f}us.")

    if trace_config is not None:
        # Parse trace.txt → JSON and print the per-tile cycle summary,
        # mirroring `parse.py` + `get_trace_summary.py` of the placed flow.
        if trace_config.physical_mlir_path is None:
            sys.exit(
                "trace requested but physical_mlir_path was not set by the JIT "
                "runtime — cannot parse trace events."
            )
        trace_config.trace_to_json(
            trace_config.physical_mlir_path, output_name=_TRACE_JSON
        )
        _print_trace_summary(_TRACE_JSON)

    print("PASS!")


if __name__ == "__main__":
    main()
