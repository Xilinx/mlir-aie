#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: ryzen_ai_npu2, xrt_python_bindings, peano
#
# RUN: %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 \
# RUN:   -std=c++20 -DNDEBUG -c %S/kernel.cc -o kernel.o
# RUN: %run_on_npu2% %python %S/test.py | FileCheck %s
#
# CHECK: PASS!

"""A trace unit that overran is reported, and one that kept up is not.

Every traced worker emits events into one trace stream. A worker that raises
events back to back overruns its trace unit, so the events after the overrun are
missing from the trace. With enable_ctrl_pkts the host reads each trace unit's
Trace_Status register after the run and raises for the units that overran.
"""

import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, Kernel, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent

Out4 = np.ndarray[(4,), np.dtype[np.int32]]

EVENTS = 4000


def build_design(
    _out: Out,
    *,
    trace_config: CompileTime[TraceConfig],
    num_workers: CompileTime[int],
    num_fast: CompileTime[int],
):
    slow = Kernel("emit_events_slow", "kernel.o", arg_types=[np.int32])
    fast = Kernel("emit_events_fast", "kernel.o", arg_types=[np.int32])
    fifo = ObjectFifo(Out4, name="out")

    def core_body(out, emit):
        elem = out.acquire(1)
        emit(EVENTS)
        elem[0] = 1
        out.release(1)

    def emit_body(emit):
        emit(EVENTS)

    workers = [Worker(core_body, [fifo.prod(), slow], trace=1)]
    for i in range(num_workers - 1):
        emit = fast if i < num_fast else slow
        workers.append(Worker(emit_body, [emit], trace=1))

    def sequence(out, out_handle):
        out_handle.drain(out, wait=True)

    runtime = Runtime(sequence, [Out4, fifo.cons()])
    program = Program(iron.get_current_device(), runtime, workers=workers)
    program.enable_trace(
        trace_config.trace_size,
        workers=workers,
        coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
        enable_ctrl_pkts=trace_config.enable_ctrl_pkts,
    )
    return program.resolve_program()


def run(design, num_workers, num_fast):
    """Run the design, returning the message of the overrun it raised, if any."""
    trace_config = TraceConfig(trace_size=8192, enable_ctrl_pkts=True)
    out = iron.zeros(4, dtype=np.int32, device="npu")
    try:
        design(
            out,
            trace_config=trace_config,
            num_workers=num_workers,
            num_fast=num_fast,
        )
    except HostRuntimeError as error:
        return str(error)
    return None


def main():
    errors = []
    design = iron.jit(build_design, object_files=[str(Path.cwd() / "kernel.o")])

    # One tile alone and four sharing the stream, none of them fast
    for num_workers in (1, 4):
        overrun = run(design, num_workers, 0)
        print(f"{num_workers} workers, none fast: {overrun or 'no overrun'}")
        if overrun:
            errors.append(f"{num_workers} quiet workers: {overrun}")

    # One fast worker among four overruns, and only that one is named
    overrun = run(design, 4, 1)
    print(f"4 workers, 1 fast: {overrun or 'no overrun'}")
    if overrun is None:
        errors.append("a worker raising events back to back did not overrun")
    elif overrun.count("core module") != 1:
        errors.append(f"expected one overrun core module: {overrun}")

    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        return 1
    print("PASS!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
