# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from aie.iron import (
    Buffer,
    CompileTime,
    Configuration,
    Kernel,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.device import NPU2Col1
from aie.utils.trace.events import CoreEvent

Out4 = np.ndarray[(4,), np.dtype[np.int32]]
Out24 = np.ndarray[(24,), np.dtype[np.int32]]
Rtp = np.ndarray[(1,), np.dtype[np.int32]]


def make_event_configuration(name, event, counts, trace_size):
    fifo = ObjectFifo(Out4, name=f"out_{name}")
    out_handle = fifo.cons()
    rtp = Buffer(Rtp, name=f"rtp_{name}", use_write_rtp=True)
    barrier = WorkerRuntimeBarrier()
    emit_events = Kernel(
        f"emit_events_{event}",
        "kernel.o",
        arg_types=[np.int32],
    )

    def core_body(out, count, ready, emit):
        ready.wait_for_value(1, greater_equal=True)
        value = count[0]
        elem = out.acquire(1)
        emit(value)
        elem[0] = value
        out.release(1)

    worker = Worker(
        core_body,
        [fifo.prod(), rtp, barrier, emit_events],
        trace=1,
    )

    def make_runtime(index, count):
        def sequence(out, out_handle):
            rtp[0] = count
            barrier.set(1)
            out_handle.drain(out, wait=True)

        return Runtime(
            sequence,
            [Out4, out_handle],
            name=f"seq_{name}{index}",
        )

    runtimes = [
        make_runtime(index, count) for index, count in enumerate(counts, start=1)
    ]
    configuration = Configuration(
        f"dev_{name}",
        NPU2Col1(),
        workers=[worker],
        runtimes=runtimes,
    )
    configuration.enable_trace(
        trace_size=trace_size,
        workers=[worker],
        coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
    )
    return configuration, runtimes


def build_design(_out: Out, *, trace_size: CompileTime[int]):
    slice_size = trace_size // 6
    dev_a, (seq_a1, seq_a2) = make_event_configuration(
        "a", 0, (7000, 9000), slice_size
    )
    dev_b, (seq_b1, seq_b2) = make_event_configuration(
        "b", 1, (8000, 10000), slice_size
    )
    calls = [
        (dev_a, seq_a1),
        (dev_b, seq_b1),
        (dev_a, seq_a2),
        (dev_b, seq_b2),
        (dev_a, seq_a2),
        (dev_b, seq_b2),
    ]

    def coordinator(out):
        for index, (configuration, runtime) in enumerate(calls):
            with configuration.configure():
                runtime.call(out.window(index * 4, (4,)))

    entry = Runtime(coordinator, [Out24])
    main = Configuration("main", NPU2Col1(), runtimes=[entry])
    return Program.compose([main, dev_a, dev_b], entry=entry).resolve_program()
