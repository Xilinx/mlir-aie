# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

import numpy as np

from aie import ir
from aie.dialects import aie
from aie.iron import Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU2Col1


def barrier_program() -> Program:
    barrier = WorkerRuntimeBarrier()

    def task(barrier):
        barrier.wait_for_value(1)
        barrier.release_with_value(1)

    def sequence(x):
        barrier.set(1)

    rt = Runtime(sequence, [np.ndarray[(16,), np.dtype[np.int32]]])
    return Program(NPU2Col1(), rt, workers=[Worker(task, fn_args=[barrier])])


def test_programs_resolved_in_one_context_move_into_one_module():
    context = ir.Context()
    first = barrier_program().resolve_program("first", context=context)
    second = barrier_program().resolve_program("second", context=context)
    assert first.context is context and second.context is context

    with context, ir.Location.unknown():
        combined = ir.Module.create()
        for module in (first, second):
            (device,) = [
                op for op in module.body.operations if isinstance(op, aie.DeviceOp)
            ]
            combined.body.append(device)
    assert combined.operation.verify()
    text = str(combined)
    assert "aie.device(npu2_1col) @first" in text
    assert "aie.device(npu2_1col) @second" in text


def test_each_program_has_its_own_context_by_default():
    assert barrier_program().resolve_program().context is not (
        barrier_program().resolve_program().context
    )
