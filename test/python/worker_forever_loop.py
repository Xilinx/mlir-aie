# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# A Worker that runs until reconfiguration wraps its body in an scf.while whose
# condition is always true, so the loop keeps no counter. A single-shot Worker
# runs its body once, in an scf.for with bound 1. A body that does some work
# once and then loops forever writes the forever loop itself with forever_.

import numpy as np

from aie.iron import Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.controlflow import forever_
from aie.iron.device import NPU2Col1


def worker_module(while_true, then_forever=False):
    barrier = WorkerRuntimeBarrier()

    def task(barrier):
        barrier.wait_for_value(1)
        if then_forever:
            for _ in forever_():
                pass

    worker = Worker(task, fn_args=[barrier], while_true=while_true)

    def sequence(x):
        barrier.set(1)

    rt = Runtime(sequence, [np.ndarray[(16,), np.dtype[np.int32]]])
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


# CHECK-LABEL: forever
# CHECK:         aie.core
# CHECK-NOT:       scf.for
# CHECK:           scf.while : () -> () {
# CHECK:             %[[TRUE:.*]] = arith.constant true
# CHECK:             scf.condition(%[[TRUE]])
# CHECK:           } do {
# CHECK:             aie.use_lock(@{{.*}}, Acquire, %{{.*}})
# CHECK:             scf.yield
# CHECK:           aie.end
print("forever")
print(worker_module(True))

# CHECK-LABEL: once
# CHECK:         aie.core
# CHECK-NOT:       scf.while
# CHECK:           scf.for %{{.*}} = %c0 to %c1 step %c1
# CHECK:             aie.use_lock(@{{.*}}, Acquire, %{{.*}})
# CHECK:           aie.end
print("once")
print(worker_module(False))

# CHECK-LABEL: once_then_forever
# CHECK:         aie.core
# CHECK:           scf.for %{{.*}} = %c0 to %c1 step %c1
# CHECK:             aie.use_lock(@{{.*}}, Acquire, %{{.*}})
# CHECK:             scf.while : () -> () {
# CHECK:               %[[TRUE:.*]] = arith.constant true
# CHECK:               scf.condition(%[[TRUE]])
# CHECK:             } do {
# CHECK-NEXT:          scf.yield
# CHECK:           aie.end
print("once_then_forever")
print(worker_module(False, then_forever=True))
