# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# A Worker that runs until reconfiguration wraps its body in an scf.while whose
# condition is always true, so the loop keeps no counter. A single-shot Worker
# runs its body once, in an scf.for with bound 1.

import numpy as np

from aie.iron import Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU2Col1

# CHECK-LABEL: aie.device(npu2_1col)
# CHECK:         aie.core
# CHECK-NOT:       scf.for
# CHECK:           scf.while : () -> () {
# CHECK:             %[[TRUE:.*]] = arith.constant true
# CHECK:             scf.condition(%[[TRUE]])
# CHECK:           } do {
# CHECK:             aie.use_lock(%{{.*}}, Acquire, %{{.*}})
# CHECK:             scf.yield
# CHECK:           }
# CHECK:           aie.end
# CHECK:         aie.core
# CHECK-NOT:       scf.while
# CHECK:           scf.for %{{.*}} = %c0 to %c1 step %c1
# CHECK:             aie.use_lock(%{{.*}}, Acquire, %{{.*}})
# CHECK:           aie.end


def worker_loops():
    forever_barrier = WorkerRuntimeBarrier()
    once_barrier = WorkerRuntimeBarrier()

    def task(barrier):
        barrier.wait_for_value(1)

    forever = Worker(task, fn_args=[forever_barrier])
    once = Worker(task, fn_args=[once_barrier], while_true=False)

    def sequence(x):
        forever_barrier.set(1)
        once_barrier.set(1)

    rt = Runtime(sequence, [np.ndarray[(16,), np.dtype[np.int32]]])
    print(Program(NPU2Col1(), rt, workers=[forever, once]).resolve_program())


worker_loops()
