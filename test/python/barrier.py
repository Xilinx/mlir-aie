# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import numpy as np
from aie.iron import (
    Worker,
    WorkerRuntimeBarrier,
    Runtime,
    Program,
)
from aie.iron.device import (
    Tile,
    NPU2Col1,
)

# CHECK: module {
# CHECK:   aie.device(npu2_1col) {
# CHECK:     %[[TILE:.*]] = aie.logical_tile<CoreTile>(?, ?)
# CHECK:     %[[LOCK:.*]] = aie.lock(%[[TILE]])
# CHECK:     aie.runtime_sequence(%arg0: memref<16xi32>) {
# CHECK:       aiex.set_lock(%[[LOCK]], 1)
# CHECK:     }
# CHECK:     %{{.*}} = aie.core(%[[TILE]]) {
# CHECK:         aie.use_lock(%[[LOCK]], Acquire, 1)
# CHECK:         aie.use_lock(%[[LOCK]], Release, 1)
# CHECK:     }
# CHECK:   }
# CHECK: }


def my_barrier():
    # Create barriers to synchronize individual workers with the runtime sequence
    workerBarrier = WorkerRuntimeBarrier()

    # Define a task for a Worker to perform
    def task(barrier):
        barrier.wait_for_value(1)
        # Perform some operation
        a = 1
        barrier.release_with_value(1)

    # Create a Worker and assign the task to it
    worker = Worker(task, fn_args=[workerBarrier])

    # Runtime operations to move data to/from the AIE-array
    external_type = np.ndarray[(16,), np.dtype[np.int32]]
    rt = Runtime()

    def sequence(x):
        rt.set_barrier(workerBarrier, 1)

    rt.sequence(sequence, [external_type])

    # Place components (assign them resources on the device) and generate an MLIR module
    return print(Program(NPU2Col1(), rt, workers=[worker]).resolve_program())


my_barrier()
