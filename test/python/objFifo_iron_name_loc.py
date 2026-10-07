# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# An ObjectFifo's op is located at the fifo's name, and the switchbox
# connections lowered from it keep that location, so diagnostics about its
# routing name the fifo.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, Tile
from aie.passmanager import PassManager


# CHECK: aie.objectfifo @of_a({{.*}} loc([[FIFO:#loc[0-9]*]])
# CHECK: [[FIFO]] = loc("of_a"({{.*}}))
# CHECK-LABEL: // lowered
# CHECK: aie.connect<DMA : 0, {{.*}}> loc([[FLOW:#loc[0-9]*]])
# CHECK: [[FLOW]] = loc("of_a"({{.*}}))
def test_fifo_name_locates_its_flow():
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]
    of_a = ObjectFifo(tile_ty, depth=2, name="of_a")

    def body(fifo):
        fifo.acquire(1)
        fifo.release(1)

    w_prod = Worker(body, fn_args=[of_a.prod()], tile=Tile(0, 2))
    w_cons = Worker(body, fn_args=[of_a.cons()], tile=Tile(0, 4))
    rt = Runtime(lambda: None, [])
    module = Program(NPU1Col1(), rt, workers=[w_prod, w_cons]).resolve_program()
    print(module.operation.get_asm(enable_debug_info=True))

    with module.context:
        PassManager.parse(
            "builtin.module(aie.device(aie-place-tiles,"
            "aie-objectFifo-stateful-transform,aie-create-pathfinder-flows))"
        ).run(module.operation)
    print("// lowered")
    print(module.operation.get_asm(enable_debug_info=True))


if __name__ == "__main__":
    test_fifo_name_locates_its_flow()
