# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Lock.set overwrites a lock's value from the runtime sequence, the host-side
peer of a Worker's acquire()/release(). A sequence that restarts a DMA chain
over a mem tile buffer uses it to re-arm the chain's producer locks."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Lock, Program, Runtime, Worker
from aie.iron.device import NPU2Col1, Tile
from aie.ir import MLIRError

host_ty = np.ndarray[(16,), np.dtype[np.int32]]


def emit_rearm(value=4):
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    prod = Lock(mem_tile, init=2, name="prod")

    def sequence(_host):
        prod.set(value)

    rt = Runtime(sequence, [host_ty])
    rt.add_lock(prod)
    return Program(NPU2Col1(), rt).resolve_program()


def emit_in_worker():
    compute_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    prod = Lock(compute_tile, init=0, name="prod")

    def core_fn(lock):
        lock.set(1)

    worker = Worker(core_fn, [prod], tile=compute_tile, while_true=False)
    rt = Runtime(lambda _host: None, [host_ty])
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


def expect_failure(emit, *args):
    try:
        emit(*args)
    except MLIRError as e:
        print(f"error: {e}")
    else:
        raise AssertionError(f"Expected {emit.__name__}{args} to fail verification")


# CHECK: %prod = aie.lock(%{{.*}}) {init = 2 : i32, sym_name = "prod"}
# CHECK: aie.runtime_sequence
# CHECK: aiex.set_lock(%prod, 4)
print(emit_rearm())

# CHECK: aiex.set_lock(%prod, 0)
print(emit_rearm(0))
# CHECK: aiex.set_lock(%prod, 63)
print(emit_rearm(63))

# CHECK: error: {{.*}}Lock value must be non-negative
expect_failure(emit_rearm, -1)
# CHECK: error: {{.*}}Lock value exceeds the maximum value of 63
expect_failure(emit_rearm, 64)
# CHECK: error: {{.*}}expects ancestor op 'aie.runtime_sequence'
expect_failure(emit_in_worker)
