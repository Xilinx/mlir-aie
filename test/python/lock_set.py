# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Lock.set overwrites a lock's value from the runtime sequence, the host-side
peer of a Worker's acquire()/release(). A sequence that restarts a DMA chain
over a mem tile buffer uses it to re-arm the chain's producer locks."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Lock, Program, Runtime
from aie.iron.device import NPU2Col1, Tile


def emit_rearm(value=4):
    host_ty = np.ndarray[(16,), np.dtype[np.int32]]
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    prod = Lock(mem_tile, init=2, name="prod")

    def sequence(_host):
        prod.set(value)

    rt = Runtime(sequence, [host_ty])
    rt.add_lock(prod)
    return Program(NPU2Col1(), rt).resolve_program()


# CHECK: %prod = aie.lock(%{{.*}}) {init = 2 : i32, sym_name = "prod"}
# CHECK: aie.runtime_sequence
# CHECK: aiex.set_lock(%prod, 4)
print(emit_rearm())

# CHECK: aiex.set_lock(%prod, 0)
print(emit_rearm(0))
# CHECK: aiex.set_lock(%prod, 63)
print(emit_rearm(63))


# CHECK: error: Lock.set value must be non-negative.
try:
    emit_rearm(-1)
except ValueError as e:
    print(f"error: {e}")
else:
    raise AssertionError("Expected a negative lock value to fail")


def emit_outside_sequence():
    host_ty = np.ndarray[(16,), np.dtype[np.int32]]
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    prod = Lock(mem_tile, init=2, name="prod")
    # Resolved, but set() after the sequence body has been emitted.
    rt = Runtime(lambda _host: None, [host_ty])
    rt.add_lock(prod)
    Program(NPU2Col1(), rt).resolve_program()
    prod.set(4)


# CHECK: error: Lock.set on prod must be called from within the function passed to Runtime(seq_fn, fn_args); inside a Worker body use acquire()/release().
try:
    emit_outside_sequence()
except RuntimeError as e:
    print(f"error: {e}")
