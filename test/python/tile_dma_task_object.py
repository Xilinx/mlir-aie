# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""A TileDmaTask built from a Flow end takes its tile, direction and channel.

Building it emits nothing; its first start writes the descriptors and every
start pushes them, so a restart is one queue push.
"""

import numpy as np
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Bd, Buffer, Flow, Program, Runtime
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1, Tile

N = 256
vec_ty = np.ndarray[(N,), np.dtype[np.int32]]


def design(sequence_body, fixed=False):
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    resident = Buffer(tile=mem, type=vec_ty, name="resident")
    if fixed:
        into = Flow(shim, mem, src_channel=0, dst_channel=1)
        out = Flow(mem, shim, src_channel=1, dst_channel=0)
    else:
        into, out = Flow(shim, mem), Flow(mem, shim)

    # Declared next to the Flows, outside the sequence body.
    load = into.task(resident, wait=True)
    store = out.chain(
        [Bd(resident, length=N // 2), Bd(resident, offset=N // 2, length=N // 2)]
    )

    def sequence(a, c):
        sequence_body(into, out, load, store, a, c)

    rt = Runtime(sequence, [vec_ty, vec_ty])
    rt.add_flow(into)
    rt.add_flow(out)
    rt.add_buffer(resident)
    return Program(NPU2Col1(), rt).resolve_program()


def restart(into, out, load, store, a, c):
    into.fill(a)
    load.start()
    load.await_()
    into.fill(a)
    load.start()
    load.await_()
    store.start()
    out.drain(c, wait=True)


# One configuration per task, one push per start. The directions and channels
# come from the Flows: the mem end of `into` receives, the mem end of `out`
# sends, each on its compiler-assigned endpoint.
# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[LOAD:.*]] = aiex.dma_configure_task_for @flow0_dst {
# CHECK: aie.dma_bd(%resident : memref<256xi32>
# CHECK: } {issue_token = true}
# CHECK: aiex.dma_start_task(%[[LOAD]])
# CHECK: aiex.dma_await_task(%[[LOAD]])
# CHECK-NOT: aiex.dma_configure_task_for @flow0_dst
# CHECK: aiex.dma_start_task(%[[LOAD]])
# CHECK: aiex.dma_await_task(%[[LOAD]])
# CHECK: %[[STORE:.*]] = aiex.dma_configure_task_for @flow1_src {
# CHECK: aie.dma_bd(%resident : memref<256xi32> len = 128)
# CHECK: aie.next_bd
# CHECK: aie.dma_bd(%resident : memref<256xi32> offset = 128 len = 128)
# CHECK: aiex.dma_start_task(%[[STORE]])
print(design(restart))


def hoisted(into, out, load, store, a, c):
    # Configure before the loop so every iteration only pushes.
    load.configure()
    for _ in range_(4):
        load.start()
        load.await_()


# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[T:.*]] = aiex.dma_configure_task(%{{.*}}, S2MM, 1)
# CHECK: scf.for
# CHECK-NOT: aiex.dma_configure_task(
# CHECK: aiex.dma_start_task(%[[T]])
# CHECK: aiex.dma_await_task(%[[T]])
print(design(hoisted, fixed=True))


def expect_error(label, fn):
    try:
        fn()
    except (RuntimeError, ValueError) as e:
        print(f"// {label}: {e}")
    else:
        raise AssertionError(f"{label} should have failed")


def await_unconfigured(into, out, load, store, a, c):
    load.await_()


def configure_twice(into, out, load, store, a, c):
    load.configure()
    load.configure()


def start_after_free(into, out, load, store, a, c):
    load.start()
    load.free()
    load.start()


def shim_end():
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    Flow(shim, mem, src_channel=0, dst_channel=0).endpoint(shim).task(
        Buffer(tile=mem, type=vec_ty)
    )


def outside_sequence():
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    core = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    Flow(mem, core).task(Buffer(tile=mem, type=vec_ty)).start()


def not_an_end():
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    core = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    Flow(mem, core).task(Buffer(tile=Tile(col=1, row=1), type=vec_ty))


expect_error("await", lambda: design(await_unconfigured))
expect_error("twice", lambda: design(configure_twice))
expect_error("free", lambda: design(start_after_free))
expect_error("shim", shim_end)
expect_error("outside", outside_sequence)
expect_error("not_an_end", not_an_end)
# CHECK: // await: TileDmaTask on {{.*}} is not configured yet; call start() or configure() inside the runtime sequence first.
# CHECK: // twice: TileDmaTask on {{.*}} is already configured; start() pushes it again.
# CHECK: // free: Task.start() after Task.free()
# CHECK: // shim: MM2S channel 0 on {{.*}} is a shim end, which moves host memory; use the Flow's fill()/drain() for it.
# CHECK: // outside: TileDmaTask.configure() must be called from within the function passed to Runtime(seq_fn, fn_args).
# CHECK: // not_an_end: Tile{{.*}} is not an end of this Flow.
