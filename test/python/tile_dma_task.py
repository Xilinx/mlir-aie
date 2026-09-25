# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""tile_dma_task builds a mem tile buffer descriptor from inside the runtime
sequence, so its length and access pattern can come from dispatch-time values.
TileDma, its structural peer, is configured once when the device loads and
cannot vary per dispatch -- which is why an operand held resident in a mem
tile had to give up dynamic shapes before this existed."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    Flow,
    Lock,
    Program,
    Runtime,
    tile_dma_chain,
    tile_dma_task,
)
from aie.iron.runtime.runtime import IronRuntimeError
from aie.iron.device import NPU2Col1, Tile


def emit_dynamic_memtile_task():
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    host_ty = np.ndarray[(4096,), np.dtype[np.int32]]

    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")

    def sequence(_host, tiles):
        # Both the transfer length and the d1 wrap come from the dispatch-time
        # tile count, so the descriptor differs on every call.
        length = tiles * 512
        task = tile_dma_task(
            mem_tile,
            DMAChannelDir.MM2S,
            0,
            buf,
            sizes=[1, 1, tiles, 512],
            strides=[0, 0, 512, 1],
            transfer_len=length,
            wait=True,
            bd_id=0,
        )
        task.await_()

    rt = Runtime(sequence, [host_ty, np.int64])
    # The buffer is reached only from the sequence body, so the Program has to
    # be told about it (a Worker's fn_args or a TileDma would do it otherwise).
    rt.add_buffer(buf)
    return Program(NPU2Col1(), rt).resolve_program()


# The descriptor is configured on the mem tile's own channel, not through a
# shim DMA allocation, and carries runtime sizes and length. The i64 length is
# range-checked before narrowing so truncation cannot turn an invalid value into
# a different valid transfer.
# CHECK: aiex.npu.assert_bd_field(%[[LEN64:.*]]) {max = 2147483647 : i32} : i64
# CHECK-NEXT: %[[LEN32:.*]] = arith.trunci %[[LEN64]] : i64 to i32
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %[[LEN32]] sizes = [1, 1, %{{.*}}, 512] strides = [0, 0, 512, 1]) {bd_id = 0 : i32}
# CHECK: aiex.dma_start_task
# CHECK: aiex.dma_await_task
print(emit_dynamic_memtile_task())


def emit_shared_length_drain():
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")
    out = Flow(mem_tile, shim, src_channel=0, dst_channel=0)

    def sequence(host, tiles):
        length = tiles * 512
        tile_dma_task(
            mem_tile,
            DMAChannelDir.MM2S,
            out.endpoint(mem_tile),
            buf,
            sizes=[1, 1, tiles, 512],
            strides=[0, 0, 512, 1],
            transfer_len=length,
        )
        out.drain(
            host,
            sizes=[1, 1, tiles, 512],
            strides=[0, 0, 512, 1],
            transfer_len=length,
            wait=True,
        )

    rt = Runtime(sequence, [buf_ty, np.int64])
    rt.add_flow(out)
    rt.add_buffer(buf)
    return Program(NPU2Col1(), rt).resolve_program()


# The same i64 length also sizes the shim drain, narrowed the same way and
# before the task opens, since a BD block admits no arithmetic.
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %{{.*}} sizes = [1, 1, %{{.*}}, 512]
# CHECK: aiex.npu.assert_bd_field(%[[DLEN64:.*]]) {max = 2147483647 : i32} : i64
# CHECK-NEXT: %[[DLEN32:.*]] = arith.trunci %[[DLEN64]] : i64 to i32
# CHECK-NEXT: aiex.dma_configure_task_for
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %[[DLEN32]] sizes = [1, 1, %{{.*}}, 512]
print(emit_shared_length_drain())


def emit_i32_dims(tiles_dims=None, chain_dims=None):
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")
    out = Flow(mem_tile, shim, src_channel=0, dst_channel=0)

    def sequence(host, tiles):
        dims = tiles_dims(tiles) if tiles_dims else [1, 1, tiles, 512]
        if chain_dims:
            tile_dma_chain(
                mem_tile,
                DMAChannelDir.MM2S,
                0,
                [Bd(buf, sizes=chain_dims(tiles), strides=[0, 0, 512, 1])],
            )
            return
        tile_dma_task(
            mem_tile,
            DMAChannelDir.MM2S,
            out.endpoint(mem_tile),
            buf,
            sizes=dims,
            strides=[0, 0, 512, 1],
        )
        out.drain(host, sizes=dims, strides=[0, 0, 512, 1], wait=True)

    rt = Runtime(sequence, [buf_ty, np.int32])
    rt.add_flow(out)
    rt.add_buffer(buf)
    try:
        return Program(NPU2Col1(), rt).resolve_program()
    except TypeError as e:
        return f"RAISED TypeError: {e}"


# An i32 dispatch-time scalar feeds the i64 sizes directly, and the omitted
# length defaults to their product: each task widens and multiplies before
# opening, since a BD block admits no arithmetic.
# CHECK: %[[T1:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK: %[[L1:.*]] = arith.trunci %{{.*}} : i64 to i32
# CHECK-NEXT: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %[[L1]] sizes = [1, 1, %[[T1]], 512]
# CHECK: %[[T2:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK: %[[L2:.*]] = arith.trunci %{{.*}} : i64 to i32
# CHECK-NEXT: aiex.dma_configure_task_for
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %[[L2]] sizes = [1, 1, %[[T2]], 512]
print(emit_i32_dims())

# CHECK: RAISED TypeError: sizes[2] must be an int or an integer SSA value from the runtime sequence, got float.
print(emit_i32_dims(tiles_dims=lambda tiles: [1, 1, 2.0, 512]))

# A chain's Bds are built inside the BD block, so there is nowhere to widen.
# CHECK: RAISED TypeError: dma_bd sizes[2] is i32 but must be i64, and a BD block cannot hold the cast.
print(emit_i32_dims(chain_dims=lambda tiles: [1, 1, tiles, 512]))


def emit_late_add_buffer():
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="late")

    def sequence(_host):
        rt.add_buffer(buf)

    rt = Runtime(sequence, [buf_ty])
    try:
        Program(NPU2Col1(), rt).resolve_program()
    except IronRuntimeError as e:
        print(f"RAISED IronRuntimeError: {e}")


# By the time the sequence body runs the Program has resolved its buffers, so a
# registration from there would be dropped; it is rejected instead.
# CHECK: RAISED IronRuntimeError: Cannot register a Buffer after DMA resolution
emit_late_add_buffer()


def emit_endpoint_task():
    buf_ty = np.ndarray[(64,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="staged")
    out = Flow(mem_tile, shim)

    def sequence(host):
        tile_dma_task(mem_tile, DMAChannelDir.MM2S, out.endpoint(mem_tile), buf)
        out.drain(host, wait=True)

    rt = Runtime(sequence, [buf_ty])
    rt.add_flow(out)
    rt.add_buffer(buf)
    return Program(NPU2Col1(), rt).resolve_program()


# A Flow whose channels the compiler assigns runs the task on its endpoint.
# CHECK: aiex.dma_configure_task_for @[[SRC:flow[0-9]*_src]] {
# CHECK-NEXT: aie.dma_bd(%staged
# CHECK: aie.route_endpoint @[[SRC]](%{{.*}}) DMA
print(emit_endpoint_task())


def emit_rejected(name, body):
    buf_ty = np.ndarray[(64,), np.dtype[np.int32]]
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name=name)
    full = Lock(tile=mem_tile, init=0, name=f"{name}_full")
    empty = Lock(tile=mem_tile, init=1, name=f"{name}_empty")

    def sequence(_host):
        body(mem_tile, buf, full, empty)

    rt = Runtime(sequence, [buf_ty])
    rt.add_buffer(buf)
    rt.add_lock(full)
    rt.add_lock(empty)
    try:
        Program(NPU2Col1(), rt).resolve_program()
    except ValueError as e:
        print(f"RAISED ValueError: {e}")


# A runtime BD takes one acquire and one release or neither; anything else is
# caught before any IR is built rather than by the lowering.
# CHECK: RAISED ValueError: tile_dma_task needs acquire and release together
emit_rejected(
    "acq_only",
    lambda t, b, full, empty: tile_dma_task(
        t, DMAChannelDir.MM2S, 0, b, acquire=Acquire(full)
    ),
)
# CHECK: RAISED ValueError: tile_dma_chain Bd 0 has 2 acquires and 0 releases
emit_rejected(
    "two_acq",
    lambda t, b, full, empty: tile_dma_chain(
        t,
        DMAChannelDir.MM2S,
        0,
        [Bd(b, acquires=[Acquire(full), Acquire(empty)])],
    ),
)
