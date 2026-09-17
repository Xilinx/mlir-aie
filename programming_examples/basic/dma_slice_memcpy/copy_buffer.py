# dma_slice_memcpy/copy_buffer.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The static_dma.py design, said as one copy_buffer() call per transfer.

static_dma.py spells out a Flow, two DmaChannels and a Bd for each direction.
All of that follows from the two ends and the locks between them, so
copy_buffer() takes those and wires the rest, leaving a design that reads as
what it does.

copy_buffer introduces no types of its own: it slices with
[`ExternalBuffer.__getitem__`][iron.ExternalBuffer], builds the same
[`Flow`][iron.Flow] / [`DmaChannel`][iron.DmaChannel] / [`Bd`][iron.Bd] objects
static_dma.py builds by hand, and registers them on the Runtime the same way.

Deliberately NOT part of the IRON library: a sketch of what such an API could
look like, kept next to the primitives it is built from.
"""

import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
)
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DmaChannel,
    ExternalBuffer,
    Flow,
    Lock,
    Program,
    Release,
    Runtime,
    TileDma,
)
from aie.iron.device import NPU2Col1, Tile


def copy_buffer(
    rt: Runtime,
    *,
    src_buffer: Buffer | ExternalBuffer,
    src_channel: int,
    dst_buffer: Buffer | ExternalBuffer,
    dst_channel: int,
    through_shim: Tile,
    src_wait_for_lock: Lock | None = None,
    src_release_lock: Lock | None = None,
    dst_wait_for_lock: Lock | None = None,
    dst_release_lock: Lock | None = None,
) -> None:
    """Copy between off-chip memory and a tile buffer, through the shim.

    One end is a tile [`Buffer`][iron.Buffer] and the other an
    [`ExternalBuffer`][iron.ExternalBuffer], whole or sliced -- which of the two
    is which decides the direction. The locks belong to the tile end, which
    acquires ``wait_for_lock`` before each buffer and releases ``release_lock``
    after it, so a producer and a consumer on the same buffer hand it back and
    forth.

    The route, a channel at each end and their buffer descriptors all follow,
    and are registered on ``rt``.
    """
    into_tile = isinstance(dst_buffer, Buffer)
    if into_tile == isinstance(src_buffer, Buffer):
        raise ValueError(
            "copy_buffer moves between off-chip memory and a tile buffer, so "
            "exactly one of src_buffer/dst_buffer must be a tile Buffer."
        )
    tile_buffer, off_chip = (
        (dst_buffer, src_buffer) if into_tile else (src_buffer, dst_buffer)
    )
    assert isinstance(tile_buffer, Buffer) and isinstance(off_chip, ExternalBuffer)
    tile = tile_buffer.tile
    if tile is None:
        raise ValueError(f"{tile_buffer} must be placed on a tile to copy to or from.")
    wait = dst_wait_for_lock if into_tile else src_wait_for_lock
    release = dst_release_lock if into_tile else src_release_lock

    ends = (through_shim, tile) if into_tile else (tile, through_shim)
    rt.add_flow(Flow(*ends, src_channel=src_channel, dst_channel=dst_channel))
    rt.add_external_buffer(off_chip)
    for lock in (wait, release):
        if lock is not None:
            rt.add_lock(lock)

    # The tile's chain loops, because its locks are what pace it; the shim's
    # takes no locks, so it has to end or it would re-send forever.
    rt.add_tile_dma(
        TileDma(
            tile=tile,
            channels=[
                DmaChannel(
                    direction=DMAChannelDir.S2MM if into_tile else DMAChannelDir.MM2S,
                    channel=dst_channel if into_tile else src_channel,
                    bds=[
                        Bd(
                            buffer=tile_buffer,
                            acquires=[Acquire(wait)] if wait else [],
                            releases=[Release(release)] if release else [],
                        )
                    ],
                )
            ],
        )
    )
    rt.add_tile_dma(
        TileDma(
            tile=through_shim,
            channels=[
                DmaChannel(
                    direction=DMAChannelDir.MM2S if into_tile else DMAChannelDir.S2MM,
                    channel=src_channel if into_tile else dst_channel,
                    loop=False,
                    bds=[Bd(buffer=off_chip, tap=off_chip.tap)],
                )
            ],
        )
    )


def dma_slice_memcpy():
    tile = Tile(0, 5)
    shim = Tile(0, 0)

    devmem = ExternalBuffer(
        np.ndarray[(16, 16, 512), np.dtype[np.int8]],
        address=0x8000_0000,
        name="devmem",
    )
    result = ExternalBuffer(
        np.ndarray[(8, 8, 512), np.dtype[np.int8]],
        address=0x8010_0000,
        name="result",
    )

    buf_free = Lock(tile, init=1, name="buf_free")
    buf_full = Lock(tile, init=0, name="buf_full")
    staging = Buffer(
        tile=tile,
        type=np.ndarray[(512,), np.dtype[np.int8]],
        name="staging",
    )

    # Nothing is dispatched: the addresses are in the design, so the runtime
    # sequence has nothing to do.
    rt = Runtime(lambda: None, [])

    copy_buffer(
        rt,
        src_buffer=devmem[0::2, 1::2, ...],
        src_channel=0,
        dst_buffer=staging,
        dst_channel=0,
        dst_wait_for_lock=buf_free,
        dst_release_lock=buf_full,
        through_shim=shim,
    )
    copy_buffer(
        rt,
        src_buffer=staging,
        src_channel=0,
        dst_buffer=result,
        dst_channel=0,
        src_wait_for_lock=buf_full,
        src_release_lock=buf_free,
        through_shim=shim,
    )

    # NPU2Col1 is built by create_class, so pyright sees the base Device
    # __init__ rather than the generated no-argument one.
    device = NPU2Col1()  # pyright: ignore[reportCallIssue]
    return Program(device, rt).resolve_program()


if __name__ == "__main__":
    print(dma_slice_memcpy())
