# dma_slice_memcpy/copy_buffer.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The dma_slice_memcpy design, said as one copy_buffer() call per transfer.

static_dma.py spells out a Flow, a DmaChannel, a Bd and a chunk count for each
direction. All of that is derivable from the two ends and the locks between
them, so copy_buffer() below takes just those and wires the rest -- leaving a
design that reads as what it does.

copy_buffer introduces no types of its own. It slices with
[`ExternalBuffer.__getitem__`][iron.ExternalBuffer], builds the same
[`Flow`][iron.Flow] / [`DmaChannel`][iron.DmaChannel] / [`Bd`][iron.Bd] objects
static_dma.py builds by hand, and registers them on the Runtime the same way --
so there is no bookkeeping left at the call site and no emit step at the end.

Deliberately NOT part of the IRON library: a sketch of what such an API could
look like, kept next to the primitives it is built from.
"""

import math

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
    rt,
    *,
    src_buffer,
    src_channel,
    dst_buffer,
    dst_channel,
    through_shim,
    src_wait_for_lock=None,
    src_release_lock=None,
    dst_wait_for_lock=None,
    dst_release_lock=None,
):
    """Copy between off-chip memory and a tile buffer, through the shim.

    One end is a tile [`Buffer`][iron.Buffer] and the other an
    [`ExternalBuffer`][iron.ExternalBuffer], whole or sliced. The locks belong
    to the tile end, which acquires ``wait_for_lock`` before each buffer and
    releases ``release_lock`` after it, so a producer and a consumer on the same
    buffer hand it back and forth.

    Everything the copy needs is derived: the route, a channel at each end, the
    tile's buffer descriptor and how many times it runs, and the shim's
    descriptor over the slice. All of it is registered on ``rt``.
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
    tile = tile_buffer.tile
    wait = dst_wait_for_lock if into_tile else src_wait_for_lock
    release = dst_release_lock if into_tile else src_release_lock

    ends = (through_shim, tile) if into_tile else (tile, through_shim)
    rt.add_flow(Flow(*ends, src_channel=src_channel, dst_channel=dst_channel))
    rt.add_external_buffer(off_chip)
    for lock in (wait, release):
        if lock is not None:
            rt.add_lock(lock)

    # The tile stages the copy one buffer at a time, so it runs its BD once per
    # chunk -- taken from the slice rather than passed in. The chain has to end
    # for that count to mean anything (see tile_dma.py).
    staged = math.prod(tile_buffer.shape)
    tile_side = DmaChannel(
        direction=DMAChannelDir.S2MM if into_tile else DMAChannelDir.MM2S,
        channel=dst_channel if into_tile else src_channel,
        loop=False,
        repeat_count=math.prod(off_chip.tap.sizes) // staged - 1,
        bds=[
            Bd(
                buffer=tile_buffer,
                length=staged,
                acquires=[Acquire(wait)] if wait else [],
                releases=[Release(release)] if release else [],
            )
        ],
    )

    # The shim moves the whole slice in one descriptor: the access pattern goes
    # over as the slice describes it.
    pattern = off_chip.tap
    shim_side = DmaChannel(
        direction=DMAChannelDir.MM2S if into_tile else DMAChannelDir.S2MM,
        channel=src_channel if into_tile else dst_channel,
        loop=False,
        bds=[
            Bd(
                buffer=off_chip,
                offset=pattern.offset,
                length=math.prod(pattern.sizes),
                sizes=list(pattern.sizes),
                strides=list(pattern.strides),
            )
        ],
    )

    # A tile has one DMA program, so a second copy touching a tile this one
    # already reached merges into it.
    rt.add_tile_dma(TileDma(tile=tile, channels=[tile_side]))
    rt.add_tile_dma(TileDma(tile=through_shim, channels=[shim_side]))


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

    tile_produce_lock = Lock(tile, lock_id=1, init=1, name="tile_produce_lock")
    tile_consume_lock = Lock(tile, lock_id=2, init=0, name="tile_consume_lock")

    tile_buffer = Buffer(
        tile=tile,
        type=np.ndarray[(512,), np.dtype[np.int8]],
        name="tile_buffer",
    )

    # Nothing is dispatched: the addresses are in the design, so the runtime
    # sequence has nothing to do.
    rt = Runtime(lambda: None, [])

    copy_buffer(
        rt,
        src_buffer=devmem[0::2, 1::2, ...],
        src_channel=0,
        dst_buffer=tile_buffer,
        dst_channel=0,
        dst_wait_for_lock=tile_produce_lock,
        dst_release_lock=tile_consume_lock,
        through_shim=shim,
    )
    copy_buffer(
        rt,
        src_buffer=tile_buffer,
        src_channel=0,
        dst_buffer=result,
        dst_channel=0,
        src_wait_for_lock=tile_consume_lock,
        src_release_lock=tile_produce_lock,
        through_shim=shim,
    )

    return Program(NPU2Col1(), rt).resolve_program()


if __name__ == "__main__":
    print(dma_slice_memcpy())
