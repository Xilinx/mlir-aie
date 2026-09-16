# dma_slice_memcpy/tile_dma.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Slice-addressed DDR -> compute-tile DMA, on IRON's explicit DMA primitives.

Introduces two ergonomic conveniences -- numpy slice notation to say which part
of DDR moves, and explicit locks on the receiving tile.

DDR is a runtime-sequence argument, so its address is patched in at dispatch
rather than frozen into the design, and the host passes ordinary tensors of
matching shape. Deriving the access pattern allocates nothing: `from_slice`
reads the geometry off a zero-storage numpy view.

The slice is bigger than the tile buffer, so it lands one buffer at a time. To
make the landing observable, the same buffer is sent straight back out to a
second DDR argument, and the two DMA channels hand the buffer back and forth
through a lock pair.

Siblings: objectfifo.py writes the same dataflow one level up; static_dma.py
reuses this file's tile side but names DDR in the design instead of taking it as
an argument.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
)
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    Flow,
    In,
    Lock,
    Out,
    Program,
    Release,
    Runtime,
    TileDma,
)
from aie.iron.device import Tile
from harness import (
    CHUNKS,
    DEVMEM_SLICE,
    DEVMEM_TY,
    SLICE_TY,
    TILE_ELEMS,
    main,
)


def tile_state(col):
    """Tiles, locks and the staging buffer.

    buf_free starts at 1 (the buffer begins empty, so the inbound channel may
    write it); buf_full starts at 0 (there is nothing to send yet).
    """
    tile_ty = np.ndarray[(TILE_ELEMS,), np.dtype[np.int8]]
    tile = Tile(col=col, row=5, tile_type=AIETileType.CoreTile)
    shim = Tile(col=col, row=0, tile_type=AIETileType.ShimNOCTile)
    buf_free = Lock(tile, lock_id=1, init=1, name="buf_free")
    buf_full = Lock(tile, lock_id=2, init=0, name="buf_full")
    tile_buffer = Buffer(tile=tile, type=tile_ty, name="comp05_tile_buffer")
    return tile, shim, buf_free, buf_full, tile_buffer


def tile_side(tile, tile_buffer, buf_free, buf_full):
    """The compute tile's DMA program.

    Both channels run the same BD CHUNKS times. S2MM takes the buffer when it is
    free and marks it full; MM2S takes it when full and marks it free, so the
    pair ping-pongs through one buffer for the whole slice. No core is involved
    -- a compute tile's DMA and locks live in its memory module and work whether
    or not the core is running.

    loop=False ends each BD chain after its one BD, which is what makes the
    chain a task that completes -- only then does repeat_count mean anything. An
    endless (loop=True) chain plus a repeat count is the deadlock to avoid:
    nothing ever finishes for the count to tick down.
    """
    return TileDma(
        tile=tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                loop=False,
                repeat_count=CHUNKS - 1,
                bds=[
                    Bd(
                        buffer=tile_buffer,
                        length=TILE_ELEMS,
                        acquires=[Acquire(buf_free)],
                        releases=[Release(buf_full)],
                    )
                ],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                loop=False,
                repeat_count=CHUNKS - 1,
                bds=[
                    Bd(
                        buffer=tile_buffer,
                        length=TILE_ELEMS,
                        acquires=[Acquire(buf_full)],
                        releases=[Release(buf_free)],
                    )
                ],
            ),
        ],
    )


@iron.jit
def dma_slice_memcpy(a_in: In, c_out: Out, *, col: CompileTime[int] = 0):
    tile, shim, buf_free, buf_full, tile_buffer = tile_state(col)

    in_flow = Flow(shim, tile, src_channel=0, dst_channel=0)
    out_flow = Flow(tile, shim, src_channel=0, dst_channel=0)

    def sequence(a, c):
        # a[...] is the access pattern the slice describes; c goes back whole.
        in_flow.fill(a, tap=a[DEVMEM_SLICE])
        out_flow.drain(c, tap=c[...], wait=True)

    rt = Runtime(sequence, [DEVMEM_TY, SLICE_TY])
    for lock in (buf_free, buf_full):
        rt.add_lock(lock)
    rt.add_tile_dma(tile_side(tile, tile_buffer, buf_free, buf_full))
    rt.add_flow(in_flow)
    rt.add_flow(out_flow)

    return Program(iron.get_current_device(), rt).resolve_program()


if __name__ == "__main__":
    main("AIE DMA Slice Memcpy (explicit DMA)", dma_slice_memcpy)
