# dma_slice_memcpy/objectfifo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Slice-addressed DDR -> staging tile -> DDR, via ObjectFifo.

The same dataflow as tile_dma.py, expressed one level up. The slice notation is
identical -- `fill`/`drain` take the access pattern a slice describes either way
-- so the whole difference is in how the staging tile is described:

    tile_dma.py                         objectfifo.py
    -----------                         -------------
    Buffer + two Locks                  (generated)
    TileDma, two DmaChannels, two Bds   ObjectFifo.forward()
    two Flows                           (generated)
    loop / repeat_count per channel     (generated)

`forward()` is the direct analogue of what tile_dma.py writes out by hand: it
stages the stream through a tile's memory with a producer/consumer lock pair, no
core involved. ObjectFifo also picks the chunk count up from the transfer rather
than being told, so nothing here counts chunks or sets a repeat. The shim BDs
come out identical to tile_dma.py's.

One real difference: `depth=2` double-buffers the staging tile, so a chunk can
arrive while the previous one leaves. tile_dma.py's single buffer serializes the
two. Depth is the knob ObjectFifo exposes for that; the explicit version would
need a second Buffer and a longer BD chain.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
)
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import AnyShimTile, Tile
from harness import main

DEVMEM_SHAPE = (16, 16, 512)
DEVMEM_SLICE = np.s_[0::2, 1::2, ...]
TILE_ELEMS = 512

# What the slice covers.
SLICE_SHAPE = (8, 8, 512)


@iron.jit
def dma_slice_memcpy(a_in: In, c_out: Out, *, col: CompileTime[int] = 0):
    chunk_ty = np.ndarray[(TILE_ELEMS,), np.dtype[np.int8]]

    # The same tile tile_dma.py stages through. forward() defaults to a memtile,
    # so the type has to be stamped explicitly to land on a compute tile.
    staging = Tile(col=col, row=5, tile_type=AIETileType.CoreTile)

    of_in = ObjectFifo(chunk_ty, depth=2, name="devmem_in")
    of_out = of_in.cons().forward(tile=staging, name="devmem_out")

    def sequence(a, c, in_h, out_h):
        # The same verbs and the same slice as tile_dma.py: the slice says what
        # moves, and the whole of c comes back.
        in_h.fill(a, tap=a[DEVMEM_SLICE])
        out_h.drain(c, tap=c[...], wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[DEVMEM_SHAPE, np.dtype[np.int8]],
            np.ndarray[SLICE_SHAPE, np.dtype[np.int8]],
            # The Runtime discovers fifos from fn_args, not from the fill/drain
            # verbs -- the body runs after fifo collection -- so the handles
            # have to be passed through.
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )

    return Program(iron.get_current_device(), rt).resolve_program()


if __name__ == "__main__":
    main(
        "AIE DMA Slice Memcpy (ObjectFifo)",
        dma_slice_memcpy,
        shape=DEVMEM_SHAPE,
        dtype=np.int8,
        key=DEVMEM_SLICE,
    )
