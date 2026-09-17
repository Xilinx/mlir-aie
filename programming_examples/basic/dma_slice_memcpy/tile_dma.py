# dma_slice_memcpy/tile_dma.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Slice-addressed DDR -> tile -> DDR, on IRON's explicit DMA primitives.

The slice says which part of DDR moves; two DMA channels hand one staging
buffer back and forth through a lock pair, and the host checks the bytes that
come back against the same slice taken in numpy.

DDR arrives at dispatch here: it is a runtime-sequence argument, so its address
is patched in per run rather than written into the design. See static_dma.py for
the same dataflow with DDR addressed by the design, objectfifo.py for it one
level up, and copy_buffer.py for the wiring derived rather than written out.
"""

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
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
from harness import main


@iron.jit
def dma_slice_memcpy(a_in: In, c_out: Out, *, col: CompileTime[int] = 0):
    tile = Tile(col, 5)
    shim = Tile(col, 0)

    # buf_free starts at 1 (the buffer begins empty, so the inbound channel may
    # write it); buf_full starts at 0 (there is nothing to send yet).
    buf_free = Lock(tile, init=1, name="buf_free")
    buf_full = Lock(tile, init=0, name="buf_full")
    # One innermost run of the slice at a time.
    staging = Buffer(
        tile=tile,
        type=np.ndarray[(512,), np.dtype[np.int8]],
        name="staging",
    )

    # The two channels ping-pong through the one buffer: S2MM takes it when
    # free and marks it full, MM2S takes it when full and marks it free. Both BD
    # chains loop, so each runs for as long as its locks allow -- what bounds
    # the transfer is the slice the runtime sequence asks for, not a count here.
    tile_dma = TileDma(
        tile=tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                bds=[
                    Bd(
                        buffer=staging,
                        acquires=[Acquire(buf_free)],
                        releases=[Release(buf_full)],
                    )
                ],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(
                        buffer=staging,
                        acquires=[Acquire(buf_full)],
                        releases=[Release(buf_free)],
                    )
                ],
            ),
        ],
    )

    into_tile = Flow(shim, tile, src_channel=0, dst_channel=0)
    out_of_tile = Flow(tile, shim, src_channel=0, dst_channel=0)

    def sequence(a, c):
        into_tile.fill(a, tap=a[0::2, 1::2, ...])
        out_of_tile.drain(c, tap=c[...], wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(16, 16, 512), np.dtype[np.int8]],
            np.ndarray[(8, 8, 512), np.dtype[np.int8]],
        ],
    )
    for lock in (buf_free, buf_full):
        rt.add_lock(lock)
    rt.add_tile_dma(tile_dma)
    rt.add_flow(into_tile)
    rt.add_flow(out_of_tile)

    return Program(iron.get_current_device(), rt).resolve_program()


if __name__ == "__main__":
    main(
        "AIE DMA Slice Memcpy (explicit DMA)",
        dma_slice_memcpy,
        shape=(16, 16, 512),
        dtype=np.int8,
        key=np.s_[0::2, 1::2, ...],
    )
