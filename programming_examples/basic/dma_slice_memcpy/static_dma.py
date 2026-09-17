# dma_slice_memcpy/static_dma.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Slice-addressed DDR -> tile -> DDR, with DDR named in the design.

The same dataflow as tile_dma.py, differing only in where DDR's address comes
from. tile_dma.py takes DDR as a runtime-sequence argument, so its address is
patched in per dispatch; here both DDR buffers are `aie.external_buffer`s at
fixed addresses moved by a static `aie.shim_dma` program, so the runtime
sequence has nothing left to do and is emitted empty.

Emit-only. With the addresses written into the design there is no host buffer to
hand it -- and nothing checks that an allocation lives there -- so there is
nothing to run or verify.

See copy_buffer.py for this design with the wiring derived rather than written.
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


def dma_slice_memcpy_static():
    tile = Tile(0, 5)
    shim = Tile(0, 0)

    # Illustrative addresses: nothing checks that an allocation lives there,
    # which is the cost of naming DDR in the design rather than taking it in.
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

    # A slice of a buffer is still a buffer, so this is what moves.
    part = devmem[0::2, 1::2, ...]

    # The tile side is the same as tile_dma.py's: the two channels ping-pong
    # through the one buffer, and their chains loop because the locks are what
    # pace them.
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

    # The shim's channels take no locks, so nothing would pace a looping chain
    # and it would re-send the slice forever. loop=False ends each chain after
    # its BD, which moves the slice exactly once.
    shim_dma = TileDma(
        tile=shim,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                loop=False,
                # The slice fits a buffer descriptor's three dimensions as
                # written, so it goes over as one descriptor.
                bds=[Bd(buffer=devmem, tap=part.tap)],
            ),
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                loop=False,
                bds=[Bd(buffer=result, tap=result.tap)],
            ),
        ],
    )

    # Nothing is dispatched: the addresses are in the design, so the runtime
    # sequence has nothing to do.
    rt = Runtime(lambda: None, [])
    for lock in (buf_free, buf_full):
        rt.add_lock(lock)
    for ddr in (devmem, result):
        rt.add_external_buffer(ddr)
    rt.add_tile_dma(tile_dma)
    rt.add_tile_dma(shim_dma)
    rt.add_flow(Flow(shim, tile, src_channel=0, dst_channel=0))
    rt.add_flow(Flow(tile, shim, src_channel=0, dst_channel=0))

    # NPU2Col1 is built by create_class, so pyright sees the base Device
    # __init__ rather than the generated no-argument one.
    device = NPU2Col1()  # pyright: ignore[reportCallIssue]
    return Program(device, rt).resolve_program()


if __name__ == "__main__":
    print(dma_slice_memcpy_static())
