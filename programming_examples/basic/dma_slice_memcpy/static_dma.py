# dma_slice_memcpy/static_dma.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The dma_slice_memcpy dataflow with DDR named in the design, not passed in.

tile_dma.py takes DDR as a runtime-sequence argument, so its address is patched
in at dispatch and the host supplies a buffer per run. This is the opposite
trade: both DDR buffers are `aie.external_buffer`s at fixed addresses, moved by a
static `aie.shim_dma` program -- so the runtime sequence has nothing left to do
and is emitted empty. That is the shape a design takes when it reproduces a
specific hardware configuration rather than being dispatched against host
allocations.

Emit-only. With the addresses written into the design there is no host buffer to
hand it -- and nothing checks that an allocation lives there -- so there is
nothing to run or verify.

See copy_buffer.py for the same design with the wiring derived rather than
written out.
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


def dma_slice_memcpy_static():
    tile = Tile(0, 5)
    shim = Tile(0, 0)

    # Illustrative addresses: nothing checks that a host allocation lives there,
    # which is the cost of naming DDR in the design rather than taking it as an
    # argument.
    devmem = ExternalBuffer(
        np.ndarray[(16, 16, 512), np.dtype[np.int8]],
        address=0x8000_0000,
        name="device_memory_devmem",
    )
    result = ExternalBuffer(
        np.ndarray[(8, 8, 512), np.dtype[np.int8]],
        address=0x8010_0000,
        name="device_memory_result",
    )

    # buf_free starts at 1 (the buffer begins empty, so the inbound channel may
    # write it); buf_full starts at 0 (there is nothing to send yet).
    buf_free = Lock(tile, lock_id=1, init=1, name="buf_free")
    buf_full = Lock(tile, lock_id=2, init=0, name="buf_full")
    tile_buffer = Buffer(
        tile=tile,
        type=np.ndarray[(512,), np.dtype[np.int8]],
        name="comp05_tile_buffer",
    )

    slice_ = devmem[0::2, 1::2, ...]
    staged = math.prod(tile_buffer.shape)
    chunks = math.prod(slice_.tap.sizes) // staged

    # Both tile channels run the same BD once per chunk. S2MM takes the buffer
    # when it is free and marks it full; MM2S takes it when full and marks it
    # free, so the pair ping-pongs through one buffer for the whole slice.
    #
    # loop=False ends each BD chain after its one BD, which is what makes the
    # chain a task that completes -- only then does repeat_count mean anything.
    tile_dma = TileDma(
        tile=tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                loop=False,
                repeat_count=chunks - 1,
                bds=[
                    Bd(
                        buffer=tile_buffer,
                        length=staged,
                        acquires=[Acquire(buf_free)],
                        releases=[Release(buf_full)],
                    )
                ],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                loop=False,
                repeat_count=chunks - 1,
                bds=[
                    Bd(
                        buffer=tile_buffer,
                        length=staged,
                        acquires=[Acquire(buf_full)],
                        releases=[Release(buf_free)],
                    )
                ],
            ),
        ],
    )

    shim_dma = TileDma(
        tile=shim,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                loop=False,
                bds=[
                    # The slice fits a buffer descriptor's three dimensions as
                    # written, so it goes over as one descriptor.
                    Bd(
                        buffer=devmem,
                        offset=slice_.tap.offset,
                        length=math.prod(slice_.tap.sizes),
                        sizes=list(slice_.tap.sizes),
                        strides=list(slice_.tap.strides),
                    )
                ],
            ),
            # The slice comes back contiguously, so one linear BD absorbs every
            # chunk the compute tile sends.
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                loop=False,
                bds=[Bd(buffer=result, length=math.prod(result.shape))],
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

    return Program(NPU2Col1(), rt).resolve_program()


if __name__ == "__main__":
    print(dma_slice_memcpy_static())
