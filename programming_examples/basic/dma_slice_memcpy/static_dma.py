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

The tile side is imported from tile_dma.py unchanged, so the whole difference
between the two files is where DDR's address comes from.

Emit-only. With the addresses written into the design there is no host buffer to
hand it -- and nothing checks that an allocation lives there -- so there is
nothing to run or verify.
"""

import argparse
import math

from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
)
from aie.iron import (
    Bd,
    BdIteration,
    DmaChannel,
    ExternalBuffer,
    Flow,
    Program,
    Runtime,
    TileDma,
)
from aie.iron.device import NPU2Col1
from harness import DEVMEM_TY, IN_TAP, SLICE_SHAPE, SLICE_TY, add_col_arg
from tile_dma import tile_side, tile_state

# Illustrative addresses: nothing checks that a host allocation lives there,
# which is the cost of naming DDR in the design rather than taking it as an
# argument.
IN_ADDR = 0x8000_0000
OUT_ADDR = 0x8010_0000

# A static shim BD's wrap is capped at 1023, so the 4096-element contiguous run
# is spelled as two dimensions of this size.
INNER = 512


def _inbound_bd(in_ddr):
    """The inbound BD, with the slice spelled out for a static shim DMA.

    tile_dma.py hands a whole tap to `fill` and lets
    `aie-decompose-large-dma-bd` legalize it. A static `aie.shim_dma` gets no
    such pass, so the geometry has to be hardware-legal as written, under two
    constraints the runtime path hides:

      * A wrap is capped at 1023, and counted in elements here rather than
        32-bit words, so the 4096-element contiguous run cannot be a single
        dimension. Spelling it as 8 x 512 costs one of the three available
        dimensions.
      * That leaves no dimension for the slice's outer axis, so it goes in the
        BD's iteration state -- which advances once per BD execution, and
        executions come from repeat_count. The two must agree, which is the same
        rule tile_dma.py's loop/repeat_count pairing follows.
    """
    outer, middle, inner = IN_TAP.sizes
    outer_stride, middle_stride, _ = IN_TAP.strides
    return Bd(
        buffer=in_ddr,
        offset=IN_TAP.offset,
        length=middle * inner,
        sizes=[middle, inner // INNER, INNER],
        strides=[middle_stride, INNER, 1],
        iteration=BdIteration(size=outer, stride=outer_stride),
    )


def dma_slice_memcpy_static(col=0):
    tile, shim, buf_free, buf_full, tile_buffer = tile_state(col)

    in_ddr = ExternalBuffer(DEVMEM_TY, address=IN_ADDR, name="device_memory_devmem")
    out_ddr = ExternalBuffer(SLICE_TY, address=OUT_ADDR, name="device_memory_result")

    shim_dma = TileDma(
        tile=shim,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                loop=False,
                repeat_count=IN_TAP.sizes[0] - 1,
                bds=[_inbound_bd(in_ddr)],
            ),
            # The slice comes back contiguously, so one linear BD absorbs every
            # chunk the compute tile sends.
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                loop=False,
                bds=[Bd(buffer=out_ddr, length=math.prod(SLICE_SHAPE))],
            ),
        ],
    )

    def sequence():
        pass

    rt = Runtime(sequence, [])
    for lock in (buf_free, buf_full):
        rt.add_lock(lock)
    for ddr in (in_ddr, out_ddr):
        rt.add_external_buffer(ddr)
    rt.add_tile_dma(tile_side(tile, tile_buffer, buf_free, buf_full))
    rt.add_tile_dma(shim_dma)
    rt.add_flow(Flow(shim, tile, src_channel=0, dst_channel=0))
    rt.add_flow(Flow(tile, shim, src_channel=0, dst_channel=0))

    return Program(NPU2Col1(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="AIE DMA Slice Memcpy (fixed-address DDR)")
    add_col_arg(p)
    print(dma_slice_memcpy_static(p.parse_args().col))


if __name__ == "__main__":
    main()
