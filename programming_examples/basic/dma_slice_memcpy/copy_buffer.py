# dma_slice_memcpy/copy_buffer.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A copy_buffer() abstraction over IRON's explicit DMA primitives.

tile_dma.py and static_dma.py write out the same transfer twice, differing only
in where DDR's address comes from, and each spells out a Flow, a DmaChannel, a
Bd and a chunk count to do it. This file shows one `copy_buffer()` covering
both, built from the primitives the other files use directly:

    dma = Dma()
    dma.copy_buffer(
        src_buffer=devmem[0::2, 1::2, ...],
        src_channel=0,
        dst_buffer=tile_buffer,
        dst_channel=0,
        dst_wait_for_lock=buf_free,
        dst_release_lock=buf_full,
        through_shim=shim,
    )
    dma.emit(rt)

Two things let one function serve both models.

`Mem` makes slicing mean the same thing whatever the memory is -- off-chip at a
fixed address, off-chip supplied at dispatch, or a tile's own buffer -- by
pairing the buffer with the access pattern a slice describes. So the call site
reads the same either way, and the function decides what to build from what it
was handed:

  * an ExternalBuffer is addressable now, so the shim's side of the transfer is
    built immediately as a static aie.shim_dma channel;
  * a buffer that arrives at dispatch is named by its type, so only the route
    and the tile's side can be built now. copy_buffer returns the Flow and the
    runtime sequence fills or drains it.

Either way the tile's side -- channel, buffer descriptor, lock pair -- is the
same, and the chunk count is derived from the slice rather than passed in.

`emit` exists because a tile has one DMA program: two TileDma objects on one
tile would emit two aie.mem regions. Calls accumulate per tile and are flushed
in one go, which is why the original this mirrors ends with an emit step too.

Deliberately NOT part of the IRON library -- this is a sketch of what such an
API could look like, kept next to the primitives it is built from.
"""

import argparse
import math
from typing import get_origin

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
)
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.util import np_ndarray_type_get_shape
from aie.iron import (
    Acquire,
    Bd,
    BdIteration,
    Buffer,
    CompileTime,
    DmaChannel,
    ExternalBuffer,
    Flow,
    In,
    Out,
    Program,
    Release,
    Runtime,
    TileDma,
)
from aie.iron.device import NPU2Col1
from harness import DEVMEM_SLICE, DEVMEM_TY, SLICE_TY, add_col_arg, main
from tile_dma import tile_state

# A static shim BD's wrap field, in elements. aie-decompose-large-dma-bd would
# legalize a pattern against this, but it declines anything inside an
# aie.shim_dma / aie.mem / aie.memtile_dma region -- the static path gets no
# such pass -- so _shim_geometry below has to arrive already legal.
MAX_WRAP = 1023


class Mem:
    """A buffer, and optionally the part of it a slice names.

    Wraps any of the three kinds of memory a copy can touch so they slice
    alike: an [`ExternalBuffer`][iron.ExternalBuffer] (off-chip, addressed by
    the design), an ``np.ndarray`` type (off-chip, supplied at dispatch), or a
    [`Buffer`][iron.Buffer] (a tile's own memory).
    """

    def __init__(self, buffer, tap=None):
        self.buffer = buffer
        # A dispatch-time buffer is named by its np.ndarray type rather than
        # by an object, so the shape comes from the type parameters.
        self.shape = (
            np_ndarray_type_get_shape(buffer)
            if get_origin(buffer) is np.ndarray
            else tuple(buffer.shape)
        )
        self.tap = (
            tap
            if tap is not None
            else TensorAccessPattern.from_slice(self.shape, np.s_[...])
        )

    @classmethod
    def of(cls, mem):
        """Accept a bare buffer too, so a whole-buffer end needs no wrapping."""
        return mem if isinstance(mem, cls) else cls(mem)

    def __getitem__(self, key):
        """Narrow to the part of the buffer ``key`` names."""
        return Mem(self.buffer, TensorAccessPattern.from_slice(self.shape, key))

    @property
    def elements(self):
        return math.prod(self.tap.sizes)

    @property
    def is_tile(self):
        return isinstance(self.buffer, Buffer)

    @property
    def is_addressable(self):
        """Whether the buffer's address is known now, rather than at dispatch."""
        return isinstance(self.buffer, ExternalBuffer)


def _split_factor(size):
    """Largest divisor of ``size`` that fits the wrap field."""
    for factor in range(min(size, MAX_WRAP), 0, -1):
        if size % factor == 0:
            return factor
    raise ValueError(f"cannot express a wrap of {size}")


def _is_contiguous(sizes, strides):
    """Whether the pattern walks one unbroken run of memory."""
    expected = 1
    for size, stride in zip(reversed(sizes), reversed(strides)):
        if stride != expected:
            return False
        expected *= size
    return True


def _shim_geometry(tap):
    """Lower ``tap`` onto a static shim BD's three dimensions plus iteration.

    A dimension wider than the wrap field is factored in two, which can leave
    four dimensions where the hardware has three. The outermost then becomes the
    BD's iteration state -- and since that advances once per BD execution, the
    channel has to run as many times, so the repeat count comes back with it.

    Only a scattered pattern needs it. The dialect exempts a contiguous shim
    transfer from the wrap cap, so that one goes over untouched.
    """
    if _is_contiguous(tap.sizes, tap.strides):
        sizes, strides = list(tap.sizes), list(tap.strides)
    else:
        sizes, strides = [], []
        for size, stride in zip(tap.sizes, tap.strides):
            if size <= MAX_WRAP:
                sizes.append(size)
                strides.append(stride)
                continue
            inner = _split_factor(size)
            sizes += [size // inner, inner]
            strides += [stride * inner, stride]

    iteration, repeat_count = None, 0
    if len(sizes) > 3:
        if len(sizes) > 4:
            raise ValueError(f"{tap} needs more dimensions than a shim BD has")
        iteration = BdIteration(size=sizes[0], stride=strides[0])
        repeat_count = sizes[0] - 1
        sizes, strides = sizes[1:], strides[1:]

    return sizes, strides, math.prod(sizes), iteration, repeat_count


class Dma:
    """Collects copies, then emits one DMA program per tile.

    A tile has a single DMA program, so channels from separate copy_buffer calls
    that land on the same tile have to be emitted together -- hence the
    accumulate-then-[`emit`][Dma.emit] shape.
    """

    def __init__(self):
        self._channels = {}
        self._flows = []

    def copy_buffer(
        self,
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
        """Move ``src_buffer`` to ``dst_buffer``, one of which is a tile buffer.

        The locks belong to whichever end is the tile: the channel acquires
        ``wait_for_lock`` before each buffer and releases ``release_lock`` after
        it, so a producer and a consumer on the same buffer hand it back and
        forth.

        Returns:
            Flow: the route built for the copy. When the off-chip end is
            supplied at dispatch, this is what the runtime sequence fills or
            drains; when it is an ExternalBuffer there is nothing left to do
            with it.
        """
        src, dst = Mem.of(src_buffer), Mem.of(dst_buffer)
        if src.is_tile == dst.is_tile:
            raise ValueError(
                "copy_buffer moves between a tile buffer and off-chip memory, "
                "so exactly one end must be a tile Buffer."
            )

        if dst.is_tile:
            tile_mem, off_chip = dst, src
            tile_dir, tile_channel = DMAChannelDir.S2MM, dst_channel
            shim_dir, shim_channel = DMAChannelDir.MM2S, src_channel
            wait, release = dst_wait_for_lock, dst_release_lock
            flow = Flow(
                through_shim,
                tile_mem.buffer.tile,
                src_channel=shim_channel,
                dst_channel=tile_channel,
            )
        else:
            tile_mem, off_chip = src, dst
            tile_dir, tile_channel = DMAChannelDir.MM2S, src_channel
            shim_dir, shim_channel = DMAChannelDir.S2MM, dst_channel
            wait, release = src_wait_for_lock, src_release_lock
            flow = Flow(
                tile_mem.buffer.tile,
                through_shim,
                src_channel=tile_channel,
                dst_channel=shim_channel,
            )
        self._flows.append(flow)

        # The tile stages the transfer one buffer at a time, so it runs its BD
        # once per chunk -- derived from the slice rather than passed in. The
        # chain has to end for that count to mean anything (see tile_dma.py).
        chunks = off_chip.elements // tile_mem.elements
        self._add(
            tile_mem.buffer.tile,
            DmaChannel(
                direction=tile_dir,
                channel=tile_channel,
                loop=False,
                repeat_count=chunks - 1,
                bds=[
                    Bd(
                        buffer=tile_mem.buffer,
                        length=tile_mem.elements,
                        acquires=[Acquire(wait)] if wait else [],
                        releases=[Release(release)] if release else [],
                    )
                ],
            ),
        )

        if off_chip.is_addressable:
            sizes, strides, length, iteration, repeat = _shim_geometry(off_chip.tap)
            self._add(
                through_shim,
                DmaChannel(
                    direction=shim_dir,
                    channel=shim_channel,
                    loop=False,
                    repeat_count=repeat,
                    bds=[
                        Bd(
                            buffer=off_chip.buffer,
                            offset=off_chip.tap.offset,
                            length=length,
                            sizes=sizes,
                            strides=strides,
                            iteration=iteration,
                        )
                    ],
                ),
            )
        return flow

    def _add(self, tile, channel):
        self._channels.setdefault(tile, []).append(channel)

    def emit(self, rt, locks=()):
        """Register everything collected so far with the Runtime."""
        for lock in locks:
            rt.add_lock(lock)
        for flow in self._flows:
            rt.add_flow(flow)
        for tile, channels in self._channels.items():
            rt.add_tile_dma(TileDma(tile=tile, channels=channels))


def _build(col, devmem, result):
    """Wire the copy both ways. Identical for both memory models."""
    tile, shim, buf_free, buf_full, tile_buffer = tile_state(col)

    dma = Dma()
    into_tile = dma.copy_buffer(
        src_buffer=devmem[DEVMEM_SLICE],
        src_channel=0,
        dst_buffer=tile_buffer,
        dst_channel=0,
        dst_wait_for_lock=buf_free,
        dst_release_lock=buf_full,
        through_shim=shim,
    )
    out_of_tile = dma.copy_buffer(
        src_buffer=tile_buffer,
        src_channel=0,
        dst_buffer=result,
        dst_channel=0,
        src_wait_for_lock=buf_full,
        src_release_lock=buf_free,
        through_shim=shim,
    )
    return dma, (buf_free, buf_full), into_tile, out_of_tile


@iron.jit
def dma_slice_memcpy(a_in: In, c_out: Out, *, col: CompileTime[int] = 0):
    # Off-chip memory that arrives at dispatch, named by its type.
    devmem, result = Mem(DEVMEM_TY), Mem(SLICE_TY)
    dma, locks, into_tile, out_of_tile = _build(col, devmem, result)

    def sequence(a, c):
        # The routes copy_buffer built; the slices it was given say what moves.
        into_tile.fill(a, tap=devmem[DEVMEM_SLICE].tap)
        out_of_tile.drain(c, tap=result.tap, wait=True)

    rt = Runtime(sequence, [DEVMEM_TY, SLICE_TY])
    dma.emit(rt, locks)
    return Program(iron.get_current_device(), rt).resolve_program()


def static_program(col=0):
    """The same _build, handed off-chip memory the design addresses itself."""
    devmem = Mem(ExternalBuffer(DEVMEM_TY, address=0x8000_0000, name="devmem"))
    result = Mem(ExternalBuffer(SLICE_TY, address=0x8010_0000, name="result"))
    dma, locks, _, _ = _build(col, devmem, result)

    def sequence():
        pass

    rt = Runtime(sequence, [])
    for ddr in (devmem.buffer, result.buffer):
        rt.add_external_buffer(ddr)
    dma.emit(rt, locks)
    return Program(NPU2Col1(), rt).resolve_program()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--static", action="store_true")
    add_col_arg(parser)
    known, _ = parser.parse_known_args()
    if known.static:
        print(static_program(known.col))
    else:
        main("AIE DMA Slice Memcpy (copy_buffer)", dma_slice_memcpy)
