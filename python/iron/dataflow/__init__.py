# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Dataflow primitives for IRON designs.

High-level (managed routing + buffers + locks):
[`ObjectFifo`][iron.ObjectFifo], [`CascadeFlow`][iron.CascadeFlow]

Lower-level (explicit routing + DMA programs; peers of the above):
[`Flow`][iron.Flow], [`PacketFlow`][iron.PacketFlow],
[`PacketDest`][iron.PacketDest], [`TileDma`][iron.TileDma],
[`DmaChannel`][iron.DmaChannel], [`Bd`][iron.Bd],
[`Acquire`][iron.Acquire], [`Release`][iron.Release]
"""

from .objectfifo import (
    ObjectFifo,
    ObjectFifoHandle,
    ObjectFifoLink,
    ObjectFifoEndpoint,
    PadDims,
    StreamDims,
)
from .cascadeflow import CascadeFlow
from .flow import Flow, PacketDest, PacketFlow
from .tile_dma import Acquire, Bd, DmaChannel, Release, TileDma
