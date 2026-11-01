# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Dataflow primitives for IRON designs.

High-level (managed routing + buffers + locks):
[`ObjectFifo`][aie.iron.ObjectFifo], [`CascadeFlow`][aie.iron.CascadeFlow]

Lower-level (explicit routing + DMA programs; peers of the above):
[`Flow`][aie.iron.Flow], [`PacketFlow`][aie.iron.PacketFlow],
[`PacketDest`][aie.iron.PacketDest], [`TileDma`][aie.iron.TileDma],
[`DmaChannel`][aie.iron.DmaChannel], [`Bd`][aie.iron.Bd],
[`Acquire`][aie.iron.Acquire], [`Release`][aie.iron.Release],
[`DmaEndpoint`][aie.iron.DmaEndpoint], [`FlowEndpoint`][aie.iron.FlowEndpoint],
[`TileDmaTask`][aie.iron.TileDmaTask]

One end managed, the other written by hand:
[`ObjectFifoPool`][aie.iron.ObjectFifoPool], [`PoolSegment`][aie.iron.PoolSegment]
"""

from ...dialects.aie import Packet, Port
from .cascadeflow import CascadeFlow
from .flow import Flow, FlowEndpoint, PacketDest, PacketFlow
from .objectfifo import (
    ObjectFifo,
    ObjectFifoEndpoint,
    ObjectFifoHandle,
    ObjectFifoLink,
    Transport,
)
from .objectfifopool import ObjectFifoPool, ObjectFifoPoolHandle, PoolSegment
from .tile_dma import (
    Acquire,
    Bd,
    BdIteration,
    DmaChannel,
    DmaEndpoint,
    Release,
    TileDma,
    TileDmaTask,
)

__all__ = [
    "ObjectFifoPool",
    "ObjectFifoPoolHandle",
    "PoolSegment",
    "ObjectFifo",
    "Packet",
    "Port",
    "Transport",
    "ObjectFifoHandle",
    "ObjectFifoLink",
    "ObjectFifoEndpoint",
    "CascadeFlow",
    "Flow",
    "FlowEndpoint",
    "PacketDest",
    "PacketFlow",
    "Acquire",
    "Bd",
    "BdIteration",
    "DmaChannel",
    "DmaEndpoint",
    "Release",
    "TileDma",
    "TileDmaTask",
]
