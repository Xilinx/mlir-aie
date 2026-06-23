# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Dataflow primitives for IRON designs.

High-level (managed routing + buffers + locks):
    :class:`ObjectFifo`, :class:`CascadeFlow`

Lower-level (explicit routing + DMA programs; peers of the above):
    :class:`Flow`, :class:`PacketFlow`, :class:`PacketDest`,
    :class:`TileDma`, :class:`DmaChannel`, :class:`Bd`,
    :class:`Acquire`, :class:`Release`
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
