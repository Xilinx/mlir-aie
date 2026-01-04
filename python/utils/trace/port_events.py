# SPDX-FileCopyrightText: Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import IntEnum
from .event_enums import CoreEvent, MemTileEvent, ShimTileEvent


# We use the packet type field in the packet header to help differentiate the tile
# that the packet came from. Since packet types don't inherently have meaning, we
# assign numerical values to each tile type: core, mem (for core), shimtilem, memtile
class PacketType(IntEnum):
    CORE = 0
    MEM = 1
    SHIMTILE = 2
    MEMTILE = 3


# Number of different trace types
NUM_TRACE_TYPES = len(PacketType)


def _get_port_events(enum_class):
    events = set()
    for i in range(8):
        for type in ["IDLE", "RUNNING", "STALLED", "TLAST"]:
            events.add(getattr(enum_class, f"PORT_{type}_{i}"))
    return events


PortEventCodes = _get_port_events(CoreEvent)
MemTilePortEventCodes = _get_port_events(MemTileEvent)
ShimTilePortEventCodes = _get_port_events(ShimTileEvent)
