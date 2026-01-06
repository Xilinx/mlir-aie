# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
#
"""Trace events enumerations for AIE architectures.

Available modules:
- aie: AIE1 architecture events
- aie2: AIE2/AIEML architecture events
- aie2p: AIE2P architecture events
"""
from enum import IntEnum
import typing

from . import aie
from . import aie2
from . import aie2p

from .aie2 import (
    CoreEvent,
    MemEvent,
    ShimTileEvent,
    MemTileEvent,
)


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


def get_events_for_device(device: str):
    if "npu1" in device:
        return aie
    elif "npu2p" in device:
        return aie2p
    else:
        return aie2p


def _get_port_events(enum_class):
    events = set()
    for i in range(8):
        for type in ["IDLE", "RUNNING", "STALLED", "TLAST"]:
            events.add(getattr(enum_class, f"PORT_{type}_{i}"))
    return events


PortEventCodes = _get_port_events(CoreEvent)
MemTilePortEventCodes = _get_port_events(MemTileEvent)
ShimTilePortEventCodes = _get_port_events(ShimTileEvent)


class GenericEvent:
    def __init__(
        self, code: typing.Union[CoreEvent, MemEvent, ShimTileEvent, MemTileEvent]
    ):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int):
            code = CoreEvent(code)
        self.code: typing.Union[CoreEvent, MemEvent, ShimTileEvent, MemTileEvent] = code

    def get_register_writes(self):
        """
        Sub-classes for specific events that require writing to a specific
        register should overwrite this method to return a dicitionary
        address -> register value.

        Note that if multiple event(-types) request writing to the same
        register, their writes will be ORed together. (This makes sense if
        configuration requires only writing some bits of the whole register.)
        """
        return {}


class BasePortEvent(GenericEvent):
    def __init__(
        self, code, port_number, master=True, enum_class=None, valid_codes=None
    ):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int) and enum_class:
            code = enum_class(code)
        if valid_codes:
            assert code in valid_codes

        self.event_number = int(code.name.split("_")[-1])
        self.port_number = port_number
        self.master = master
        super().__init__(code)

    def get_register_writes(self):
        def master(port):
            return port | (1 << 5)

        def slave(port):
            return port

        # 0x3FF00: Stream switch event port selection 0
        # 0x3FF04: Stream switch event port selection 1
        base_addr = self.get_base_address()
        address = base_addr if self.event_number < 4 else base_addr + 4
        value = master(self.port_number) if self.master else slave(self.port_number)

        value = (value & 0xFF) << 8 * (self.event_number % 4)

        ret = {base_addr: 0, base_addr + 4: 0}
        ret[address] = value

        return ret

    def get_base_address(self):
        raise NotImplementedError


class PortEvent(BasePortEvent):
    def __init__(self, code, port_number, master=True):
        super().__init__(code, port_number, master, CoreEvent, PortEventCodes)

    def get_base_address(self):
        return 0x3FF00


class MemTilePortEvent(BasePortEvent):
    def __init__(self, code, port_number, master=True):
        super().__init__(code, port_number, master, MemTileEvent, MemTilePortEventCodes)

    def get_base_address(self):
        return 0xB0F00


class ShimTilePortEvent(BasePortEvent):
    def __init__(self, code, port_number, master=True):
        super().__init__(
            code, port_number, master, ShimTileEvent, ShimTilePortEventCodes
        )

    def get_base_address(self):
        return 0x3FF00
