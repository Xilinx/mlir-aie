# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
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

from aie.dialects.aie import WireBundle, DMAChannelDir


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
    if "xcvc1902" in device:
        return aie
    elif "npu2p" in device:
        return aie2p
    else:
        return aie2


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
    """
    Base class for port monitoring events.

    Port events (PORT_RUNNING_N, PORT_IDLE_N, PORT_STALLED_N, PORT_TLAST_N) monitor
    activity on stream switch ports. The suffix N (0-7) determines which hardware
    monitor slot is configured to watch the specified physical port.

    Args:
        code: The port event (e.g., CoreEvent.PORT_RUNNING_0). The suffix determines
              which monitor slot (0-7) is configured.
        port: Port bundle type (WireBundle.DMA, WireBundle.North, etc.)
        channel: Channel number within the bundle (e.g., 0 for DMA channel 0)
        master: True for input to tile (S2MM), False for output from tile (MM2S)

    Example:
        # Monitor DMA channel 0 input with PORT_RUNNING_0
        PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True)

        # Monitor DMA channel 1 output with PORT_RUNNING_1
        PortEvent(CoreEvent.PORT_RUNNING_1, port=WireBundle.DMA, channel=1, master=False)
    """

    def __init__(
        self,
        code,
        port: WireBundle,
        channel: int,
        master: bool = True,
        enum_class=None,
        valid_codes=None,
    ):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int) and enum_class:
            code = enum_class(code)
        if valid_codes:
            assert code in valid_codes

        # Extract slot number from event name: PORT_RUNNING_0 -> 0
        self.slot = int(code.name.split("_")[-1])
        self.port = port
        self.channel = channel
        self.master = master
        super().__init__(code)

    @property
    def direction(self) -> DMAChannelDir:
        """Get direction as DMAChannelDir enum."""
        return DMAChannelDir.S2MM if self.master else DMAChannelDir.MM2S


class PortEvent(BasePortEvent):
    """
    Configure a port monitor slot for core tile tracing.

    Example:
        # Monitor DMA channel 0 input with PORT_RUNNING_0
        PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True)
    """

    def __init__(
        self,
        code,
        port: WireBundle,
        channel: int,
        master: bool = True,
    ):
        super().__init__(
            code,
            port=port,
            channel=channel,
            master=master,
            enum_class=CoreEvent,
            valid_codes=PortEventCodes,
        )


class MemTilePortEvent(BasePortEvent):
    """
    Configure a port monitor slot for mem tile tracing.

    Example:
        # Monitor DMA channel 0 output with PORT_RUNNING_0
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=False)
    """

    def __init__(
        self,
        code,
        port: WireBundle,
        channel: int,
        master: bool = True,
    ):
        super().__init__(
            code,
            port=port,
            channel=channel,
            master=master,
            enum_class=MemTileEvent,
            valid_codes=MemTilePortEventCodes,
        )


class ShimTilePortEvent(BasePortEvent):
    """
    Configure a port monitor slot for shim tile tracing.

    Example:
        # Monitor South port channel 2 input with PORT_RUNNING_0
        ShimTilePortEvent(ShimTileEvent.PORT_RUNNING_0, port=WireBundle.South, channel=2, master=True)
    """

    def __init__(
        self,
        code,
        port: WireBundle,
        channel: int,
        master: bool = True,
    ):
        super().__init__(
            code,
            port=port,
            channel=channel,
            master=master,
            enum_class=ShimTileEvent,
            valid_codes=ShimTilePortEventCodes,
        )
