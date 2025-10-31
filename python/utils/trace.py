# trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import typing
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.aie import get_target_model
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from enum import IntEnum


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


# fmt: off
PortEventCodes = { CoreEvent.PORT_IDLE_0, CoreEvent.PORT_IDLE_1,
                   CoreEvent.PORT_IDLE_2, CoreEvent.PORT_IDLE_3,
                   CoreEvent.PORT_IDLE_4, CoreEvent.PORT_IDLE_5,
                   CoreEvent.PORT_IDLE_6, CoreEvent.PORT_IDLE_7,
                   CoreEvent.PORT_RUNNING_0, CoreEvent.PORT_RUNNING_1,
                   CoreEvent.PORT_RUNNING_2, CoreEvent.PORT_RUNNING_3,
                   CoreEvent.PORT_RUNNING_4, CoreEvent.PORT_RUNNING_5,
                   CoreEvent.PORT_RUNNING_6, CoreEvent.PORT_RUNNING_7,
                   CoreEvent.PORT_STALLED_0, CoreEvent.PORT_STALLED_1,
                   CoreEvent.PORT_STALLED_2, CoreEvent.PORT_STALLED_3,
                   CoreEvent.PORT_STALLED_4, CoreEvent.PORT_STALLED_5,
                   CoreEvent.PORT_STALLED_6, CoreEvent.PORT_STALLED_7,
                   CoreEvent.PORT_TLAST_0, CoreEvent.PORT_TLAST_1,
                   CoreEvent.PORT_TLAST_2, CoreEvent.PORT_TLAST_3,
                   CoreEvent.PORT_TLAST_4, CoreEvent.PORT_TLAST_5,
                   CoreEvent.PORT_TLAST_6, CoreEvent.PORT_TLAST_7, }

MemTilePortEventCodes = { MemTileEvent.PORT_IDLE_0, MemTileEvent.PORT_IDLE_1,
                   MemTileEvent.PORT_IDLE_2, MemTileEvent.PORT_IDLE_3,
                   MemTileEvent.PORT_IDLE_4, MemTileEvent.PORT_IDLE_5,
                   MemTileEvent.PORT_IDLE_6, MemTileEvent.PORT_IDLE_7,
                   MemTileEvent.PORT_RUNNING_0, MemTileEvent.PORT_RUNNING_1,
                   MemTileEvent.PORT_RUNNING_2, MemTileEvent.PORT_RUNNING_3,
                   MemTileEvent.PORT_RUNNING_4, MemTileEvent.PORT_RUNNING_5,
                   MemTileEvent.PORT_RUNNING_6, MemTileEvent.PORT_RUNNING_7,
                   MemTileEvent.PORT_STALLED_0, MemTileEvent.PORT_STALLED_1,
                   MemTileEvent.PORT_STALLED_2, MemTileEvent.PORT_STALLED_3,
                   MemTileEvent.PORT_STALLED_4, MemTileEvent.PORT_STALLED_5,
                   MemTileEvent.PORT_STALLED_6, MemTileEvent.PORT_STALLED_7,
                   MemTileEvent.PORT_TLAST_0, MemTileEvent.PORT_TLAST_1,
                   MemTileEvent.PORT_TLAST_2, MemTileEvent.PORT_TLAST_3,
                   MemTileEvent.PORT_TLAST_4, MemTileEvent.PORT_TLAST_5,
                   MemTileEvent.PORT_TLAST_6, MemTileEvent.PORT_TLAST_7, }


ShimTilePortEventCodes = { ShimTileEvent.PORT_IDLE_0, ShimTileEvent.PORT_IDLE_1,
                   ShimTileEvent.PORT_IDLE_2, ShimTileEvent.PORT_IDLE_3,
                   ShimTileEvent.PORT_IDLE_4, ShimTileEvent.PORT_IDLE_5,
                   ShimTileEvent.PORT_IDLE_6, ShimTileEvent.PORT_IDLE_7,
                   ShimTileEvent.PORT_RUNNING_0, ShimTileEvent.PORT_RUNNING_1,
                   ShimTileEvent.PORT_RUNNING_2, ShimTileEvent.PORT_RUNNING_3,
                   ShimTileEvent.PORT_RUNNING_4, ShimTileEvent.PORT_RUNNING_5,
                   ShimTileEvent.PORT_RUNNING_6, ShimTileEvent.PORT_RUNNING_7,
                   ShimTileEvent.PORT_STALLED_0, ShimTileEvent.PORT_STALLED_1,
                   ShimTileEvent.PORT_STALLED_2, ShimTileEvent.PORT_STALLED_3,
                   ShimTileEvent.PORT_STALLED_4, ShimTileEvent.PORT_STALLED_5,
                   ShimTileEvent.PORT_STALLED_6, ShimTileEvent.PORT_STALLED_7,
                   ShimTileEvent.PORT_TLAST_0, ShimTileEvent.PORT_TLAST_1,
                   ShimTileEvent.PORT_TLAST_2, ShimTileEvent.PORT_TLAST_3,
                   ShimTileEvent.PORT_TLAST_4, ShimTileEvent.PORT_TLAST_5,
                   ShimTileEvent.PORT_TLAST_6, ShimTileEvent.PORT_TLAST_7, }

# fmt: on


# We use the packet type field in the packet header to help differentiate the tile
# that the packet came from. Since packet types don't inherently have meaning, we
# assign numerical values to each tile type: core, mem (for core), shimtilem, memtile
class PacketType(IntEnum):
    CORE = 0
    MEM = 1
    SHIMTILE = 2
    MEMTILE = 3


class PortEvent(GenericEvent):
    def __init__(self, code, port_number, master=True):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int):
            code = CoreEvent(code)
        assert code in PortEventCodes
        # fmt: off
        self.event_number = (
                 0 if code in { CoreEvent.PORT_IDLE_0,    CoreEvent.PORT_RUNNING_0, 
                                CoreEvent.PORT_STALLED_0, CoreEvent.PORT_TLAST_0    }
            else 1 if code in { CoreEvent.PORT_IDLE_1,    CoreEvent.PORT_RUNNING_1,
                                CoreEvent.PORT_STALLED_1, CoreEvent.PORT_TLAST_1,   }
            else 2 if code in { CoreEvent.PORT_IDLE_2,    CoreEvent.PORT_RUNNING_2,
                                CoreEvent.PORT_STALLED_2, CoreEvent.PORT_TLAST_2    }
            else 3 if code in { CoreEvent.PORT_IDLE_3,    CoreEvent.PORT_RUNNING_3,
                                CoreEvent.PORT_STALLED_3, CoreEvent.PORT_TLAST_3    }
            else 4 if code in { CoreEvent.PORT_IDLE_4,    CoreEvent.PORT_RUNNING_4,
                                CoreEvent.PORT_STALLED_4, CoreEvent.PORT_TLAST_4    }
            else 5 if code in { CoreEvent.PORT_IDLE_5,    CoreEvent.PORT_RUNNING_5,
                                CoreEvent.PORT_STALLED_5, CoreEvent.PORT_TLAST_5    }
            else 6 if code in { CoreEvent.PORT_IDLE_6,    CoreEvent.PORT_RUNNING_6,
                                CoreEvent.PORT_STALLED_6, CoreEvent.PORT_TLAST_6    }
            else 7
        )
        # fmt: on
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
        address = 0x3FF00 if self.event_number < 4 else 0x3FF04
        value = master(self.port_number) if self.master else slave(self.port_number)

        value = (value & 0xFF) << 8 * (self.event_number % 4)

        ret = {0x3FF00: 0, 0x3FF04: 0}
        ret[address] = value

        return ret


class MemTilePortEvent(GenericEvent):
    def __init__(self, code, port_number, master=True):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int):
            code = MemTileEvent(code)
        assert code in MemTilePortEventCodes
        # fmt: off
        self.event_number = (
                 0 if code in { MemTileEvent.PORT_IDLE_0,    MemTileEvent.PORT_RUNNING_0, 
                                MemTileEvent.PORT_STALLED_0, MemTileEvent.PORT_TLAST_0    }
            else 1 if code in { MemTileEvent.PORT_IDLE_1,    MemTileEvent.PORT_RUNNING_1,
                                MemTileEvent.PORT_STALLED_1, MemTileEvent.PORT_TLAST_1,   }
            else 2 if code in { MemTileEvent.PORT_IDLE_2,    MemTileEvent.PORT_RUNNING_2,
                                MemTileEvent.PORT_STALLED_2, MemTileEvent.PORT_TLAST_2    }
            else 3 if code in { MemTileEvent.PORT_IDLE_3,    MemTileEvent.PORT_RUNNING_3,
                                MemTileEvent.PORT_STALLED_3, MemTileEvent.PORT_TLAST_3    }
            else 4 if code in { MemTileEvent.PORT_IDLE_4,    MemTileEvent.PORT_RUNNING_4,
                                MemTileEvent.PORT_STALLED_4, MemTileEvent.PORT_TLAST_4    }
            else 5 if code in { MemTileEvent.PORT_IDLE_5,    MemTileEvent.PORT_RUNNING_5,
                                MemTileEvent.PORT_STALLED_5, MemTileEvent.PORT_TLAST_5    }
            else 6 if code in { MemTileEvent.PORT_IDLE_6,    MemTileEvent.PORT_RUNNING_6,
                                MemTileEvent.PORT_STALLED_6, MemTileEvent.PORT_TLAST_6    }
            else 7
        )
        # fmt: on
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
        address = 0xB0F00 if self.event_number < 4 else 0xB0F04
        value = master(self.port_number) if self.master else slave(self.port_number)

        value = (value & 0xFF) << 8 * (self.event_number % 4)

        ret = {0xB0F00: 0, 0xB0F04: 0}
        ret[address] = value

        return ret


class ShimTilePortEvent(GenericEvent):
    def __init__(self, code, port_number, master=True):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int):
            code = ShimTileEvent(code)
        assert code in ShimTilePortEventCodes
        # fmt: off
        self.event_number = (
                 0 if code in { ShimTileEvent.PORT_IDLE_0,    ShimTileEvent.PORT_RUNNING_0, 
                                ShimTileEvent.PORT_STALLED_0, ShimTileEvent.PORT_TLAST_0    }
            else 1 if code in { ShimTileEvent.PORT_IDLE_1,    ShimTileEvent.PORT_RUNNING_1,
                                ShimTileEvent.PORT_STALLED_1, ShimTileEvent.PORT_TLAST_1,   }
            else 2 if code in { ShimTileEvent.PORT_IDLE_2,    ShimTileEvent.PORT_RUNNING_2,
                                ShimTileEvent.PORT_STALLED_2, ShimTileEvent.PORT_TLAST_2    }
            else 3 if code in { ShimTileEvent.PORT_IDLE_3,    ShimTileEvent.PORT_RUNNING_3,
                                ShimTileEvent.PORT_STALLED_3, ShimTileEvent.PORT_TLAST_3    }
            else 4 if code in { ShimTileEvent.PORT_IDLE_4,    ShimTileEvent.PORT_RUNNING_4,
                                ShimTileEvent.PORT_STALLED_4, ShimTileEvent.PORT_TLAST_4    }
            else 5 if code in { ShimTileEvent.PORT_IDLE_5,    ShimTileEvent.PORT_RUNNING_5,
                                ShimTileEvent.PORT_STALLED_5, ShimTileEvent.PORT_TLAST_5    }
            else 6 if code in { ShimTileEvent.PORT_IDLE_6,    ShimTileEvent.PORT_RUNNING_6,
                                ShimTileEvent.PORT_STALLED_6, ShimTileEvent.PORT_TLAST_6    }
            else 7
        )
        # fmt: on
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
        address = 0x3FF00 if self.event_number < 4 else 0x3FF04
        value = master(self.port_number) if self.master else slave(self.port_number)

        value = (value & 0xFF) << 8 * (self.event_number % 4)

        ret = {0x3FF00: 0, 0x3FF04: 0}
        ret[address] = value

        return ret


# TODO Should be expanded to be check for valid shim tile based on device
# Checks if tile is a shim tile (for now, assumes row 0 is only shim tile, true for aie1/aie2/aie2p)
def isShimTile(tile):
    return int(tile.row) == 0


# TODO Should be expanded to be check for valid shim tile based on device
# Checks if tile is a Mem tile (for now, assumes row 1 is only mem tile, true for aie1/aie2/aie2p)
def isMemTile(tile):
    return int(tile.row) == 1


# TODO Should be expanded to be check for valid shim tile based on device
# Checks if tile is a Core tile (for now, assumes any row > 1 is core tile, true for aie1/aie2/aie2p)
# though we're not checking max row value so this isn't 100% accurate
def isCoreTile(tile):
    return int(tile.row) > 1


def pack4bytes(b3, b2, b1, b0):
    w = (b3 & 0xFF) << 24
    w |= (b2 & 0xFF) << 16
    w |= (b1 & 0xFF) << 8
    w |= (b0 & 0xFF) << 0
    return w


# This function configures the a tile's memory trace unit given a set of configurations as described below:
#
# function arguments:
# * `tile` - ocre tile to configure
# * `start` - start event. We generally use a global broadcast signal to synchronize the start event
#             for multiple cores.
# * `stop` - stop event. We generally use a global broadcast signal to synchronize the stop event for
#            multiple cores.
# * `events` - array of 8 events to trace
# * `enable_packet` - enables putting event data into packets
# * `packet_id` - packet id or flow id used to route the packets through the stream switch
# * `packet_type` - packet type is an arbitrary field but we use it to describe the tile type the
#                   packets are coming from
def configure_coremem_tracing_aie2(
    tile,
    start=MemEvent.TRUE,
    stop=MemEvent.NONE,
    events=[
        MemEvent.DMA_S2MM_0_START_TASK,
        MemEvent.DMA_S2MM_0_FINISHED_BD,
        MemEvent.CONFLICT_DM_BANK_0,
        MemEvent.CONFLICT_DM_BANK_1,
        MemEvent.CONFLICT_DM_BANK_2,
        MemEvent.CONFLICT_DM_BANK_3,
        MemEvent.EDGE_DETECTION_EVENT_0,
        MemEvent.EDGE_DETECTION_EVENT_1,
    ],
    enable_packet=0,
    packet_id=0,
    packet_type=PacketType.MEM,
):
    if isinstance(start, int):
        start = MemEvent(start)
    if isinstance(stop, int):
        stop = MemEvent(stop)
    if len(events) > 8:
        raise RuntimeError(
            f"At most 8 events can be traced at once, have {len(events)}."
        )
    events = (events + [MemEvent.NONE] * 8)[:8]

    # Reorder events so they match the event order for display
    ordered_events = [events[p] for p in [3, 2, 1, 0, 7, 6, 5, 4]]

    # Assure all selected events are valid
    ordered_events = [
        e if isinstance(e, GenericEvent) else GenericEvent(e) for e in ordered_events
    ]

    # Require ports to be specifically given for port events.
    for event in ordered_events:
        if event.code in PortEventCodes and not isinstance(event, PortEvent):
            raise RuntimeError(
                f"Tracing: {event.code.name} is a PortEvent and requires a port to be specified alongside it. \n"
                "To select master port N, specify the event as follows: "
                f"PortEvent(CoreEvent.{event.code.name}, N, master=True), "
                "and analogously with master=False for slave ports. "
                "For example: "
                f"configure_simple_tracing_aie2( ..., events=[PortEvent(CoreEvent.{event.code.name}, 1, master=True)])"
            )

    # 140D0: Trace Control 0
    #          0xAABB---C
    #            AA        <- Event to stop trace capture
    #              BB      <- Event to start trace capture
    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
    # Configure so that "Event 1" (always true) causes tracing to start
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x140D0,
        value=pack4bytes(stop.value, start.value, 0, 0),
    )
    # 0x140D4: Trace Control 1
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x140D4,
        value=((packet_type & 0x7) << 12) | (packet_id & 0x1F) if enable_packet else 0,
    )
    # 0x140E0: Trace Event Group 1  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x140E0,
        value=pack4bytes(*(e.code.value for e in ordered_events[0:4])),
    )
    # 0x140E4: Trace Event Group 2  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x140E4,
        value=pack4bytes(*(e.code.value for e in ordered_events[4:8])),
    )

    # Event specific register writes
    all_reg_writes = {}
    for e in ordered_events:
        reg_writes = e.get_register_writes()
        for addr, value in reg_writes.items():
            if addr in all_reg_writes:
                all_reg_writes[addr] |= value
            else:
                all_reg_writes[addr] = value
    for addr, value in all_reg_writes.items():
        npu_write32(
            column=int(tile.col), row=int(tile.row), address=addr, value=value
        )  # For backwards compatibility, allow integers for start/stop events


# This function configures the a tile's trace unit given a set of configurations as described below:
#
# function arguments:
# * `tile` - ocre tile to configure
# * `start` - start event. We generally use a global broadcast signal to synchronize the start event
#             for multiple cores.
# * `stop` - stop event. We generally use a global broadcast signal to synchronize the stop event for
#            multiple cores.
# * `events` - array of 8 events to trace
# * `enable_packet` - enables putting event data into packets
# * `packet_id` - packet id or flow id used to route the packets through the stream switch
# * `packet_type` - packet type is an arbitrary field but we use it to describe the tile type the
#                   packets are coming from
def configure_coretile_tracing_aie2(
    tile,
    start,
    stop,
    events=[
        CoreEvent.INSTR_EVENT_0,
        CoreEvent.INSTR_EVENT_1,
        CoreEvent.INSTR_VECTOR,
        PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
        PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
        CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
        CoreEvent.INSTR_LOCK_RELEASE_REQ,
        CoreEvent.LOCK_STALL,
    ],
    enable_packet=0,
    packet_id=0,
    packet_type=PacketType.CORE,
):
    # For backwards compatibility, allow integers for start/stop events
    if isinstance(start, int):
        start = CoreEvent(start)
    if isinstance(stop, int):
        stop = CoreEvent(stop)

    # Pad the input so we have exactly 8 events.
    if len(events) > 8:
        raise RuntimeError(
            f"At most 8 events can be traced at once, have {len(events)}."
        )
    events = (events + [CoreEvent.NONE] * 8)[:8]

    # Reorder events so they match the event order for display
    ordered_events = [events[p] for p in [3, 2, 1, 0, 7, 6, 5, 4]]

    # Assure all selected events are valid
    ordered_events = [
        e if isinstance(e, GenericEvent) else GenericEvent(e) for e in ordered_events
    ]

    # Require ports to be specifically given for port events.
    for event in ordered_events:
        if event.code in PortEventCodes and not isinstance(event, PortEvent):
            raise RuntimeError(
                f"Tracing: {event.code.name} is a PortEvent and requires a port to be specified alongside it. \n"
                "To select master port N, specify the event as follows: "
                f"PortEvent(CoreEvent.{event.code.name}, N, master=True), "
                "and analogously with master=False for slave ports. "
                "For example: "
                f"configure_simple_tracing_aie2( ..., events=[PortEvent(CoreEvent.{event.code.name}, 1, master=True)])"
            )

    # 0x340D0: Trace Control 0
    #          0xAABB---C
    #            AA        <- Event to stop trace capture
    #              BB      <- Event to start trace capture
    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
    # Configure so that "Event 1" (always true) causes tracing to start
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340D0,
        value=pack4bytes(stop.value, start.value, 0, 0),
    )
    # 0x340D4: Trace Control 1
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340D4,
        value=((packet_type & 0x7) << 12) | (packet_id & 0x1F) if enable_packet else 0,
    )
    # 0x340E0: Trace Event Group 1  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340E0,
        value=pack4bytes(*(e.code.value for e in ordered_events[0:4])),
    )
    # 0x340E4: Trace Event Group 2  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340E4,
        value=pack4bytes(*(e.code.value for e in ordered_events[4:8])),
    )

    # Event specific register writes
    all_reg_writes = {}
    for e in ordered_events:
        reg_writes = e.get_register_writes()
        for addr, value in reg_writes.items():
            if addr in all_reg_writes:
                all_reg_writes[addr] |= value
            else:
                all_reg_writes[addr] = value
    for addr, value in all_reg_writes.items():
        npu_write32(column=int(tile.col), row=int(tile.row), address=addr, value=value)


# Configures the memtile for tracing given start/stop events, trace events, and optional
# packet config as applicable.
def configure_memtile_tracing_aie2(
    tile,
    start=MemTileEvent.TRUE,
    stop=MemTileEvent.NONE,
    events=[
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),  # master(0)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, 1, True),  # master(1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, 0, False),  # slave(0)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, 1, False),  # slave(1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 2, False),  # slave(2)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 3, False),  # slave(3)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 4, False),  # slave(4)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 5, False),  # slave(5)
    ],
    enable_packet=0,
    packet_id=0,
    packet_type=PacketType.MEMTILE,
):
    # For backwards compatibility, allow integers for start/stop events
    if isinstance(start, int):
        start = MemTileEvent(start)
    if isinstance(stop, int):
        stop = MemTileEvent(stop)

    # Pad the input so we have exactly 8 events.
    if len(events) > 8:
        raise RuntimeError(
            f"At most 8 events can be traced at once, have {len(events)}."
        )
    events = (events + [MemTileEvent.NONE] * 8)[:8]

    # Reorder events so they match the event order for display
    ordered_events = [events[p] for p in [3, 2, 1, 0, 7, 6, 5, 4]]

    # Assure all selected events are valid
    ordered_events = [
        e if isinstance(e, GenericEvent) else GenericEvent(e) for e in ordered_events
    ]

    # Require ports to be specifically given for port events.
    for event in ordered_events:
        if event.code in PortEventCodes and not isinstance(event, PortEvent):
            raise RuntimeError(
                f"Tracing: {event.code.name} is a PortEvent and requires a port to be specified alongside it. \n"
                "To select master port N, specify the event as follows: "
                f"PortEvent(CoreEvent.{event.code.name}, N, master=True), "
                "and analogously with master=False for slave ports. "
                "For example: "
                f"configure_simple_tracing_aie2( ..., events=[PortEvent(CoreEvent.{event.code.name}, 1, master=True)])"
            )

    # 0x340D0: Trace Control 0
    #          0xAABB---C
    #            AA        <- Event to stop trace capture
    #              BB      <- Event to start trace capture
    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
    # Configure so that "Event 1" (always true) causes tracing to start
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x940D0,
        value=pack4bytes(stop.value, start.value, 0, 0),
    )
    # 0x340D4: Trace Control 1
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x940D4,
        value=((packet_type & 0x7) << 12) | (packet_id & 0x1F) if enable_packet else 0,
    )
    # 0x340E0: Trace Event Group 1  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x940E0,
        value=pack4bytes(*(e.code.value for e in ordered_events[0:4])),
    )
    # 0x340E4: Trace Event Group 2  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x940E4,
        value=pack4bytes(*(e.code.value for e in ordered_events[4:8])),
    )

    # Event specific register writes
    all_reg_writes = {}
    for e in ordered_events:
        reg_writes = e.get_register_writes()
        for addr, value in reg_writes.items():
            if addr in all_reg_writes:
                all_reg_writes[addr] |= value
            else:
                all_reg_writes[addr] = value
    for addr, value in all_reg_writes.items():
        npu_write32(
            column=int(tile.col), row=int(tile.row), address=addr, value=value
        )  # For backwards compatibility, allow integers for start/stop events


# Configures the shimtile for tracing given start/stop events, trace events, and optional
# packet config as applicalbe.
def configure_shimtile_tracing_aie2(
    tile,
    start=ShimTileEvent.TRUE,
    stop=ShimTileEvent.NONE,
    events=[
        ShimTileEvent.DMA_S2MM_0_START_TASK,
        ShimTileEvent.DMA_S2MM_1_START_TASK,
        ShimTileEvent.DMA_MM2S_0_START_TASK,
        ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
        ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
        ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
    ],
    enable_packet=0,
    packet_id=0,
    packet_type=PacketType.SHIMTILE,
):
    if isinstance(start, int):
        start = ShimTileEvent(start)
    if isinstance(stop, int):
        stop = ShimTileEvent(stop)

    # Pad the input so we have exactly 8 events.
    if len(events) > 8:
        raise RuntimeError(
            f"At most 8 events can be traced at once, have {len(events)}."
        )
    events = (events + [ShimTileEvent.NONE] * 8)[:8]

    # Reorder events so they match the event order for display
    ordered_events = [events[p] for p in [3, 2, 1, 0, 7, 6, 5, 4]]

    # Assure all selected events are valid
    ordered_events = [
        e if isinstance(e, GenericEvent) else GenericEvent(e) for e in ordered_events
    ]

    # Require ports to be specifically given for port events.
    for event in ordered_events:
        if event.code in PortEventCodes and not isinstance(event, PortEvent):
            raise RuntimeError(
                f"Tracing: {event.code.name} is a PortEvent and requires a port to be specified alongside it. \n"
                "To select master port N, specify the event as follows: "
                f"PortEvent(CoreEvent.{event.code.name}, N, master=True), "
                "and analogously with master=False for slave ports. "
                "For example: "
                f"configure_simple_tracing_aie2( ..., events=[PortEvent(CoreEvent.{event.code.name}, 1, master=True)])"
            )

    # 0x340D0: Trace Control 0
    #          0xAABB---C
    #            AA        <- Event to stop trace capture
    #              BB      <- Event to start trace capture
    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
    # Configure so that "Event 1" (always true) causes tracing to start
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340D0,
        value=pack4bytes(stop.value, start.value, 0, 0),
    )
    # 0x340D4: Trace Control 1
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340D4,
        value=((packet_type & 0x7) << 12) | (packet_id & 0x1F) if enable_packet else 0,
    )
    # 0x340E0: Trace Event Group 1  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340E0,
        value=pack4bytes(*(e.code.value for e in ordered_events[0:4])),
    )
    # 0x340E4: Trace Event Group 2  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340E4,
        value=pack4bytes(*(e.code.value for e in ordered_events[4:8])),
    )

    # Event specific register writes
    all_reg_writes = {}
    for e in ordered_events:
        reg_writes = e.get_register_writes()
        for addr, value in reg_writes.items():
            if addr in all_reg_writes:
                all_reg_writes[addr] |= value
            else:
                all_reg_writes[addr] = value
    for addr, value in all_reg_writes.items():
        npu_write32(column=int(tile.col), row=int(tile.row), address=addr, value=value)


# Configure timer in core tile's memory to reset based on `event`
def configure_timer_ctrl_coremem_aie2(tile, event):
    addr = 0x14000
    eventValue = (event.value & 0x7F) << 8
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=eventValue,
    )


# Configure timer in core tile to reset based on `event`
def configure_timer_ctrl_coretile_aie2(tile, event):
    addr = 0x34000
    eventValue = (event.value & 0x7F) << 8
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=eventValue,
    )


# Configure timer in memtile to reset based on `event`
def configure_timer_ctrl_memtile_aie2(tile, event):
    addr = 0x94000
    eventValue = (event.value & 0x7F) << 8
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=eventValue,
    )


# Configure timer in core tile to reset based on `event`
def configure_timer_ctrl_shimtile_aie2(tile, event):
    addr = 0x34000
    eventValue = (event.value & 0x7F) << 8
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=eventValue,
    )


# Configure broadcast event based on an internal triggered event.
#
# function arguments:
# * `num` - broadcaast number we want to broadcast on
# * `event` - the triggering broadcast event
def configure_broadcast_core_aie2(tile, num, event):
    if isShimTile(tile):
        base_addr = 0x34010
    elif isMemTile(tile):
        base_addr = 0x94010
    elif isCoreTile(tile):
        base_addr = 0x34010
    else:
        raise ValueError(
            "Invalid tile("
            + str(tile.col)
            + ","
            + str(tile.row)
            + "). Check tile coordinates are within a valid range."
        )

    addr = base_addr + num * 4
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=event.value,
    )


# Generate an `event` at the given `tile`. This event can broadcasted and
# use by all tiles in the device to synchronize to.
def configure_event_gen_core_aie2(tile, event):
    addr = 0x34008
    eventValue = event.value & 0x7F
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=eventValue,
    )


# Configure shim tile's DMA for tracing.
# This configures the shim tile / bd to process a specficic `packet id`
# and `packet type`. It also configures the address patch. Note that we
# can call this multiple times for each `packet id`/ `packet type` but
# mapped to the same `ddr_id`, `size`, and `offset` and the packets will
# be written to the output location as they come in for all
# `packet id`/ `packet type` listed
def configure_shimtile_dma_tracing_aie2(
    shim,
    channel=1,
    bd_id=13,
    ddr_id=4,
    size=8192,
    offset=0,
    enable_token=0,
    enable_packet=0,
    packet_id=0,
    packet_type=PacketType.CORE,
    shim_burst_length=64,
):

    dev = shim.parent.attributes["device"]
    tm = get_target_model(dev)

    # Shim has to be a shim tile
    assert tm.is_shim_noc_tile(shim.col, shim.row)

    # configure_shimtile_bd_aie2(shim, channel, bd_id, ddr_id, size, offset, 1, 0, 0)
    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
    # out to host DDR memory
    npu_writebd(
        bd_id=bd_id,
        buffer_length=size,
        buffer_offset=offset,
        enable_packet=enable_packet,
        out_of_order_id=0,
        packet_id=packet_id,
        packet_type=packet_type,
        column=int(shim.col),
        d0_size=0,
        d0_stride=0,
        d0_zero_after=0,
        d0_zero_before=0,
        d1_size=0,
        d1_stride=0,
        d1_zero_after=0,
        d1_zero_before=0,
        d2_size=0,
        d2_stride=0,
        d2_zero_after=0,
        d2_zero_before=0,
        burst_length=shim_burst_length,
        iteration_current=0,
        iteration_size=0,
        iteration_stride=0,
        lock_acq_enable=0,
        lock_acq_id=0,
        lock_acq_val=0,
        lock_rel_id=0,
        lock_rel_val=0,
        next_bd=0,
        row=0,
        use_next_bd=0,
        valid_bd=1,
    )
    addr = (int(shim.col) << tm.get_column_shift()) | (0x1D004 + bd_id * 0x20)
    npu_address_patch(addr=addr, arg_idx=ddr_id, arg_plus=offset)

    # configure S2MM channel
    npu_write32(
        column=int(shim.col),
        row=int(shim.row),
        address=0x1D204 if channel == 0 else 0x1D20C,
        value=((enable_token & 0x1) << 31) | bd_id,
    )


# Wrapper to configure the core tile and shim tile for packet tracing. This does
# the following:
# 1. Configure core tile based on start/ stop, events, and flow id. The flow id
#    needs to be unique per flow.
# 2. Configure timer based on broadcast event (default is 15). This ensures all
#    tiles keying off this event has a synchronized timer so their trace are
#    synchronized. This event is also used as the start event for tracing.
# 3. Configure shim tile to receive this flow and move the data to offset/ size.
#
def configure_coremem_packet_tracing_aie2(
    tile,
    shim,
    flow_id=0,
    bd_id=15,
    size=8192,
    offset=0,
    enable_token=0,
    # brdcst_event=0x7A,  # event 122 - broadcast 15 # TODO
    channel=1,
    ddr_id=4,
    start=MemEvent.BROADCAST_15,
    stop=MemEvent.BROADCAST_14,
    events=[
        MemEvent.DMA_S2MM_0_START_TASK,
        MemEvent.DMA_S2MM_0_FINISHED_BD,
        MemEvent.CONFLICT_DM_BANK_0,
        MemEvent.CONFLICT_DM_BANK_1,
        MemEvent.CONFLICT_DM_BANK_2,
        MemEvent.CONFLICT_DM_BANK_3,
        MemEvent.EDGE_DETECTION_EVENT_0,
        MemEvent.EDGE_DETECTION_EVENT_1,
    ],
    shim_burst_length=64,
):
    configure_coremem_tracing_aie2(
        tile=tile,
        start=start,
        stop=stop,
        events=events,
        enable_packet=1,
        packet_id=flow_id,
        packet_type=PacketType.MEM,
    )
    configure_timer_ctrl_coremem_aie2(tile, start)
    configure_shimtile_dma_tracing_aie2(
        shim=shim,
        channel=channel,
        bd_id=bd_id,
        ddr_id=ddr_id,
        size=size,
        offset=offset,
        enable_token=enable_token,
        enable_packet=1,
        packet_id=flow_id,
        packet_type=PacketType.MEM,
        shim_burst_length=shim_burst_length,
    )


# Wrapper to configure the core tile and shim tile for packet tracing. This does
# the following:
# 1. Configure core tile based on start/ stop, events, and flow id. The flow id
#    needs to be unique per flow.
# 2. Configure timer based on broadcast event (default is 15). This ensures all
#    tiles keying off this event has a synchronized timer so their trace are
#    synchronized. This event is also used as the start event for tracing.
# 3. Configure shim tile to receive this flow and move the data to offset/ size.
#
def configure_coretile_packet_tracing_aie2(
    tile,
    shim,
    flow_id=0,
    bd_id=15,
    size=8192,
    offset=0,
    enable_token=0,
    # brdcst_event=0x7A,  # event 122 - broadcast 15 # TODO
    channel=1,
    ddr_id=4,
    start=CoreEvent.BROADCAST_15,
    stop=CoreEvent.BROADCAST_14,
    events=[
        CoreEvent.INSTR_EVENT_0,
        CoreEvent.INSTR_EVENT_1,
        CoreEvent.INSTR_VECTOR,
        PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
        PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
        CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
        CoreEvent.INSTR_LOCK_RELEASE_REQ,
        CoreEvent.LOCK_STALL,
    ],
    shim_burst_length=64,
):
    configure_coretile_tracing_aie2(
        tile=tile,
        start=start,
        stop=stop,
        events=events,
        enable_packet=1,
        packet_id=flow_id,
        packet_type=PacketType.CORE,
    )
    configure_timer_ctrl_coretile_aie2(tile, start)
    configure_shimtile_dma_tracing_aie2(
        shim=shim,
        channel=channel,
        bd_id=bd_id,
        ddr_id=ddr_id,
        size=size,
        offset=offset,
        enable_token=enable_token,
        enable_packet=1,
        packet_id=flow_id,
        packet_type=PacketType.CORE,
        shim_burst_length=shim_burst_length,
    )


# Configures mem tile for packet trcing. This is very similar to configure_coretile_packet_tracing_aie2
# and maybe they can be combined if we pass the tile type to select the correct address offsets.
# As it stands, we call configure_memtile_tracing_aie2 and configure_timer_ctrl_memtile_aie2 instead
# of the core tile variants. The default events we care about are also different for the memtile.
def configure_memtile_packet_tracing_aie2(
    tile,
    shim,
    flow_id=0,
    bd_id=15,
    size=8192,
    offset=0,
    enable_token=0,
    # brdcst_event=0x9D,  # event 157 - broadcast 15
    channel=1,
    ddr_id=4,
    start=MemTileEvent.BROADCAST_15,
    stop=MemTileEvent.BROADCAST_14,
    events=[
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),  # master(0)
        # MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, 1, True),   # master(1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, 14, False),  # slave(14/ north1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, 0, False),  # slave(0)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, 1, False),  # slave(1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 2, False),  # slave(2)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 3, False),  # slave(3)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 4, False),  # slave(4)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 5, False),  # slave(5)
        # MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 13, False),  # slave(13/ north0)
        # MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 14, False),  # slave(14/ north1)
        # MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 17, False),  # slave(17/ trace)
        # MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 8, True),  # masteer(9/ south1)
    ],
    shim_burst_length=64,
):
    configure_memtile_tracing_aie2(
        tile=tile,
        start=start,
        stop=stop,
        events=events,
        enable_packet=1,
        packet_id=flow_id,
        packet_type=PacketType.MEMTILE,
    )
    configure_timer_ctrl_memtile_aie2(tile, start)
    configure_shimtile_dma_tracing_aie2(
        shim,
        channel,
        bd_id,
        ddr_id,
        size,
        offset,
        enable_token,
        1,
        flow_id,
        PacketType.MEMTILE,
        shim_burst_length=shim_burst_length,
    )


# Configures shim tile for packet tracing. This is very simila rot configure_coretile_packet_tracing_aie2
# and maybe they can be combined if we pass the tile type to select the correct address offsets.
# As it stands, we call configure_shimtile_tracing_aie2 and configure_timer_ctrl_memtile_aie2 instead
# of the core tile variants. The default events we care about are also different for the memtile.
def configure_shimtile_packet_tracing_aie2(
    tile,
    shim,
    flow_id=0,
    bd_id=15,
    size=8192,
    offset=0,
    enable_token=0,
    # brdcst_event=0x9D,  # event 157 - broadcast 15
    channel=1,
    ddr_id=4,
    start=ShimTileEvent.USER_EVENT_1,
    stop=ShimTileEvent.USER_EVENT_0,
    events=[
        ShimTileEvent.DMA_S2MM_0_START_TASK,
        ShimTileEvent.DMA_S2MM_1_START_TASK,
        ShimTileEvent.DMA_MM2S_0_START_TASK,
        ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
        ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
        ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
    ],
    shim_burst_length=64,
):
    configure_shimtile_tracing_aie2(
        tile=tile,
        start=start,
        stop=stop,
        events=events,
        enable_packet=1,
        packet_id=flow_id,
        packet_type=PacketType.SHIMTILE,
    )
    configure_timer_ctrl_shimtile_aie2(tile, start)
    configure_shimtile_dma_tracing_aie2(
        shim,
        channel,
        bd_id,
        ddr_id,
        size,
        offset,
        enable_token,
        1,
        flow_id,
        PacketType.SHIMTILE,
        shim_burst_length=shim_burst_length,
    )


# Wrapper around packeflows to itereate over tiles_to_trace and route them to the shim
# for outputing the trace to L3 memory. This uses default values for the packet id
# that increases for each tile we trace, starting with 1. This should match the tile
# trace config that's set by configure_coretile_packet_tracing_aie2.
#
# NOTE: that because we do it this way, we inherently cannot trace more than 31 tiles.
#
# Function arguments:
# * `tiles to trace` - array of tiles to trace
# * `shim tile` - Single shim tile to configure for writing trace packets to DDR
def configure_packet_tracing_flow(tiles_to_trace, shim):

    exist_traces = []
    for i in range(len(tiles_to_trace)):

        if tiles_to_trace[i] not in exist_traces:
            packetflow(
                i + 1,
                tiles_to_trace[i],
                WireBundle.Trace,
                0,
                dests={"dest": shim, "port": WireBundle.DMA, "channel": 1},
                keep_pkt_header=True,
            )

            exist_traces.append(tiles_to_trace[i])
        else:
            # Ct's memory trace?
            packetflow(
                i + 1,
                tiles_to_trace[i],
                WireBundle.Trace,
                1,
                dests={"dest": shim, "port": WireBundle.DMA, "channel": 1},
                keep_pkt_header=True,
            )


# Configure the shim tile to support packet tracing via:
# 1. Set an event generation to create a custom user event 1 (127, 0x7f)
# 2. Custom event also triggers a broadcast event (by default broadcast 15)
# 3. Custom event also resets timer (will be true for all tiles) so all timers are synchronized
# The actual shim dma config is done via configure_shimtile_tracing_aie2 but this tends to be done
# for each tile we're tracing.
#
# Function arguments:
# * `brdcst_num` - which broadcast number to use (1-15)
# * `user_event` - Which user event do we want to generate which will be used to reset local timers
#                  and be broadcasted out on the `broadcast_num`
def configure_shim_trace_start_aie2(
    shim,
    brdcst_num=15,
    user_event=ShimTileEvent.USER_EVENT_1,  # 0x7F, 127: user even t#1
):
    configure_timer_ctrl_coretile_aie2(
        shim, user_event
    )  # TODO: should have call configure_timer_ctrl_shimtile_aie2() instead?
    configure_broadcast_core_aie2(shim, brdcst_num, user_event)
    configure_event_gen_core_aie2(shim, user_event)


# Generate a done event (broadcasted shim user event) that the other tile will use as stop event
def gen_trace_done_aie2(
    shim,
    brdcst_num=14,
    user_event=ShimTileEvent.USER_EVENT_0,  # 0x7E, 126
):
    configure_broadcast_core_aie2(shim, brdcst_num, user_event)
    configure_event_gen_core_aie2(shim, user_event)


# This wrapper function iterates over the `tiles_to_trace` array and calls the right version of
# `configure_*tile_packet_tracing_aie2`. A key distinction is made to choose the right start and stop
# event depending on the tile type. We pass in 3 sets of optional event arguments that allows them
# to be customized depending on the tile type.
#
# NOTE: We configure the shimdma to bd 15 and channel 1 be default which has the potential to
# conflict if that channel or bd is used by the lowering steps to configure the shim data movers.
# We also configure each tile in the array tiles_to_trace with a incrementing packet id which matches
# the packet ids used by `configure_packet_tracing_flow`.
#
# * `tiles to trace` - array of tiles to trace
# * `shim tile` - Single shim tile to configure for writing trace packets to DDR
# * `size` - trace buffer size (in bytes)
# * `offset` - offest (in bytes) where trace buffer data should begin. By default, this is 0 but can be >0 if we share a buffer with an output.
# * `enable_token` - enable token generation for shimdma. Not recommended since we generally have our dma size > trace data size which we don't always know how big it needs to be.
# * `ddr_id` - which XRT buffer to use where 0 -> group_id(3) ... 4 -> group_id(7). We generally put trace last so we use ddr_id=4.
# * `start_user_event` - which user event do we use as a start event
# * `stop_user_event` - which user event do we use as a stop event
# * `start_broadcast_num` - which broadcast number do we send the start user event
# * `stop_broadcast_num` - which broadcast number do we send the stop user event
# * `coretile_events` - which 8 events do we use for all coretiles in array
# * `memtile_events` - which 8 events do we use for all memtiles in array
# * `shimtile_events` - which 8 events do we use for all shimtiles in array
# * `shim_burst_length` - burst size for shim dma. Default is 64B but can be
# *                        be set to 64B, 128B, 256B, and 512B
def configure_packet_tracing_aie2(
    tiles_to_trace,
    shim,
    trace_size,
    trace_offset=0,
    enable_token=0,
    ddr_id=4,
    start_user_event=ShimTileEvent.USER_EVENT_1,
    stop_user_event=ShimTileEvent.USER_EVENT_0,
    start_broadcast_num=15,
    stop_broadcast_num=14,
    coretile_events=[
        CoreEvent.INSTR_EVENT_0,
        CoreEvent.INSTR_EVENT_1,
        CoreEvent.INSTR_VECTOR,
        PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
        PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
        CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
        CoreEvent.INSTR_LOCK_RELEASE_REQ,
        CoreEvent.LOCK_STALL,
    ],
    memtile_events=[
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),  # master(0)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, 14, False),  # slave(14/ north1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, 0, False),  # slave(0)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, 1, False),  # slave(1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 2, False),  # slave(2)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 3, False),  # slave(3)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 4, False),  # slave(4)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 5, False),  # slave(5)
    ],
    shimtile_events=[
        ShimTileEvent.DMA_S2MM_0_START_TASK,
        ShimTileEvent.DMA_S2MM_1_START_TASK,
        ShimTileEvent.DMA_MM2S_0_START_TASK,
        ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
        ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
        ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
    ],
    coremem_events=[
        MemEvent.DMA_S2MM_0_START_TASK,
        MemEvent.DMA_MM2S_0_START_TASK,
        MemEvent.CONFLICT_DM_BANK_0,
        MemEvent.CONFLICT_DM_BANK_1,
        MemEvent.CONFLICT_DM_BANK_2,
        MemEvent.CONFLICT_DM_BANK_3,
        MemEvent.EDGE_DETECTION_EVENT_0,
        MemEvent.EDGE_DETECTION_EVENT_1,
    ],
    shim_burst_length=64,
):

    if coretile_events == None:
        coretile_events = [
            CoreEvent.INSTR_EVENT_0,
            CoreEvent.INSTR_EVENT_1,
            CoreEvent.INSTR_VECTOR,
            PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
            PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
            CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
            CoreEvent.INSTR_LOCK_RELEASE_REQ,
            CoreEvent.LOCK_STALL,
        ]
    if memtile_events == None:
        memtile_events = [
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),  # master(0)
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_1, 14, False
            ),  # slave(14/ north1)
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, 0, False),  # slave(0)
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, 1, False),  # slave(1)
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 2, False),  # slave(2)
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 3, False),  # slave(3)
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 4, False),  # slave(4)
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 5, False),  # slave(5)
        ]
    if shimtile_events == None:
        shimtile_events = [
            ShimTileEvent.DMA_S2MM_0_START_TASK,
            ShimTileEvent.DMA_S2MM_1_START_TASK,
            ShimTileEvent.DMA_MM2S_0_START_TASK,
            ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
            ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
            ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
            ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
            ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
        ]
    if coremem_events == None:
        coremem_events = [
            MemEvent.DMA_S2MM_0_START_TASK,
            MemEvent.DMA_MM2S_0_START_TASK,
            MemEvent.CONFLICT_DM_BANK_0,
            MemEvent.CONFLICT_DM_BANK_1,
            MemEvent.CONFLICT_DM_BANK_2,
            MemEvent.CONFLICT_DM_BANK_3,
            MemEvent.EDGE_DETECTION_EVENT_0,
            MemEvent.EDGE_DETECTION_EVENT_1,
        ]

    start_core_mem_broadcast_event = MemEvent(107 + start_broadcast_num)
    stop_core_mem_broadcast_event = MemEvent(107 + stop_broadcast_num)
    start_core_broadcast_event = CoreEvent(107 + start_broadcast_num)
    stop_core_broadcast_event = CoreEvent(107 + stop_broadcast_num)
    start_memtile_broadcast_event = MemTileEvent(142 + start_broadcast_num)
    stop_memtile_broadcast_event = MemTileEvent(142 + stop_broadcast_num)
    start_shimtile_broadcast_event = ShimTileEvent(110 + start_broadcast_num)
    stop_shimtile_broadcast_event = ShimTileEvent(110 + stop_broadcast_num)

    exist_core_tile_traces = []
    for i in range(len(tiles_to_trace)):
        if isShimTile(tiles_to_trace[i]):
            if tiles_to_trace[i] == shim:
                configure_shimtile_tracing_aie2(
                    tile=tiles_to_trace[i],
                    start=start_user_event,
                    stop=stop_user_event,
                    events=shimtile_events,
                    enable_packet=1,
                    packet_id=i + 1,
                    packet_type=PacketType.SHIMTILE,
                )
                configure_timer_ctrl_shimtile_aie2(tiles_to_trace[i], start_user_event)
            else:
                configure_shimtile_tracing_aie2(
                    tile=tiles_to_trace[i],
                    start=start_shimtile_broadcast_event,
                    stop=stop_shimtile_broadcast_event,
                    events=shimtile_events,
                    enable_packet=1,
                    packet_id=i + 1,
                    packet_type=PacketType.SHIMTILE,
                )
                configure_timer_ctrl_shimtile_aie2(
                    tiles_to_trace[i], start_shimtile_broadcast_event
                )
        elif isMemTile(tiles_to_trace[i]):
            configure_memtile_tracing_aie2(
                tile=tiles_to_trace[i],
                start=start_memtile_broadcast_event,
                stop=stop_memtile_broadcast_event,
                events=memtile_events,
                enable_packet=1,
                packet_id=i + 1,
                packet_type=PacketType.MEMTILE,
            )
            configure_timer_ctrl_memtile_aie2(
                tiles_to_trace[i], start_memtile_broadcast_event
            )
        elif isCoreTile(tiles_to_trace[i]):
            if tiles_to_trace[i] not in exist_core_tile_traces:
                configure_coretile_tracing_aie2(
                    tile=tiles_to_trace[i],
                    start=start_core_broadcast_event,
                    stop=stop_core_broadcast_event,
                    events=coretile_events,
                    enable_packet=1,
                    packet_id=i + 1,
                    packet_type=PacketType.CORE,
                )
                configure_timer_ctrl_coretile_aie2(
                    tiles_to_trace[i], start_core_broadcast_event
                )
                exist_core_tile_traces.append(tiles_to_trace[i])
            else:
                configure_coremem_tracing_aie2(
                    tile=tiles_to_trace[i],
                    start=start_core_mem_broadcast_event,
                    stop=stop_core_mem_broadcast_event,
                    events=coremem_events,
                    enable_packet=1,
                    packet_id=i + 1,
                    packet_type=PacketType.MEM,
                )
                configure_timer_ctrl_coremem_aie2(
                    tiles_to_trace[i], start_core_mem_broadcast_event
                )
        else:
            raise ValueError(
                "Invalid tile("
                + str(tiles_to_trace[i].col)
                + ","
                + str(tiles_to_trace[i].row)
                + "). Check tile coordinates are within a valid range."
            )
    configure_shimtile_dma_tracing_aie2(
        shim=shim,
        channel=1,
        bd_id=15,
        ddr_id=ddr_id,
        size=trace_size,
        offset=trace_offset,
        enable_token=enable_token,
        enable_packet=1,
        shim_burst_length=shim_burst_length,
    )

    configure_shim_trace_start_aie2(shim, start_broadcast_num, start_user_event)


# **** DEPRECATED ****
#
# This does a simple circuit switched trace config for a given tile
# and shim. Since we're not doing packete switching, we're not synchronizing
# any timers. This works fine for a trace of a single tile though it does use
# a stream for routing the trace (which is the same as multi-tile tracing
# except that can be shared with trace packets)
#

#
# Configure tracing, see https://github.com/Xilinx/mlir-aie/blob/resnet/docs/Tracing.md
# This is a very simple model of tracing, which has some big assumptions:
# 1) Trace data is collected over circuit switched connections, not packet-switched
# 2) A ShimDMA S2MM channel is dedicated to the trace data
# 3) Trace data is small enough to fit in a fixed-size buffer, which is collected with the
# outputs of the design
# 4) The usual model of '2 inputs, 1 output' is followed, and the
# trace data is appended to the other outputs

# tile: The tile we're tracing
# shim: The shim tile to output data with.
# bd_id: The BD in the shim tile to use.
# channel: The S2MM channel to use (0 or 1).
# size: The size of the trace data
# offset: The offset of the trace data in the (single) output buffer.
# start: The event number to start tracing on
# stop: The event number to stop tracing on
# events: A list of events to trace.  Up to 8 events are allowed in aie2, more are ignored

# Some events:
# TRUE                       (0x01)
# STREAM_STALL               (0x18)
# LOCK_STALL                 (0x1A)
# EVENTS_CORE_INSTR_EVENT_1  (0x22)
# EVENTS_CORE_INSTR_EVENT_0  (0x21)
# INSTR_VECTOR               (0x25)  Core executes a vecotr MAC, ADD or compare instruction
# INSTR_LOCK_ACQUIRE_REQ     (0x2C)  Core executes a lock acquire instruction
# INSTR_LOCK_RELEASE_REQ     (0x2D)  Core executes a lock release instruction
# EVENTS_CORE_PORT_RUNNING_1 (0x4F)
# EVENTS_CORE_PORT_RUNNING_0 (0x4B)


# Event numbers should be less than 128.
# Big assumption: The bd_id and channel are unused.  If they are used by something else, then
# everything will probably break.


def configure_simple_tracing_aie2(
    tile,
    shim,
    channel=1,
    bd_id=13,
    ddr_id=4,
    size=8192,
    offset=0,
    start=CoreEvent.TRUE,
    stop=CoreEvent.NONE,  # NOTE: No stop event can cause errors in trace generation
    events=[
        CoreEvent.INSTR_EVENT_0,
        CoreEvent.INSTR_EVENT_1,
        CoreEvent.INSTR_VECTOR,
        PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
        PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
        CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
        CoreEvent.INSTR_LOCK_RELEASE_REQ,
        CoreEvent.LOCK_STALL,
    ],
):
    configure_coretile_tracing_aie2(tile, start, stop, events)
    configure_shimtile_dma_tracing_aie2(shim, channel, bd_id, ddr_id, size, offset)
