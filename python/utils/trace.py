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
from aie.utils.trace_events_enum import CoreEvent, MemEvent, PLEvent, MemTileEvent
from enum import IntEnum


class GenericEvent:
    def __init__(self, code: typing.Union[CoreEvent, MemEvent, PLEvent, MemTileEvent]):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int):
            code = CoreEvent(code)
        self.code: typing.Union[CoreEvent, MemEvent, PLEvent, MemTileEvent] = code

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


# fmt: on

class PacketType(IntEnum):
    CORE = 0
    MEM = 1
    SHIMTILE = 2
    MEMTILE = 3

class PortEvent(GenericEvent):
    def __init__(self, code, port_number, master=True):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int):
            code = MemTileEvent(code)
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

def extract_trace(out_buf, out_buf_shape, out_buf_dtype, trace_size):
    trace_size_words = trace_size // 4
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint32)
    output_prefix = (
        out_buf_flat[:-trace_size_words].view(out_buf_dtype).reshape(out_buf_shape)
    )
    trace_suffix = out_buf_flat[-trace_size_words:]
    return output_prefix, trace_suffix


def write_out_trace(trace, file_name):
    out_str = "\n".join(f"{i:0{8}x}" for i in trace if i != 0)
    with open(file_name, "w") as f:
        f.write(out_str)


def pack4bytes(b3, b2, b1, b0):
    w = (b3 & 0xFF) << 24
    w |= (b2 & 0xFF) << 16
    w |= (b1 & 0xFF) << 8
    w |= (b0 & 0xFF) << 0
    return w



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


def configure_coretile_tracing_aie2(
    tile,
    start=CoreEvent.TRUE,
    stop=CoreEvent.NONE,
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
    packet_type=PacketType.CORE
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
        value= ((packet_type & 0x7) << 12) | (packet_id & 0x1F) if enable_packet else 0
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
# packet config as applicalbe.
def configure_memtile_tracing_aie2(
    tile,
    start=MemTileEvent.TRUE,
    stop=MemTileEvent.NONE,
    events=[
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),   # master(0)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, 1, True),   # master(1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, 0, False),  # slave(0)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, 1, False),  # slave(1)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 2, False),  # slave(2)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 3, False),  # slave(3)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 4, False),  # slave(4)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 5, False),  # slave(5)
    ],
    enable_packet=0,
    packet_id=0,
    packet_type=PacketType.MEMTILE
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
        value= ((packet_type & 0x7) << 12) | (packet_id & 0x1F) if enable_packet else 0
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
        npu_write32(column=int(tile.col), row=int(tile.row), address=addr, value=value)


# Configure timer in core tile to reset based on `event`
def configure_timer_ctrl_core_aie2(tile, event):
    addr = 0x34000
    event = (event & 0x7F) << 8
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=event,
    )


# Configure timer in memtile to reset based on `event`
def configure_timer_ctrl_memtile_aie2(tile, event):
    addr = 0x94000
    event = (event & 0x7F) << 8
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=event,
    )


# Configure broadcast event based on an internal triggered event.
# `num` is the broadcaast number we want to broadcast on 
# and `event` is the triggering broadcast event.
def configure_broadcast_core_aie2(tile, num, event):
    addr = 0x34010 + num * 4
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=event,
    )


# Create an event generation at the shim tile
# This is used to create a custom event to synchronize over
def configure_event_gen_core_aie2(tile, event):
    addr = 0x34008
    event = event & 0x7F
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=addr,
        value=event,
    )


# Configure shim tile for tracing.
# This configures the shim tile / bd to process a specficic packet id and packet type.
# It also configures the address patch. 
def configure_shimtile_tracing_aie2(
    shim,
    channel=1,
    bd_id=13,
    ddr_id=2,
    size=8192,
    offset=0,
	enable_token=0,
    enable_packet=0,
    packet_id=0,
    packet_type=PacketType.CORE,
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
        d1_size=0,
        d1_stride=0,
        d2_stride=0,
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
        value= ((enable_token & 0x1) << 31) | bd_id,
    )


# def configure_shimtile_bd_aie2(
#     shim,
#     channel=1,
#     bd_id=13,
#     ddr_id=2,
#     size=8192,
#     offset=0,
#     enable_packet=0,
#     packet_id=0,
#     packet_type=0
# ):
#     npu_writebd(
#         bd_id=bd_id,
#         buffer_length=size,
#         buffer_offset=offset,
#         enable_packet=enable_packet,
#         out_of_order_id=0,
#         packet_id=packet_id,
#         packet_type=packet_type,
#         column=int(shim.col),
#         d0_size=0,
#         d0_stride=0,
#         d1_size=0,
#         d1_stride=0,
#         d2_stride=0,
#         iteration_current=0,
#         iteration_size=0,
#         iteration_stride=0,
#         lock_acq_enable=0,
#         lock_acq_id=0,
#         lock_acq_val=0,
#         lock_rel_id=0,
#         lock_rel_val=0,
#         next_bd=0,
#         row=0,
#         use_next_bd=0,
#         valid_bd=1,
#     )
#     addr = (int(shim.col) << tm.get_column_shift()) | (0x1D004 + bd_id * 0x20)
#     npu_address_patch(addr=addr, arg_idx=ddr_id, arg_plus=offset)


# This does a simple circuit switched trace config for a given tile
# and shim. Since we're not doing packete switching, we're not synchronizing 
# any timers. This works fine for a trace of a single tile though it does use 
# a stream for routing the trace (which is the same as multi-tile tracing
# except that can be shared with trace packets)
def configure_simple_tracing_aie2(
    tile,
    shim,
    channel=1,
    bd_id=13,
    ddr_id=2,
    size=8192,
    offset=0,
    start=CoreEvent.TRUE,
    stop=CoreEvent.NONE,
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
    configure_shimtile_tracing_aie2(shim, channel, bd_id, ddr_id, size, offset)


# Wrapper to configure the core tile and shim tile for packet tracing. This does 
# the following:
# 1. Configure core tile based on start/ stop, events, and flow id. The flow id
#    needs to be unique per flow.
# 2. Configure timer based on broadcast event (default is 15). This ensures all
#    tiles keying off this event has a synchronized timer so their trace are
#    synchronized. This event is also used as the start event for tracing.
# 3. Configure shim tile to receive this flow and move the data to offset/ size.
# 
def configure_core_packet_tracing_aie2(
    tile,
    shim,
    flow_id=0,
    bd_id=15,
    size=8192,
    offset=0,
	enable_token=0,
    brdcst_event=0x7a, # event 122 - broadcast 15
    channel=1,
    ddr_id=2,
    stop=CoreEvent.NONE,
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
    configure_coretile_tracing_aie2(tile, brdcst_event, stop, events, 1, flow_id, PacketType.CORE)
    configure_timer_ctrl_core_aie2(tile, brdcst_event)
    configure_shimtile_tracing_aie2(shim, channel, bd_id, ddr_id, size, offset, enable_token,
                                    1, flow_id, PacketType.CORE)


# Configures mem tile for packet trcing. This is very simila rot configure_core_packet_tracing_aie2
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
    brdcst_event=0x9d, # event 157 - broadcast 15
    channel=1,
    ddr_id=2,
    stop=MemTileEvent.NONE,
    events=[
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),   # master(0)
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
):
    configure_memtile_tracing_aie2(tile, brdcst_event, stop, events, 1, flow_id, PacketType.MEMTILE)
    configure_timer_ctrl_memtile_aie2(tile, brdcst_event)
    configure_shimtile_tracing_aie2(shim, channel, bd_id, ddr_id, size, offset, enable_token,
                                    1, flow_id, PacketType.MEMTILE)


# Wrapper around packeflows to itereate over tiles to trace and route them
# to the shim for outputing the trace to L3 memory. This uses default values for the packet id
# that increases for each tile we trace. This should match the tile trace config that's set by
# configure_core_packet_tracing_aie2
def configure_packet_tracing_flow(
    tiles_to_trace, 
    shim
):
    for i in range(len(tiles_to_trace)):
        packetflow(i+1, tiles_to_trace[i], WireBundle.Trace, 0, shim, WireBundle.DMA, 1, keep_pkt_header=True)
    

# Configure the shim tile to support packet tracing via:
# 1. Set an event generation to create a custom user event 1 (127, 0x7f)
# 2. Custom event also triggers a broadcast event (by default broadcast 15)
# 3. Custom event also resets timer (will be true for all tiles) so all timers are synchronized
# The actual shim dma config is done via configure_shimtile_tracing_aie2 but this tends to be done
# for each tile we're tracing. 
def configure_shim_packet_tracing_aie2(
    shim,
    brdcst_num=15,
    user_event=0x7f, # 127: user even t#1
):
    configure_timer_ctrl_core_aie2(shim, user_event)
    configure_broadcast_core_aie2(shim, brdcst_num, user_event)
    configure_event_gen_core_aie2(shim, user_event)


# Wrapper to iterate over tiles to trace and configure their default packet tracing config
# along with the shim config for packet tracing
def configure_packet_tracing_aie2(
    tiles_to_trace,
    shim,
    trace_size,
    trace_offset,
	enable_token=0,
):
    for i in range(len(tiles_to_trace)):
        configure_core_packet_tracing_aie2(tiles_to_trace[i], shim, i+1, 15-i, trace_size, trace_offset, enable_token)
    configure_shim_packet_tracing_aie2(shim)

