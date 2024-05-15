# trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import typing
from aie.dialects.aiex import *
from aie.dialects.aie import get_target_model
from aie.utils.trace_events_enum import CoreEvent, MemEvent, PLEvent

class GenericEvent:
    def __init__(self, code: typing.Union[CoreEvent, MemEvent, PLEvent]):
        # For backwards compatibility, allow integer as event
        if isinstance(code, int):
            code = CoreEvent(code)
        self.code : typing.Union[CoreEvent, MemEvent, PLEvent] = code
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

class PortEvent(GenericEvent):
    def __init__(self, code, port_number, master=True):
        assert code in {CoreEvent.PORT_IDLE_0, CoreEvent.PORT_IDLE_1,
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
                        CoreEvent.PORT_TLAST_6, CoreEvent.PORT_TLAST_7}
        self.event_number = (0 if code in {CoreEvent.PORT_IDLE_0, CoreEvent.PORT_RUNNING_0, CoreEvent.PORT_STALLED_0, CoreEvent.PORT_TLAST_0} else 
                             1 if code in {CoreEvent.PORT_IDLE_1, CoreEvent.PORT_RUNNING_1, CoreEvent.PORT_STALLED_1, CoreEvent.PORT_TLAST_1} else 
                             2 if code in {CoreEvent.PORT_IDLE_2, CoreEvent.PORT_RUNNING_2, CoreEvent.PORT_STALLED_2, CoreEvent.PORT_TLAST_2} else 
                             3 if code in {CoreEvent.PORT_IDLE_3, CoreEvent.PORT_RUNNING_3, CoreEvent.PORT_STALLED_3, CoreEvent.PORT_TLAST_3} else 
                             4 if code in {CoreEvent.PORT_IDLE_4, CoreEvent.PORT_RUNNING_4, CoreEvent.PORT_STALLED_4, CoreEvent.PORT_TLAST_4} else 
                             5 if code in {CoreEvent.PORT_IDLE_5, CoreEvent.PORT_RUNNING_5, CoreEvent.PORT_STALLED_5, CoreEvent.PORT_TLAST_5} else 
                             6 if code in {CoreEvent.PORT_IDLE_6, CoreEvent.PORT_RUNNING_6, CoreEvent.PORT_STALLED_6, CoreEvent.PORT_TLAST_6} else 
                                          {CoreEvent.PORT_IDLE_7, CoreEvent.PORT_RUNNING_7, CoreEvent.PORT_STALLED_7, CoreEvent.PORT_TLAST_7})
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
        address = (0x3FF00 if self.event_number < 4 else
                   0x3FF04)
        value = (master(self.port_number) if self.master else 
                 slave(self.port_number))
        
        value =  (value & 0xFF) << 8*(self.event_number%4)

        return {address: value}


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
    events=[CoreEvent.INSTR_EVENT_1,
            CoreEvent.INSTR_EVENT_0,
            CoreEvent.INSTR_VECTOR,
            CoreEvent.INSTR_LOCK_RELEASE_REQ,
            CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
            CoreEvent.LOCK_STALL,
            PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
            PortEvent(CoreEvent.PORT_RUNNING_1, 1, False) # slave(1)
            ]
):
    dev = shim.parent.attributes["device"]
    tm = get_target_model(dev)

    # Shim has to be a shim tile
    assert tm.is_shim_noc_tile(shim.col, shim.row)

    # For backwards compatibility, allow integers for start/stop events
    if isinstance(start, int):
        start = CoreEvent(start)
    if isinstance(stop, int):
        stop = CoreEvent(stop)

    # Pad the input so we have exactly 8 events.
    events = (events + [CoreEvent.NONE] * 8)[:8]

    # Assure all selected events are valid
    events = [e if isinstance(e, GenericEvent)
              else GenericEvent(e)
              for e in events]

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
    # This is used to control packet routing.  For the moment
    # only deal with the simple case of circuit routing.
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340D4,
        value=0,
    )
    # 0x340E0: Trace Event Group 1  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340E0,
        value=pack4bytes(*(e.code.value for e in events[0:4])),
    )
    # 0x340E4: Trace Event Group 2  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    npu_write32(
        column=int(tile.col),
        row=int(tile.row),
        address=0x340E4,
        value=pack4bytes(*(e.code.value for e in events[4:8])),
    )

    # Event specific register writes
    all_reg_writes = {}
    for e in events:
        reg_writes = e.get_register_writes()
        for addr, value in reg_writes.items():
            if addr in all_reg_writes:
                all_reg_writes[addr] |= value
            else:
                all_reg_writes[addr] = value
    for addr, value in all_reg_writes.items():
        npu_write32(
            column=int(tile.col),
            row=int(tile.row),
            address=addr,
            value=value
        )

    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
    # out to host DDR memory
    npu_writebd(
        bd_id=bd_id,
        buffer_length=size,
        buffer_offset=offset,
        enable_packet=0,
        out_of_order_id=0,
        packet_id=0,
        packet_type=0,
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
        value=bd_id,
    )
