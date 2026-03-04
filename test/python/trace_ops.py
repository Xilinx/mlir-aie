#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %python %s | FileCheck %s

from aie.dialects.aie import (
    AIEDevice,
    ComboLogic,
    DMAChannelDir,
    EdgeTrigger,
    TraceMode,
    TracePacketType,
    WireBundle,
    device,
    tile,
    trace,
    trace_combo_event,
    trace_edge_event,
    trace_event,
    trace_mode,
    trace_packet,
    trace_port,
    trace_start,
    trace_start_config,
    trace_stop,
)
from aie.dialects.aiex import runtime_sequence
from util import construct_and_print_module


# CHECK-LABEL: traceBasic
# CHECK: %[[TILE:.*]] = aie.tile(0, 2)
# CHECK: aie.trace @my_trace(%[[TILE]]) {
# CHECK:   aie.trace.mode "Event-Time"
# CHECK:   aie.trace.packet id = 1 type = core
# CHECK:   aie.trace.event <"INSTR_EVENT_0">
# CHECK:   aie.trace.event <"INSTR_VECTOR">
# CHECK:   aie.trace.start event = <"TRUE">
# CHECK:   aie.trace.stop event = <"NONE">
# CHECK: }
@construct_and_print_module
def traceBasic():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "my_trace")
        def body():
            trace_mode(TraceMode.EventTime)
            trace_packet(1, TracePacketType.Core)
            trace_event("INSTR_EVENT_0")
            trace_event("INSTR_VECTOR")
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: traceAllModes
# CHECK: aie.trace @trace_et
# CHECK:   aie.trace.mode "Event-Time"
# CHECK: aie.trace @trace_ep
# CHECK:   aie.trace.mode "Event-PC"
# CHECK: aie.trace @trace_ex
# CHECK:   aie.trace.mode Execution
@construct_and_print_module
def traceAllModes():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "trace_et")
        def body1():
            trace_mode(TraceMode.EventTime)
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        @trace(t, "trace_ep")
        def body2():
            trace_mode(TraceMode.EventPC)
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        @trace(t, "trace_ex")
        def body3():
            trace_mode(TraceMode.Execution)
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: tracePacketTypes
# CHECK: aie.trace.packet id = 1 type = core
# CHECK: aie.trace.packet id = 2 type = mem
# CHECK: aie.trace.packet id = 3 type = shimtile
# CHECK: aie.trace.packet id = 4 type = memtile
@construct_and_print_module
def tracePacketTypes():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "t1")
        def body1():
            trace_packet(1, TracePacketType.Core)
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        @trace(t, "t2")
        def body2():
            trace_packet(2, TracePacketType.Mem)
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        @trace(t, "t3")
        def body3():
            trace_packet(3, TracePacketType.ShimTile)
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        @trace(t, "t4")
        def body4():
            trace_packet(4, TracePacketType.MemTile)
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: traceEventWithLabel
# CHECK: aie.trace.event <"INSTR_EVENT_0"> label = "start_marker"
# CHECK: aie.trace.event <"INSTR_EVENT_1">
@construct_and_print_module
def traceEventWithLabel():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "my_trace")
        def body():
            trace_event("INSTR_EVENT_0", label="start_marker")
            trace_event("INSTR_EVENT_1")
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: tracePort
# CHECK: aie.trace.port<0> port = DMA channel = 0 direction = S2MM
# CHECK: aie.trace.port<1> port = DMA channel = 1 direction = MM2S
# CHECK: aie.trace.port<2> port = North channel = 0 direction = S2MM
@construct_and_print_module
def tracePort():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "my_trace")
        def body():
            trace_port(0, WireBundle.DMA, 0, DMAChannelDir.S2MM)
            trace_port(1, WireBundle.DMA, 1, DMAChannelDir.MM2S)
            trace_port(2, WireBundle.North, 0, DMAChannelDir.S2MM)
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: traceStartStopBroadcast
# CHECK: aie.trace.start broadcast = 15
# CHECK: aie.trace.stop broadcast = 14
@construct_and_print_module
def traceStartStopBroadcast():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "my_trace")
        def body():
            trace_start(broadcast=15)
            trace_stop(broadcast=14)


# CHECK-LABEL: traceStartStopEvent
# CHECK: aie.trace.start event = <"BROADCAST_15">
# CHECK: aie.trace.stop event = <"BROADCAST_14">
@construct_and_print_module
def traceStartStopEvent():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "my_trace")
        def body():
            trace_start(event="BROADCAST_15")
            trace_stop(event="BROADCAST_14")


# CHECK-LABEL: traceComboEvent
# CHECK: aie.trace.combo_event<0> <"LOCK_STALL"> AND <"STREAM_STALL">
# CHECK: aie.trace.combo_event<1> <"INSTR_EVENT_0"> OR <"INSTR_VECTOR">
# CHECK: aie.trace.combo_event<2> <"COMBO_EVENT_0"> AND_NOT <"COMBO_EVENT_1">
@construct_and_print_module
def traceComboEvent():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "my_trace")
        def body():
            trace_combo_event(0, "LOCK_STALL", ComboLogic.AND, "STREAM_STALL")
            trace_combo_event(1, "INSTR_EVENT_0", ComboLogic.OR, "INSTR_VECTOR")
            trace_combo_event(2, "COMBO_EVENT_0", ComboLogic.AND_NOT, "COMBO_EVENT_1")
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: traceEdgeEvent
# CHECK: aie.trace.edge_event<0> event = <"LOCK_STALL"> trigger = RISING
# CHECK: aie.trace.edge_event<1> event = <"DMA_S2MM_0_START_TASK"> trigger = BOTH
@construct_and_print_module
def traceEdgeEvent():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "my_trace")
        def body():
            trace_edge_event(0, "LOCK_STALL", EdgeTrigger.RISING)
            trace_edge_event(1, "DMA_S2MM_0_START_TASK", EdgeTrigger.BOTH)
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: traceStartConfig
# CHECK: aie.trace @cfg_trace
# CHECK: aie.runtime_sequence @seq()
# CHECK:   aie.trace.start_config @cfg_trace
@construct_and_print_module
def traceStartConfig():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "cfg_trace")
        def body():
            trace_mode(TraceMode.EventTime)
            trace_packet(1, TracePacketType.Core)
            trace_event("INSTR_EVENT_0")
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        @runtime_sequence()
        def seq():
            trace_start_config("cfg_trace")


# CHECK-LABEL: traceBufferSize
# CHECK: aie.trace @sized_trace(%{{.*}}) buffer_size = 8192
@construct_and_print_module
def traceBufferSize():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        @trace(t, "sized_trace", buffer_size=8192)
        def body():
            trace_event("INSTR_EVENT_0")
            trace_start(event="TRUE")
            trace_stop(event="NONE")


# CHECK-LABEL: traceFullExample
# CHECK: %[[T02:.*]] = aie.tile(0, 2)
# CHECK: %[[T00:.*]] = aie.tile(0, 0)
# CHECK: aie.trace @core_trace(%[[T02]]) {
# CHECK:   aie.trace.mode "Event-Time"
# CHECK:   aie.trace.packet id = 1 type = core
# CHECK:   aie.trace.event <"INSTR_EVENT_0">
# CHECK:   aie.trace.event <"MEMORY_STALL">
# CHECK:   aie.trace.port<0> port = DMA channel = 0 direction = S2MM
# CHECK:   aie.trace.start event = <"BROADCAST_15">
# CHECK:   aie.trace.stop event = <"BROADCAST_14">
# CHECK: }
# CHECK: aie.trace @shim_trace(%[[T00]]) {
# CHECK:   aie.trace.packet id = 2 type = shimtile
# CHECK:   aie.trace.event <"DMA_S2MM_0_START_TASK">
# CHECK:   aie.trace.start event = <"TRUE">
# CHECK:   aie.trace.stop event = <"NONE">
# CHECK: }
# CHECK: aie.runtime_sequence @seq() {
# CHECK:   aie.trace.start_config @core_trace
# CHECK:   aie.trace.start_config @shim_trace
# CHECK: }
@construct_and_print_module
def traceFullExample():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t02 = tile(0, 2)
        t00 = tile(0, 0)

        @trace(t02, "core_trace")
        def core_body():
            trace_mode(TraceMode.EventTime)
            trace_packet(1, TracePacketType.Core)
            trace_event("INSTR_EVENT_0")
            trace_event("MEMORY_STALL")
            trace_port(0, WireBundle.DMA, 0, DMAChannelDir.S2MM)
            trace_start(event="BROADCAST_15")
            trace_stop(event="BROADCAST_14")

        @trace(t00, "shim_trace")
        def shim_body():
            trace_packet(2, TracePacketType.ShimTile)
            trace_event("DMA_S2MM_0_START_TASK")
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        @runtime_sequence()
        def seq():
            trace_start_config("core_trace")
            trace_start_config("shim_trace")
