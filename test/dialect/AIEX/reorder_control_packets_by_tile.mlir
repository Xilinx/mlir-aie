//===- reorder_control_packets_by_tile.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-reorder-control-packets-by-tile | FileCheck %s

// Each run of control packets is grouped by tile (column, then row), each
// tile's packets in their order. MM2S queue pushes (core tile 0x1DE14, memory
// tile 0xA0634) follow every other packet of the run, and core enables
// (Core_Control 0x32000 with bit 0 set) come last; both are stamped
// aiex.ctrl_pkt_enable_phase. These packets name no device, so no channel is proved
// gated. A channel's control register (0x1DE10), a core disable and a core
// reset stay with the tile's other writes. A non-packet op ends a run.

// CHECK-LABEL: aie.runtime_sequence
// Tile (0, 1) gets its switch, then its lock.
// CHECK-NEXT: address = 1769472 : ui32, data = array<i32: 11>
// CHECK-NEXT: address = 1835008 : ui32, data = array<i32: 12>
// Tile (0, 2) gets core disable, reset pulse, channel reset, switch, then BD.
// CHECK-NEXT: address = 2301952 : ui32, data = array<i32: 0>
// CHECK-NEXT: address = 2301952 : ui32, data = array<i32: 2>
// CHECK-NEXT: address = 2219536 : ui32, data = array<i32: 2>
// CHECK-NEXT: address = 2355204 : ui32, data = array<i32: 21>
// CHECK-NEXT: address = 2215936 : ui32, data = array<i32: 22>
// Tile (1, 2) gets its switch.
// CHECK-NEXT: address = 35909636 : ui32, data = array<i32: 31>
// Queue pushes, by tile.
// CHECK-NEXT: address = 1705524 : ui32, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// CHECK-NEXT: address = 2219540 : ui32, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// Core enables, by tile.
// CHECK-NEXT: address = 2301952 : ui32, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// CHECK-NEXT: address = 35856384 : ui32, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// The next run is reordered on its own.
// CHECK-NEXT: aiex.npu.load_pdi
// CHECK-NEXT: address = 2355204 : ui32, data = array<i32: 0>
// CHECK-NEXT: address = 35909636 : ui32, data = array<i32: 0>
// CHECK-NOT:  aiex.control_packet

aie.device(npu2) @cfg {
}

aie.device(npu2) {
  aie.runtime_sequence() {
    // (1, 2) core enable, ahead of everything.
    aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    // (0, 2) core disable and reset pulse.
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    // (0, 2) MM2S 0 control register reset.
    aiex.control_packet {address = 2219536 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    // (1, 2) switch.
    aiex.control_packet {address = 35909636 : ui32, data = array<i32: 31>, opcode = 0 : i32, stream_id = 0 : i32}
    // (0, 1) switch.
    aiex.control_packet {address = 1769472 : ui32, data = array<i32: 11>, opcode = 0 : i32, stream_id = 0 : i32}
    // (0, 2) switch and BD.
    aiex.control_packet {address = 2355204 : ui32, data = array<i32: 21>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2215936 : ui32, data = array<i32: 22>, opcode = 0 : i32, stream_id = 0 : i32}
    // (0, 2) MM2S 0 queue push.
    aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    // (0, 1) MM2S 0 queue push, then a lock.
    aiex.control_packet {address = 1705524 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 1835008 : ui32, data = array<i32: 12>, opcode = 0 : i32, stream_id = 0 : i32}
    // (0, 2) core enable.
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.npu.load_pdi {device_ref = @cfg}
    aiex.control_packet {address = 35909636 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355204 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}
