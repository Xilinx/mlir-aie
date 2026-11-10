//===- reorder_ctrl_packets_by_tile.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile | FileCheck %s
// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile=by-tile=false | FileCheck %s --check-prefix=NOSORT

// Each run of adjacent groups of one kind and device is sorted by tile
// (column, then row), each tile's packets in their order. MM2S queue pushes
// (core tile 0x1DE14, memory tile 0xA0634) follow every other packet of the
// run, and core enables (Core_Control 0x32000 with bit 0 set) come last; both
// form the enable phase, after an aiex.control_packet_barrier. A channel's
// control register (0x1DE10), a core disable and a core reset stay with the
// tile's other writes. A teardown group starts a run of its own, and control
// packets outside groups are left as they are.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-NEXT: aiex.control_packet_group @cfg(Config) {
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
// CHECK-NEXT: {{^ *}}}
// Queue pushes, by tile, then core enables, by tile.
// CHECK-NEXT: aiex.control_packet_barrier
// CHECK-NEXT: aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT: address = 1705524 : ui32, data = array<i32: 1>
// CHECK-NEXT: address = 2219540 : ui32, data = array<i32: 1>
// CHECK-NEXT: address = 2301952 : ui32, data = array<i32: 1>
// CHECK-NEXT: address = 35856384 : ui32, data = array<i32: 1>
// CHECK-NEXT: {{^ *}}}
// The teardown is a run of its own.
// CHECK-NEXT: aiex.control_packet_group @cfg(Teardown) {
// CHECK-NEXT: address = 2355204 : ui32, data = array<i32: 0>
// CHECK-NEXT: address = 35909636 : ui32, data = array<i32: 0>
// CHECK-NEXT: {{^ *}}}
// A packet outside groups stays where it is.
// CHECK-NEXT: address = 35909636 : ui32, data = array<i32: 7>
// CHECK-NEXT: address = 2355204 : ui32, data = array<i32: 7>
// CHECK-NOT:  aiex.control_packet

// NOSORT-LABEL: aie.runtime_sequence
// NOSORT-NEXT: aiex.control_packet_group @cfg(Config) {
// NOSORT-NEXT: address = 2301952 : ui32, data = array<i32: 0>
// NOSORT:      address = 1835008 : ui32, data = array<i32: 12>
// NOSORT-NEXT: {{^ *}}}
// NOSORT-NEXT: aiex.control_packet_barrier
// NOSORT-NEXT: aiex.control_packet_group @cfg(Config) {
// NOSORT-NEXT: address = 2219540 : ui32, data = array<i32: 1>
// NOSORT-NEXT: address = 1705524 : ui32, data = array<i32: 1>
// NOSORT-NEXT: address = 35856384 : ui32, data = array<i32: 1>
// NOSORT-NEXT: address = 2301952 : ui32, data = array<i32: 1>

aie.device(npu2) @cfg {
}

aie.device(npu2) {
  aie.runtime_sequence() {
    aiex.control_packet_group @cfg(Config) {
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
    }
    aiex.control_packet_group @cfg(Config) {
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
    }
    aiex.control_packet_group @cfg(Teardown) {
      aiex.control_packet {address = 35909636 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aiex.control_packet {address = 35909636 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2355204 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}
