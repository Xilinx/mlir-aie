//===- reorder_ctrl_packets_gated.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile | FileCheck %s
// RUN: aie-opt %s -aie-reorder-ctrl-packets-by-tile=gated-enable=true | FileCheck %s --check-prefix=GATED

// A queue push to a channel aie-annotate-ctrl-packet-start-gating proved gated
// (the device of the packet's group marks that channel's aie.dma_start
// aiex.started_by) stays with its tile's writes. Pushes to other
// channels and core enables form the enable phase, after a barrier. The
// addresses are 1705524 and 1705532 for memory tile (0, 1)'s MM2S 0 and 1
// queues, 2219540 for core tile (0, 2)'s MM2S 0 queue, 2301952 (= 1) for
// (0, 2)'s core enable, and 35909636 for tile (1, 2)'s switch. @other's group
// is a run of its own, and a packet outside groups stays as it is.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-NEXT:  aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:    address = 1769472 : ui32, data = array<i32: 11>
// CHECK-NEXT:    address = 1705524 : ui32, data = array<i32: 1>
// CHECK-NEXT:    address = 2355204 : ui32, data = array<i32: 21>
// CHECK-NEXT:    address = 35909636 : ui32, data = array<i32: 31>
// CHECK-NEXT:  {{^ *}}}
// CHECK-NEXT:  aiex.control_packet_barrier
// CHECK-NEXT:  aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:    address = 1705532 : ui32, data = array<i32: 1>
// CHECK-NEXT:    address = 2219540 : ui32, data = array<i32: 2>
// CHECK-NEXT:    address = 2301952 : ui32, data = array<i32: 1>
// CHECK-NEXT:  {{^ *}}}
// CHECK-NEXT:  aiex.control_packet_group @other(Config) {
// CHECK-NEXT:    address = 1705524 : ui32, data = array<i32: 2>
// CHECK-NEXT:  {{^ *}}}
// CHECK-NEXT:  address = 2219540 : ui32, data = array<i32: 1>

// With gated-enable, the enable of core (0, 2), which @cfg marks gated, goes
// last among column 0's packets.
// GATED-LABEL: aie.runtime_sequence
// GATED-NEXT:  aiex.control_packet_group @cfg(Config) {
// GATED-NEXT:    address = 1769472 : ui32, data = array<i32: 11>
// GATED-NEXT:    address = 1705524 : ui32, data = array<i32: 1>
// GATED-NEXT:    address = 2355204 : ui32, data = array<i32: 21>
// GATED-NEXT:    address = 2301952 : ui32, data = array<i32: 1>
// GATED-NEXT:    address = 35909636 : ui32, data = array<i32: 31>
// GATED-NEXT:  {{^ *}}}
// GATED-NEXT:  aiex.control_packet_barrier
// GATED-NEXT:  aiex.control_packet_group @cfg(Config) {
// GATED-NEXT:    address = 1705532 : ui32, data = array<i32: 1>
// GATED-NEXT:    address = 2219540 : ui32, data = array<i32: 2>
// GATED-NEXT:  {{^ *}}}

module {
  aie.device(npu2) @cfg {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %dma = aie.memtile_dma(%t01) {
      %0 = aie.dma_start(MM2S, 0, ^end, ^next) {aiex.started_by = []}
    ^next:
      %1 = aie.dma_start(MM2S, 1, ^end, ^end)
    ^end:
      aie.end
    }
    %mem = aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^end, ^end)
    ^end:
      aie.end
    }
    %core = aie.core(%t02) {
      aie.end
    } {aiex.started_by = []}
  }

  aie.device(npu2) @other {
    %t01 = aie.tile(0, 1)
  }

  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2355204 : ui32, data = array<i32: 21>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2219540 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1769472 : ui32, data = array<i32: 11>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1705532 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1705524 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35909636 : ui32, data = array<i32: 31>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet_group @other(Config) {
        aiex.control_packet {address = 1705524 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
